import asyncio
import functools
from collections.abc import Callable
from typing import Any

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.exceptions import (
    GuardBlockedError,
    GuardShieldError,
    HumanGateSyncError,
)
from agentguard.core.metrics import GuardMetrics
from agentguard.core.session import SessionContext


class Guard:
    def __init__(self, shields: list[BaseShield] | None = None) -> None:
        self.shields = shields or []
        self.metrics = GuardMetrics()

    @classmethod
    def from_dict(cls, config: dict) -> "Guard":
        """Build a Guard from a plain dict (e.g. parsed YAML/JSON).

        ::

            Guard.from_dict({"shields": [
                {"type": "PromptShield", "mode": "strict"},
                {"type": "SecretsShield", "on_detect": "redact"},
                {"type": "CostLimit", "max_usd": 5.0},
            ]})

        Each entry's ``type`` names a shield exported from the ``agentguard``
        package; the remaining keys are passed as constructor kwargs.
        """
        import agentguard as _ag

        shields: list[BaseShield] = []
        for raw in config.get("shields", []):
            entry = dict(raw)
            try:
                name = entry.pop("type")
            except KeyError as exc:
                raise ValueError("each shield entry needs a 'type'") from exc
            shield_cls = getattr(_ag, name, None)
            if not (isinstance(shield_cls, type) and issubclass(shield_cls, BaseShield)):
                raise ValueError(f"unknown shield type: {name!r}")
            try:
                shields.append(shield_cls(**entry))
            except TypeError as exc:
                raise ValueError(f"bad config for shield {name!r}: {exc}") from exc
        return cls(shields=shields)

    def stats(self) -> dict:
        """Return a snapshot of scan/block counters for monitoring."""
        return self.metrics.snapshot()

    def _raise_block(
        self,
        shield: BaseShield,
        result: ShieldResult,
        default_reason: str,
        default_code: str,
    ) -> None:
        code = result.reason_code or default_code
        self.metrics.record_block(shield.__class__.__name__, code)
        raise GuardBlockedError(
            result.reason or default_reason, code, shield.__class__.__name__
        )

    # ------------------------------------------------------------------ #
    # Decorators                                                           #
    # ------------------------------------------------------------------ #

    def protect(self, fn: Callable) -> Callable:
        """Decorator for async agent functions."""
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@guard.protect requires an async function. "
                "Use @guard.protect_sync for sync functions."
            )

        @functools.wraps(fn)
        async def wrapper(
            query: str,
            *args: Any,
            _guard_ctx: SessionContext | None = None,
            **kwargs: Any,
        ) -> Any:
            ctx = _guard_ctx or SessionContext()
            sanitized = await self._scan_input(query, ctx)
            result = await fn(sanitized, *args, **kwargs)
            return await self._scan_output(str(result), ctx)

        return wrapper

    def protect_sync(self, fn: Callable) -> Callable:
        """Decorator for sync agent functions."""
        if asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@guard.protect_sync requires a sync function. "
                "Use @guard.protect for async functions."
            )

        async_only = [s for s in self.shields if getattr(s, "requires_async", False)]
        if async_only:
            names = ", ".join(sorted(s.__class__.__name__ for s in async_only))
            raise HumanGateSyncError(
                f"{names} require a live event loop and cannot run under "
                "@guard.protect_sync (the per-call loop could never receive the "
                "approval). Use the async @guard.protect instead."
            )

        @functools.wraps(fn)
        def wrapper(
            query: str,
            *args: Any,
            _guard_ctx: SessionContext | None = None,
            **kwargs: Any,
        ) -> Any:
            ctx = _guard_ctx or SessionContext()
            sanitized = asyncio.run(self._scan_input(query, ctx))
            result = fn(sanitized, *args, **kwargs)
            return asyncio.run(self._scan_output(str(result), ctx))

        return wrapper

    # ------------------------------------------------------------------ #
    # Explicit run                                                         #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        agent_fn: Callable,
        query: str,
        ctx: SessionContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent through all shields. Alternative to the decorator."""
        ctx = ctx or SessionContext()
        sanitized = await self._scan_input(query, ctx)

        if asyncio.iscoroutinefunction(agent_fn):
            result = await agent_fn(sanitized, **kwargs)
        else:
            result = agent_fn(sanitized, **kwargs)

        return await self._scan_output(str(result), ctx)

    async def scan_tool_call(
        self,
        tool_name: str,
        params: dict,
        ctx: SessionContext | None = None,
    ) -> ShieldResult:
        """Scan a tool call through all shields. Called by GuardedTool."""
        ctx = ctx or SessionContext()
        self.metrics.record_scan("tool_calls")
        for shield in self.shields:
            try:
                result = await shield.scan_tool_call(tool_name, params, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                self._raise_block(shield, result, "Tool call blocked", "TOOL_BLOCKED")
        return ShieldResult(allowed=True)

    async def scan_tool_output(
        self,
        tool_name: str,
        output: str,
        ctx: SessionContext | None = None,
    ) -> str:
        """Scan content returned by a tool before it re-enters the agent.

        This is the indirect-prompt-injection chokepoint: tool results and
        retrieved documents are attacker-controlled and must be inspected just
        like user input. Returns the (possibly sanitised) output. Called by
        GuardedTool after the wrapped tool returns.
        """
        ctx = ctx or SessionContext()
        self.metrics.record_scan("tool_outputs")
        current = output
        for shield in self.shields:
            try:
                result = await shield.scan_tool_output(tool_name, current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                self._raise_block(
                    shield, result, "Tool output blocked", "TOOL_OUTPUT_BLOCKED"
                )
            if result.modified_input is not None:
                current = result.modified_input
        return current

    # ------------------------------------------------------------------ #
    # Internal scan pipelines                                              #
    # ------------------------------------------------------------------ #

    async def _scan_input(self, text: str, ctx: SessionContext) -> str:
        self.metrics.record_scan("inputs")
        current = text
        for shield in self.shields:
            try:
                result = await shield.scan_input(current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                self._raise_block(shield, result, "Input blocked", "INPUT_BLOCKED")
            if result.modified_input is not None:
                current = result.modified_input
        ctx.request_count += 1
        return current

    async def _scan_output(self, text: str, ctx: SessionContext) -> str:
        self.metrics.record_scan("outputs")
        current = text
        for shield in self.shields:
            try:
                result = await shield.scan_output(current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                self._raise_block(shield, result, "Output blocked", "OUTPUT_BLOCKED")
            if result.modified_input is not None:
                current = result.modified_input
        return current
