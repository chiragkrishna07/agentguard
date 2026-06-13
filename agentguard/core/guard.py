import asyncio
import functools
from collections.abc import Callable
from typing import Any

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.exceptions import GuardBlockedError, GuardShieldError
from agentguard.core.session import SessionContext


class Guard:
    def __init__(self, shields: list[BaseShield] | None = None) -> None:
        self.shields = shields or []

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
        for shield in self.shields:
            try:
                result = await shield.scan_tool_call(tool_name, params, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                raise GuardBlockedError(
                    result.reason or "Tool call blocked",
                    result.reason_code or "TOOL_BLOCKED",
                    shield.__class__.__name__,
                )
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
        current = output
        for shield in self.shields:
            try:
                result = await shield.scan_tool_output(tool_name, current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                raise GuardBlockedError(
                    result.reason or "Tool output blocked",
                    result.reason_code or "TOOL_OUTPUT_BLOCKED",
                    shield.__class__.__name__,
                )
            if result.modified_input is not None:
                current = result.modified_input
        return current

    # ------------------------------------------------------------------ #
    # Internal scan pipelines                                              #
    # ------------------------------------------------------------------ #

    async def _scan_input(self, text: str, ctx: SessionContext) -> str:
        current = text
        for shield in self.shields:
            try:
                result = await shield.scan_input(current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                raise GuardBlockedError(
                    result.reason or "Input blocked",
                    result.reason_code or "INPUT_BLOCKED",
                    shield.__class__.__name__,
                )
            if result.modified_input is not None:
                current = result.modified_input
        ctx.request_count += 1
        return current

    async def _scan_output(self, text: str, ctx: SessionContext) -> str:
        current = text
        for shield in self.shields:
            try:
                result = await shield.scan_output(current, ctx)
            except GuardBlockedError:
                raise
            except Exception as exc:
                raise GuardShieldError(shield.__class__.__name__, str(exc)) from exc
            if not result.allowed:
                raise GuardBlockedError(
                    result.reason or "Output blocked",
                    result.reason_code or "OUTPUT_BLOCKED",
                    shield.__class__.__name__,
                )
            if result.modified_input is not None:
                current = result.modified_input
        return current
