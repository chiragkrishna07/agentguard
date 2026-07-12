"""
HumanGate — async human-in-the-loop approval for high-risk actions.

IMPORTANT: This shield is async-only. Approval is delivered by an external task
calling ``approve()``/``deny()`` while the gated request awaits, which requires a
live event loop. Running it through ``Guard.protect_sync`` therefore raises
``HumanGateSyncError`` (the transient loop created per call could never receive
the approval); use the async ``@guard.protect`` instead.
"""

import asyncio
import fnmatch
import hashlib
import hmac
import json
import math
import secrets
import threading
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, Optional

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

if TYPE_CHECKING:
    from agentguard.notifiers.base import BaseNotifier


class HumanGate(BaseShield):
    """Block execution on matching triggers until a human approves or denies.

    Trigger formats
    ---------------
    ``"tool_call:<glob>"``   — fires when a tool name matches the glob pattern
    ``"cost_exceeds:<usd>"`` — fires when session cost exceeds the float value
    ``"pii_detected"``       — fire by setting ``ctx.metadata["pii_detected"] = True``
                               (PIIRedactor does this automatically)
    """

    scan_tool_arguments_as_input = False
    requires_async = True

    def __init__(
        self,
        triggers: list[str],
        notifier: Optional["BaseNotifier"] = None,
        timeout_seconds: float = 300,
        on_timeout: Literal["block", "allow"] = "block",
        *,
        include_param_values: bool = False,
        param_sanitizer: Callable[[dict[str, Any]], Any] | None = None,
        max_pending_gates: int = 1_000,
        identity_mode: Literal["hmac", "omit", "raw"] = "hmac",
        include_param_keys: bool = False,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be > 0")
        if on_timeout not in ("block", "allow"):
            raise ValueError("on_timeout must be 'block' or 'allow'")
        if not isinstance(include_param_values, bool):
            raise ValueError("include_param_values must be boolean")
        if param_sanitizer is not None and not callable(param_sanitizer):
            raise ValueError("param_sanitizer must be callable or None")
        if (
            isinstance(max_pending_gates, bool)
            or not isinstance(max_pending_gates, int)
            or max_pending_gates < 1
        ):
            raise ValueError("max_pending_gates must be >= 1")
        if identity_mode not in ("hmac", "omit", "raw"):
            raise ValueError("identity_mode must be 'hmac', 'omit', or 'raw'")
        if not isinstance(include_param_keys, bool):
            raise ValueError("include_param_keys must be boolean")
        self._validate_triggers(triggers)
        if notifier is None:
            from agentguard.notifiers.cli import CLINotifier

            notifier = CLINotifier()

        self.triggers: list[str] = triggers
        self.notifier: "BaseNotifier" = notifier
        self.timeout_seconds: float = timeout_seconds
        self.on_timeout: Literal["block", "allow"] = on_timeout
        self.include_param_values: bool = include_param_values
        self.param_sanitizer: Callable[[dict[str, Any]], Any] | None = param_sanitizer
        self.max_pending_gates: int = max_pending_gates
        self.identity_mode: Literal["hmac", "omit", "raw"] = identity_mode
        self.include_param_keys: bool = include_param_keys
        self._events: dict[
            str, tuple[asyncio.Event, asyncio.AbstractEventLoop]
        ] = {}
        self._decisions: dict[str, bool] = {}
        self._state_lock: threading.Lock = threading.Lock()
        # A keyed digest permits correlation inside this gate instance without
        # making low-entropy tool arguments brute-forceable from notifications.
        self._fingerprint_key: bytes = secrets.token_bytes(32)

    @staticmethod
    def _validate_triggers(triggers: list[str]) -> None:
        if not triggers:
            raise ValueError("triggers must not be empty")
        for trigger in triggers:
            if trigger == "pii_detected":
                continue
            if trigger.startswith("tool_call:") and trigger[len("tool_call:") :]:
                continue
            if trigger.startswith("cost_exceeds:"):
                try:
                    threshold = float(trigger[len("cost_exceeds:") :])
                except ValueError as exc:
                    raise ValueError(f"invalid HumanGate trigger: {trigger!r}") from exc
                if math.isfinite(threshold) and threshold >= 0:
                    continue
            raise ValueError(f"invalid HumanGate trigger: {trigger!r}")

    # ------------------------------------------------------------------ #
    # Trigger matching                                                      #
    # ------------------------------------------------------------------ #

    def _tool_triggered(self, tool_name: str) -> bool:
        for t in self.triggers:
            if t.startswith("tool_call:"):
                if fnmatch.fnmatchcase(tool_name.casefold(), t[len("tool_call:") :].casefold()):
                    return True
        return False

    def _cost_triggered(self, ctx: SessionContext) -> bool:
        for t in self.triggers:
            if t.startswith("cost_exceeds:"):
                try:
                    threshold = float(t[len("cost_exceeds:") :])
                    if ctx.cost_usd > threshold:
                        return True
                except ValueError:
                    pass
        return False

    def _pii_triggered(self, ctx: SessionContext) -> bool:
        return "pii_detected" in self.triggers and bool(ctx.metadata.get("pii_detected"))

    # ------------------------------------------------------------------ #
    # Approval workflow                                                    #
    # ------------------------------------------------------------------ #

    async def _await_decision(self, gate_id: str, context: dict) -> bool:
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if len(self._events) >= self.max_pending_gates:
                return False
            self._events[gate_id] = (event, loop)
        try:
            await self.notifier.notify(gate_id, context)
            try:
                await asyncio.wait_for(event.wait(), timeout=self.timeout_seconds)
                return self._decisions.get(gate_id, False)
            except asyncio.TimeoutError:
                return self.on_timeout == "allow"
        finally:
            with self._state_lock:
                self._events.pop(gate_id, None)
                self._decisions.pop(gate_id, None)

    async def _decide(self, gate_id: str, approved: bool) -> bool:
        with self._state_lock:
            pending = self._events.get(gate_id)
            if pending is None:
                return False
            self._decisions[gate_id] = approved
        event, owner_loop = pending
        if owner_loop is asyncio.get_running_loop():
            event.set()
        elif owner_loop.is_running():
            owner_loop.call_soon_threadsafe(event.set)
        else:
            with self._state_lock:
                self._events.pop(gate_id, None)
                self._decisions.pop(gate_id, None)
            return False
        return True

    async def approve(self, gate_id: str) -> bool:
        """Approve an active gate; return false for unknown/expired IDs."""
        return await self._decide(gate_id, True)

    async def deny(self, gate_id: str) -> bool:
        """Deny an active gate; return false for unknown/expired IDs."""
        return await self._decide(gate_id, False)

    def _notification_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.param_sanitizer is not None:
            return {"sanitized_params": self.param_sanitizer(params)}
        if self.include_param_values:
            return {"params": params}
        try:
            encoded = json.dumps(
                params,
                sort_keys=True,
                separators=(",", ":"),
                default=lambda value: f"<{type(value).__name__}>",
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            encoded = b"<unserializable>"
        fingerprint = hmac.new(self._fingerprint_key, encoded, hashlib.sha256).hexdigest()[:16]
        summary: dict[str, Any] = {
            "param_count": len(params),
            "params_fingerprint": fingerprint,
        }
        if self.include_param_keys:
            summary["param_keys"] = sorted(str(key)[:128] for key in params)[:100]
        return summary

    def _session_identity(self, ctx: SessionContext) -> dict[str, str]:
        if self.identity_mode == "omit":
            return {}
        if self.identity_mode == "raw":
            return {"session_id": ctx.session_id}
        digest = hmac.new(
            self._fingerprint_key,
            f"session\0{ctx.session_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        return {"session_id": f"hmac:{digest}"}

    # ------------------------------------------------------------------ #
    # Shield hooks                                                         #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        if not (self._cost_triggered(ctx) or self._pii_triggered(ctx)):
            return ShieldResult(allowed=True)

        gate_id = f"gate-{secrets.token_urlsafe(24)}"
        approved = await self._await_decision(
            gate_id,
            {
                "type": "input",
                "reason": "cost_exceeds or pii_detected trigger",
                "cost_so_far": round(ctx.cost_usd, 4),
                **self._session_identity(ctx),
            },
        )
        if not approved:
            return ShieldResult(
                allowed=False,
                reason="Human approval denied or timed out",
                reason_code="HUMAN_GATE_DENIED",
            )
        return ShieldResult(allowed=True)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: SessionContext
    ) -> ShieldResult:
        if not self._tool_triggered(tool_name):
            return ShieldResult(allowed=True)

        gate_id = f"gate-{secrets.token_urlsafe(24)}"
        approved = await self._await_decision(
            gate_id,
            {
                "type": "tool_call",
                "tool_name": tool_name,
                **self._session_identity(ctx),
                **self._notification_params(params),
            },
        )
        if not approved:
            return ShieldResult(
                allowed=False,
                reason=f"Human approval required for '{tool_name}' — denied or timed out",
                reason_code="HUMAN_GATE_DENIED",
            )
        return ShieldResult(allowed=True)
