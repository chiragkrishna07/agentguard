"""
HumanGate — async human-in-the-loop approval for high-risk actions.

IMPORTANT: This shield is async-only. Using it from a synchronous context
will raise HumanGateSyncError.
"""
import asyncio
import fnmatch
import uuid
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.exceptions import HumanGateSyncError
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

    def __init__(
        self,
        triggers: List[str],
        notifier: Optional["BaseNotifier"] = None,
        timeout_seconds: int = 300,
        on_timeout: Literal["block", "allow"] = "block",
    ) -> None:
        if notifier is None:
            from agentguard.notifiers.cli import CLINotifier
            notifier = CLINotifier()

        self.triggers = triggers
        self.notifier = notifier
        self.timeout_seconds = timeout_seconds
        self.on_timeout = on_timeout
        self._events: Dict[str, asyncio.Event] = {}
        self._decisions: Dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Trigger matching                                                      #
    # ------------------------------------------------------------------ #

    def _tool_triggered(self, tool_name: str) -> bool:
        for t in self.triggers:
            if t.startswith("tool_call:"):
                if fnmatch.fnmatch(tool_name, t[len("tool_call:"):]):
                    return True
        return False

    def _cost_triggered(self, ctx: SessionContext) -> bool:
        for t in self.triggers:
            if t.startswith("cost_exceeds:"):
                try:
                    threshold = float(t[len("cost_exceeds:"):])
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
        self._events[gate_id] = event
        try:
            await self.notifier.notify(gate_id, context)
            try:
                await asyncio.wait_for(event.wait(), timeout=self.timeout_seconds)
                return self._decisions.get(gate_id, False)
            except asyncio.TimeoutError:
                return self.on_timeout == "allow"
        finally:
            self._events.pop(gate_id, None)
            self._decisions.pop(gate_id, None)

    async def approve(self, gate_id: str) -> None:
        """Call this from your notifier callback to approve a gate."""
        self._decisions[gate_id] = True
        if gate_id in self._events:
            self._events[gate_id].set()

    async def deny(self, gate_id: str) -> None:
        """Call this from your notifier callback to deny a gate."""
        self._decisions[gate_id] = False
        if gate_id in self._events:
            self._events[gate_id].set()

    # ------------------------------------------------------------------ #
    # Shield hooks                                                         #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        if not (self._cost_triggered(ctx) or self._pii_triggered(ctx)):
            return ShieldResult(allowed=True)

        gate_id = f"gate-{uuid.uuid4().hex[:8]}"
        approved = await self._await_decision(
            gate_id,
            {
                "type": "input",
                "reason": "cost_exceeds or pii_detected trigger",
                "session_id": ctx.session_id,
                "cost_so_far": round(ctx.cost_usd, 4),
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

        gate_id = f"gate-{uuid.uuid4().hex[:8]}"
        approved = await self._await_decision(
            gate_id,
            {
                "type": "tool_call",
                "tool_name": tool_name,
                "params": str(params),
                "session_id": ctx.session_id,
            },
        )
        if not approved:
            return ShieldResult(
                allowed=False,
                reason=f"Human approval required for '{tool_name}' — denied or timed out",
                reason_code="HUMAN_GATE_DENIED",
            )
        return ShieldResult(allowed=True)
