"""
SizeLimit — cap the size of text flowing through the agent.

Oversized input is a cheap denial-of-wallet / context-stuffing vector: a giant
payload inflates token cost and can bury a jailbreak deep in the context.
Oversized tool output is the same problem from the retrieval side. This shield
bounds each flow, either blocking or truncating.
"""
import warnings
from typing import Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class SizeLimit(BaseShield):
    """Bound input / output / tool-output length in characters.

    Parameters
    ----------
    max_input_chars / max_output_chars / max_tool_output_chars:
        Per-flow ceilings. ``None`` disables that flow's check.
    on_exceed:
        ``"block"`` (default) rejects the request; ``"truncate"`` clips the text
        to the limit (and warns) so the agent keeps running on a bounded slice.
    """

    def __init__(
        self,
        max_input_chars: int | None = 20_000,
        max_output_chars: int | None = None,
        max_tool_output_chars: int | None = 20_000,
        on_exceed: Literal["block", "truncate"] = "block",
    ) -> None:
        if on_exceed not in ("block", "truncate"):
            raise ValueError("on_exceed must be 'block' or 'truncate'")
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.max_tool_output_chars = max_tool_output_chars
        self.on_exceed = on_exceed

    def _check(self, text: str, limit: int | None, flow: str) -> ShieldResult:
        if limit is None or len(text) <= limit:
            return ShieldResult(allowed=True)
        msg = f"{flow} length {len(text)} exceeds limit {limit}"
        if self.on_exceed == "truncate":
            warnings.warn(f"[AgentGuard SizeLimit] {msg}; truncating", stacklevel=4)
            return ShieldResult(allowed=True, modified_input=text[:limit])
        return ShieldResult(allowed=False, reason=msg, reason_code="SIZE_LIMIT_EXCEEDED")

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._check(text, self.max_input_chars, "Input")

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._check(text, self.max_output_chars, "Output")

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        return self._check(output, self.max_tool_output_chars, f"Tool '{tool_name}' output")
