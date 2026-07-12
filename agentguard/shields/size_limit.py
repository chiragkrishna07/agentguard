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
    max_input_bytes / max_output_bytes / max_tool_output_bytes:
        Optional UTF-8 byte ceilings for transport/storage protection. Byte
        limits matter when a small character count can still encode to a much
        larger payload.
    on_exceed:
        ``"block"`` (default) rejects the request; ``"truncate"`` clips the text
        to the limit (and warns) so the agent keeps running on a bounded slice.
    """

    structured_values_only = True

    def __init__(
        self,
        max_input_chars: int | None = 20_000,
        max_output_chars: int | None = None,
        max_tool_output_chars: int | None = 20_000,
        on_exceed: Literal["block", "truncate"] = "block",
        *,
        max_input_bytes: int | None = None,
        max_output_bytes: int | None = None,
        max_tool_output_bytes: int | None = None,
    ) -> None:
        if on_exceed not in ("block", "truncate"):
            raise ValueError("on_exceed must be 'block' or 'truncate'")
        limits = {
            "max_input_chars": max_input_chars,
            "max_output_chars": max_output_chars,
            "max_tool_output_chars": max_tool_output_chars,
            "max_input_bytes": max_input_bytes,
            "max_output_bytes": max_output_bytes,
            "max_tool_output_bytes": max_tool_output_bytes,
        }
        for name, value in limits.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be >= 0 or None")
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.max_tool_output_chars = max_tool_output_chars
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_tool_output_bytes = max_tool_output_bytes
        self.on_exceed = on_exceed

    def _check(
        self,
        text: str,
        char_limit: int | None,
        byte_limit: int | None,
        flow: str,
    ) -> ShieldResult:
        char_exceeded = char_limit is not None and len(text) > char_limit
        encoded = text.encode("utf-8")
        byte_exceeded = byte_limit is not None and len(encoded) > byte_limit
        if not char_exceeded and not byte_exceeded:
            return ShieldResult(allowed=True)
        details = []
        if char_exceeded:
            details.append(f"{len(text)} chars exceeds {char_limit}")
        if byte_exceeded:
            details.append(f"{len(encoded)} UTF-8 bytes exceeds {byte_limit}")
        msg = f"{flow} size " + "; ".join(details)
        if self.on_exceed == "truncate":
            warnings.warn(f"[AgentGuard SizeLimit] {msg}; truncating", stacklevel=4)
            truncated = text[:char_limit] if char_exceeded and char_limit is not None else text
            if byte_exceeded and byte_limit is not None:
                truncated = truncated.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")
            return ShieldResult(allowed=True, modified_input=truncated)
        return ShieldResult(allowed=False, reason=msg, reason_code="SIZE_LIMIT_EXCEEDED")

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._check(text, self.max_input_chars, self.max_input_bytes, "Input")

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._check(text, self.max_output_chars, self.max_output_bytes, "Output")

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        return self._check(
            output,
            self.max_tool_output_chars,
            self.max_tool_output_bytes,
            f"Tool '{tool_name}' output",
        )
