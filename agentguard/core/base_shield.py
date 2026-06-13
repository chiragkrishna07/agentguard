from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentguard.core.session import SessionContext


@dataclass
class ShieldResult:
    allowed: bool
    modified_input: str | None = None
    reason: str | None = None
    reason_code: str | None = None


class BaseShield(ABC):
    async def scan_input(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: "SessionContext"
    ) -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: "SessionContext"
    ) -> ShieldResult:
        """Scan content returned by a tool before it re-enters the agent.

        This is where indirect prompt injection lives: a tool or retrieval step
        returns attacker-controlled text (a web page, an email, a document)
        containing hidden instructions. Shields override this to block or
        sanitise that content. ``modified_input`` rewrites the tool output.
        """
        return ShieldResult(allowed=True)
