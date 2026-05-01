from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agentguard.core.session import SessionContext


@dataclass
class ShieldResult:
    allowed: bool
    modified_input: Optional[str] = None
    reason: Optional[str] = None
    reason_code: Optional[str] = None


class BaseShield(ABC):
    async def scan_input(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: "SessionContext"
    ) -> ShieldResult:
        return ShieldResult(allowed=True)
