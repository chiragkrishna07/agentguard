from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agentguard.core.session import SessionContext

GuardFlow = Literal["input", "output", "tool_call", "tool_output"]


@dataclass
class ShieldResult:
    allowed: bool
    modified_input: str | None = None
    reason: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class GuardDecision:
    """Content-free summary emitted after a guard pipeline makes a decision.

    Observer shields can use this hook for audit/telemetry without receiving
    the raw input, output, tool arguments, or tool result.  New fields can be
    added compatibly as AgentGuard grows; observers should only depend on the
    fields they need.
    """

    flow: GuardFlow
    allowed: bool
    shield_name: str | None = None
    reason_code: str | None = None
    tool_name: str | None = None


class BaseShield(ABC):
    # Shields that need a live, externally-driven event loop (e.g. HumanGate,
    # which awaits an approval delivered by another task) set this True. The
    # sync entry point refuses to run them — see Guard.protect_sync.
    requires_async: bool = False
    # Content shields normally apply their input policy to model-generated tool
    # arguments as well. Operational shields (rate/cost/audit/human approval)
    # opt out so a single request is not counted or approved twice.
    scan_tool_arguments_as_input: bool = True
    # Opt in when detection needs a duplicated ``key=value`` view in addition
    # to primary values (for contextual DLP/injection rules). Accounting and
    # observer shields keep the default to avoid double-counting content.
    needs_structured_context: bool = False
    # Rewriters such as SizeLimit truncation may opt into string values only;
    # protected key/scalar context cannot be safely truncated.
    structured_values_only: bool = False
    # Tool-boundary controls whose state is meaningless when a new anonymous
    # SessionContext is created for every call opt in here. GuardedTool then
    # requires an explicit wrapper or per-call context rather than silently
    # disabling the intended session budget.
    requires_tool_session_context: bool = False
    @property
    def requires_buffered_output(self) -> bool:
        """Whether streaming must buffer the complete output for this policy."""
        return False

    def select_structured_context_key(self, key_path: tuple[str, ...]) -> str | None:
        """Choose the schema key paired with a nested value for detection.

        The nearest key is the general default. DLP shields may override this
        to retain a security-relevant ancestor such as ``api_key`` or
        ``passport_number`` through arbitrary wrapper objects. Only one
        context duplicate is emitted per value, keeping traversal linear.
        """
        return key_path[-1] if key_path else None

    async def scan_input(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: "SessionContext") -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_output_preview(self, text: str, ctx: "SessionContext") -> ShieldResult:
        """Inspect a provisional cumulative streaming prefix.

        The default reuses normal output policy. Stateful accounting/audit
        shields override this hook so incremental streaming can re-scan a
        growing prefix without double charging or logging it as final output.
        """
        return await self.scan_output(text, ctx)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: "SessionContext"
    ) -> ShieldResult:
        return ShieldResult(allowed=True)

    async def scan_tool_arguments(
        self, tool_name: str, text: str, ctx: "SessionContext"
    ) -> ShieldResult:
        """Sanitize text contained in model-generated tool arguments.

        By default this reuses the shield's input content policy.  Shields that
        account for whole requests or perform human approval set
        ``scan_tool_arguments_as_input = False`` and still participate in the
        separate :meth:`scan_tool_call` policy phase.
        """
        if not self.scan_tool_arguments_as_input:
            return ShieldResult(allowed=True)
        return await self.scan_input(text, ctx)

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

    async def on_decision(self, decision: GuardDecision, ctx: "SessionContext") -> None:
        """Observe a final guard decision without access to raw content.

        The default is deliberately a no-op, so existing third-party shields
        remain source- and runtime-compatible.  Audit/metrics shields can
        override this to record blocks made by shields later in the pipeline.
        """

    async def on_input_committed(self, text: str, ctx: "SessionContext") -> None:
        """Commit state only after the complete input pipeline succeeds.

        ``text`` contains the final sanitized string-value view. Stateful
        shields must use this hook rather than retaining raw text during
        :meth:`scan_input`, because a later shield may still block the flow.
        """
