"""
Ready-made shield stacks.

Most teams don't want to hand-pick and tune every shield on day one — they want
a sensible default that's safe in production. These factories return a `Guard`
wired with a curated stack. Pass overrides through ``**kwargs`` or rebuild from
the individual shields when you outgrow them.

    from agentguard.presets import recommended

    guard = recommended(max_usd=10.0)

    @guard.protect
    async def agent(query: str) -> str:
        ...
"""

from agentguard.core.guard import Guard
from agentguard.shields.audit_logger import AuditLogger
from agentguard.shields.circuit_breaker import CircuitBreaker
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.dangerous_command import DangerousCommandShield
from agentguard.shields.memory_policy import MemoryPolicyShield
from agentguard.shields.network_policy import NetworkPolicyShield
from agentguard.shields.pii_redactor import PIIRedactor
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.rate_limit import RateLimit
from agentguard.shields.secrets import SecretsShield
from agentguard.shields.size_limit import SizeLimit
from agentguard.shields.tool_budget import ToolCallBudget
from agentguard.shields.tool_integrity import ToolIntegrityShield
from agentguard.shields.tool_validator import ToolValidator


def minimal() -> Guard:
    """Just two local essentials: injection and secret defense.

    No Presidio, moderation service, or ML model download is required.
    """
    return Guard(
        shields=[
            PromptShield(mode="strict", use_canary=False),
            SecretsShield(on_detect="redact"),
        ]
    )


def recommended(
    max_usd: float | None = 25.0,
    model: str = "gpt-4o",
    audit: bool = True,
    *,
    requests_per_minute: int | None = 60,
    network_policy: bool = True,
    tool_budget: bool = True,
    circuit_breaker: bool = True,
    tool_integrity: bool = True,
    memory_policy: bool = True,
    dangerous_commands: bool = True,
) -> Guard:
    """A balanced production stack covering the full request lifecycle.

    Covers bounded input/output, request rate, direct/indirect injection,
    secret/PII redaction, tool definition integrity, tool argument validation,
    destructive tool arguments, runaway tool loops, public HTTPS egress,
    durable-memory policy, containment after repeated denials, an optional cost
    ceiling, and a privacy-preserving audit trail. Applications must still
    configure explicit tool/domain allowlists and real authorization for their
    own capabilities.

    ``DangerousCommandShield`` runs here in blocklist mode, which catches known
    destructive shapes but cannot enumerate every one. If your agent drives a
    shell or a database, pass ``allowed_commands=[...]`` to a hand-built stack
    instead — an allowlist is the control that actually holds.

    ``CircuitBreaker`` leads the stack so a contained session is denied before
    any other work happens. It is given a cooldown here rather than requiring an
    explicit reset, because a preset is used by deployments that may have no
    operator watching for a wedged breaker. Pass ``circuit_breaker=False`` to
    drop it, or build the stack by hand for a stricter no-auto-rearm policy.
    """
    shields: list = []
    if circuit_breaker:
        shields.append(
            CircuitBreaker(max_blocks=10, window_seconds=300.0, cooldown_seconds=300.0)
        )
    shields.append(
        SizeLimit(
            max_input_chars=20_000,
            max_output_chars=50_000,
            max_tool_output_chars=20_000,
            max_input_bytes=80_000,
            max_output_bytes=200_000,
            max_tool_output_bytes=80_000,
        )
    )
    if requests_per_minute is not None:
        shields.append(
            RateLimit(
                requests_per_minute=requests_per_minute,
                per="session",
                burst=min(10, requests_per_minute),
            )
        )
    shields.extend(
        [
            PromptShield(mode="strict", on_indirect="block"),
            SecretsShield(on_detect="redact"),
            PIIRedactor(mode="redact", redact_output=True, scan_tool_output=True),
        ]
    )
    if memory_policy:
        shields.append(MemoryPolicyShield())
    if tool_integrity:
        shields.append(ToolIntegrityShield())
    shields.append(ToolValidator())
    if dangerous_commands:
        shields.append(DangerousCommandShield())
    if tool_budget:
        shields.append(ToolCallBudget())
    if network_policy:
        shields.append(NetworkPolicyShield())
    if max_usd is not None:
        shields.append(CostLimit(max_usd=max_usd, model=model))
    if audit:
        shields.append(AuditLogger(output="stdout"))
    return Guard(shields=shields)


def paranoid(
    max_usd: float | None = 25.0,
    model: str = "gpt-4o",
    *,
    requests_per_minute: int = 30,
) -> Guard:
    """Maximum scrutiny: paranoid injection mode, secrets hard-blocked, PII
    redacted everywhere, tool definitions pinned and calls restricted to
    registered tools, durable memory held to a declared origin, containment
    latched until an operator resets it, cost-capped, fully audited. Expect more
    false positives — use when the cost of a miss dwarfs the cost of a block.

    Two settings here need a deployment decision rather than a default:
    ``CircuitBreaker`` has no cooldown, so a trip holds until
    :meth:`~agentguard.CircuitBreaker.reset`; and ``require_registration=True``
    means every tool must pass through
    :meth:`~agentguard.Guard.scan_tool_definitions` before it can be called.

    ``DangerousCommandShield`` is still a blocklist here, because the allowlist
    it would rather enforce is the set of commands *your* agent legitimately
    runs, and only you know that set. Supplying ``allowed_commands`` is the
    single highest-value change you can make to this stack.
    """
    shields: list = [
        CircuitBreaker(max_blocks=5, window_seconds=300.0, cooldown_seconds=None),
        SizeLimit(
            max_input_chars=10_000,
            max_output_chars=25_000,
            max_tool_output_chars=10_000,
            max_input_bytes=40_000,
            max_output_bytes=100_000,
            max_tool_output_bytes=40_000,
        ),
        RateLimit(
            requests_per_minute=requests_per_minute,
            per="session",
            burst=min(5, requests_per_minute),
        ),
        PromptShield(mode="paranoid", on_indirect="block"),
        SecretsShield(on_detect="block"),
        PIIRedactor(mode="redact", redact_output=True, scan_tool_output=True),
        MemoryPolicyShield(
            require_origin=True,
            max_writes_per_session=100,
            max_chars_per_session=100_000,
        ),
        ToolIntegrityShield(require_registration=True),
        ToolValidator(),
        DangerousCommandShield(require_sql_where=True),
        ToolCallBudget(
            max_calls_per_session=50,
            max_calls_per_tool=20,
            max_distinct_tools=15,
            max_consecutive_identical=3,
        ),
        NetworkPolicyShield(),
    ]
    if max_usd is not None:
        shields.append(CostLimit(max_usd=max_usd, model=model, on_limit="block"))
    shields.append(AuditLogger(output="stdout"))
    return Guard(shields=shields)
