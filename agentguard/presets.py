"""
Ready-made shield stacks.

Most teams don't want to hand-pick and tune seven shields on day one — they want
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
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.pii_redactor import PIIRedactor
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.secrets import SecretsShield
from agentguard.shields.size_limit import SizeLimit


def minimal() -> Guard:
    """Just the two zero-dependency essentials: injection + secret defense.

    No tiktoken, no Presidio, no ML download. Safe to drop in anywhere.
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
) -> Guard:
    """A balanced production stack covering the full request lifecycle.

    Input: injection detection (with tool-output scanning for indirect
    injection) + PII redaction + secret redaction. Output: PII and secret
    leakage redaction. Optional cost ceiling and audit trail.
    """
    shields: list = [
        SizeLimit(max_input_chars=20_000, max_tool_output_chars=20_000),
        PromptShield(mode="strict", on_indirect="block"),
        SecretsShield(on_detect="redact"),
        PIIRedactor(mode="redact", redact_output=True, scan_tool_output=True),
    ]
    if max_usd is not None:
        shields.append(CostLimit(max_usd=max_usd, model=model))
    if audit:
        shields.append(AuditLogger(output="stdout"))
    return Guard(shields=shields)


def paranoid(max_usd: float | None = 25.0, model: str = "gpt-4o") -> Guard:
    """Maximum scrutiny: paranoid injection mode, secrets hard-blocked, PII
    redacted everywhere, cost-capped, fully audited. Expect more false
    positives — use when the cost of a miss dwarfs the cost of a block.
    """
    shields: list = [
        PromptShield(mode="paranoid", on_indirect="block"),
        SecretsShield(on_detect="block"),
        PIIRedactor(mode="redact", redact_output=True, scan_tool_output=True),
    ]
    if max_usd is not None:
        shields.append(CostLimit(max_usd=max_usd, model=model, on_limit="block"))
    shields.append(AuditLogger(output="stdout"))
    return Guard(shields=shields)
