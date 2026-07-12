import asyncio

import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.content_policy import (
    ContentPolicyShield,
    ContentRule,
    ContentVerdict,
)


@pytest.mark.asyncio
async def test_score_classifier_applies_per_category_thresholds_all_directions():
    seen = []

    async def classify(text, direction, ctx):
        seen.append((text, direction, ctx.session_id))
        return {"violence": 0.8, "self_harm": 0.2}

    shield = ContentPolicyShield(
        classifier=classify,
        thresholds={"violence": 0.9, "self_harm": 0.1},
    )
    ctx = SessionContext()
    result = await shield.scan_tool_output("web", "content", ctx)
    assert not result.allowed
    assert ctx.metadata["content_policy_violation"]["categories"] == ["self_harm"]
    assert seen[0][1] == "tool_output"


@pytest.mark.asyncio
async def test_boolean_score_mapping():
    shield = ContentPolicyShield(classifier=lambda *_: {"policy_a": False, "policy_b": True})
    result = await shield.scan_input("x", SessionContext())
    assert not result.allowed
    assert "policy_b" in (result.reason or "")


@pytest.mark.asyncio
async def test_content_verdict_can_sanitize_allowed_text():
    shield = ContentPolicyShield(
        classifier=lambda *_: ContentVerdict(allowed=True, modified_text="safe")
    )
    result = await shield.scan_output("raw", SessionContext())
    assert result.allowed
    assert result.modified_input == "safe"


@pytest.mark.asyncio
async def test_deterministic_rules_work_from_dict_config():
    shield = ContentPolicyShield(
        rules=[
            {
                "category": "regulated_export",
                "pattern": r"\bEXPORT-CONTROLLED-42\b",
                "directions": ("output",),
                "reason": "regulated content may not leave this boundary",
            }
        ]
    )
    assert (await shield.scan_input("EXPORT-CONTROLLED-42", SessionContext())).allowed
    result = await shield.scan_output("EXPORT-CONTROLLED-42", SessionContext())
    assert not result.allowed
    assert result.reason_code == "CONTENT_RULE_VIOLATION"


@pytest.mark.asyncio
async def test_classifier_timeout_fails_closed_by_default():
    async def slow(*_):
        await asyncio.sleep(0.05)
        return {}

    shield = ContentPolicyShield(classifier=slow, classifier_timeout_seconds=0.001)
    with pytest.raises(asyncio.TimeoutError):
        await shield.scan_input("x", SessionContext())


@pytest.mark.asyncio
async def test_classifier_error_warn_mode_allows_and_records_type():
    def broken(*_):
        raise RuntimeError("provider unavailable with secret details")

    shield = ContentPolicyShield(classifier=broken, on_error="warn")
    ctx = SessionContext()
    with pytest.warns(UserWarning, match="RuntimeError"):
        result = await shield.scan_input("x", ctx)
    assert result.allowed
    assert ctx.metadata["content_policy_error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_directions_can_disable_output_scanning():
    shield = ContentPolicyShield(
        classifier=lambda *_: {"blocked": True}, directions=("input",)
    )
    assert (await shield.scan_output("x", SessionContext())).allowed


@pytest.mark.asyncio
async def test_tool_arguments_have_distinct_policy_direction():
    seen = []

    def classify(text, direction, ctx):
        seen.append(direction)
        return {"tool_only": direction == "tool_call"}

    shield = ContentPolicyShield(classifier=classify)
    result = await shield.scan_tool_arguments("send", "payload", SessionContext())
    assert not result.allowed
    assert seen == ["tool_call"]


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        ContentPolicyShield()
    with pytest.raises(ValueError):
        ContentPolicyShield(classifier=lambda *_: {}, default_threshold=1.1)
    with pytest.raises(ValueError):
        ContentPolicyShield(rules=[ContentRule(category="x", pattern="[")])
    with pytest.raises(ValueError):
        ContentPolicyShield(classifier="moderate")
