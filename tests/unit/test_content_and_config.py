import pytest

from agentguard.core.content import apply_to_text, extract_text
from agentguard.core.guard import Guard
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.secrets import SecretsShield


class TestExtractText:
    def test_string(self):
        assert extract_text("hello") == "hello"

    def test_none(self):
        assert extract_text(None) == ""

    def test_part_dict(self):
        assert extract_text({"type": "text", "text": "hi"}) == "hi"

    def test_list_of_parts_skips_non_text(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text", "text": "second"},
        ]
        assert extract_text(content) == "first\nsecond"


class TestApplyToText:
    @pytest.mark.asyncio
    async def test_string_transformed(self):
        out = await apply_to_text("abc", lambda t: _upper(t))
        assert out == "ABC"

    @pytest.mark.asyncio
    async def test_list_preserves_structure(self):
        content = [
            {"type": "text", "text": "abc"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ]
        out = await apply_to_text(content, lambda t: _upper(t))
        assert out[0]["text"] == "ABC"
        assert out[1] == {"type": "image_url", "image_url": {"url": "http://x"}}


async def _upper(t: str) -> str:
    return t.upper()


class TestGuardFromDict:
    def test_builds_shields_in_order(self):
        guard = Guard.from_dict(
            {
                "shields": [
                    {"type": "PromptShield", "use_canary": False},
                    {"type": "SecretsShield", "on_detect": "redact"},
                    {"type": "CostLimit", "max_usd": 5.0},
                ]
            }
        )
        assert isinstance(guard.shields[0], PromptShield)
        assert isinstance(guard.shields[1], SecretsShield)
        assert isinstance(guard.shields[2], CostLimit)
        assert guard.shields[2].max_usd == 5.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            Guard.from_dict({"shields": [{"type": "Nope"}]})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError):
            Guard.from_dict({"shields": [{"mode": "strict"}]})

    def test_non_shield_export_rejected(self):
        # SessionContext is exported but isn't a shield.
        with pytest.raises(ValueError):
            Guard.from_dict({"shields": [{"type": "SessionContext"}]})

    def test_empty_config(self):
        assert Guard.from_dict({}).shields == []
