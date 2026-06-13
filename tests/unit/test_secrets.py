import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.secrets import SecretsShield


@pytest.fixture
def ctx():
    return SessionContext()


class TestSecretsConfig:
    def test_invalid_on_detect_raises(self):
        with pytest.raises(ValueError):
            SecretsShield(on_detect="invalid")


class TestSecretsDetection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,secret",
        [
            ("aws", "AKIAIOSFODNN7EXAMPLE"),
            ("github", "ghp_" + "a" * 36),
            ("openai", "sk-" + "a" * 24),
            ("anthropic", "sk-ant-" + "b" * 24),
            ("google", "AIza" + "C" * 35),
            ("slack", "xoxb-123456789012-abcdefghijkl"),
            ("stripe", "sk_live_" + "0" * 24),
            ("jwt", "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4"),
        ],
    )
    async def test_redacts_known_secret_types(self, ctx, label, secret):
        shield = SecretsShield(on_detect="redact")
        result = await shield.scan_input(f"the credential is {secret} ok", ctx)
        assert result.modified_input is not None
        assert secret not in result.modified_input
        assert "[REDACTED_" in result.modified_input

    @pytest.mark.asyncio
    async def test_private_key_block_detected(self, ctx):
        shield = SecretsShield(on_detect="redact")
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
        result = await shield.scan_input(text, ctx)
        assert result.modified_input is not None
        assert "BEGIN RSA PRIVATE KEY" not in result.modified_input

    @pytest.mark.asyncio
    async def test_anthropic_labeled_correctly_not_openai(self, ctx):
        shield = SecretsShield(on_detect="redact")
        result = await shield.scan_input("sk-ant-" + "x" * 24, ctx)
        assert "[REDACTED_ANTHROPIC_KEY]" in result.modified_input

    @pytest.mark.asyncio
    async def test_clean_text_untouched(self, ctx):
        shield = SecretsShield()
        result = await shield.scan_input("the weather is sunny in Tokyo today", ctx)
        assert result.allowed is True
        assert result.modified_input is None

    @pytest.mark.asyncio
    async def test_block_mode(self, ctx):
        shield = SecretsShield(on_detect="block")
        result = await shield.scan_input("aws AKIAIOSFODNN7EXAMPLE here", ctx)
        assert result.allowed is False
        assert result.reason_code == "SECRET_DETECTED"

    @pytest.mark.asyncio
    async def test_mask_mode_preserves_length(self, ctx):
        shield = SecretsShield(on_detect="mask")
        secret = "ghp_" + "a" * 36
        result = await shield.scan_input(secret, ctx)
        assert result.modified_input == "*" * len(secret)

    @pytest.mark.asyncio
    async def test_multiple_secrets_all_redacted(self, ctx):
        shield = SecretsShield(on_detect="redact")
        text = f"aws=AKIAIOSFODNN7EXAMPLE gh=ghp_{'a' * 36}"
        result = await shield.scan_input(text, ctx)
        assert "AKIA" not in result.modified_input
        assert "ghp_" not in result.modified_input


class TestSecretsDirections:
    @pytest.mark.asyncio
    async def test_output_scanning(self, ctx):
        shield = SecretsShield(on_detect="redact")
        result = await shield.scan_output("here is the key sk-" + "z" * 24, ctx)
        assert "[REDACTED_OPENAI_KEY]" in result.modified_input

    @pytest.mark.asyncio
    async def test_tool_output_scanning(self, ctx):
        shield = SecretsShield(on_detect="redact")
        result = await shield.scan_tool_output("read_file", "AKIAIOSFODNN7EXAMPLE", ctx)
        assert "AKIA" not in result.modified_input

    @pytest.mark.asyncio
    async def test_direction_can_be_disabled(self, ctx):
        shield = SecretsShield(on_detect="redact", scan_directions=("input",))
        out = await shield.scan_output("sk-" + "z" * 24, ctx)
        assert out.modified_input is None

    @pytest.mark.asyncio
    async def test_custom_pattern(self, ctx):
        shield = SecretsShield(custom_patterns={"INTERNAL": r"INT-[0-9]{6}"})
        result = await shield.scan_input("ref INT-123456 done", ctx)
        assert "[REDACTED_INTERNAL]" in result.modified_input
