import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.pii_redactor import PIIRedactor


@pytest.fixture
def ctx():
    return SessionContext()


class TestPIIRedactorInvalidConfig:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            PIIRedactor(mode="invalid_mode")

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError):
            PIIRedactor(engine="unknown")


class TestPIIRedactorRegexEngine:
    @pytest.mark.asyncio
    async def test_redacts_ssn(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("My SSN is 123-45-6789, please advise.", ctx)
        assert result.allowed is True
        assert result.modified_input is not None
        assert "123-45-6789" not in result.modified_input
        assert "[REDACTED_SSN]" in result.modified_input

    @pytest.mark.asyncio
    async def test_redacts_email(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("Email me at alice@example.com for details.", ctx)
        assert result.allowed is True
        assert result.modified_input is not None
        assert "alice@example.com" not in result.modified_input

    @pytest.mark.asyncio
    async def test_redacts_credit_card(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("Card number: 4111 1111 1111 1111", ctx)
        assert result.modified_input is not None
        assert "4111" not in result.modified_input

    @pytest.mark.asyncio
    async def test_clean_input_returns_none_modified(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("What is the weather like today?", ctx)
        assert result.allowed is True
        assert result.modified_input is None

    @pytest.mark.asyncio
    async def test_mask_mode_uses_stars(self, ctx):
        shield = PIIRedactor(mode="mask", engine="regex")
        result = await shield.scan_input("My SSN is 123-45-6789.", ctx)
        assert result.modified_input is not None
        assert "***" in result.modified_input

    @pytest.mark.asyncio
    async def test_tokenize_mode_stores_in_context(self, ctx):
        shield = PIIRedactor(mode="tokenize", engine="regex")
        result = await shield.scan_input("Email: bob@example.com", ctx)
        assert result.modified_input is not None
        assert "bob@example.com" not in result.modified_input
        assert len(ctx._token_map) >= 1

    @pytest.mark.asyncio
    async def test_tokenize_mode_resolves_in_output(self, ctx):
        shield = PIIRedactor(mode="tokenize", engine="regex")
        input_result = await shield.scan_input("Email: carol@example.com", ctx)
        token = list(ctx._token_map.keys())[0]

        output_result = await shield.scan_output(f"Sending to {token} now.", ctx)
        assert output_result.allowed is True
        assert output_result.modified_input is not None
        assert "carol@example.com" in output_result.modified_input

    @pytest.mark.asyncio
    async def test_specific_entities_only(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex", entities=["EMAIL"])
        result = await shield.scan_input("SSN 123-45-6789 and email test@test.com", ctx)
        # SSN should remain, email should be redacted
        assert result.modified_input is not None
        assert "123-45-6789" in result.modified_input
        assert "test@test.com" not in result.modified_input
