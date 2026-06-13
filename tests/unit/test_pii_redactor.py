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
        await shield.scan_input("Email: carol@example.com", ctx)
        token = list(ctx._token_map.keys())[0]

        output_result = await shield.scan_output(f"Sending to {token} now.", ctx)
        assert output_result.allowed is True
        assert output_result.modified_input is not None
        assert "carol@example.com" in output_result.modified_input

    @pytest.mark.asyncio
    async def test_overlapping_matches_do_not_corrupt_output(self, ctx):
        # An email whose host is all digits also matches the PHONE_US pattern.
        # The overlapping spans must not corrupt the output or leak fragments.
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("contact john.doe@1234567890.com please", ctx)
        assert result.modified_input is not None
        # No fragment of the original PII may survive, and no mangled label.
        assert "1234567890" not in result.modified_input
        assert "john.doe" not in result.modified_input
        assert ".com" not in result.modified_input
        assert "E_US]" not in result.modified_input  # the old corruption signature
        assert "[REDACTED_EMAIL]" in result.modified_input
        assert result.modified_input == "contact [REDACTED_EMAIL] please"

    @pytest.mark.asyncio
    async def test_multiple_distinct_pii_all_redacted(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input(
            "SSN 123-45-6789, card 4111 1111 1111 1111, ip 10.0.0.1", ctx
        )
        assert result.modified_input is not None
        assert "123-45-6789" not in result.modified_input
        assert "4111" not in result.modified_input
        assert "10.0.0.1" not in result.modified_input

    @pytest.mark.asyncio
    async def test_specific_entities_only(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex", entities=["EMAIL"])
        result = await shield.scan_input("SSN 123-45-6789 and email test@test.com", ctx)
        # SSN should remain, email should be redacted
        assert result.modified_input is not None
        assert "123-45-6789" in result.modified_input
        assert "test@test.com" not in result.modified_input
