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
    async def test_luhn_invalid_number_not_treated_as_card(self, ctx):
        # A 16-digit number that fails the Luhn checksum (e.g. an order id).
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("order ref 1234 5678 9012 3456 shipped", ctx)
        assert result.modified_input is None  # nothing redacted

    @pytest.mark.asyncio
    async def test_luhn_valid_number_redacted(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("pay with 4242 4242 4242 4242 today", ctx)
        assert result.modified_input is not None
        assert "4242" not in result.modified_input

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,value",
        [
            ("ipv6", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
            ("mac", "00:1A:2B:3C:4D:5E"),
            ("itin", "912-78-1234"),
        ],
    )
    async def test_new_entities_redacted(self, ctx, label, value):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input(f"value is {value} ok", ctx)
        assert result.modified_input is not None
        assert value not in result.modified_input

    @pytest.mark.asyncio
    async def test_itin_labeled_distinctly_from_ssn(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        assert "[REDACTED_ITIN]" in (
            await shield.scan_input("912-78-1234", ctx)
        ).modified_input
        assert "[REDACTED_SSN]" in (
            await shield.scan_input("123-45-6789", ctx)
        ).modified_input

    @pytest.mark.asyncio
    async def test_time_not_matched_as_mac_or_ipv6(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_input("the meeting is at 10:30 on server1", ctx)
        assert result.modified_input is None

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
    async def test_output_redaction_off_by_default(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex")
        result = await shield.scan_output("The SSN is 123-45-6789", ctx)
        assert result.modified_input is None

    @pytest.mark.asyncio
    async def test_output_redaction_when_enabled(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex", redact_output=True)
        result = await shield.scan_output("The leaked SSN is 123-45-6789", ctx)
        assert result.modified_input is not None
        assert "123-45-6789" not in result.modified_input
        assert "[REDACTED_SSN]" in result.modified_input

    @pytest.mark.asyncio
    async def test_tokenize_output_takes_precedence_over_redaction(self, ctx):
        # In tokenize mode, scan_output must restore the user's PII, not redact it.
        shield = PIIRedactor(mode="tokenize", engine="regex", redact_output=True)
        await shield.scan_input("Email: dave@example.com", ctx)
        token = list(ctx._token_map.keys())[0]
        out = await shield.scan_output(f"Replying to {token}", ctx)
        assert "dave@example.com" in out.modified_input

    @pytest.mark.asyncio
    async def test_tool_output_redaction(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex", scan_tool_output=True)
        result = await shield.scan_tool_output(
            "read_db", "row: ssn 123-45-6789, email x@y.com", ctx
        )
        assert result.modified_input is not None
        assert "123-45-6789" not in result.modified_input

    @pytest.mark.asyncio
    async def test_specific_entities_only(self, ctx):
        shield = PIIRedactor(mode="redact", engine="regex", entities=["EMAIL"])
        result = await shield.scan_input("SSN 123-45-6789 and email test@test.com", ctx)
        # SSN should remain, email should be redacted
        assert result.modified_input is not None
        assert "123-45-6789" in result.modified_input
        assert "test@test.com" not in result.modified_input
