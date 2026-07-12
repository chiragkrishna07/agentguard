import json
from types import SimpleNamespace

import pytest

from agentguard.core.exceptions import GuardBlockedError
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext
from agentguard.shields.pii_redactor import PIIRedactor
from agentguard.tools import GuardedTool


@pytest.fixture
def ctx():
    return SessionContext()


class TestTravelAndInternationalPII:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text,rule_id,secret",
        [
            ("Passport number L898902C3", "PASSPORT", "L898902C3"),
            ("Aadhaar 1234 5678 9012", "AADHAAR", "1234 5678 9012"),
            ("identifier 2345 6789 0124", "AADHAAR", "2345 6789 0124"),
            ("Aadhaar XXXX XXXX 1234", "AADHAAR_MASKED", "XXXX XXXX 1234"),
            (
                "Aadhaar virtual ID 1234 5678 9012 3456",
                "AADHAAR_VID",
                "1234 5678 9012 3456",
            ),
            ("PAN ABCDE1234F", "PAN", "ABCDE1234F"),
            ("tax id ABCPA1234F", "PAN", "ABCPA1234F"),
            ("GSTIN 27AAPFU0939F1ZV", "GSTIN", "27AAPFU0939F1ZV"),
            ("EPIC ABC1234567", "VOTER_ID_IN", "ABC1234567"),
            ("NINO AB 12 34 56 C", "NINO_UK", "AB 12 34 56 C"),
            ("EIN 12-3456789", "EIN_US", "12-3456789"),
            ("SIN 046 454 286", "SIN_CA", "046 454 286"),
            ("NHS number 943 476 5919", "NHS_NUMBER", "943 476 5919"),
            ("CPF 529.982.247-25", "CPF_BR", "529.982.247-25"),
            ("CNPJ 04.252.011/0001-10", "CNPJ_BR", "04.252.011/0001-10"),
            ("TFN 123 456 782", "TFN_AU", "123 456 782"),
            ("Medicare 4019 84589 1", "MEDICARE_AU", "4019 84589 1"),
            ("UPI alice@okhdfcbank", "UPI_ID", "alice@okhdfcbank"),
            ("Call +44 (20) 7123 4567", "PHONE_INTERNATIONAL", "+44 (20) 7123 4567"),
            ("mobile +91 98765 43210", "PHONE_IN", "+91 98765 43210"),
            ("IBAN GB82 WEST 1234 5698 7654 32", "IBAN", "GB82 WEST 1234 5698 7654 32"),
            ("bank account: 123456789012", "BANK_ACCOUNT", "123456789012"),
            (
                "driver's license D123-4567-8901",
                "DRIVERS_LICENSE",
                "D123-4567-8901",
            ),
            ("national ID: S1234567D", "NATIONAL_ID", "S1234567D"),
        ],
    )
    async def test_sensitive_identifier_is_redacted(self, text, rule_id, secret, ctx):
        result = await PIIRedactor().scan_input(text, ctx)

        assert result.modified_input is not None
        assert secret not in result.modified_input
        assert f"[REDACTED_{rule_id}]" in result.modified_input
        assert rule_id in ctx.metadata["pii_rule_ids"]

    @pytest.mark.asyncio
    async def test_full_passport_mrz_is_removed(self, ctx):
        mrz = (
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        )
        result = await PIIRedactor().scan_input(f"scan:\n{mrz}\nend", ctx)
        assert result.modified_input == "scan:\n[REDACTED_PASSPORT_MRZ]\nend"

    @pytest.mark.asyncio
    async def test_date_is_dob_only_with_local_birth_context(self, ctx):
        dob = await PIIRedactor().scan_input("Date of birth: 12/25/1990", ctx)
        assert dob.modified_input == "Date of birth: [REDACTED_DATE_OF_BIRTH]"

        travel_ctx = SessionContext()
        travel = await PIIRedactor().scan_input(
            "Depart 12/25/2026 and return 2026-12-30", travel_ctx
        )
        assert travel.modified_input is None
        assert "pii_detected" not in travel_ctx.metadata

        nearby_ctx = SessionContext()
        nearby = await PIIRedactor().scan_input(
            "Date of birth is missing; departure date is 12/25/2026", nearby_ctx
        )
        assert nearby.modified_input is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "Flight L898902C3 departs 12/25/2026",
            "booking reference 9876543210",
            "ticket ABCDE1234F",
            "order 2345 6789 0125",
            "IBAN GB82 WEST 1234 5698 7654 33",
            "DOB 02/30/1990",
        ],
    )
    async def test_ambiguous_or_checksum_invalid_values_are_not_redacted(self, text, ctx):
        result = await PIIRedactor().scan_input(text, ctx)
        assert result.modified_input is None


class TestStructuredJSONRedaction:
    @pytest.mark.asyncio
    async def test_numeric_card_redaction_keeps_json_valid(self, ctx):
        raw = (
            '{"payment":{"card":4111111111111111,"amount":12500,'
            '"approved":true,"note":null}}'
        )
        result = await PIIRedactor().scan_input(raw, ctx)
        parsed = json.loads(result.modified_input)

        assert parsed["payment"]["card"] == "[REDACTED_CREDIT_CARD]"
        assert parsed["payment"]["amount"] == 12500
        assert isinstance(parsed["payment"]["amount"], int)
        assert parsed["payment"]["approved"] is True
        assert parsed["payment"]["note"] is None

    @pytest.mark.asyncio
    async def test_property_names_supply_tight_context_for_ambiguous_values(self, ctx):
        raw = json.dumps(
            {
                "passport_number": "A1234567",
                "aadhaar": 123456789012,
                "phone": 9876543210,
                "departure_date": "12/25/2026",
            }
        )
        result = await PIIRedactor().scan_input(raw, ctx)
        parsed = json.loads(result.modified_input)

        assert parsed["passport_number"] == "[REDACTED_PASSPORT]"
        assert parsed["aadhaar"] == "[REDACTED_AADHAAR]"
        assert parsed["phone"] == "[REDACTED_PHONE_IN]"
        assert parsed["departure_date"] == "12/25/2026"

    @pytest.mark.asyncio
    async def test_nested_arrays_and_strings_are_sanitized_recursively(self, ctx):
        raw = json.dumps(
            {
                "travellers": [
                    {"email": "alice@example.com", "age": 31},
                    {"email": "bob@example.net", "age": 29},
                ]
            }
        )
        parsed = json.loads((await PIIRedactor().scan_input(raw, ctx)).modified_input)

        assert parsed["travellers"][0]["email"] == "[REDACTED_EMAIL]"
        assert parsed["travellers"][1]["email"] == "[REDACTED_EMAIL]"
        assert parsed["travellers"][0]["age"] == 31

    @pytest.mark.asyncio
    async def test_clean_json_preserves_original_serialization(self, ctx):
        raw = '{ "count": 3, "ok": true, "items": [1, 2] }'
        result = await PIIRedactor().scan_input(raw, ctx)
        assert result.modified_input is None
        assert "pii_detected" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_json_tool_output_is_sanitized(self, ctx):
        shield = PIIRedactor(scan_tool_output=True)
        result = await shield.scan_tool_output(
            "lookup_customer", '{"email":"person@example.com","loyalty_points":100}', ctx
        )
        parsed = json.loads(result.modified_input)
        assert parsed == {"email": "[REDACTED_EMAIL]", "loyalty_points": 100}

    @pytest.mark.asyncio
    async def test_json_agent_output_is_sanitized_when_enabled(self, ctx):
        shield = PIIRedactor(redact_output=True)
        result = await shield.scan_output(
            '{"card":4242424242424242,"status":"confirmed"}', ctx
        )
        parsed = json.loads(result.modified_input)
        assert parsed["card"] == "[REDACTED_CREDIT_CARD]"
        assert parsed["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_json_numeric_tokenization_stays_valid_and_clears_after_resolution(
        self, ctx
    ):
        shield = PIIRedactor(mode="tokenize")
        tokenized = await shield.scan_input('{"card":4111111111111111}', ctx)
        parsed_tokenized = json.loads(tokenized.modified_input)

        assert parsed_tokenized["card"].startswith("[AGENTGUARD_CREDIT_CARD_")
        resolved = await shield.scan_output(tokenized.modified_input, ctx)
        assert json.loads(resolved.modified_input)["card"] == "4111111111111111"
        assert ctx._token_map == {}


class TestPIIMetadataAndTokenLifecycle:
    @pytest.mark.asyncio
    async def test_metadata_accumulates_across_directions(self, ctx):
        shield = PIIRedactor(redact_output=True, scan_tool_output=True)
        await shield.scan_input("alice@example.com", ctx)
        await shield.scan_output("SSN 123-45-6789", ctx)
        await shield.scan_tool_output("db", "IP 10.0.0.1", ctx)

        assert ctx.metadata["pii_detected"] is True
        assert ctx.metadata["pii_rule_ids"] == ["EMAIL", "IP_ADDRESS", "SSN"]
        assert ctx.metadata["pii_detection_count"] == 3

    @pytest.mark.asyncio
    async def test_resolved_tokens_are_cleared_by_default(self, ctx):
        shield = PIIRedactor(mode="tokenize")
        tokenized = await shield.scan_input("email alice@example.com", ctx)
        token = next(iter(ctx._token_map))

        resolved = await shield.scan_output(tokenized.modified_input, ctx)
        assert "alice@example.com" in resolved.modified_input
        assert token not in ctx._token_map

    @pytest.mark.asyncio
    async def test_resolved_token_cleanup_can_be_disabled(self, ctx):
        shield = PIIRedactor(mode="tokenize", clear_resolved_tokens=False)
        tokenized = await shield.scan_input("email alice@example.com", ctx)
        token = next(iter(ctx._token_map))

        await shield.scan_output(tokenized.modified_input, ctx)
        assert token in ctx._token_map

    @pytest.mark.asyncio
    async def test_token_store_is_bounded_and_records_eviction(self, ctx):
        shield = PIIRedactor(mode="tokenize", max_tokenized_values=2)
        await shield.scan_input("one@example.com", ctx)
        first_value = next(iter(ctx._token_map.values()))
        await shield.scan_input("two@example.com", ctx)
        await shield.scan_input("three@example.com", ctx)

        assert len(ctx._token_map) == 2
        assert first_value not in ctx._token_map.values()
        assert ctx.metadata["pii_token_evictions"] == 1

    @pytest.mark.asyncio
    async def test_explicit_session_cleanup_removes_unused_values(self, ctx):
        shield = PIIRedactor(mode="tokenize")
        await shield.scan_input("alice@example.com and bob@example.net", ctx)

        assert shield.clear_tokenized_values(ctx) == 2
        assert ctx._token_map == {}

    def test_token_limit_is_validated(self):
        with pytest.raises(ValueError, match="positive integer"):
            PIIRedactor(max_tokenized_values=0)


class TestPIIToolArgumentDLP:
    @pytest.mark.asyncio
    async def test_policy_is_off_by_default_for_legitimate_booking(self, ctx):
        params = {"traveler": {"passport_number": "A1234567"}}
        sanitized = await Guard([PIIRedactor()]).scan_tool_arguments(
            "book_flight", params, ctx
        )

        assert sanitized == params
        assert "pii_detected" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_block_policy_prevents_nested_pii_from_reaching_tool(self, ctx):
        called = False

        async def send_profile(profile):
            nonlocal called
            called = True
            return profile

        shield = PIIRedactor(tool_argument_policy="block")
        tool = GuardedTool(send_profile, Guard([shield]), ctx)
        with pytest.raises(GuardBlockedError) as exc_info:
            await tool(profile={"identity": {"passport_number": "A1234567"}})

        assert exc_info.value.reason_code == "PII_IN_TOOL_ARGUMENTS"
        assert "A1234567" not in str(exc_info.value)
        assert called is False
        assert ctx.metadata["pii_tool_argument_detected"] is True

    @pytest.mark.asyncio
    async def test_block_policy_detects_numeric_card_without_stringifying_execution(self, ctx):
        shield = PIIRedactor(tool_argument_policy="block")
        with pytest.raises(GuardBlockedError) as exc_info:
            await Guard([shield]).scan_tool_arguments(
                "charge", {"payment": {"card": 4111111111111111, "amount": 125}}, ctx
            )

        assert exc_info.value.reason_code == "PII_IN_TOOL_ARGUMENTS"

    @pytest.mark.asyncio
    async def test_redact_policy_reaches_nested_tool_execution(self, ctx):
        seen = None

        async def analyze_profile(profile):
            nonlocal seen
            seen = profile
            return "ok"

        shield = PIIRedactor(tool_argument_policy="redact")
        tool = GuardedTool(analyze_profile, Guard([shield]), ctx)
        await tool(profile={"contact": {"email": "alice@example.com"}, "age": 31})

        assert seen == {"contact": {"email": "[REDACTED_EMAIL]"}, "age": 31}

    @pytest.mark.asyncio
    async def test_presidio_tool_policy_uses_rule_ids_without_optional_anonymizer(
        self, ctx
    ):
        class FakeAnalyzer:
            def analyze(self, **kwargs):
                return [SimpleNamespace(start=0, end=5, entity_type="PERSON")]

        shield = PIIRedactor(engine="presidio", tool_argument_policy="redact")
        shield._analyzer = FakeAnalyzer()
        result = await shield.scan_tool_arguments("send", "Alice", ctx)

        assert result.modified_input == "[REDACTED_PERSON]"
        assert ctx.metadata["pii_rule_ids"] == ["PERSON"]

    def test_tool_argument_policy_is_validated(self):
        with pytest.raises(ValueError, match="tool_argument_policy"):
            PIIRedactor(tool_argument_policy="audit")
