import pytest

from agentguard.core.exceptions import GuardBlockedError
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext
from agentguard.shields.secrets import SecretsShield
from agentguard.tools import GuardedTool


@pytest.fixture
def ctx():
    return SessionContext()


class TestUnicodeEvasion:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invisible",
        ["\u034f", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\ufe0f"],
    )
    async def test_zero_width_obfuscation_maps_back_to_source(self, invisible, ctx):
        secret = f"s{invisible}k-{invisible}" + "a" * 24
        result = await SecretsShield().scan_input(f"key={secret}", ctx)

        assert result.modified_input == "key=[REDACTED_OPENAI_KEY]"
        assert invisible not in result.modified_input

    @pytest.mark.asyncio
    async def test_zero_width_aws_prefix(self, ctx):
        secret = "A\u200bK\u200bI\u200bAIOSFODNN7EXAMPLE"
        result = await SecretsShield().scan_input(secret, ctx)
        assert result.modified_input == "[REDACTED_AWS_ACCESS_KEY]"

    @pytest.mark.asyncio
    async def test_full_width_ascii_is_detected_without_normalising_output(self, ctx):
        secret = "ｓｋ－" + "ａ" * 24
        result = await SecretsShield().scan_input(secret, ctx)
        assert result.modified_input == "[REDACTED_OPENAI_KEY]"

    @pytest.mark.asyncio
    async def test_benign_unicode_is_not_modified(self, ctx):
        text = "東京への旅行と café の予約"
        result = await SecretsShield().scan_input(text, ctx)
        assert result.modified_input is None


class TestCompleteCredentials:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key_type",
        ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY", "ENCRYPTED PRIVATE KEY", "PGP PRIVATE KEY BLOCK"],
    )
    async def test_complete_private_key_block_is_removed(self, key_type, ctx):
        pem = (
            f"-----BEGIN {key_type}-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC...\n"
            f"-----END {key_type}-----"
        )
        result = await SecretsShield().scan_input(f"before\n{pem}\nafter", ctx)

        assert result.modified_input == "before\n[REDACTED_PRIVATE_KEY]\nafter"
        assert "MIIEvQ" not in result.modified_input
        assert "END" not in result.modified_input

    @pytest.mark.asyncio
    async def test_truncated_private_key_header_fails_safe(self, ctx):
        result = await SecretsShield().scan_input(
            "-----BEGIN PRIVATE KEY-----\npartial stream", ctx
        )
        assert result.modified_input == "[REDACTED_PRIVATE_KEY]"

    @pytest.mark.asyncio
    async def test_unterminated_private_key_never_leaks_body_or_tail(self, ctx):
        text = (
            "prefix\n-----BEGIN PRIVATE KEY-----\n"
            "MII-SYNTHETIC-SENSITIVE-BODY\ninternal tail"
        )
        result = await SecretsShield().scan_input(text, ctx)
        assert result.modified_input == "prefix\n[REDACTED_PRIVATE_KEY]"
        assert "SENSITIVE" not in result.modified_input

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text,rule_id,secret",
        [
            (
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
                "BEARER_TOKEN",
                "abcdefghijklmnopqrstuvwxyz012345",
            ),
            (
                "postgresql://service:S3cretPassword@db.example.com/travel",
                "DATABASE_URL",
                "postgresql://service:S3cretPassword@db.example.com/travel",
            ),
            ("api_key: live-value-123456789", "GENERIC_CREDENTIAL", "live-value-123456789"),
            (
                '"client_secret": "client-secret-123456"',
                "GENERIC_CREDENTIAL",
                "client-secret-123456",
            ),
            (
                'password="comma,and;semicolon!123"',
                "GENERIC_CREDENTIAL",
                "comma,and;semicolon!123",
            ),
            (
                "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "AWS_SECRET_ACCESS_KEY",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            ),
            ("Authorization: Basic dXNlcjpTdXBlclNlY3JldA==", "BASIC_AUTH_CREDENTIAL", "dXNlcjpTdXBlclNlY3JldA=="),
        ],
    )
    async def test_contextual_generic_credentials(self, text, rule_id, secret, ctx):
        result = await SecretsShield().scan_input(text, ctx)

        assert result.modified_input is not None
        assert secret not in result.modified_input
        assert f"[REDACTED_{rule_id}]" in result.modified_input
        assert rule_id in ctx.metadata["secret_rule_ids"]


class TestGenericCredentialControls:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "Bearer of good news",
            "Bearer authentication is described in RFC 6750",
            "Basic authentication is required by this endpoint",
            "api_key: ${API_KEY}",
            "api_token: <INSERT_TOKEN>",
            "password: changeme",
            "client_secret: YOUR_SECRET_HERE",
        ],
    )
    async def test_clear_placeholders_and_prose_are_ignored(self, text, ctx):
        result = await SecretsShield().scan_input(text, ctx)
        assert result.modified_input is None
        assert "secret_detected" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_generic_tier_can_be_disabled(self, ctx):
        shield = SecretsShield(detect_generic_credentials=False)
        generic = await shield.scan_input("api_key: live-value-123456789", ctx)
        provider = await shield.scan_input("sk-" + "a" * 24, ctx)

        assert generic.modified_input is None
        assert provider.modified_input == "[REDACTED_OPENAI_KEY]"

    def test_generic_minimum_is_validated(self):
        with pytest.raises(ValueError, match="at least 8"):
            SecretsShield(generic_min_length=7)

    def test_scan_directions_are_validated(self):
        with pytest.raises(ValueError, match="Unknown scan direction"):
            SecretsShield(scan_directions=("input", "sideways"))

    @pytest.mark.asyncio
    async def test_custom_pattern_can_select_only_secret_group(self, ctx):
        shield = SecretsShield(
            custom_patterns={
                "TENANT_KEY": r"tenant-key=(?P<secret>[A-Z0-9]{12})"
            }
        )
        result = await shield.scan_input("tenant-key=ABCDEF123456", ctx)
        assert result.modified_input == "tenant-key=[REDACTED_TENANT_KEY]"


class TestSecretMetadata:
    @pytest.mark.asyncio
    async def test_metadata_accumulates_rule_ids_and_counts(self, ctx):
        shield = SecretsShield()
        await shield.scan_input("sk-" + "a" * 24, ctx)
        await shield.scan_output("ghp_" + "b" * 36, ctx)

        assert ctx.metadata["secret_detected"] is True
        assert ctx.metadata["secret_rule_ids"] == ["GITHUB_TOKEN", "OPENAI_KEY"]
        assert ctx.metadata["secret_detection_count"] == 2

    @pytest.mark.asyncio
    async def test_disabled_direction_does_not_set_metadata(self, ctx):
        shield = SecretsShield(scan_directions=("input",))
        result = await shield.scan_output("sk-" + "a" * 24, ctx)
        assert result.modified_input is None
        assert "secret_detected" not in ctx.metadata


class TestSecretToolArgumentDLP:
    @pytest.mark.asyncio
    async def test_nested_generic_secret_blocks_tool_execution_by_default(self, ctx):
        called = False

        async def send_request(request):
            nonlocal called
            called = True
            return request

        tool = GuardedTool(send_request, Guard([SecretsShield()]), ctx)
        with pytest.raises(GuardBlockedError) as exc_info:
            await tool(request={"headers": {"api_key": "live-value-123456789"}})

        assert exc_info.value.reason_code == "SECRET_IN_TOOL_ARGUMENTS"
        assert "live-value" not in str(exc_info.value)
        assert called is False
        assert ctx.metadata["secret_tool_argument_detected"] is True

    @pytest.mark.asyncio
    async def test_redact_policy_reaches_nested_tool_execution(self, ctx):
        seen = None

        async def send_request(request):
            nonlocal seen
            seen = request
            return "ok"

        shield = SecretsShield(tool_argument_policy="redact")
        tool = GuardedTool(send_request, Guard([shield]), ctx)
        await tool(request={"auth": {"api_key": "live-value-123456789"}})

        assert seen == {"auth": {"api_key": "[REDACTED_GENERIC_CREDENTIAL]"}}

    @pytest.mark.asyncio
    async def test_mask_policy_preserves_secret_value_length(self, ctx):
        shield = SecretsShield(tool_argument_policy="mask")
        guard = Guard([shield])
        secret = "sk-" + "z" * 24
        sanitized = await guard.scan_tool_arguments(
            "send", {"nested": [{"authorization": secret}]}, ctx
        )

        assert sanitized["nested"][0]["authorization"] == "*" * len(secret)

    @pytest.mark.asyncio
    async def test_off_policy_leaves_tool_arguments_unchanged(self, ctx):
        shield = SecretsShield(tool_argument_policy="off")
        params = {"api_key": "live-value-123456789"}
        sanitized = await Guard([shield]).scan_tool_arguments("send", params, ctx)

        assert sanitized == params
        assert "secret_detected" not in ctx.metadata

    def test_tool_argument_policy_is_validated(self):
        with pytest.raises(ValueError, match="tool_argument_policy"):
            SecretsShield(tool_argument_policy="mutate")
