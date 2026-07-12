import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.size_limit import SizeLimit


@pytest.fixture
def ctx():
    return SessionContext()


class TestSizeLimit:
    def test_invalid_on_exceed_raises(self):
        with pytest.raises(ValueError):
            SizeLimit(on_exceed="nope")

    def test_negative_limits_rejected(self):
        with pytest.raises(ValueError):
            SizeLimit(max_input_chars=-1)
        with pytest.raises(ValueError):
            SizeLimit(max_input_bytes=-1)
        with pytest.raises(ValueError):
            SizeLimit(max_input_chars=1.5)

    @pytest.mark.asyncio
    async def test_under_limit_passes(self, ctx):
        shield = SizeLimit(max_input_chars=100)
        result = await shield.scan_input("short", ctx)
        assert result.allowed is True
        assert result.modified_input is None

    @pytest.mark.asyncio
    async def test_over_limit_blocks(self, ctx):
        shield = SizeLimit(max_input_chars=10, on_exceed="block")
        result = await shield.scan_input("x" * 50, ctx)
        assert result.allowed is False
        assert result.reason_code == "SIZE_LIMIT_EXCEEDED"

    @pytest.mark.asyncio
    async def test_truncate_mode(self, ctx):
        shield = SizeLimit(max_input_chars=10, on_exceed="truncate")
        result = await shield.scan_input("x" * 50, ctx)
        assert result.allowed is True
        assert result.modified_input == "x" * 10

    @pytest.mark.asyncio
    async def test_none_disables_flow(self, ctx):
        shield = SizeLimit(max_input_chars=None)
        result = await shield.scan_input("x" * 100_000, ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_tool_output_limit(self, ctx):
        shield = SizeLimit(max_tool_output_chars=5, on_exceed="block")
        result = await shield.scan_tool_output("fetch", "x" * 10, ctx)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_utf8_byte_limit_blocks_multibyte_payload(self, ctx):
        shield = SizeLimit(max_input_chars=10, max_input_bytes=3)
        result = await shield.scan_input("🙂", ctx)
        assert result.allowed is False
        assert "UTF-8 bytes" in result.reason

    @pytest.mark.asyncio
    async def test_utf8_byte_truncation_never_emits_partial_codepoint(self, ctx):
        shield = SizeLimit(
            max_input_chars=None,
            max_input_bytes=5,
            on_exceed="truncate",
        )
        with pytest.warns(UserWarning):
            result = await shield.scan_input("🙂🙂", ctx)
        assert result.modified_input == "🙂"
        assert len(result.modified_input.encode("utf-8")) <= 5
