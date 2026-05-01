import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.tool_validator import ToolValidator


@pytest.fixture
def ctx():
    return SessionContext()


class TestToolValidatorNameRules:
    @pytest.mark.asyncio
    async def test_blocked_pattern_denies_tool(self, ctx):
        shield = ToolValidator(blocked=["delete_*"])
        result = await shield.scan_tool_call("delete_users", {}, ctx)
        assert result.allowed is False
        assert result.reason_code == "TOOL_NOT_ALLOWED"

    @pytest.mark.asyncio
    async def test_allowed_pattern_permits_tool(self, ctx):
        shield = ToolValidator(allowed=["search_*", "read_*"])
        result = await shield.scan_tool_call("search_hotels", {"city": "Tokyo"}, ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_not_in_allowed_list_denies(self, ctx):
        shield = ToolValidator(allowed=["search_*"])
        result = await shield.scan_tool_call("delete_records", {}, ctx)
        assert result.allowed is False
        assert result.reason_code == "TOOL_NOT_ALLOWED"

    @pytest.mark.asyncio
    async def test_blocked_takes_precedence_over_allowed(self, ctx):
        shield = ToolValidator(allowed=["delete_*"], blocked=["delete_production_*"])
        result = await shield.scan_tool_call("delete_production_db", {}, ctx)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_no_restrictions_allows_everything(self, ctx):
        shield = ToolValidator()
        result = await shield.scan_tool_call("anything_goes", {"x": 1}, ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_glob_wildcard_matches(self, ctx):
        shield = ToolValidator(blocked=["admin_*"])
        assert (await shield.scan_tool_call("admin_reset", {}, ctx)).allowed is False
        assert (await shield.scan_tool_call("admin_export", {}, ctx)).allowed is False
        assert (await shield.scan_tool_call("search_users", {}, ctx)).allowed is True


class TestToolValidatorParamRules:
    @pytest.mark.asyncio
    async def test_max_value_enforced(self, ctx):
        shield = ToolValidator(
            param_rules={"transfer_funds": {"amount": {"type": float, "max": 1000.0}}}
        )
        result = await shield.scan_tool_call("transfer_funds", {"amount": 9999.0}, ctx)
        assert result.allowed is False
        assert result.reason_code == "TOOL_PARAM_INVALID"

    @pytest.mark.asyncio
    async def test_within_max_allowed(self, ctx):
        shield = ToolValidator(
            param_rules={"transfer_funds": {"amount": {"type": float, "max": 1000.0}}}
        )
        result = await shield.scan_tool_call("transfer_funds", {"amount": 100.0}, ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_wrong_type_blocked(self, ctx):
        shield = ToolValidator(
            param_rules={"search": {"query": {"type": str}}}
        )
        result = await shield.scan_tool_call("search", {"query": 12345}, ctx)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_maxlen_enforced(self, ctx):
        shield = ToolValidator(
            param_rules={"search": {"query": {"maxlen": 10}}}
        )
        result = await shield.scan_tool_call("search", {"query": "a" * 100}, ctx)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_regex_pattern_enforced(self, ctx):
        shield = ToolValidator(
            param_rules={"transfer": {"account": {"type": str, "pattern": r"[A-Z]{2}\d+"}}}
        )
        result = await shield.scan_tool_call("transfer", {"account": "not-valid"}, ctx)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_missing_param_skipped(self, ctx):
        shield = ToolValidator(
            param_rules={"search": {"query": {"type": str, "max": 100}}}
        )
        # 'query' not provided — rule should not fail on missing optional param
        result = await shield.scan_tool_call("search", {}, ctx)
        assert result.allowed is True


class TestToolValidatorWarnMode:
    @pytest.mark.asyncio
    async def test_warn_mode_logs_but_allows(self, ctx):
        import warnings
        shield = ToolValidator(blocked=["delete_*"], on_violation="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await shield.scan_tool_call("delete_users", {}, ctx)
        assert result.allowed is True
        assert len(w) >= 1
