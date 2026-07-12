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
    async def test_bool_does_not_bypass_integer_type_or_numeric_bound(self, ctx):
        shield = ToolValidator(param_rules={"x": {"amount": {"type": int, "max": 2}}})
        assert not (await shield.scan_tool_call("x", {"amount": True}, ctx)).allowed

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

    @pytest.mark.asyncio
    async def test_param_rules_are_case_insensitive_and_support_globs(self, ctx):
        shield = ToolValidator(
            param_rules={
                "*": {"tenant": {"required": True, "type": str}},
                "TRANSFER_*": {"amount": {"required": True, "max": 100}},
            }
        )
        result = await shield.scan_tool_call(
            "transfer_funds", {"tenant": "acme", "amount": "101"}, ctx
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_dotted_paths_and_closed_param_set(self, ctx):
        shield = ToolValidator(
            param_rules={
                "book": {
                    "traveler.name": {"required": True, "type": str},
                    "price": {"type": (int, float), "min": 0},
                }
            },
            allow_extra_params=False,
        )
        assert (
            await shield.scan_tool_call(
                "BOOK", {"traveler": {"name": "A"}, "price": 12}, ctx
            )
        ).allowed
        result = await shield.scan_tool_call(
            "book", {"traveler": {"name": "A"}, "price": 12, "debug": True}, ctx
        )
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_closed_param_set_rejects_nested_extras(self, ctx):
        shield = ToolValidator(
            param_rules={
                "pay": {
                    "payment.amount": {"required": True, "max": 10},
                    "payment.currency": {"required": True, "choices": ["USD"]},
                }
            },
            allow_extra_params=False,
        )

        result = await shield.scan_tool_call(
            "pay",
            {
                "payment": {
                    "amount": 1,
                    "currency": "USD",
                    "skip_approval": True,
                    "admin": True,
                }
            },
            ctx,
        )

        assert not result.allowed
        assert "payment.admin" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_literal_and_nested_dotted_values_are_ambiguous(self, ctx):
        shield = ToolValidator(
            param_rules={"pay": {"payment.amount": {"required": True, "max": 10}}}
        )

        result = await shield.scan_tool_call(
            "pay",
            {
                "payment.amount": 1,
                "payment": {"amount": 1_000_000},
            },
            ctx,
        )

        assert not result.allowed
        assert "ambiguous" in (result.reason or "").lower()

    @pytest.mark.asyncio
    async def test_null_nan_enum_minlen_and_predicate_fail_closed(self, ctx):
        base = {"type": str, "minlen": 2, "choices": ["ok"]}
        shield = ToolValidator(param_rules={"x": {"value": base}})
        assert not (await shield.scan_tool_call("x", {"value": None}, ctx)).allowed
        assert not (await shield.scan_tool_call("x", {"value": "x"}, ctx)).allowed
        assert (await shield.scan_tool_call("x", {"value": "ok"}, ctx)).allowed

        numeric = ToolValidator(param_rules={"x": {"n": {"max": 10}}})
        assert not (await numeric.scan_tool_call("x", {"n": float("nan")}, ctx)).allowed

        predicate = ToolValidator(
            param_rules={"x": {"n": {"predicate": lambda n: n % 2 == 0}}}
        )
        assert not (await predicate.scan_tool_call("x", {"n": 3}, ctx)).allowed

    @pytest.mark.asyncio
    async def test_async_param_predicate(self, ctx):
        async def positive(value):
            return value > 0

        shield = ToolValidator(param_rules={"x": {"n": {"predicate": positive}}})
        assert (await shield.scan_tool_call("x", {"n": 1}, ctx)).allowed
        assert not (await shield.scan_tool_call("x", {"n": -1}, ctx)).allowed


class TestToolValidatorAuthorization:
    @pytest.mark.asyncio
    async def test_require_identity(self, ctx):
        shield = ToolValidator(require_user_id=True)
        result = await shield.scan_tool_call("read", {}, ctx)
        assert not result.allowed
        assert result.reason_code == "TOOL_IDENTITY_REQUIRED"
        ctx.user_id = "u-1"
        assert (await shield.scan_tool_call("read", {}, ctx)).allowed

    @pytest.mark.asyncio
    async def test_async_authorization_callback_receives_context(self, ctx):
        async def authorize(tool, params, session):
            return session.user_id == params.get("owner")

        shield = ToolValidator(authorize=authorize)
        ctx.user_id = "u-1"
        assert (await shield.scan_tool_call("read", {"owner": "u-1"}, ctx)).allowed
        result = await shield.scan_tool_call("read", {"owner": "u-2"}, ctx)
        assert not result.allowed
        assert result.reason_code == "TOOL_AUTHORIZATION_DENIED"

    @pytest.mark.asyncio
    async def test_per_tool_validator_can_return_reason(self, ctx):
        shield = ToolValidator(
            validators={"transfer_*": lambda params, _ctx: "currency denied"}
        )
        result = await shield.scan_tool_call("transfer_funds", {"currency": "X"}, ctx)
        assert not result.allowed
        assert result.reason == "currency denied"

    @pytest.mark.asyncio
    async def test_malformed_authorizer_result_fails_closed(self, ctx):
        shield = ToolValidator(authorize=lambda *_: 1)
        with pytest.raises(TypeError, match="authorization callbacks"):
            await shield.scan_tool_call("read", {}, ctx)

    @pytest.mark.asyncio
    async def test_name_and_payload_bounds(self, ctx):
        shield = ToolValidator(max_tool_name_chars=3, max_params=1, max_total_chars=10)
        assert not (await shield.scan_tool_call("long", {}, ctx)).allowed
        assert not (await shield.scan_tool_call("x", {"a": 1, "b": 2}, ctx)).allowed
        assert not (await shield.scan_tool_call("x", {"a": "0123456789"}, ctx)).allowed


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        ToolValidator(on_violation="maybe")
    with pytest.raises(ValueError):
        ToolValidator(param_rules={"x": {"p": {"pattern": "["}}})
    with pytest.raises(ValueError):
        ToolValidator(param_rules={"x": {"p": {"type": "str"}}})
    with pytest.raises(ValueError):
        ToolValidator(param_rules={"x": {"p": {"min": 2, "max": 1}}})
    with pytest.raises(ValueError):
        ToolValidator(authorize="not-callable")


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
