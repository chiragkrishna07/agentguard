from unittest.mock import MagicMock, patch

import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.cost_limit import CostLimit


@pytest.fixture
def ctx():
    return SessionContext()


def _mock_encoder(token_count: int):
    enc = MagicMock()
    enc.encode.return_value = list(range(token_count))
    return enc


class TestCostLimit:
    def test_invalid_configuration(self):
        with pytest.raises(ValueError):
            CostLimit(max_usd=0)
        with pytest.raises(ValueError):
            CostLimit(max_usd=1, per="tenant")
        with pytest.raises(ValueError):
            CostLimit(max_usd=1, on_limit="maybe")
        with pytest.raises(ValueError):
            CostLimit(max_usd=1, pricing={"broken": {"input": 1}})
        with pytest.raises(ValueError):
            CostLimit(max_usd=float("nan"))
        with pytest.raises(ValueError):
            CostLimit(
                max_usd=1,
                pricing={"broken": {"input": float("nan"), "output": 1}},
            )

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, ctx):
        shield = CostLimit(max_usd=10.0, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(10)):
            result = await shield.scan_input("hello", ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_blocks_when_limit_exceeded(self, ctx):
        shield = CostLimit(max_usd=0.000001, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(10_000)):
            result = await shield.scan_input("x" * 10000, ctx)
        assert result.allowed is False
        assert result.reason_code == "COST_LIMIT_EXCEEDED"

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_calls(self, ctx):
        shield = CostLimit(max_usd=99.0, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(100)):
            await shield.scan_input("first", ctx)
            first_cost = ctx.cost_usd
            await shield.scan_input("second", ctx)
            assert ctx.cost_usd > first_cost

    @pytest.mark.asyncio
    async def test_output_cost_adds_to_session(self, ctx):
        shield = CostLimit(max_usd=10.0, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(50)):
            await shield.scan_output("response text", ctx)
        assert ctx.cost_usd > 0

    @pytest.mark.asyncio
    async def test_output_that_crosses_budget_is_blocked(self, ctx):
        shield = CostLimit(max_usd=0.00001, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(10_000)):
            result = await shield.scan_output("expensive response", ctx)
        assert result.allowed is False
        assert result.reason_code == "COST_LIMIT_EXCEEDED"
        assert ctx.cost_usd > shield.max_usd
        assert ctx.metadata["cost_limit"]["direction"] == "output"
        assert ctx.metadata["cost_limit"]["actual_total_usd"] == pytest.approx(
            ctx.cost_usd
        )

    def test_estimate_and_remaining_do_not_mutate(self, ctx):
        shield = CostLimit(max_usd=10, model="gpt-4o")
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(100)):
            estimate = shield.estimate_cost(input_text="x", output_text="y")
        assert estimate > 0
        assert shield.remaining_usd(ctx) == 10
        assert ctx.cost_usd == 0

    @pytest.mark.asyncio
    async def test_warn_mode_does_not_block(self, ctx):
        shield = CostLimit(max_usd=0.000001, on_limit="warn", model="gpt-4o")
        import warnings
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(10_000)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = await shield.scan_input("x" * 10000, ctx)
        assert result.allowed is True
        assert len(w) >= 1

    @pytest.mark.asyncio
    async def test_custom_pricing_used(self, ctx):
        custom = {"my-model": {"input": 100.0, "output": 100.0}}
        shield = CostLimit(max_usd=0.0001, model="my-model", pricing=custom)
        with patch.object(shield, "_get_encoder", return_value=_mock_encoder(100)):
            result = await shield.scan_input("test", ctx)
        assert result.allowed is False
