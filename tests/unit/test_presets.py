import pytest

from agentguard.core.exceptions import HumanGateSyncError
from agentguard.core.guard import Guard
from agentguard.presets import minimal, paranoid, recommended
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.human_gate import HumanGate
from agentguard.shields.prompt_shield import PromptShield


def _names(guard):
    return [s.__class__.__name__ for s in guard.shields]


class TestPresets:
    def test_minimal_is_zero_dependency(self):
        guard = minimal()
        assert _names(guard) == ["PromptShield", "SecretsShield"]

    def test_recommended_full_stack(self):
        guard = recommended()
        assert _names(guard) == [
            "SizeLimit",
            "RateLimit",
            "PromptShield",
            "SecretsShield",
            "PIIRedactor",
            "ToolValidator",
            "ToolCallBudget",
            "NetworkPolicyShield",
            "CostLimit",
            "AuditLogger",
        ]

    def test_recommended_without_cost_or_audit(self):
        guard = recommended(
            max_usd=None,
            audit=False,
            requests_per_minute=None,
            network_policy=False,
            tool_budget=False,
        )
        assert "CostLimit" not in _names(guard)
        assert "AuditLogger" not in _names(guard)
        assert "RateLimit" not in _names(guard)
        assert "NetworkPolicyShield" not in _names(guard)
        assert "ToolCallBudget" not in _names(guard)

    def test_recommended_cost_uses_model_and_limit(self):
        guard = recommended(max_usd=12.0, model="gpt-4o-mini")
        cost = next(s for s in guard.shields if isinstance(s, CostLimit))
        assert cost.max_usd == 12.0
        assert cost.model == "gpt-4o-mini"

    def test_paranoid_blocks_secrets_and_uses_paranoid_mode(self):
        guard = paranoid()
        ps = next(s for s in guard.shields if isinstance(s, PromptShield))
        assert ps.mode == "paranoid"

    @pytest.mark.asyncio
    async def test_recommended_end_to_end_redacts_output(self):
        guard = recommended(max_usd=None, audit=False)

        @guard.protect
        async def agent(q):
            return "leaked ssn 123-45-6789"

        out = await agent("hello")
        assert "123-45-6789" not in out


class TestHumanGateSyncGuard:
    def test_protect_sync_rejects_human_gate(self):
        guard = Guard(shields=[HumanGate(triggers=["tool_call:*"])])
        with pytest.raises(HumanGateSyncError):

            @guard.protect_sync
            def agent(q):
                return q

    def test_protect_sync_ok_without_async_shields(self):
        guard = Guard(shields=[PromptShield(use_canary=False)])

        @guard.protect_sync
        def agent(q):
            return "ok"

        assert agent("hello") == "ok"
