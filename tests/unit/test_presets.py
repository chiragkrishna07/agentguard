import pytest

from agentguard.core.exceptions import GuardBlockedError, HumanGateSyncError
from agentguard.core.guard import Guard
from agentguard.presets import minimal, paranoid, recommended
from agentguard.shields.circuit_breaker import CircuitBreaker
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.human_gate import HumanGate
from agentguard.shields.memory_policy import MemoryPolicyShield
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.tool_integrity import ToolIntegrityShield


def _names(guard):
    return [s.__class__.__name__ for s in guard.shields]


class TestPresets:
    def test_minimal_is_zero_dependency(self):
        guard = minimal()
        assert _names(guard) == ["PromptShield", "SecretsShield"]

    def test_recommended_full_stack(self):
        guard = recommended()
        assert _names(guard) == [
            "CircuitBreaker",
            "SizeLimit",
            "RateLimit",
            "PromptShield",
            "SecretsShield",
            "PIIRedactor",
            "MemoryPolicyShield",
            "ToolIntegrityShield",
            "ToolValidator",
            "DangerousCommandShield",
            "ToolCallBudget",
            "NetworkPolicyShield",
            "CostLimit",
            "AuditLogger",
        ]

    def test_recommended_contains_first(self):
        """A contained session must be denied before any other shield runs."""
        assert _names(recommended())[0] == "CircuitBreaker"

    def test_recommended_without_cost_or_audit(self):
        guard = recommended(
            max_usd=None,
            audit=False,
            requests_per_minute=None,
            network_policy=False,
            tool_budget=False,
            circuit_breaker=False,
            tool_integrity=False,
            memory_policy=False,
            dangerous_commands=False,
        )
        assert "CostLimit" not in _names(guard)
        assert "AuditLogger" not in _names(guard)
        assert "RateLimit" not in _names(guard)
        assert "NetworkPolicyShield" not in _names(guard)
        assert "ToolCallBudget" not in _names(guard)
        assert "CircuitBreaker" not in _names(guard)
        assert "ToolIntegrityShield" not in _names(guard)
        assert "MemoryPolicyShield" not in _names(guard)
        assert "DangerousCommandShield" not in _names(guard)

    async def test_recommended_blocks_a_destructive_tool_call(self):
        """Roster order is only worth asserting if the call is actually reached."""
        guard = recommended(max_usd=None, audit=False)
        with pytest.raises(GuardBlockedError) as exc:
            await guard.scan_tool_call("run_shell", {"command": "rm -rf /"})
        assert exc.value.reason_code == "DANGEROUS_SHELL_COMMAND"

    async def test_recommended_allows_an_ordinary_tool_call(self):
        guard = recommended(max_usd=None, audit=False)
        await guard.scan_tool_call("run_shell", {"command": "ls -la /tmp"})

    def test_recommended_breaker_rearms_on_its_own(self):
        """A preset may run unattended, so its breaker must not wedge forever."""
        breaker = next(s for s in recommended().shields if isinstance(s, CircuitBreaker))
        assert breaker.cooldown_seconds is not None

    def test_paranoid_breaker_latches_until_reset(self):
        breaker = next(s for s in paranoid().shields if isinstance(s, CircuitBreaker))
        assert breaker.cooldown_seconds is None
        assert breaker.max_blocks == 5

    def test_paranoid_requires_registered_tools_and_memory_origin(self):
        shields = paranoid().shields
        integrity = next(s for s in shields if isinstance(s, ToolIntegrityShield))
        memory = next(s for s in shields if isinstance(s, MemoryPolicyShield))
        assert integrity.require_registration is True
        assert memory.require_origin is True

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
