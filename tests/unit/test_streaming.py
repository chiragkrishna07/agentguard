from unittest.mock import MagicMock, patch

import pytest

from agentguard.core.exceptions import GuardBlockedError
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext
from agentguard.shields.content_policy import ContentPolicyShield
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.pii_redactor import PIIRedactor
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.secrets import SecretsShield
from agentguard.streaming import StreamGuard


async def _gen(parts):
    for p in parts:
        yield p


CHUNKS = ["my key is sk-", "abcdefghij", "klmnopqrstuv", " and ssn 123-", "45-6789 ok"]


class TestStreamGuardConfig:
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            StreamGuard(Guard(), mode="nope")

    def test_invalid_holdback(self):
        with pytest.raises(ValueError):
            StreamGuard(Guard(), holdback=0)

    def test_invalid_resource_limits(self):
        with pytest.raises(ValueError):
            StreamGuard(Guard(), max_buffer_chars=0)
        with pytest.raises(ValueError):
            StreamGuard(Guard(), max_buffer_bytes=0)
        with pytest.raises(ValueError):
            StreamGuard(Guard(), max_chunks=0)

    def test_incremental_refuses_full_output_policies_by_default(self):
        calls = 0

        def classify(text, direction, ctx):
            nonlocal calls
            calls += 1
            return {"unsafe": 0.0}

        guard = Guard([ContentPolicyShield(classifier=classify)])
        with pytest.raises(ValueError, match="buffer mode"):
            StreamGuard(guard, mode="incremental")
        assert calls == 0


class TestStreamGuard:
    @pytest.mark.asyncio
    async def test_buffer_mode_redacts_across_chunks(self):
        guard = Guard(shields=[SecretsShield(), PIIRedactor(redact_output=True)])
        out = await StreamGuard(guard, mode="buffer").collect(_gen(CHUNKS))
        assert "sk-abcdefghijklmnopqrstuv" not in out
        assert "123-45-6789" not in out
        assert "[REDACTED_OPENAI_KEY]" in out
        assert "[REDACTED_SSN]" in out

    @pytest.mark.asyncio
    async def test_incremental_mode_matches_buffer_result(self):
        guard = Guard(shields=[SecretsShield(), PIIRedactor(redact_output=True)])
        buffered = await StreamGuard(guard, mode="buffer").collect(_gen(CHUNKS))
        incremental = await StreamGuard(
            guard,
            mode="incremental",
            holdback=16,
            allow_unsafe_incremental=True,
        ).collect(_gen(CHUNKS))
        assert incremental == buffered

    @pytest.mark.asyncio
    async def test_incremental_emits_progressively(self):
        # With whitespace-delimited tokens, output streams out before the end.
        guard = Guard(shields=[])
        sg = StreamGuard(guard, mode="incremental", holdback=3)
        pieces = [p async for p in sg.scan(_gen(["aaa ", "bbb ", "ccc ", "ddd"]))]
        assert "".join(pieces) == "aaa bbb ccc ddd"
        assert len(pieces) >= 2  # streamed, not all at the end

    @pytest.mark.asyncio
    async def test_incremental_no_leak_when_token_straddles_boundary(self):
        # Regression: a secret split across chunks must never be emitted raw.
        guard = Guard(shields=[SecretsShield(on_detect="redact")])
        chunks = ["here is ", "ghp_aaaaaaaaaa", "aaaaaaaaaaaaaa", "aaaaaaaaaaaa done"]
        out = await StreamGuard(
            guard,
            mode="incremental",
            holdback=8,
            allow_unsafe_incremental=True,
        ).collect(_gen(chunks))
        assert "ghp_" not in out
        assert "[REDACTED_GITHUB_TOKEN]" in out
        # identical to the safe buffer-mode result
        buffered = await StreamGuard(guard, mode="buffer").collect(_gen(chunks))
        assert out == buffered

    @pytest.mark.asyncio
    async def test_blocking_shield_aborts_stream(self):
        ctx = SessionContext()
        ctx.metadata["canary_token"] = "AGENTGUARD-CANARY-TEST"
        guard = Guard(shields=[PromptShield(use_canary=True)])
        sg = StreamGuard(guard, ctx=ctx, mode="buffer")
        with pytest.raises(GuardBlockedError):
            await sg.collect(_gen(["safe ", "AGENTGUARD-CANARY-TEST", " leak"]))

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        out = await StreamGuard(Guard()).collect(_gen([]))
        assert out == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["buffer", "incremental"])
    async def test_stream_buffer_is_bounded_in_both_modes(self, mode):
        guard = Guard()
        stream = StreamGuard(guard, mode=mode, max_buffer_chars=5)
        with pytest.raises(GuardBlockedError) as exc:
            await stream.collect(_gen(["123", "456"]))
        assert exc.value.reason_code == "STREAM_SIZE_LIMIT_EXCEEDED"
        assert guard.stats()["blocks_by_code"]["STREAM_SIZE_LIMIT_EXCEEDED"] == 1

    @pytest.mark.asyncio
    async def test_stream_chunk_count_is_bounded(self):
        stream = StreamGuard(Guard(), max_chunks=1)
        with pytest.raises(GuardBlockedError) as exc:
            await stream.collect(_gen(["a", "b"]))
        assert exc.value.reason_code == "STREAM_CHUNK_LIMIT_EXCEEDED"

    @pytest.mark.asyncio
    async def test_non_string_chunk_fails_closed_without_stringifying(self):
        stream = StreamGuard(Guard())
        with pytest.raises(GuardBlockedError) as exc:
            await stream.collect(_gen([{"secret": "value"}]))
        assert exc.value.reason_code == "STREAM_CHUNK_TYPE_INVALID"

    @pytest.mark.asyncio
    async def test_incremental_preview_does_not_double_charge_cost(self):
        encoder = MagicMock()
        encoder.encode.return_value = list(range(10))
        cost = CostLimit(max_usd=10, model="gpt-4o")
        ctx = SessionContext()
        guard = Guard([cost])
        stream = StreamGuard(guard, ctx=ctx, mode="incremental", holdback=2)
        with patch.object(cost, "_get_encoder", return_value=encoder):
            output = await stream.collect(_gen(["one ", "two ", "three"]))
        assert output == "one two three"
        expected_once = (10 / 1_000_000) * cost.pricing["gpt-4o"]["output"]
        assert ctx.cost_usd == pytest.approx(expected_once)
        assert guard.stats()["outputs_scanned"] == 1
