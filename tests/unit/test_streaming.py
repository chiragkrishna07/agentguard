import pytest

from agentguard.core.exceptions import GuardBlockedError
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext
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
        incremental = await StreamGuard(guard, mode="incremental", holdback=16).collect(
            _gen(CHUNKS)
        )
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
        out = await StreamGuard(guard, mode="incremental", holdback=8).collect(
            _gen(chunks)
        )
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
