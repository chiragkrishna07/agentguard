"""
StreamGuard — run the output shields over a streamed LLM response.

Streaming is awkward for a guard: once a token is shown to the user you can't
un-show it, but redaction needs to see the whole match. StreamGuard offers two
honest strategies:

- ``mode="buffer"`` (default, safe): accumulate the full stream, scan once, then
  yield the sanitised result. Correct for any match length; gives up streaming
  latency.
- ``mode="incremental"``: emit only a *frozen* prefix of the output — the text
  up to the last whitespace boundary at least ``holdback`` characters back from
  the live end. Because a frozen prefix always ends at whitespace, no
  whitespace-delimited token (API keys, SSNs, emails, JWTs) is ever split or
  half-emitted, regardless of its length. Lower latency than buffering. Caveat:
  matches that *contain* whitespace (a space-separated credit card, a multi-line
  PEM block) are only reliably redacted in buffer mode — use buffer mode when
  that matters. Shields which declare full-output requirements make incremental
  construction fail by default; ``allow_unsafe_incremental=True`` is an explicit
  compatibility override.

A blocking output shield (e.g. a triggered canary) raises ``GuardBlockedError``
out of the generator, aborting the stream.
"""

import copy
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from agentguard.core.base_shield import GuardDecision
from agentguard.core.exceptions import GuardBlockedError
from agentguard.core.session import SessionContext

if TYPE_CHECKING:
    from agentguard.core.guard import Guard


class StreamGuard:
    def __init__(
        self,
        guard: "Guard",
        ctx: SessionContext | None = None,
        mode: Literal["buffer", "incremental"] = "buffer",
        holdback: int = 256,
        *,
        max_buffer_chars: int = 1_000_000,
        max_buffer_bytes: int = 4_000_000,
        max_chunks: int = 100_000,
        allow_unsafe_incremental: bool = False,
    ) -> None:
        if mode not in ("buffer", "incremental"):
            raise ValueError("mode must be 'buffer' or 'incremental'")
        if holdback < 1:
            raise ValueError("holdback must be >= 1")
        for name, value in {
            "max_buffer_chars": max_buffer_chars,
            "max_buffer_bytes": max_buffer_bytes,
            "max_chunks": max_chunks,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be >= 1")
        if not isinstance(allow_unsafe_incremental, bool):
            raise ValueError("allow_unsafe_incremental must be boolean")
        buffered_shields = [
            shield.__class__.__name__
            for shield in guard.shields
            if shield.requires_buffered_output
        ]
        if mode == "incremental" and buffered_shields and not allow_unsafe_incremental:
            names = ", ".join(buffered_shields)
            raise ValueError(
                "incremental streaming would release content before full-output "
                f"policies complete ({names}); use buffer mode or explicitly set "
                "allow_unsafe_incremental=True"
            )
        self.guard = guard
        self.ctx = ctx
        self.mode = mode
        self.holdback = holdback
        self.max_buffer_chars = max_buffer_chars
        self.max_buffer_bytes = max_buffer_bytes
        self.max_chunks = max_chunks
        self.allow_unsafe_incremental = allow_unsafe_incremental

    async def _bounded_chunks(
        self, chunks: AsyncIterator[str], ctx: SessionContext
    ) -> AsyncIterator[str]:
        chars = 0
        byte_count = 0
        chunk_count = 0
        async for chunk in chunks:
            if not isinstance(chunk, str):
                await self._block(
                    ctx,
                    "Stream yielded a non-string chunk",
                    "STREAM_CHUNK_TYPE_INVALID",
                )
            chunk_count += 1
            chars += len(chunk)
            byte_count += len(chunk.encode("utf-8"))
            if chunk_count > self.max_chunks:
                await self._block(
                    ctx,
                    f"Stream exceeds {self.max_chunks} chunks",
                    "STREAM_CHUNK_LIMIT_EXCEEDED",
                )
            if chars > self.max_buffer_chars or byte_count > self.max_buffer_bytes:
                await self._block(
                    ctx,
                    "Stream exceeds the configured character or byte buffer limit",
                    "STREAM_SIZE_LIMIT_EXCEEDED",
                )
            yield chunk

    async def _block(self, ctx: SessionContext, reason: str, code: str) -> None:
        self.guard.metrics.record_block(self.__class__.__name__, code)
        await self.guard._notify_decision(
            GuardDecision(
                flow="output",
                allowed=False,
                shield_name=self.__class__.__name__,
                reason_code=code,
            ),
            ctx,
        )
        raise GuardBlockedError(reason, code, self.__class__.__name__)

    async def scan(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield sanitised output chunks from a stream of raw chunks."""
        ctx = self.ctx or SessionContext()
        bounded = self._bounded_chunks(chunks, ctx)

        if self.mode == "buffer":
            buffer = "".join([chunk async for chunk in bounded])
            yield await self.guard._scan_output(buffer, ctx)
            return

        buffer = ""
        emitted = 0  # index into the sanitised *frozen* prefix already yielded
        async for chunk in bounded:
            buffer += chunk
            cut = self._frozen_cut(buffer)
            if cut <= 0:
                continue
            # The frozen prefix ends at whitespace, so every token inside it is
            # complete; scanning it gives a result that can only be *extended*
            # (never rewritten) as more whole tokens are appended — which makes
            # `emitted` a safe, monotonic index.
            sanitized = await self.guard.scan_output_preview(
                buffer[:cut], self._preview_context(ctx)
            )
            if len(sanitized) > emitted:
                yield sanitized[emitted:]
                emitted = len(sanitized)

        # Flush: scan the entire buffer and emit whatever is left.
        sanitized = await self.guard._scan_output(buffer, ctx)
        if len(sanitized) > emitted:
            yield sanitized[emitted:]

    @staticmethod
    def _preview_context(ctx: SessionContext) -> SessionContext:
        """Clone mutable session data so provisional scans have no state effects."""
        try:
            metadata = copy.deepcopy(ctx.metadata)
        except (TypeError, ValueError):
            metadata = dict(ctx.metadata)
        preview = SessionContext(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            cost_usd=ctx.cost_usd,
            request_count=ctx.request_count,
            metadata=metadata,
        )
        preview._token_map = dict(ctx._token_map)
        return preview

    def _frozen_cut(self, buffer: str) -> int:
        """Largest index <= len-holdback that sits just after a whitespace char.

        Cutting there guarantees the frozen prefix ends on a token boundary, so a
        token still being streamed is never split across the emit boundary.
        """
        limit = min(len(buffer) - self.holdback, len(buffer))
        for c in range(limit, 0, -1):
            if buffer[c - 1].isspace():
                return c
        return 0

    async def collect(self, chunks: AsyncIterator[str]) -> str:
        """Convenience: consume the scanned stream and return the full string."""
        return "".join([out async for out in self.scan(chunks)])
