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
  that matters.

A blocking output shield (e.g. a triggered canary) raises ``GuardBlockedError``
out of the generator, aborting the stream.
"""
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

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
    ) -> None:
        if mode not in ("buffer", "incremental"):
            raise ValueError("mode must be 'buffer' or 'incremental'")
        if holdback < 1:
            raise ValueError("holdback must be >= 1")
        self.guard = guard
        self.ctx = ctx
        self.mode = mode
        self.holdback = holdback

    async def scan(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield sanitised output chunks from a stream of raw chunks."""
        ctx = self.ctx or SessionContext()

        if self.mode == "buffer":
            buffer = "".join([chunk async for chunk in chunks])
            yield await self.guard._scan_output(buffer, ctx)
            return

        buffer = ""
        emitted = 0  # index into the sanitised *frozen* prefix already yielded
        async for chunk in chunks:
            buffer += chunk
            cut = self._frozen_cut(buffer)
            if cut <= 0:
                continue
            # The frozen prefix ends at whitespace, so every token inside it is
            # complete; scanning it gives a result that can only be *extended*
            # (never rewritten) as more whole tokens are appended — which makes
            # `emitted` a safe, monotonic index.
            sanitized = await self.guard._scan_output(buffer[:cut], ctx)
            if len(sanitized) > emitted:
                yield sanitized[emitted:]
                emitted = len(sanitized)

        # Flush: scan the entire buffer and emit whatever is left.
        sanitized = await self.guard._scan_output(buffer, ctx)
        if len(sanitized) > emitted:
            yield sanitized[emitted:]

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
