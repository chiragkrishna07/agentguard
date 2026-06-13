"""
StreamGuard — run the output shields over a streamed LLM response.

Streaming is awkward for a guard: once a token is shown to the user you can't
un-show it, but redaction needs to see the whole match. StreamGuard offers two
honest strategies:

- ``mode="buffer"`` (default, safe): accumulate the full stream, scan once, then
  yield the sanitised result. Correct for any match length; gives up streaming
  latency.
- ``mode="incremental"``: re-scan the growing buffer after each chunk and emit
  the *stable* sanitised prefix, always holding back the last ``holdback``
  characters so a secret/PII token near the tail is never half-emitted. Lower
  latency, but only guaranteed for matches up to ``holdback`` characters — use
  buffer mode if you must catch arbitrarily long secrets (e.g. PEM blocks).

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
        emitted = 0
        async for chunk in chunks:
            buffer += chunk
            sanitized = await self.guard._scan_output(buffer, ctx)
            stable_end = max(0, len(sanitized) - self.holdback)
            if stable_end > emitted:
                yield sanitized[emitted:stable_end]
                emitted = stable_end

        # Flush whatever remained inside the holdback window.
        sanitized = await self.guard._scan_output(buffer, ctx)
        if len(sanitized) > emitted:
            yield sanitized[emitted:]

    async def collect(self, chunks: AsyncIterator[str]) -> str:
        """Convenience: consume the scanned stream and return the full string."""
        return "".join([out async for out in self.scan(chunks)])
