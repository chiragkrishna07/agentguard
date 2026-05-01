import time
from typing import Dict, Literal, Tuple

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class RateLimit(BaseShield):
    """Token-bucket rate limiter.

    Each bucket starts full at `burst` tokens and refills at
    `requests_per_minute / 60` tokens per second.
    """

    def __init__(
        self,
        requests_per_minute: int,
        per: Literal["session", "global"] = "session",
        burst: int = 1,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self.requests_per_minute = requests_per_minute
        self.per = per
        self.burst = max(1, burst)
        # key → (tokens_available, last_refill_monotonic)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _bucket_key(self, ctx: SessionContext) -> str:
        return ctx.session_id if self.per == "session" else "__global__"

    def _try_consume(self, key: str) -> bool:
        now = time.monotonic()
        refill_rate = self.requests_per_minute / 60.0

        if key not in self._buckets:
            # First request: start with a full bucket and consume one token
            self._buckets[key] = (float(self.burst) - 1.0, now)
            return True

        tokens, last = self._buckets[key]
        tokens = min(float(self.burst), tokens + (now - last) * refill_rate)

        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False

        self._buckets[key] = (tokens - 1.0, now)
        return True

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        key = self._bucket_key(ctx)
        if not self._try_consume(key):
            return ShieldResult(
                allowed=False,
                reason=f"Rate limit exceeded: max {self.requests_per_minute} requests/minute",
                reason_code="RATE_LIMIT_EXCEEDED",
            )
        return ShieldResult(allowed=True)
