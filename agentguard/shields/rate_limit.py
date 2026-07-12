import math
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from numbers import Real
from typing import Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class RateLimit(BaseShield):
    """Token-bucket rate limiter.

    Each bucket starts full at `burst` tokens and refills at
    `requests_per_minute / 60` tokens per second.
    """

    scan_tool_arguments_as_input = False

    def __init__(
        self,
        requests_per_minute: int,
        per: Literal["session", "user", "global"] = "session",
        burst: int = 1,
        *,
        max_buckets: int = 10_000,
        bucket_ttl_seconds: float = 3_600,
        key_fn: Callable[[SessionContext], str] | None = None,
        on_limit: Literal["block", "warn"] = "block",
    ) -> None:
        if (
            isinstance(requests_per_minute, bool)
            or not isinstance(requests_per_minute, Real)
            or not math.isfinite(float(requests_per_minute))
            or requests_per_minute <= 0
        ):
            raise ValueError("requests_per_minute must be > 0")
        if per not in ("session", "user", "global"):
            raise ValueError("per must be 'session', 'user', or 'global'")
        if isinstance(burst, bool) or not isinstance(burst, int) or burst < 1:
            raise ValueError("burst must be >= 1")
        if isinstance(max_buckets, bool) or not isinstance(max_buckets, int) or max_buckets < 1:
            raise ValueError("max_buckets must be >= 1")
        if (
            isinstance(bucket_ttl_seconds, bool)
            or not isinstance(bucket_ttl_seconds, Real)
            or not math.isfinite(float(bucket_ttl_seconds))
            or bucket_ttl_seconds <= 0
        ):
            raise ValueError("bucket_ttl_seconds must be > 0")
        if on_limit not in ("block", "warn"):
            raise ValueError("on_limit must be 'block' or 'warn'")
        if key_fn is not None and not callable(key_fn):
            raise ValueError("key_fn must be callable or None")
        self.requests_per_minute: int = requests_per_minute
        self.per: Literal["session", "user", "global"] = per
        self.burst: int = burst
        self.max_buckets: int = max_buckets
        self.bucket_ttl_seconds: float = bucket_ttl_seconds
        self.key_fn: Callable[[SessionContext], str] | None = key_fn
        self.on_limit: Literal["block", "warn"] = on_limit
        # key → (tokens_available, last_refill_monotonic)
        # OrderedDict makes the state bound an O(1) LRU eviction instead of an
        # attacker-controlled, ever-growing session-id map.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        # Guards bucket read-modify-write so the limiter stays correct when a
        # single shield instance is shared across OS threads (e.g. a threaded
        # WSGI server, or protect_sync called from multiple threads).
        self._lock: threading.Lock = threading.Lock()

    def _bucket_key(self, ctx: SessionContext) -> str | None:
        if self.key_fn is not None:
            key = self.key_fn(ctx)
            return str(key) if key is not None else None
        if self.per == "session":
            return ctx.session_id
        if self.per == "user":
            return ctx.user_id
        return "__global__"

    def _evict(self, now: float) -> None:
        while self._buckets:
            _, (_, last) = next(iter(self._buckets.items()))
            if now - last <= self.bucket_ttl_seconds:
                break
            self._buckets.popitem(last=False)
        while len(self._buckets) >= self.max_buckets:
            self._buckets.popitem(last=False)

    def _try_consume(self, key: str) -> tuple[bool, float]:
        with self._lock:
            now = time.monotonic()
            refill_rate = self.requests_per_minute / 60.0

            if key not in self._buckets:
                self._evict(now)
                # First request: start with a full bucket and consume one token
                self._buckets[key] = (float(self.burst) - 1.0, now)
                return True, 0.0

            tokens, last = self._buckets[key]
            tokens = min(float(self.burst), tokens + (now - last) * refill_rate)
            self._buckets.move_to_end(key)

            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False, max(0.0, (1.0 - tokens) / refill_rate)

            self._buckets[key] = (tokens - 1.0, now)
            return True, 0.0

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        key = self._bucket_key(ctx)
        if not key:
            return ShieldResult(
                allowed=False,
                reason="Rate-limit identity key is required",
                reason_code="RATE_LIMIT_IDENTITY_REQUIRED",
            )
        allowed, retry_after = self._try_consume(key)
        if not allowed:
            ctx.metadata["rate_limit"] = {
                "limited": True,
                "retry_after_seconds": round(retry_after, 3),
                "scope": self.per,
            }
            reason = (
                f"Rate limit exceeded: max {self.requests_per_minute} requests/minute; "
                f"retry after {retry_after:.3f}s"
            )
            if self.on_limit == "warn":
                warnings.warn(f"[AgentGuard RateLimit] {reason}", stacklevel=4)
                return ShieldResult(allowed=True)
            return ShieldResult(
                allowed=False,
                reason=reason,
                reason_code="RATE_LIMIT_EXCEEDED",
            )
        return ShieldResult(allowed=True)
