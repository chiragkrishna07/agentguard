import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.rate_limit import RateLimit


@pytest.fixture
def ctx():
    return SessionContext()


class TestRateLimit:
    def test_invalid_rpm_raises(self):
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=0)

    def test_invalid_configuration(self):
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=1, burst=0)
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=1, per="tenant")
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=1, max_buckets=0)
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=float("nan"))
        with pytest.raises(ValueError):
            RateLimit(requests_per_minute=1, key_fn="tenant")

    @pytest.mark.asyncio
    async def test_first_request_always_allowed(self, ctx):
        shield = RateLimit(requests_per_minute=10, burst=1)
        result = await shield.scan_input("hello", ctx)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_burst_allows_multiple_quick_requests(self, ctx):
        shield = RateLimit(requests_per_minute=60, burst=3)
        results = [await shield.scan_input("msg", ctx) for _ in range(3)]
        assert all(r.allowed for r in results)

    @pytest.mark.asyncio
    async def test_exceeded_burst_is_blocked(self, ctx):
        shield = RateLimit(requests_per_minute=1, burst=1)
        await shield.scan_input("first", ctx)
        result = await shield.scan_input("second", ctx)
        assert result.allowed is False
        assert result.reason_code == "RATE_LIMIT_EXCEEDED"

    @pytest.mark.asyncio
    async def test_per_session_isolation(self):
        shield = RateLimit(requests_per_minute=1, burst=1, per="session")
        ctx1 = SessionContext()
        ctx2 = SessionContext()

        await shield.scan_input("req", ctx1)
        # ctx1 bucket is empty; ctx2 starts fresh
        result = await shield.scan_input("req", ctx2)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_per_user_spans_sessions_and_requires_identity(self):
        shield = RateLimit(requests_per_minute=1, burst=1, per="user")
        anonymous = await shield.scan_input("x", SessionContext())
        assert anonymous.allowed is False
        assert anonymous.reason_code == "RATE_LIMIT_IDENTITY_REQUIRED"

        one = SessionContext(user_id="same-user")
        two = SessionContext(user_id="same-user")
        assert (await shield.scan_input("x", one)).allowed
        assert not (await shield.scan_input("x", two)).allowed

    @pytest.mark.asyncio
    async def test_custom_key_function(self):
        shield = RateLimit(
            requests_per_minute=1,
            burst=1,
            key_fn=lambda ctx: str(ctx.metadata.get("tenant_id", "")),
        )
        ctx = SessionContext(metadata={"tenant_id": "t-1"})
        assert (await shield.scan_input("x", ctx)).allowed
        assert not (await shield.scan_input("x", ctx)).allowed

    @pytest.mark.asyncio
    async def test_bucket_storage_is_bounded(self):
        shield = RateLimit(requests_per_minute=1, max_buckets=2)
        for _ in range(5):
            assert (await shield.scan_input("x", SessionContext())).allowed
        assert len(shield._buckets) == 2

    @pytest.mark.asyncio
    async def test_limit_records_retry_metadata(self):
        shield = RateLimit(requests_per_minute=1, burst=1)
        ctx = SessionContext()
        assert (await shield.scan_input("x", ctx)).allowed
        result = await shield.scan_input("x", ctx)
        assert not result.allowed
        assert ctx.metadata["rate_limit"]["retry_after_seconds"] > 0

    @pytest.mark.asyncio
    async def test_global_scope_shared_across_sessions(self):
        shield = RateLimit(requests_per_minute=1, burst=1, per="global")
        ctx1 = SessionContext()
        ctx2 = SessionContext()

        await shield.scan_input("req", ctx1)
        result = await shield.scan_input("req", ctx2)
        # Both share the same global bucket — second should be blocked
        assert result.allowed is False

    def test_concurrent_threads_never_exceed_burst(self):
        # A single shield shared across OS threads must not over-admit.
        import threading

        shield = RateLimit(requests_per_minute=1, burst=5, per="global")
        ctx = SessionContext()
        allowed = []
        lock = threading.Lock()

        def worker():
            import asyncio

            r = asyncio.run(shield.scan_input("x", ctx))
            if r.allowed:
                with lock:
                    allowed.append(1)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With burst=5 and a negligibly small refill over the test window,
        # at most a handful beyond 5 could be admitted via refill; the lock
        # guarantees we never grossly over-admit (which the race would).
        assert sum(allowed) <= 6
