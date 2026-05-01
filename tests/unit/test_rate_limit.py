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
    async def test_global_scope_shared_across_sessions(self):
        shield = RateLimit(requests_per_minute=1, burst=1, per="global")
        ctx1 = SessionContext()
        ctx2 = SessionContext()

        await shield.scan_input("req", ctx1)
        result = await shield.scan_input("req", ctx2)
        # Both share the same global bucket — second should be blocked
        assert result.allowed is False
