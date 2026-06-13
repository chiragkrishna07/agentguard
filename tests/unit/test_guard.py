from unittest.mock import AsyncMock

import pytest

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.exceptions import GuardBlockedError, GuardShieldError
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BlockingShield(BaseShield):
    async def scan_input(self, text, ctx):
        return ShieldResult(allowed=False, reason="Always blocked", reason_code="TEST_BLOCK")


class PassShield(BaseShield):
    async def scan_input(self, text, ctx):
        return ShieldResult(allowed=True)


class UpperCaseShield(BaseShield):
    async def scan_input(self, text, ctx):
        return ShieldResult(allowed=True, modified_input=text.upper())


class ErroringShield(BaseShield):
    async def scan_input(self, text, ctx):
        raise RuntimeError("internal shield failure")


class OrderTracker(BaseShield):
    def __init__(self, name: str, log: list):
        self.name = name
        self.log = log

    async def scan_input(self, text, ctx):
        self.log.append(self.name)
        return ShieldResult(allowed=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGuardRun:
    @pytest.mark.asyncio
    async def test_no_shields_passes_through(self):
        guard = Guard()
        agent = AsyncMock(return_value="hello response")
        result = await guard.run(agent, "hello")
        agent.assert_called_once_with("hello")
        assert result == "hello response"

    @pytest.mark.asyncio
    async def test_blocking_shield_raises_and_agent_not_called(self):
        guard = Guard(shields=[BlockingShield()])
        agent = AsyncMock(return_value="response")
        with pytest.raises(GuardBlockedError) as exc_info:
            await guard.run(agent, "hello")
        assert exc_info.value.reason_code == "TEST_BLOCK"
        agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_modifying_shield_changes_query_to_agent(self):
        guard = Guard(shields=[UpperCaseShield()])
        agent = AsyncMock(return_value="ok")
        await guard.run(agent, "hello")
        agent.assert_called_once_with("HELLO")

    @pytest.mark.asyncio
    async def test_erroring_shield_fails_closed(self):
        guard = Guard(shields=[ErroringShield()])
        agent = AsyncMock(return_value="ok")
        with pytest.raises(GuardShieldError):
            await guard.run(agent, "hello")
        agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_shields_run_in_declared_order(self):
        log = []
        guard = Guard(shields=[
            OrderTracker("first", log),
            OrderTracker("second", log),
            OrderTracker("third", log),
        ])
        await guard.run(AsyncMock(return_value="ok"), "test")
        assert log == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_sync_agent_works_via_run(self):
        guard = Guard(shields=[PassShield()])
        sync_agent = lambda q, **kw: f"sync: {q}"  # noqa: E731
        result = await guard.run(sync_agent, "hi")
        assert result == "sync: hi"


class TestGuardProtectDecorator:
    @pytest.mark.asyncio
    async def test_protect_async_passes_through(self):
        guard = Guard(shields=[PassShield()])

        @guard.protect
        async def my_agent(query: str) -> str:
            return f"answer to {query}"

        result = await my_agent("test question")
        assert "test question" in result

    def test_protect_raises_on_sync_function(self):
        guard = Guard()
        with pytest.raises(TypeError):
            @guard.protect
            def sync_fn(q):
                return q

    def test_protect_sync_raises_on_async_function(self):
        guard = Guard()
        with pytest.raises(TypeError):
            @guard.protect_sync
            async def async_fn(q):
                return q

    @pytest.mark.asyncio
    async def test_protect_injects_guard_ctx(self):
        guard = Guard(shields=[PassShield()])
        ctx = SessionContext(user_id="u-123")

        @guard.protect
        async def my_agent(query: str) -> str:
            return "ok"

        await my_agent("q", _guard_ctx=ctx)
        assert ctx.request_count == 1


class TestGuardScanToolCall:
    @pytest.mark.asyncio
    async def test_tool_blocked_by_tool_validator(self):
        from agentguard.shields.tool_validator import ToolValidator

        guard = Guard(shields=[ToolValidator(blocked=["delete_*"])])
        with pytest.raises(GuardBlockedError) as exc_info:
            await guard.scan_tool_call("delete_users", {})
        assert exc_info.value.reason_code == "TOOL_NOT_ALLOWED"

    @pytest.mark.asyncio
    async def test_tool_allowed_passes_through(self):
        from agentguard.shields.tool_validator import ToolValidator

        guard = Guard(shields=[ToolValidator(allowed=["search_*"])])
        result = await guard.scan_tool_call("search_hotels", {"city": "Tokyo"})
        assert result.allowed is True
