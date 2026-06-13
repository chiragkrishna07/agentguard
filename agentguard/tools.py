"""
GuardedTool — wrap any callable so its invocation passes through ToolValidator
and HumanGate shields before execution.
"""
import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext


class GuardedTool:
    """Wraps a tool function (sync or async) with AgentGuard shield scanning.

    Usage
    -----
    guarded = GuardedTool(my_tool_fn, guard, ctx)
    result  = await guarded(city="Tokyo", nights=2)
    """

    def __init__(
        self,
        fn: Callable,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
    ) -> None:
        self._fn = fn
        self._guard = guard
        self._ctx = ctx
        self.__name__: str = getattr(fn, "__name__", repr(fn))
        self.__doc__ = fn.__doc__

    async def __call__(self, **kwargs: Any) -> Any:
        from agentguard.core.session import SessionContext as _SessionContext

        ctx = self._ctx or _SessionContext()
        await self._guard.scan_tool_call(self.__name__, kwargs, ctx)

        if asyncio.iscoroutinefunction(self._fn):
            result = await self._fn(**kwargs)
        else:
            result = self._fn(**kwargs)

        # Inspect the returned content for indirect prompt injection and other
        # threats before it flows back into the agent. Only string results are
        # rewritten in place; for other types we scan their text form for
        # block decisions but return the original object untouched.
        if isinstance(result, str):
            return await self._guard.scan_tool_output(self.__name__, result, ctx)
        if isinstance(result, (dict, list)):
            await self._guard.scan_tool_output(self.__name__, str(result), ctx)
        return result

    def __repr__(self) -> str:
        return f"GuardedTool(fn={self.__name__!r})"
