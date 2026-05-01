"""
GuardedTool — wrap any callable so its invocation passes through ToolValidator
and HumanGate shields before execution.
"""
import asyncio
from typing import TYPE_CHECKING, Any, Callable, Optional

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
            return await self._fn(**kwargs)
        return self._fn(**kwargs)

    def __repr__(self) -> str:
        return f"GuardedTool(fn={self.__name__!r})"
