"""
GuardLangGraph — AgentGuard adapter for LangGraph agents.

Usage
-----
from agentguard.adapters.langgraph import GuardLangGraph

adapter = GuardLangGraph(guard)

# Wrap a node function
@adapter.wrap_node
async def call_model(state):
    ...

# Wrap a tool function
search = adapter.wrap_tool(search_fn)
"""
import asyncio
import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext
    from agentguard.tools import GuardedTool


class GuardLangGraph:
    def __init__(
        self,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
    ) -> None:
        self.guard = guard
        from agentguard.core.session import SessionContext as _Ctx
        self.ctx = ctx or _Ctx()

    def wrap_node(self, fn: Callable) -> Callable:
        """Scan the last user message in state['messages'] through input shields."""

        @functools.wraps(fn)
        async def wrapper(state: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                content = last.content if hasattr(last, "content") else str(last)
                sanitized = await self.guard._scan_input(content, self.ctx)
                if hasattr(last, "content"):
                    last.content = sanitized
                else:
                    messages[-1] = sanitized

            if asyncio.iscoroutinefunction(fn):
                return await fn(state, *args, **kwargs)
            return fn(state, *args, **kwargs)

        return wrapper

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool
        return GuardedTool(fn, self.guard, self.ctx)
