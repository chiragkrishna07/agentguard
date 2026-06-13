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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from agentguard.core.content import scan_joined_text

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
        async def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                scan = lambda t: self.guard._scan_input(t, self.ctx)  # noqa: E731
                if hasattr(last, "content"):
                    # LangChain message — content may be str or a list of parts.
                    last.content = await scan_joined_text(last.content, scan)
                else:
                    messages[-1] = await scan_joined_text(last, scan)

            if asyncio.iscoroutinefunction(fn):
                return await fn(state, *args, **kwargs)
            return fn(state, *args, **kwargs)

        return wrapper

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool
        return GuardedTool(fn, self.guard, self.ctx)
