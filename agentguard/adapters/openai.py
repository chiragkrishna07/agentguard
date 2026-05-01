"""
GuardOpenAI — AgentGuard adapter for the raw OpenAI SDK.

Usage
-----
from agentguard.adapters.openai import GuardOpenAI
from openai import AsyncOpenAI

adapter = GuardOpenAI(guard)
client  = AsyncOpenAI()

# Scans user message before sending and scans response before returning
response = await adapter.create(client, model="gpt-4o", messages=[...])

# Wrap a tool function
search = adapter.wrap_tool(search_fn)
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext


class GuardOpenAI:
    def __init__(
        self,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
    ) -> None:
        self.guard = guard
        from agentguard.core.session import SessionContext as _Ctx
        self.ctx = ctx or _Ctx()

    async def create(self, client: Any, **kwargs: Any) -> Any:
        """Drop-in replacement for client.chat.completions.create with guard scanning."""
        messages: List[Dict[str, Any]] = list(kwargs.get("messages", []))
        if messages:
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") == "user":
                sanitized = await self.guard._scan_input(last["content"], self.ctx)
                messages[-1] = {**last, "content": sanitized}
                kwargs = {**kwargs, "messages": messages}

        response = await client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        await self.guard._scan_output(content, self.ctx)
        return response

    def wrap_tool(self, fn: Callable) -> "GuardedTool":  # type: ignore[name-defined]
        from agentguard.tools import GuardedTool
        return GuardedTool(fn, self.guard, self.ctx)
