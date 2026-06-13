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
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext
    from agentguard.tools import GuardedTool


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
        messages: list[dict[str, Any]] = list(kwargs.get("messages", []))
        if messages:
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") == "user":
                sanitized = await self.guard._scan_input(last["content"], self.ctx)
                messages[-1] = {**last, "content": sanitized}
                kwargs = {**kwargs, "messages": messages}

        response = await client.chat.completions.create(**kwargs)

        message = response.choices[0].message

        # Validate any tool calls the model decided to make BEFORE the caller
        # executes them — this is where ToolValidator / HumanGate get their say
        # on model-chosen actions and arguments.
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", "")
            raw_args = getattr(fn, "arguments", "") or "{}"
            try:
                params = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                params = {"_raw": raw_args}
            if not isinstance(params, dict):
                params = {"_value": params}
            await self.guard.scan_tool_call(name, params, self.ctx)

        content = message.content or ""
        await self.guard._scan_output(content, self.ctx)
        return response

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool
        return GuardedTool(fn, self.guard, self.ctx)
