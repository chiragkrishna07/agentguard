"""
GuardCrewAI — AgentGuard adapter for CrewAI.

Usage
-----
from agentguard.adapters.crewai import GuardCrewAI

adapter = GuardCrewAI(guard)

# Scan inputs before crew kickoff
result = await adapter.kickoff(crew, inputs={"topic": "AI security"})

# Wrap a tool function
search = adapter.wrap_tool(search_fn)
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext


class GuardCrewAI:
    def __init__(
        self,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
    ) -> None:
        self.guard = guard
        from agentguard.core.session import SessionContext as _Ctx
        self.ctx = ctx or _Ctx()

    async def kickoff(self, crew: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Scan text inputs before crew kickoff. Works with both sync and async crews."""
        import asyncio

        inputs = dict(inputs or {})
        for key in ("query", "input", "topic", "task"):
            if key in inputs and isinstance(inputs[key], str):
                inputs[key] = await self.guard._scan_input(inputs[key], self.ctx)
                break  # Only scan the first text field found

        if asyncio.iscoroutinefunction(crew.kickoff):
            return await crew.kickoff(inputs=inputs)
        return crew.kickoff(inputs=inputs)

    def wrap_tool(self, fn: Callable) -> "GuardedTool":  # type: ignore[name-defined]
        from agentguard.tools import GuardedTool
        return GuardedTool(fn, self.guard, self.ctx)
