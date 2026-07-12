"""
GuardCrewAI — AgentGuard adapter for CrewAI.

Usage
-----
from agentguard.adapters.crewai import GuardCrewAI

adapter = GuardCrewAI(guard)
result = await adapter.kickoff(crew, inputs={"topic": "AI security"})

# Wrap a tool function
search = adapter.wrap_tool(search_fn)
"""

import inspect
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext
    from agentguard.tools import GuardedTool


_SESSION_KEYS = ("thread_id", "session_id", "conversation_id", "crew_id")

class GuardCrewAI:
    def __init__(
        self,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
        *,
        max_session_contexts: int = 1024,
        trust_input_identity: bool = False,
    ) -> None:
        if max_session_contexts < 1:
            raise ValueError("max_session_contexts must be at least 1")
        if not isinstance(trust_input_identity, bool):
            raise ValueError("trust_input_identity must be boolean")
        self.guard = guard
        self.ctx = ctx
        self._max_session_contexts = max_session_contexts
        self._trust_input_identity = trust_input_identity
        self._contexts: OrderedDict[tuple[str | None, str], SessionContext] = OrderedDict()
        self._contexts_lock = threading.Lock()

    def _context_for_inputs(
        self,
        inputs: dict[str, Any],
        explicit: Optional["SessionContext"],
    ) -> "SessionContext":
        if explicit is not None:
            return explicit
        if self.ctx is not None:
            return self.ctx

        from agentguard.core.session import SessionContext as _Ctx

        # Crew inputs are usually model/user-controlled application data.
        # Never turn an arbitrary ``session_id``/``user_id`` field into an
        # authorization or rate-limit identity unless the caller explicitly
        # opts in after deriving those fields from authenticated runtime state.
        if not self._trust_input_identity:
            return _Ctx()

        configurable = inputs.get("configurable", {})
        if not isinstance(configurable, dict):
            configurable = {}
        raw_id = next(
            (configurable[key] for key in _SESSION_KEYS if configurable.get(key) is not None),
            None,
        )
        if raw_id is None:
            raw_id = next(
                (inputs[key] for key in _SESSION_KEYS if inputs.get(key) is not None),
                None,
            )
        if raw_id is None:
            return _Ctx()

        session_id = str(raw_id)
        user_id = configurable.get("user_id", inputs.get("user_id"))
        normalized_user_id = str(user_id) if user_id is not None else None
        cache_key = (normalized_user_id, session_id)
        with self._contexts_lock:
            ctx = self._contexts.get(cache_key)
            if ctx is None:
                ctx = _Ctx(
                    session_id=session_id,
                    user_id=normalized_user_id,
                )
                self._contexts[cache_key] = ctx
                if len(self._contexts) > self._max_session_contexts:
                    self._contexts.popitem(last=False)
            else:
                self._contexts.move_to_end(cache_key)
            return ctx

    async def _scan_result(self, result: Any, ctx: "SessionContext") -> Any:
        # Core scans every public field on Pydantic/dataclass/content-bearing
        # objects. Avoid a field whitelist: CrewOutput variants may place
        # security-relevant text in ``pydantic``, ``tasks_output``, metadata,
        # citations, or future fields in addition to ``raw``/``json_dict``.
        return await self.guard.scan_output(result, ctx)

    async def kickoff(
        self,
        crew: Any,
        inputs: dict[str, Any] | None = None,
        *,
        _guard_ctx: Optional["SessionContext"] = None,
    ) -> Any:
        """Guard complete CrewAI inputs and the typed kickoff result."""
        input_values = dict(inputs or {})
        ctx = self._context_for_inputs(input_values, _guard_ctx)
        sanitized_inputs = await self.guard.scan_input(input_values, ctx)

        result = crew.kickoff(inputs=sanitized_inputs)
        if inspect.isawaitable(result):
            result = await result
        return await self._scan_result(result, ctx)

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool

        return GuardedTool(fn, self.guard, self.ctx)
