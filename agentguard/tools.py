"""
GuardedTool — wrap any callable so its invocation passes through ToolValidator
and HumanGate shields before execution.
"""

import inspect
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

    A wrapper without a fixed context may receive the reserved
    ``_guard_ctx=SessionContext(...)`` keyword per call. Stateful tool shields
    require one of these explicit context forms.
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
        self._requires_session_context = any(
            shield.requires_tool_session_context for shield in guard.shields
        )
        self.__name__: str = getattr(fn, "__name__", repr(fn))
        self.__doc__ = fn.__doc__

    def _bind_call(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], inspect.BoundArguments | None]:
        """Bind invocation arguments into a sanitizable, named structure."""
        try:
            signature = inspect.signature(self._fn)
        except (TypeError, ValueError):
            # Some extension/builtin callables do not expose a signature.  Keep
            # both channels distinct so rebuild cannot lose a caller value.
            return {"_args": args, "_kwargs": kwargs}, None

        # ``bind`` validates positional-only and required arguments before any
        # policy decision. A failed bind must never fall back to an ambiguous
        # reconstruction and must never execute the tool.
        bound = signature.bind(*args, **kwargs)
        params: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            parameter = signature.parameters[name]
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                # A generic ``def tool(**kwargs)`` must not hide every actual
                # argument beneath a synthetic "kwargs" key from validators.
                params.update(value)
            else:
                params[name] = value
        return params, bound

    @staticmethod
    def _rebuild_call(
        sanitized: dict[str, Any],
        bound: inspect.BoundArguments | None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Rebuild positional-only, ``*args`` and ``**kwargs`` fail-closed."""
        if bound is None:
            if set(sanitized) != {"_args", "_kwargs"}:
                raise TypeError("sanitized tool arguments changed fallback structure")
            args = sanitized["_args"]
            kwargs = sanitized["_kwargs"]
            if not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise TypeError("sanitized tool arguments changed invocation types")
            return args, kwargs

        signature = bound.signature
        rebuilt: dict[str, Any] = {}
        consumed: set[str] = set()
        for name, original in bound.arguments.items():
            parameter = signature.parameters[name]
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                original_keys = set(original)
                if not original_keys.issubset(sanitized):
                    raise TypeError("sanitized tool arguments lost keyword arguments")
                rebuilt[name] = {key: sanitized[key] for key in original}
                consumed.update(original_keys)
            else:
                if name not in sanitized:
                    raise TypeError(f"sanitized tool arguments lost parameter {name!r}")
                value = sanitized[name]
                if parameter.kind is inspect.Parameter.VAR_POSITIONAL and not isinstance(
                    value, tuple
                ):
                    raise TypeError("sanitized tool arguments changed *args type")
                rebuilt[name] = value
                consumed.add(name)

        if consumed != set(sanitized):
            raise TypeError("sanitized tool arguments introduced unknown parameters")
        bound.arguments.clear()
        bound.arguments.update(rebuilt)
        return bound.args, bound.kwargs

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from agentguard.core.exceptions import GuardShieldError, GuardToolError
        from agentguard.core.session import SessionContext

        call_ctx = kwargs.pop("_guard_ctx", None)
        if call_ctx is not None and not isinstance(call_ctx, SessionContext):
            raise TypeError("_guard_ctx must be a SessionContext")
        if self._ctx is not None and call_ctx is not None and call_ctx is not self._ctx:
            raise ValueError("a fixed-context GuardedTool cannot override _guard_ctx")
        ctx = self._ctx or call_ctx
        if ctx is None and self._requires_session_context:
            raise GuardShieldError(
                self.__class__.__name__,
                "an explicit SessionContext is required by a stateful tool shield",
            ) from None
        ctx = ctx or SessionContext()
        params, bound = self._bind_call(args, kwargs)
        sanitized = await self._guard.scan_tool_arguments(self.__name__, params, ctx)
        safe_args, safe_kwargs = self._rebuild_call(sanitized, bound)

        try:
            result = self._fn(*safe_args, **safe_kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            # Tool/provider exceptions regularly echo request bodies, remote
            # responses, secrets, and even indirect-injection text. Never let
            # that raw channel bypass output scanning by default.
            ctx.metadata["tool_execution_error"] = {
                "tool_name": self.__name__,
                "exception_type": type(exc).__name__,
            }
            if self._guard.expose_internal_errors:
                raise GuardToolError(self.__name__, str(exc)) from exc
            raise GuardToolError(self.__name__) from None

        # Inspect the returned content for indirect prompt injection and other
        # threats before it flows back into the agent. Structured rewrites are
        # propagated while preserving dict/list/tuple and scalar field types.
        return await self._guard.scan_tool_output(self.__name__, result, ctx)

    def __repr__(self) -> str:
        return f"GuardedTool(fn={self.__name__!r})"
