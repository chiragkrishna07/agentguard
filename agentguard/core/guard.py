import asyncio
import functools
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from numbers import Number
from types import SimpleNamespace
from typing import Any, Awaitable, NoReturn, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel

from agentguard.core.base_shield import BaseShield, GuardDecision, GuardFlow, ShieldResult
from agentguard.core.exceptions import (
    GuardBlockedError,
    GuardShieldError,
    HumanGateSyncError,
)
from agentguard.core.metrics import GuardMetrics
from agentguard.core.session import SessionContext

_ContentT = TypeVar("_ContentT")
_CONTENT_ATTRIBUTES = (
    "content",
    "text",
    "page_content",
    "raw",
    "json_dict",
    "pydantic",
    "tasks_output",
    "output",
    "result",
)


class _StructureError(ValueError):
    def __init__(self, message: str, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


def _is_safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, Number, date, datetime, time, UUID, Enum))


def _object_field_items(value: Any) -> list[tuple[str, Any]] | None:
    """Return safely readable fields for supported content-bearing objects."""
    if isinstance(value, BaseModel):
        raw = vars(value)
        items = [(name, item) for name, item in raw.items() if not name.startswith("_")]
        extra = getattr(value, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            items.extend((name, item) for name, item in extra.items() if name not in raw)
        return items

    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, object.__getattribute__(value, field.name)) for field in fields(value)]

    try:
        raw = vars(value)
    except TypeError:
        return None
    selected = [name for name in _CONTENT_ATTRIBUTES if name in raw]
    if not selected:
        return None
    # Once an object advertises a recognized content field, all public fields
    # are part of the untrusted payload. Document metadata, citations, parsed
    # attributes, and peer-agent annotations can carry the same injection or
    # secret as ``page_content`` itself.
    return [(name, item) for name, item in raw.items() if not name.startswith("_")]


def _container_items(value: Any) -> list[tuple[str | None, Any]] | None:
    if isinstance(value, dict):
        return [(key if isinstance(key, str) else None, item) for key, item in value.items()]
    object_items = _object_field_items(value)
    if object_items is not None:
        return list(object_items)
    return None


def _text_leaves(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
    max_chars: int | None,
    max_bytes: int | None,
) -> list[str]:
    """Return string values using a bounded, cycle-aware traversal.

    This value-only view is used to rebuild sanitised values. A separate
    contextual surface exposes dictionary keys and numbers for block decisions
    without ever rewriting schema or scalar types.
    """
    leaves: list[str] = []
    active_containers: set[int] = set()
    # (entering, object, depth). Exit frames remove containers from the active
    # ancestor set, allowing harmless shared substructures while rejecting true
    # cycles.
    stack: list[tuple[bool, Any, int]] = [(True, value, 0)]
    nodes = 0
    text_chars = 0
    text_bytes = 0

    def charge_text(text: str) -> None:
        nonlocal text_chars, text_bytes
        text_chars += len(text)
        if max_chars is not None and text_chars > max_chars:
            raise _StructureError(
                f"Structured content exceeds the {max_chars} character limit",
                "STRUCTURE_CHAR_LIMIT_EXCEEDED",
            )
        # Check the character ceiling first so a single hostile multi-gigabyte
        # string is rejected before allocating another equally large UTF-8
        # buffer merely to measure it.
        encoded_length = len(text.encode("utf-8"))
        text_bytes += encoded_length
        if max_bytes is not None and text_bytes > max_bytes:
            raise _StructureError(
                f"Structured content exceeds the {max_bytes} UTF-8 byte limit",
                "STRUCTURE_BYTE_LIMIT_EXCEEDED",
            )

    while stack:
        entering, current, depth = stack.pop()
        if not entering:
            active_containers.remove(id(current))
            continue

        nodes += 1
        if nodes > max_nodes:
            raise _StructureError(
                f"Structured content exceeds the {max_nodes} node limit",
                "STRUCTURE_NODE_LIMIT_EXCEEDED",
            )
        if depth > max_depth:
            raise _StructureError(
                f"Structured content exceeds the {max_depth} level depth limit",
                "STRUCTURE_DEPTH_EXCEEDED",
            )

        if isinstance(current, str):
            charge_text(current)
            leaves.append(current)
            continue
        if _is_safe_scalar(current):
            continue

        object_items = _container_items(current)
        is_sequence = isinstance(current, (list, tuple))
        if object_items is None and not is_sequence:
            raise _StructureError(
                f"Unsupported content type: {type(current).__name__}",
                "UNSUPPORTED_CONTENT_TYPE",
            )

        if object_items is not None:
            # Schema keys are part of the shield detection surface. Charge
            # them before surface construction so enormous attacker-controlled
            # keys cannot bypass SizeLimit and force an unbounded aggregate.
            for key, _ in object_items:
                if key is not None:
                    charge_text(key)

        container_id = id(current)
        if container_id in active_containers:
            raise _StructureError(
                "Cyclic structured content is not supported",
                "STRUCTURE_CYCLE_DETECTED",
            )
        active_containers.add(container_id)
        stack.append((False, current, depth))
        children = [item for _, item in object_items] if object_items is not None else current
        for child in reversed(children):
            stack.append((True, child, depth + 1))

    return leaves


def _replace_text_leaves(value: _ContentT, replacements: Any) -> _ContentT:
    """Rebuild ``value`` with string leaves consumed from ``replacements``."""
    if isinstance(value, str):
        return cast(_ContentT, next(replacements))
    if isinstance(value, dict):
        return cast(
            _ContentT,
            {key: _replace_text_leaves(item, replacements) for key, item in value.items()},
        )
    if isinstance(value, list):
        return cast(
            _ContentT,
            [_replace_text_leaves(item, replacements) for item in value],
        )
    if isinstance(value, tuple):
        items = tuple(_replace_text_leaves(item, replacements) for item in value)
        if hasattr(value, "_fields"):
            # Preserve namedtuple subclasses where possible.
            try:
                return cast(_ContentT, type(value)(*items))
            except TypeError:
                pass
        return cast(_ContentT, items)
    object_items = _object_field_items(value)
    if object_items is not None:
        updates = {name: _replace_text_leaves(item, replacements) for name, item in object_items}
        if isinstance(value, BaseModel):
            model_copy = getattr(value, "model_copy", None)
            if callable(model_copy):
                return cast(_ContentT, model_copy(update=updates, deep=False))
            legacy_copy = getattr(value, "copy", None)
            if callable(legacy_copy):
                return cast(_ContentT, legacy_copy(update=updates, deep=False))
            raise TypeError("Pydantic model does not support safe copying")

        if isinstance(value, SimpleNamespace):
            raw = dict(vars(value))
            raw.update(updates)
            return cast(_ContentT, SimpleNamespace(**raw))

        try:
            rebuilt = object.__new__(type(value))
        except (TypeError, AttributeError) as exc:
            raise TypeError(f"Cannot safely rebuild content object {type(value).__name__}") from exc

        if hasattr(rebuilt, "__dict__"):
            try:
                raw = vars(value)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot safely read content object {type(value).__name__}"
                ) from exc
            rebuilt.__dict__.update(raw)
            rebuilt.__dict__.update(updates)
            return cast(_ContentT, rebuilt)

        # Slot-based dataclasses have no __dict__, but their declared fields
        # can be assigned without invoking constructors or property setters.
        if is_dataclass(value):
            try:
                for field in fields(value):
                    object.__setattr__(
                        rebuilt,
                        field.name,
                        updates.get(
                            field.name,
                            object.__getattribute__(value, field.name),
                        ),
                    )
            except (AttributeError, TypeError) as exc:
                raise TypeError(f"Cannot safely rebuild dataclass {type(value).__name__}") from exc
            return cast(_ContentT, rebuilt)
        raise TypeError(f"Cannot safely rebuild content object {type(value).__name__}")
    return value


@dataclass(frozen=True)
class _SurfacePart:
    text: str
    # ``primary`` directly represents a string leaf. ``string_context`` is a
    # key=value duplicate whose value rewrite can be mapped to that leaf.
    # ``protected`` exposes a key/numeric scalar for detection only.
    kind: str
    value_index: int | None = None
    prefix: str = ""


def _contextual_text_surface(
    value: Any,
    leaves: list[str],
    key_selector: Callable[[tuple[str, ...]], str | None],
) -> list[_SurfacePart]:
    """Build a value-first surface followed by key-aware detection context.

    Keeping primary values adjacent catches attacks split across fields. The
    context trailer lets DLP rules recognize ``api_key=<value>`` without
    rewriting keys. A context-only string redaction maps back to its primary
    value; modified schema/numeric context fails closed.
    """
    parts = [
        _SurfacePart(text=text, kind="primary", value_index=index)
        for index, text in enumerate(leaves)
    ]
    context: list[_SurfacePart] = []
    value_index = 0
    stack: list[tuple[Any, tuple[str, ...]]] = [(value, ())]
    while stack:
        current, key_path = stack.pop()
        if isinstance(current, str):
            key_context = key_selector(key_path)
            if key_context is not None:
                prefix = f"{key_context}="
                context.append(
                    _SurfacePart(
                        text=f"{prefix}{current}",
                        kind="string_context",
                        value_index=value_index,
                        prefix=prefix,
                    )
                )
            value_index += 1
            continue

        object_items = _container_items(current)
        if object_items is not None:
            if key_path:
                # Keep container keys visible without a dangling assignment.
                # Otherwise a following child context can look like
                # ``key=key=value`` to contextual DLP patterns.
                context.append(_SurfacePart(text=key_path[-1], kind="protected"))
            for key, item in reversed(object_items):
                child_path = key_path + (key,) if key is not None else key_path
                stack.append((item, child_path))
        elif isinstance(current, (list, tuple)):
            if key_path:
                context.append(_SurfacePart(text=key_path[-1], kind="protected"))
            # Preserve a parent assignment through sequence wrappers:
            # ``api_key=["secret"]`` must retain the ``api_key`` hint.
            for item in reversed(current):
                stack.append((item, key_path))
        elif isinstance(current, Number) and not isinstance(current, bool):
            key_context = key_selector(key_path)
            prefix = f"{key_context}=" if key_context is not None else ""
            context.append(_SurfacePart(text=f"{prefix}{current}", kind="protected"))
        elif key_path:
            # Even when a bool/None/application object is not textual content,
            # its dynamic key can itself carry an injection or secret.
            context.append(_SurfacePart(text=key_path[-1], kind="protected"))

    return [*parts, *context]


def _plain_text_surface(value: Any, leaves: list[str]) -> list[_SurfacePart]:
    """Expose each value, key, and number once for accounting/observation."""
    parts = [
        _SurfacePart(text=text, kind="primary", value_index=index)
        for index, text in enumerate(leaves)
    ]
    protected: list[_SurfacePart] = []
    stack = [value]
    while stack:
        current = stack.pop()
        object_items = _container_items(current)
        if object_items is not None:
            for key, item in reversed(object_items):
                stack.append(item)
                if key is not None:
                    protected.append(_SurfacePart(text=f"{key}=", kind="protected"))
        elif isinstance(current, (list, tuple)):
            for item in reversed(current):
                stack.append(item)
        elif isinstance(current, Number) and not isinstance(current, bool):
            protected.append(_SurfacePart(text=str(current), kind="protected"))
    return [*parts, *protected]


def _structured_separator(leaves: list[str]) -> str:
    """Choose an all-whitespace separator absent from every string leaf.

    A whitespace-only boundary lets prompt rules see phrases split across
    fields while still allowing modified aggregate text to be mapped back to
    the original structure without parsing or serialising user data.
    """
    whitespace = "\x1c\x1d\x1e\x1f\u2028\u2029"
    length = 1
    while True:
        # Usually one uncommon whitespace control is enough, keeping size/cost
        # accounting close to the actual payload.  Grow only for adversarial
        # inputs containing every candidate sequence.
        for _ in range(16):
            random_bits = uuid.uuid4().int
            separator = "".join(
                whitespace[(random_bits >> (index * 3)) % len(whitespace)]
                for index in range(length)
            )
            if not any(separator in leaf for leaf in leaves):
                return separator
        length += 1


async def _resolve_awaitable(value: Awaitable[Any]) -> Any:
    return await value


class Guard:
    def __init__(
        self,
        shields: list[BaseShield] | None = None,
        *,
        max_structure_depth: int = 64,
        max_structure_nodes: int = 50_000,
        max_structure_chars: int | None = 1_000_000,
        max_structure_bytes: int | None = 4_000_000,
        expose_internal_errors: bool = False,
    ) -> None:
        if (
            isinstance(max_structure_depth, bool)
            or not isinstance(max_structure_depth, int)
            or not 1 <= max_structure_depth <= 256
        ):
            raise ValueError("max_structure_depth must be between 1 and 256")
        if (
            isinstance(max_structure_nodes, bool)
            or not isinstance(max_structure_nodes, int)
            or max_structure_nodes < 1
        ):
            raise ValueError("max_structure_nodes must be at least 1")
        for name, value in {
            "max_structure_chars": max_structure_chars,
            "max_structure_bytes": max_structure_bytes,
        }.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be at least 1 or None")
        if not isinstance(expose_internal_errors, bool):
            raise TypeError("expose_internal_errors must be a bool")
        configured_shields = list(shields or [])
        if any(not isinstance(shield, BaseShield) for shield in configured_shields):
            raise TypeError("every shield must be a BaseShield instance")
        self.shields = configured_shields
        self.max_structure_depth = max_structure_depth
        self.max_structure_nodes = max_structure_nodes
        self.max_structure_chars = max_structure_chars
        self.max_structure_bytes = max_structure_bytes
        self.expose_internal_errors = expose_internal_errors
        self.metrics = GuardMetrics()

    @classmethod
    def from_dict(cls, config: dict) -> "Guard":
        """Build a Guard from a plain dict (e.g. parsed YAML/JSON).

        ::

            Guard.from_dict({"shields": [
                {"type": "PromptShield", "mode": "strict"},
                {"type": "SecretsShield", "on_detect": "redact"},
                {"type": "CostLimit", "max_usd": 5.0},
            ]})

        Each entry's ``type`` names a shield exported from the ``agentguard``
        package; the remaining keys are passed as constructor kwargs.
        """
        import agentguard as _ag

        shields: list[BaseShield] = []
        for raw in config.get("shields", []):
            entry = dict(raw)
            try:
                name = entry.pop("type")
            except KeyError as exc:
                raise ValueError("each shield entry needs a 'type'") from exc
            shield_cls = getattr(_ag, name, None)
            if not (isinstance(shield_cls, type) and issubclass(shield_cls, BaseShield)):
                raise ValueError(f"unknown shield type: {name!r}")
            try:
                shields.append(shield_cls(**entry))
            except TypeError as exc:
                raise ValueError(f"bad config for shield {name!r}: {exc}") from exc
        return cls(shields=shields)

    def stats(self) -> dict:
        """Return a snapshot of scan/block counters for monitoring."""
        return self.metrics.snapshot()

    def _internal_error_detail(self, exc: Exception) -> str:
        if self.expose_internal_errors:
            return str(exc)
        return f"internal {exc.__class__.__name__} (details hidden)"

    def _raise_internal_error(
        self,
        shield_name: str,
        exc: Exception,
        *,
        prefix: str | None = None,
    ) -> NoReturn:
        detail = self._internal_error_detail(exc)
        if prefix:
            detail = f"{prefix}: {detail}"
        error = GuardShieldError(shield_name, detail)
        if self.expose_internal_errors:
            raise error from exc
        raise error from None

    async def _notify_decision(
        self,
        decision: GuardDecision,
        ctx: SessionContext,
    ) -> None:
        """Notify observer shields without ever exposing raw content.

        An observer failure on an allowed request fails closed.  If the request
        is already blocked, the original block remains authoritative and hook
        failures are retained as content-free diagnostic metadata instead of
        accidentally turning a policy block into a different exception.
        """
        for observer in self.shields:
            try:
                await observer.on_decision(decision, ctx)
            except Exception as exc:
                if decision.allowed:
                    self._raise_internal_error(
                        observer.__class__.__name__,
                        exc,
                        prefix="decision observer failed",
                    )
                errors = ctx.metadata.setdefault("decision_hook_errors", [])
                errors.append(observer.__class__.__name__)

    async def _raise_block(
        self,
        shield: BaseShield,
        result: ShieldResult,
        default_reason: str,
        default_code: str,
        flow: GuardFlow,
        ctx: SessionContext,
        tool_name: str | None = None,
    ) -> None:
        code = result.reason_code or default_code
        shield_name = shield.__class__.__name__
        self.metrics.record_block(shield_name, code)
        await self._notify_decision(
            GuardDecision(
                flow=flow,
                allowed=False,
                shield_name=shield_name,
                reason_code=code,
                tool_name=tool_name,
            ),
            ctx,
        )
        raise GuardBlockedError(result.reason or default_reason, code, shield_name)

    async def _handle_raised_block(
        self,
        exc: GuardBlockedError,
        flow: GuardFlow,
        ctx: SessionContext,
        tool_name: str | None = None,
    ) -> None:
        self.metrics.record_block(exc.shield_name, exc.reason_code)
        await self._notify_decision(
            GuardDecision(
                flow=flow,
                allowed=False,
                shield_name=exc.shield_name,
                reason_code=exc.reason_code,
                tool_name=tool_name,
            ),
            ctx,
        )

    async def _validated_text_leaves(
        self,
        content: Any,
        flow: GuardFlow,
        ctx: SessionContext,
        tool_name: str | None,
    ) -> list[str]:
        try:
            return _text_leaves(
                content,
                max_depth=self.max_structure_depth,
                max_nodes=self.max_structure_nodes,
                max_chars=self.max_structure_chars,
                max_bytes=self.max_structure_bytes,
            )
        except _StructureError as exc:
            shield_name = self.__class__.__name__
            self.metrics.record_block(shield_name, exc.reason_code)
            await self._notify_decision(
                GuardDecision(
                    flow=flow,
                    allowed=False,
                    shield_name=shield_name,
                    reason_code=exc.reason_code,
                    tool_name=tool_name,
                ),
                ctx,
            )
            error = GuardBlockedError(str(exc), exc.reason_code, shield_name)
            if self.expose_internal_errors:
                raise error from exc
            raise error from None

    # ------------------------------------------------------------------ #
    # Decorators                                                           #
    # ------------------------------------------------------------------ #

    def protect(self, fn: Callable) -> Callable:
        """Decorator for async agent functions."""
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@guard.protect requires an async function. "
                "Use @guard.protect_sync for sync functions."
            )

        @functools.wraps(fn)
        async def wrapper(
            query: Any,
            *args: Any,
            _guard_ctx: SessionContext | None = None,
            **kwargs: Any,
        ) -> Any:
            ctx = _guard_ctx or SessionContext()
            sanitized = await self.scan_input(query, ctx)
            result = await fn(sanitized, *args, **kwargs)
            return await self.scan_output(result, ctx)

        return wrapper

    def protect_sync(self, fn: Callable) -> Callable:
        """Decorator for sync agent functions."""
        if asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@guard.protect_sync requires a sync function. "
                "Use @guard.protect for async functions."
            )

        async_only = [s for s in self.shields if getattr(s, "requires_async", False)]
        if async_only:
            names = ", ".join(sorted(s.__class__.__name__ for s in async_only))
            raise HumanGateSyncError(
                f"{names} require a live event loop and cannot run under "
                "@guard.protect_sync (the per-call loop could never receive the "
                "approval). Use the async @guard.protect instead."
            )

        @functools.wraps(fn)
        def wrapper(
            query: Any,
            *args: Any,
            _guard_ctx: SessionContext | None = None,
            **kwargs: Any,
        ) -> Any:
            ctx = _guard_ctx or SessionContext()
            sanitized = asyncio.run(self.scan_input(query, ctx))
            result = fn(sanitized, *args, **kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(_resolve_awaitable(result))
            return asyncio.run(self.scan_output(result, ctx))

        return wrapper

    # ------------------------------------------------------------------ #
    # Explicit run                                                         #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        agent_fn: Callable,
        query: Any,
        ctx: SessionContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent through all shields. Alternative to the decorator."""
        ctx = ctx or SessionContext()
        sanitized = await self.scan_input(query, ctx)

        result = agent_fn(sanitized, **kwargs)
        if inspect.isawaitable(result):
            result = await result

        return await self.scan_output(result, ctx)

    async def scan_tool_call(
        self,
        tool_name: str,
        params: dict,
        ctx: SessionContext | None = None,
    ) -> ShieldResult:
        """Scan a tool call through all shields. Called by GuardedTool."""
        ctx = ctx or SessionContext()
        self.metrics.record_scan("tool_calls")
        await self._scan_tool_policy(tool_name, params, ctx)
        return ShieldResult(allowed=True)

    async def _scan_tool_policy(
        self,
        tool_name: str,
        params: dict,
        ctx: SessionContext,
    ) -> None:
        for shield in self.shields:
            try:
                result = await shield.scan_tool_call(tool_name, params, ctx)
            except GuardBlockedError as exc:
                await self._handle_raised_block(exc, "tool_call", ctx, tool_name)
                if self.expose_internal_errors:
                    raise
                raise exc from None
            except Exception as exc:
                self._raise_internal_error(shield.__class__.__name__, exc)
            if not result.allowed:
                await self._raise_block(
                    shield,
                    result,
                    "Tool call blocked",
                    "TOOL_BLOCKED",
                    "tool_call",
                    ctx,
                    tool_name,
                )
        await self._notify_decision(
            GuardDecision(flow="tool_call", allowed=True, tool_name=tool_name), ctx
        )

    async def scan_tool_arguments(
        self,
        tool_name: str,
        params: dict[str, Any],
        ctx: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Sanitize model-generated arguments, then apply tool-call policy.

        This is the execution-boundary API used by :class:`GuardedTool` and
        framework adapters.  Content shields see all nested string values in a
        single scan, rewrites are returned in the same dict/list/tuple shape,
        and operational request shields are not counted a second time.  The
        legacy :meth:`scan_tool_call` validation-only API remains unchanged.
        """
        if not isinstance(params, dict):
            raise TypeError("tool params must be a dict")
        ctx = ctx or SessionContext()
        self.metrics.record_scan("tool_calls")

        async def scanner(shield: BaseShield, text: str) -> ShieldResult:
            return await shield.scan_tool_arguments(tool_name, text, ctx)

        sanitized = await self._scan_content(
            params,
            ctx,
            flow="tool_call",
            metric=None,
            scanner=scanner,
            default_reason="Tool arguments blocked",
            default_code="TOOL_ARGUMENTS_BLOCKED",
            tool_name=tool_name,
            notify_allowed=False,
        )
        await self._scan_tool_policy(tool_name, sanitized, ctx)
        return sanitized

    async def scan_tool_output(
        self,
        tool_name: str,
        output: _ContentT,
        ctx: SessionContext | None = None,
    ) -> _ContentT:
        """Scan content returned by a tool before it re-enters the agent.

        This is the indirect-prompt-injection chokepoint: tool results and
        retrieved documents are attacker-controlled and must be inspected just
        like user input. Returns the (possibly sanitised) output. Called by
        GuardedTool after the wrapped tool returns.
        """
        ctx = ctx or SessionContext()

        async def scanner(shield: BaseShield, text: str) -> ShieldResult:
            return await shield.scan_tool_output(tool_name, text, ctx)

        return await self._scan_content(
            output,
            ctx,
            flow="tool_output",
            metric="tool_outputs",
            scanner=scanner,
            default_reason="Tool output blocked",
            default_code="TOOL_OUTPUT_BLOCKED",
            tool_name=tool_name,
        )

    async def scan_input(
        self,
        content: _ContentT,
        ctx: SessionContext | None = None,
    ) -> _ContentT:
        """Scan input while preserving JSON-like structured value types.

        Strings are scanned directly.  For dictionaries, lists, and tuples,
        rewrites are mapped back onto the same shape. Dictionary keys and
        numeric scalars are visible to shields for block decisions but are
        never rewritten, preventing redaction from corrupting schema or
        changing scalar types.
        """
        ctx = ctx or SessionContext()

        async def scanner(shield: BaseShield, text: str) -> ShieldResult:
            return await shield.scan_input(text, ctx)

        scanned = await self._scan_content(
            content,
            ctx,
            flow="input",
            metric="inputs",
            scanner=scanner,
            default_reason="Input blocked",
            default_code="INPUT_BLOCKED",
            notify_allowed=False,
        )
        final_text = "\n".join(
            _text_leaves(
                scanned,
                max_depth=self.max_structure_depth,
                max_nodes=self.max_structure_nodes,
                max_chars=self.max_structure_chars,
                max_bytes=self.max_structure_bytes,
            )
        )
        for shield in self.shields:
            try:
                await shield.on_input_committed(final_text, ctx)
            except Exception as exc:
                error = GuardShieldError(
                    shield.__class__.__name__,
                    f"input commit failed: {self._internal_error_detail(exc)}",
                )
                if self.expose_internal_errors:
                    raise error from exc
                raise error from None
        await self._notify_decision(GuardDecision(flow="input", allowed=True), ctx)
        ctx.request_count += 1
        return scanned

    async def scan_output(
        self,
        content: _ContentT,
        ctx: SessionContext | None = None,
    ) -> _ContentT:
        """Scan output and return the same JSON-like container shape."""
        ctx = ctx or SessionContext()

        async def scanner(shield: BaseShield, text: str) -> ShieldResult:
            return await shield.scan_output(text, ctx)

        return await self._scan_content(
            content,
            ctx,
            flow="output",
            metric="outputs",
            scanner=scanner,
            default_reason="Output blocked",
            default_code="OUTPUT_BLOCKED",
        )

    async def scan_output_preview(
        self,
        content: _ContentT,
        ctx: SessionContext | None = None,
    ) -> _ContentT:
        """Scan a provisional streaming prefix without final accounting.

        Blocking decisions still raise and are audited. Successful previews do
        not increment output metrics or emit an allowed decision. Applications
        normally use this indirectly through ``StreamGuard``.
        """
        ctx = ctx or SessionContext()

        async def scanner(shield: BaseShield, text: str) -> ShieldResult:
            return await shield.scan_output_preview(text, ctx)

        return await self._scan_content(
            content,
            ctx,
            flow="output",
            metric=None,
            scanner=scanner,
            default_reason="Output preview blocked",
            default_code="OUTPUT_BLOCKED",
            notify_allowed=False,
        )

    # ------------------------------------------------------------------ #
    # Internal scan pipelines                                              #
    # ------------------------------------------------------------------ #

    async def _scan_content(
        self,
        content: _ContentT,
        ctx: SessionContext,
        *,
        flow: GuardFlow,
        metric: str | None,
        scanner: Callable[[BaseShield, str], Awaitable[ShieldResult]],
        default_reason: str,
        default_code: str,
        tool_name: str | None = None,
        notify_allowed: bool = True,
    ) -> _ContentT:
        if metric is not None:
            self.metrics.record_scan(metric)
        current = content
        # Validate even when no shields are installed: traversal itself is an
        # attacker-controlled resource surface.
        leaves = await self._validated_text_leaves(current, flow, ctx, tool_name)
        try:
            # Snapshot supported mutable structures before the first shield can
            # await. This closes argument/output TOCTOU races: later caller or
            # tool mutations cannot change what executes or what the agent sees
            # after policy approved a different object graph.
            current = _replace_text_leaves(current, iter(leaves))
        except Exception as exc:
            self._raise_internal_error(
                self.__class__.__name__,
                exc,
                prefix="content snapshot failed",
            )
        for shield in self.shields:
            try:
                if shield.structured_values_only:
                    surface = [
                        _SurfacePart(text=text, kind="primary", value_index=index)
                        for index, text in enumerate(leaves)
                    ]
                elif shield.needs_structured_context:
                    surface = _contextual_text_surface(
                        current,
                        leaves,
                        shield.select_structured_context_key,
                    )
                else:
                    surface = _plain_text_surface(current, leaves)
            except Exception as exc:
                self._raise_internal_error(
                    shield.__class__.__name__,
                    exc,
                    prefix="structured detection surface failed",
                )
            surface_text = [part.text for part in surface]
            separator = _structured_separator(surface_text)
            aggregate = separator.join(surface_text)
            try:
                result = await scanner(shield, aggregate)
            except GuardBlockedError as exc:
                await self._handle_raised_block(exc, flow, ctx, tool_name)
                if self.expose_internal_errors:
                    raise
                raise exc from None
            except Exception as exc:
                self._raise_internal_error(shield.__class__.__name__, exc)
            if not result.allowed:
                await self._raise_block(
                    shield,
                    result,
                    default_reason,
                    default_code,
                    flow,
                    ctx,
                    tool_name,
                )
            if result.modified_input is not None:
                if not isinstance(result.modified_input, str):
                    raise GuardShieldError(
                        shield.__class__.__name__,
                        "modified_input must be a string",
                    )
                if not surface_text:
                    # There is nowhere type-safe to apply a text rewrite.  The
                    # shield still got to block/observe the empty text surface.
                    continue
                rewritten_surface = result.modified_input.split(separator)
                if len(rewritten_surface) > len(surface_text):
                    raise GuardShieldError(
                        shield.__class__.__name__,
                        "structured rewrite introduced AgentGuard's field separator",
                    )
                # Whole-content neutralisation and aggregate truncation can
                # intentionally remove field boundaries.  Keep the safe text
                # in the leading fields and clear all omitted fields so no raw
                # tail can survive the rewrite.
                rewritten_surface.extend([""] * (len(surface_text) - len(rewritten_surface)))
                rewritten = rewritten_surface[: len(leaves)]
                unsafe_context = False
                for part, context_rewrite in zip(
                    surface[len(leaves) :], rewritten_surface[len(leaves) :]
                ):
                    if context_rewrite == part.text:
                        continue
                    if part.kind == "string_context" and part.value_index is not None:
                        if context_rewrite.startswith(part.prefix):
                            candidate = context_rewrite[len(part.prefix) :]
                        elif (
                            context_rewrite.startswith("[REDACTED_")
                            and context_rewrite.endswith("]")
                        ) or (context_rewrite and set(context_rewrite) == {"*"}):
                            # Contextual credential rules may replace the whole
                            # ``key=value`` match. The replacement itself is a
                            # safe value; schema remains unchanged in the tree.
                            candidate = context_rewrite
                        else:
                            unsafe_context = True
                            break
                        primary = rewritten[part.value_index]
                        original = leaves[part.value_index]
                        if primary != original:
                            # The canonical value view was already sanitized.
                            # Contextual duplicates may legitimately produce a
                            # different token (for example tokenize-mode PII).
                            continue
                        if primary == original or primary == candidate:
                            rewritten[part.value_index] = candidate
                            continue
                    unsafe_context = True
                    break

                if unsafe_context:
                    await self._raise_block(
                        shield,
                        ShieldResult(
                            allowed=False,
                            reason=(
                                "A structured-content sanitizer attempted to "
                                "rewrite a schema key or non-string scalar; "
                                "the flow was blocked to preserve types"
                            ),
                            reason_code="STRUCTURE_TYPE_PRESERVATION_BLOCK",
                        ),
                        "Structured content blocked",
                        "STRUCTURE_BLOCKED",
                        flow,
                        ctx,
                        tool_name,
                    )
                try:
                    current = _replace_text_leaves(current, iter(rewritten))
                except Exception as exc:
                    self._raise_internal_error(
                        shield.__class__.__name__,
                        exc,
                        prefix="structured content rebuild failed",
                    )
                leaves = await self._validated_text_leaves(current, flow, ctx, tool_name)

        if notify_allowed:
            await self._notify_decision(
                GuardDecision(flow=flow, allowed=True, tool_name=tool_name),
                ctx,
            )
        return current

    # Private aliases retained for adapters and third-party integrations built
    # against pre-0.13 AgentGuard.  New code should use the public methods.
    async def _scan_input(self, content: _ContentT, ctx: SessionContext) -> _ContentT:
        return await self.scan_input(content, ctx)

    async def _scan_output(self, content: _ContentT, ctx: SessionContext) -> _ContentT:
        return await self.scan_output(content, ctx)
