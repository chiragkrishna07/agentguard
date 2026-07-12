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

import functools
import inspect
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agentguard.core.content import extract_text, strict_json_object
from agentguard.core.exceptions import GuardBlockedError

if TYPE_CHECKING:
    from agentguard.core.guard import Guard
    from agentguard.core.session import SessionContext
    from agentguard.tools import GuardedTool


_INPUT_KEYS = ("user_message", "query", "input")
_SESSION_KEYS = ("thread_id", "session_id", "conversation_id")
_CONTROL_STATE_KEYS = {
    "config",
    "configurable",
    "__interrupt__",
}


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role", message.get("type"))
    else:
        role = getattr(message, "role", getattr(message, "type", None))
    return role.lower() if isinstance(role, str) else None


def _message_content(message: Any) -> Any:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _contains_text(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if isinstance(content, dict):
        return any(isinstance(content.get(key), str) for key in ("text", "content"))
    if isinstance(content, (list, tuple)):
        return any(_contains_text(item) for item in content)
    return False


def _replace_content_text(content: Any, sanitized: str) -> Any:
    """Put combined sanitized text back without disturbing media parts."""
    if isinstance(content, str):
        return sanitized
    if isinstance(content, dict):
        for key in ("text", "content"):
            if isinstance(content.get(key), str):
                return {**content, key: sanitized}
        return content
    if isinstance(content, (list, tuple)):
        used = False
        items = []
        for item in content:
            if _contains_text(item):
                replacement = sanitized if not used else ""
                items.append(_replace_content_text(item, replacement))
                used = True
            else:
                items.append(item)
        return tuple(items) if isinstance(content, tuple) else items
    return content


def _replace_attr(obj: Any, name: str, value: Any) -> Any:
    if isinstance(obj, dict):
        return {**obj, name: value}
    try:
        setattr(obj, name, value)
        return obj
    except (AttributeError, TypeError, ValueError):
        model_copy = getattr(obj, "model_copy", None)
        if callable(model_copy):
            return model_copy(update={name: value})
        copy = getattr(obj, "copy", None)
        if callable(copy):
            try:
                return copy(update={name: value})
            except (TypeError, ValueError):
                pass
    raise TypeError(f"cannot safely attach sanitized {name!r} to {type(obj).__name__}")


def _replace_message_content(message: Any, sanitized: Any) -> Any:
    if isinstance(message, str):
        return sanitized
    old_content = _message_content(message)
    return _replace_attr(
        message,
        "content",
        _replace_content_text(old_content, sanitized),
    )


def _is_message_object(value: Any) -> bool:
    return not isinstance(value, (str, dict, list, tuple)) and hasattr(value, "content")


def _is_message_dict(value: Any) -> bool:
    if not isinstance(value, dict) or "content" not in value:
        return False
    role = _message_role(value)
    return role in {"user", "human", "assistant", "ai", "tool", "system"} or bool(
        value.get("tool_calls")
    )


@dataclass(frozen=True)
class _MessageSlot:
    index: int


def _mask_output_messages(
    value: Any,
    messages: list[Any],
    contents: list[Any],
) -> Any:
    if _is_message_object(value) or _is_message_dict(value):
        index = len(messages)
        messages.append(value)
        content = _message_content(value)
        contents.append(extract_text(content) if _contains_text(content) else None)
        return _MessageSlot(index)
    if isinstance(value, dict):
        return {key: _mask_output_messages(item, messages, contents) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_output_messages(item, messages, contents) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask_output_messages(item, messages, contents) for item in value)
    return value


def _restore_output_messages(value: Any, messages: list[Any]) -> Any:
    if isinstance(value, _MessageSlot):
        return messages[value.index]
    if isinstance(value, dict):
        return {key: _restore_output_messages(item, messages) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_output_messages(item, messages) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_output_messages(item, messages) for item in value)
    return value


def _collect_output_text(value: Any) -> list[str]:
    """Collect strings, treating LangChain message content as one text unit."""
    if isinstance(value, str):
        return [value]
    if _is_message_object(value):
        content = getattr(value, "content", None)
        return [extract_text(content)] if _contains_text(content) else []
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_collect_output_text(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_collect_output_text(item))
        return result
    return []


def _replace_output_text(value: Any, replacements: Iterator[str]) -> Any:
    if isinstance(value, str):
        return next(replacements)
    if _is_message_object(value):
        content = getattr(value, "content", None)
        if not _contains_text(content):
            return value
        return _replace_message_content(value, next(replacements))
    if isinstance(value, dict):
        return {key: _replace_output_text(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_output_text(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_output_text(item, replacements) for item in value)
    return value


class GuardLangGraph:
    def __init__(
        self,
        guard: "Guard",
        ctx: Optional["SessionContext"] = None,
        *,
        max_session_contexts: int = 1024,
        trusted_state_keys: tuple[str, ...] | None = None,
        trust_state_identity: bool = False,
    ) -> None:
        if max_session_contexts < 1:
            raise ValueError("max_session_contexts must be at least 1")
        if not isinstance(trust_state_identity, bool):
            raise ValueError("trust_state_identity must be boolean")
        self.guard = guard
        self.ctx = ctx
        self._max_session_contexts = max_session_contexts
        self._trust_state_identity = trust_state_identity
        self._trusted_state_keys = {
            *_CONTROL_STATE_KEYS,
            *(trusted_state_keys or ()),
        }
        if trust_state_identity:
            self._trusted_state_keys.update({*_SESSION_KEYS, "user_id", "tenant_id"})
        self._contexts: OrderedDict[tuple[str | None, str], SessionContext] = OrderedDict()
        self._contexts_lock = threading.Lock()

    @staticmethod
    def _config_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        config = kwargs.get("config")
        if isinstance(config, dict):
            return config
        for item in args:
            if isinstance(item, dict) and isinstance(item.get("configurable"), dict):
                return item
        return {}

    def _context_for_call(
        self,
        state: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "SessionContext":
        if self.ctx is not None:
            return self.ctx

        from agentguard.core.session import SessionContext as _Ctx

        config = self._config_from_call(args, kwargs)
        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            configurable = {}

        raw_id = next(
            (configurable[key] for key in _SESSION_KEYS if configurable.get(key) is not None),
            None,
        )
        if raw_id is None and self._trust_state_identity:
            raw_id = next(
                (state[key] for key in _SESSION_KEYS if state.get(key) is not None),
                None,
            )

        # Without a caller-supplied identity, isolation is safer than silently
        # sharing rate limits, PII token maps, and canaries between users.
        if raw_id is None:
            return _Ctx()

        session_id = str(raw_id)
        user_id = configurable.get("user_id")
        if user_id is None and self._trust_state_identity:
            user_id = state.get("user_id")
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

    async def _scan_state_input(
        self, state: dict[str, Any], ctx: "SessionContext"
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        message_items = list(messages) if isinstance(messages, (list, tuple)) else []
        selected_indexes: list[int] = []
        context_indexes: list[int] = []
        for index, message in enumerate(message_items):
            role = _message_role(message)
            if role in ("user", "human") or (role is None and index == len(message_items) - 1):
                content = _message_content(message)
                if _contains_text(content):
                    selected_indexes.append(index)
            elif role not in {"system", "developer"} and _contains_text(
                _message_content(message)
            ):
                # Assistant, tool, peer-agent, and unknown-role history is
                # persisted context, not trusted policy. Treat it like a tool
                # result before scanning/committing the current user turn.
                context_indexes.append(index)

        selected_keys = [key for key in _INPUT_KEYS if key in state]
        sanitized_state = dict(state)

        excluded = {"messages", *selected_keys, *self._trusted_state_keys}
        for key, value in state.items():
            if key in excluded:
                continue
            sanitized_state[key] = await self.guard.scan_tool_output(
                f"langgraph_state:{key}", value, ctx
            )

        if context_indexes:
            context_text = tuple(
                extract_text(_message_content(message_items[index]))
                for index in context_indexes
            )
            sanitized_context = await self.guard.scan_tool_output(
                "langgraph_untrusted_message_history", context_text, ctx
            )
            for index, sanitized in zip(context_indexes, sanitized_context):
                message_items[index] = _replace_message_content(
                    message_items[index], sanitized
                )

        sources: list[Any] = [
            extract_text(_message_content(message_items[index])) for index in selected_indexes
        ]
        sources.extend(state[key] for key in selected_keys)
        if sources:
            sanitized_sources = await self.guard.scan_input(sources, ctx)
            cursor = iter(sanitized_sources)
            for index in selected_indexes:
                message_items[index] = _replace_message_content(message_items[index], next(cursor))
            for key in selected_keys:
                sanitized_state[key] = next(cursor)

        if selected_indexes or context_indexes:
            sanitized_state["messages"] = (
                tuple(message_items) if isinstance(messages, tuple) else message_items
            )
        return sanitized_state

    async def _scan_node_output(self, result: Any, ctx: "SessionContext") -> Any:
        result = await self._sanitize_tool_calls(result, ctx)
        messages: list[Any] = []
        contents: list[Any] = []
        masked_result = _mask_output_messages(result, messages, contents)
        projection = {
            "message_texts": tuple(contents),
            "structured_result": masked_result,
        }
        sanitized = await self.guard.scan_output(projection, ctx)
        sanitized_messages = [
            _replace_message_content(message, content) if content is not None else message
            for message, content in zip(messages, sanitized["message_texts"])
        ]
        return _restore_output_messages(sanitized["structured_result"], sanitized_messages)

    async def _sanitize_one_tool_call(self, call: Any, ctx: "SessionContext") -> Any:
        if isinstance(call, dict):
            function = call.get("function")
            if function is not None:
                updated_function = await self._sanitize_one_tool_call(function, ctx)
                return {**call, "function": updated_function}
            name = call.get("name", "")
            raw_args = call.get("args", call.get("arguments", {}))
            argument_field = "args" if "args" in call else "arguments"
        else:
            function = getattr(call, "function", None)
            if function is not None:
                updated_function = await self._sanitize_one_tool_call(function, ctx)
                return _replace_attr(call, "function", updated_function)
            name = getattr(call, "name", "")
            if hasattr(call, "args"):
                argument_field = "args"
                raw_args = getattr(call, "args")
            else:
                argument_field = "arguments"
                raw_args = getattr(call, "arguments", {})

        arguments_were_json = isinstance(raw_args, str)
        if not isinstance(raw_args, (str, dict)):
            raise GuardBlockedError(
                "Model-generated tool arguments were not a JSON object",
                "TOOL_ARGUMENT_JSON_INVALID",
                self.__class__.__name__,
            ) from None
        try:
            params = strict_json_object(raw_args)
        except ValueError:
            raise GuardBlockedError(
                "Model-generated tool arguments were invalid JSON",
                "TOOL_ARGUMENT_JSON_INVALID",
                self.__class__.__name__,
            ) from None

        sanitized = await self.guard.scan_tool_arguments(str(name), params, ctx)
        safe_args: Any
        if arguments_were_json:
            safe_args = json.dumps(
                sanitized,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        else:
            safe_args = sanitized
        return _replace_attr(call, argument_field, safe_args)

    async def _sanitize_tool_calls(self, value: Any, ctx: "SessionContext") -> Any:
        """Recursively validate and rewrite model-emitted LangChain tool calls."""
        if _is_message_object(value):
            raw_calls = getattr(value, "tool_calls", None) or []
            updated_value = value
            if raw_calls:
                updated = [await self._sanitize_one_tool_call(call, ctx) for call in raw_calls]
                updated_value = _replace_attr(
                    updated_value,
                    "tool_calls",
                    tuple(updated) if isinstance(raw_calls, tuple) else updated,
                )
            legacy_call = getattr(updated_value, "function_call", None)
            if legacy_call is not None:
                updated_value = _replace_attr(
                    updated_value,
                    "function_call",
                    await self._sanitize_one_tool_call(legacy_call, ctx),
                )
            return updated_value
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "tool_calls" and isinstance(item, (list, tuple)):
                    updated = [await self._sanitize_one_tool_call(call, ctx) for call in item]
                    result[key] = tuple(updated) if isinstance(item, tuple) else updated
                elif key == "function_call" and item is not None:
                    result[key] = await self._sanitize_one_tool_call(item, ctx)
                else:
                    result[key] = await self._sanitize_tool_calls(item, ctx)
            return result
        if isinstance(value, list):
            return [await self._sanitize_tool_calls(item, ctx) for item in value]
        if isinstance(value, tuple):
            return tuple([await self._sanitize_tool_calls(item, ctx) for item in value])
        return value

    def wrap_node(self, fn: Callable) -> Callable:
        """Guard user state before a node and recursively scan its output."""

        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            ctx = self._context_for_call(state, args, kwargs)
            sanitized_state = await self._scan_state_input(state, ctx)

            result = fn(sanitized_state, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return await self._scan_node_output(result, ctx)

        return wrapper

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool

        return GuardedTool(fn, self.guard, self.ctx)
