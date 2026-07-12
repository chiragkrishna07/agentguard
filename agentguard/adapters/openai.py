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

from agentguard.core.content import (
    contains_text,
    extract_text,
    replace_text,
    strict_json_object,
)
from agentguard.core.exceptions import GuardBlockedError

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
        # A constructor context is an explicit request to share session state.
        # Without one, each create call gets an isolated context unless the
        # caller passes ``_guard_ctx=...`` for deliberate multi-turn reuse.
        self.ctx = ctx

    @staticmethod
    def _replace_attr(obj: Any, name: str, value: Any) -> Any:
        """Replace an SDK-model field, including immutable Pydantic models."""
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

    async def _sanitize_function_call(self, call: Any, ctx: "SessionContext") -> Any:
        name = getattr(call, "name", "")
        raw_args = getattr(call, "arguments", "") or "{}"
        if not isinstance(raw_args, str):
            raise GuardBlockedError(
                "Model-generated tool arguments were not JSON text",
                "TOOL_ARGUMENT_JSON_INVALID",
                self.__class__.__name__,
            )
        try:
            params = strict_json_object(raw_args)
        except ValueError:
            raise GuardBlockedError(
                "Model-generated tool arguments were invalid JSON",
                "TOOL_ARGUMENT_JSON_INVALID",
                self.__class__.__name__,
            ) from None
        sanitized = await self.guard.scan_tool_arguments(str(name), params, ctx)
        safe_args = json.dumps(
            sanitized,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        return self._replace_attr(call, "arguments", safe_args)

    async def create(self, client: Any, **kwargs: Any) -> Any:
        """Drop-in replacement for client.chat.completions.create with guard scanning."""
        from agentguard.core.session import SessionContext as _Ctx

        call_ctx = kwargs.pop("_guard_ctx", None)
        if call_ctx is not None and not isinstance(call_ctx, _Ctx):
            raise TypeError("_guard_ctx must be a SessionContext")
        ctx = call_ctx or self.ctx or _Ctx()

        input_messages: list[dict[str, Any]] = list(kwargs.get("messages", []))
        # Persisted assistant/tool messages are untrusted memory. Inspect and
        # rewrite them before the current user turn so a poisoned history that
        # blocks cannot still be committed into PromptShield's rolling state.
        context_indexes = [
            index
            for index, message in enumerate(input_messages)
            if isinstance(message, dict)
            and message.get("role") in {"assistant", "tool"}
            and contains_text(message.get("content"))
        ]
        if context_indexes:
            context_text = tuple(
                extract_text(input_messages[index].get("content"))
                for index in context_indexes
            )
            sanitized_context = await self.guard.scan_tool_output(
                "openai_untrusted_message_history", context_text, ctx
            )
            for index, sanitized in zip(context_indexes, sanitized_context):
                message = input_messages[index]
                input_messages[index] = {
                    **message,
                    "content": replace_text(message.get("content"), sanitized),
                }

        user_indexes = [
            index
            for index, message in enumerate(input_messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and contains_text(message.get("content"))
        ]
        if user_indexes:
            user_text = tuple(
                extract_text(input_messages[index].get("content")) for index in user_indexes
            )
            sanitized_user_text = await self.guard.scan_input(user_text, ctx)
            for index, sanitized in zip(user_indexes, sanitized_user_text):
                message = input_messages[index]
                input_messages[index] = {
                    **message,
                    "content": replace_text(message.get("content"), sanitized),
                }

        if user_indexes or context_indexes:
            kwargs = {**kwargs, "messages": input_messages}

        response = await client.chat.completions.create(**kwargs)

        # Validate any tool calls the model decided to make BEFORE the caller
        # executes them — this is where ToolValidator / HumanGate get their say
        # on model-chosen actions and arguments.
        raw_choices = getattr(response, "choices", []) or []
        choices = list(raw_choices)
        original_messages: list[Any] = []
        response_messages: list[Any] = []
        for choice in choices:
            response_message = getattr(choice, "message", None)
            if response_message is None:
                continue
            original_messages.append(response_message)
            raw_tool_calls = getattr(response_message, "tool_calls", None) or []
            updated_tool_calls = []
            for call in raw_tool_calls:
                fn = getattr(call, "function", None)
                if fn is None:
                    updated_tool_calls.append(call)
                    continue
                updated_fn = await self._sanitize_function_call(fn, ctx)
                updated_tool_calls.append(self._replace_attr(call, "function", updated_fn))

            if raw_tool_calls:
                response_message = self._replace_attr(
                    response_message,
                    "tool_calls",
                    tuple(updated_tool_calls)
                    if isinstance(raw_tool_calls, tuple)
                    else updated_tool_calls,
                )
            legacy_function_call = getattr(response_message, "function_call", None)
            if legacy_function_call is not None:
                response_message = self._replace_attr(
                    response_message,
                    "function_call",
                    await self._sanitize_function_call(legacy_function_call, ctx),
                )
            response_messages.append(response_message)

        # Scan all candidate outputs in one pipeline decision and, critically,
        # write sanitized content back into the response object.  Earlier
        # versions called the output shields but discarded their rewrite.
        if response_messages:
            contents = [getattr(message, "content", None) for message in response_messages]
            sanitized_contents = await self.guard.scan_output(contents, ctx)
            updated_messages = [
                self._replace_attr(message, "content", content)
                for message, content in zip(response_messages, sanitized_contents)
            ]

            updated_by_id = {
                id(original): updated
                for original, updated in zip(original_messages, updated_messages)
            }
            updated_choices = []
            for choice in choices:
                original_message = getattr(choice, "message", None)
                updated_message = updated_by_id.get(id(original_message))
                updated_choices.append(
                    self._replace_attr(choice, "message", updated_message)
                    if updated_message is not None
                    else choice
                )
            response = self._replace_attr(
                response,
                "choices",
                tuple(updated_choices) if isinstance(raw_choices, tuple) else updated_choices,
            )

        return response

    def wrap_tool(self, fn: Callable) -> "GuardedTool":
        from agentguard.tools import GuardedTool

        return GuardedTool(fn, self.guard, self.ctx)
