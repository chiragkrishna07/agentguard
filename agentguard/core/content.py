"""Helpers for chat message content that may be structured.

Modern chat APIs allow a message's ``content`` to be a list of typed parts
(``{"type": "text", "text": ...}`` alongside images/audio) instead of a plain
string. The shields operate on text, so adapters use these helpers to pull the
text out for scanning and to write sanitised text back without disturbing the
non-text parts.
"""

import json
import math
from collections.abc import Awaitable, Callable
from typing import Any


def strict_json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse/validate model tool arguments as finite, duplicate-free JSON."""

    def reject_constant(_: str) -> Any:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    if isinstance(raw, str):
        try:
            value = json.loads(
                raw,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid JSON tool arguments") from exc
    elif isinstance(raw, dict):
        value = raw
    else:
        raise ValueError("tool arguments must be a JSON object")

    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")

    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > 50_000 or depth > 64:
            raise ValueError("JSON tool arguments exceed structural limits")
        if current is None or isinstance(current, (str, bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("non-finite JSON number")
            continue
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                stack.append((item, depth + 1))
            continue
        if isinstance(current, list):
            for item in current:
                stack.append((item, depth + 1))
            continue
        raise ValueError("unsupported value in JSON tool arguments")
    return value


def extract_text(content: Any) -> str:
    """Flatten message content into a single scannable string.

    Handles plain strings, a single part dict, and a list of parts. Non-text
    parts (images, etc.) contribute nothing. Newline-joined so adjacent parts
    don't merge into spurious tokens.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content"):
            if isinstance(content.get(key), str):
                return content[key]
        return ""
    if isinstance(content, (list, tuple)):
        parts = [extract_text(part) for part in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def _is_text_part(part: Any) -> bool:
    return isinstance(part, dict) and isinstance(part.get("text"), str)


def contains_text(content: Any) -> bool:
    """Return whether supported chat content contains a textual segment."""
    if isinstance(content, str):
        return True
    if isinstance(content, dict):
        return any(isinstance(content.get(key), str) for key in ("text", "content"))
    if isinstance(content, (list, tuple)):
        return any(contains_text(item) for item in content)
    return False


def replace_text(content: Any, sanitized: str) -> Any:
    """Replace combined text while preserving media parts and container types."""
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
            if contains_text(item):
                replacement = sanitized if not used else ""
                items.append(replace_text(item, replacement))
                used = True
            else:
                items.append(item)
        return tuple(items) if isinstance(content, tuple) else items
    return content


async def scan_joined_text(content: Any, fn: Callable[[str], Awaitable[str]]) -> Any:
    """Scan the *combined* text of a message, then rebuild the content.

    Unlike :func:`apply_to_text` (which scans each part in isolation), this joins
    every text part and scans the whole thing, so an injection split across two
    parts ("ignore all" + "previous instructions") is still caught. The result
    collapses the text parts into a single sanitised part and keeps non-text
    parts (images, etc.) untouched.
    """
    if isinstance(content, str):
        return await fn(content)
    if isinstance(content, (list, tuple)):
        text_parts = [p for p in content if _is_text_part(p)]
        others = [p for p in content if not _is_text_part(p)]
        if not text_parts:
            return content
        sanitized = await fn("\n".join(p["text"] for p in text_parts))
        return [{"type": "text", "text": sanitized}, *others]
    if isinstance(content, dict):
        return await apply_to_text(content, fn)
    return content


async def apply_to_text(content: Any, fn: Callable[[str], Awaitable[str]]) -> Any:
    """Run async ``fn`` over each text segment, preserving overall structure.

    - ``str``           → ``await fn(str)``
    - ``[parts...]``    → each text-bearing part's text replaced by ``fn(text)``;
                          non-text parts are left untouched.
    - single part dict  → its text replaced.

    A shield that blocks raises out of ``fn``, which is the desired behaviour.
    """
    if isinstance(content, str):
        return await fn(content)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if isinstance(content.get(key), str):
                return {**content, key: await fn(content[key])}
        return content
    if isinstance(content, (list, tuple)):
        new_parts = []
        for part in content:
            new_parts.append(await apply_to_text(part, fn))
        return type(content)(new_parts) if isinstance(content, tuple) else new_parts
    return content
