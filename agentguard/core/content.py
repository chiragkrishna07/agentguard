"""Helpers for chat message content that may be structured.

Modern chat APIs allow a message's ``content`` to be a list of typed parts
(``{"type": "text", "text": ...}`` alongside images/audio) instead of a plain
string. The shields operate on text, so adapters use these helpers to pull the
text out for scanning and to write sanitised text back without disturbing the
non-text parts.
"""
from collections.abc import Awaitable, Callable
from typing import Any


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


async def apply_to_text(
    content: Any, fn: Callable[[str], Awaitable[str]]
) -> Any:
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
