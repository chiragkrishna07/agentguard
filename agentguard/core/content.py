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
