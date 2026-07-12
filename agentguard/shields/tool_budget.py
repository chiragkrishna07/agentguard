"""Budgets and loop detection for autonomous tool use."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import threading
import warnings
from collections.abc import Mapping
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class ToolCallBudget(BaseShield):
    """Bound tool activity within a :class:`SessionContext`.

    This is a deterministic circuit breaker for runaway agents. It limits total
    calls, calls per tool, distinct tools, repeated identical calls, and the
    structural size of arguments. Counts represent attempted calls, including
    calls a later shield may deny; this prevents repeated probing from escaping
    the budget through shield ordering.

    ``max_calls_per_tool`` may be an integer or a mapping of case-insensitive
    tool globs to limits. A mapping can include ``"*"`` as its fallback.
    Set any behavioral limit to ``None`` to disable it.
    """

    requires_tool_session_context = True

    def __init__(
        self,
        *,
        max_calls_per_session: int | None = 100,
        max_calls_per_tool: int | Mapping[str, int] | None = 50,
        max_distinct_tools: int | None = 25,
        max_consecutive_identical: int | None = 5,
        max_argument_bytes: int | None = 65_536,
        max_argument_depth: int | None = 20,
        max_argument_nodes: int | None = 5_000,
        state_key: str = "agentguard.tool_budget",
        on_violation: Literal["block", "warn"] = "block",
    ) -> None:
        self._validate_positive("max_calls_per_session", max_calls_per_session)
        self._validate_per_tool(max_calls_per_tool)
        self._validate_positive("max_distinct_tools", max_distinct_tools)
        self._validate_positive("max_consecutive_identical", max_consecutive_identical)
        self._validate_positive("max_argument_bytes", max_argument_bytes)
        self._validate_positive("max_argument_depth", max_argument_depth)
        self._validate_positive("max_argument_nodes", max_argument_nodes)
        if not state_key:
            raise ValueError("state_key must not be empty")
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'")

        self.max_calls_per_session = max_calls_per_session
        self.max_calls_per_tool = max_calls_per_tool
        self.max_distinct_tools = max_distinct_tools
        self.max_consecutive_identical = max_consecutive_identical
        self.max_argument_bytes = max_argument_bytes
        self.max_argument_depth = max_argument_depth
        self.max_argument_nodes = max_argument_nodes
        self.state_key = state_key
        self.on_violation = on_violation
        self._lock = threading.Lock()

    @staticmethod
    def _validate_positive(name: str, value: int | None) -> None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be >= 1 or None")

    @classmethod
    def _validate_per_tool(cls, value: int | Mapping[str, int] | None) -> None:
        if isinstance(value, Mapping):
            for pattern, limit in value.items():
                if not pattern:
                    raise ValueError("max_calls_per_tool patterns must not be empty")
                cls._validate_positive(f"max_calls_per_tool[{pattern!r}]", limit)
        else:
            cls._validate_positive("max_calls_per_tool", value)

    def _tool_limit(self, tool_name: str) -> int | None:
        configured = self.max_calls_per_tool
        if configured is None or isinstance(configured, int):
            return configured
        matched: int | None = None
        name = tool_name.casefold()
        for pattern, limit in configured.items():
            if fnmatch.fnmatchcase(name, pattern.casefold()):
                matched = limit
        return matched

    @staticmethod
    def _measure_and_canonicalise(
        value: Any,
        *,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> tuple[int, int, int, Any]:
        """Return byte estimate, max depth, node count, and safe fingerprint data."""
        if value is None or isinstance(value, (bool, int, float)):
            text = repr(value)
            return len(text.encode("utf-8")), depth, 1, value
        if isinstance(value, str):
            return len(value.encode("utf-8")), depth, 1, value
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            return len(raw), depth, 1, {"__bytes_sha256__": hashlib.sha256(raw).hexdigest()}

        seen = seen or set()
        if isinstance(value, Mapping):
            object_id = id(value)
            if object_id in seen:
                return 0, depth, 1, {"__cycle__": "mapping"}
            seen.add(object_id)
            try:
                size = 0
                nodes = 1
                max_depth = depth
                canonical: list[tuple[str, Any]] = []
                for key, child in value.items():
                    safe_key = key if isinstance(key, str) else f"<{type(key).__name__}>"
                    size += len(safe_key.encode("utf-8"))
                    child_size, child_depth, child_nodes, child_value = (
                        ToolCallBudget._measure_and_canonicalise(
                            child, depth=depth + 1, seen=seen
                        )
                    )
                    size += child_size
                    nodes += child_nodes
                    max_depth = max(max_depth, child_depth)
                    canonical.append((safe_key, child_value))
            finally:
                # ``seen`` tracks active ancestors, not every object ever
                # visited. Shared substructures must be charged once per
                # occurrence or an attacker can alias a large subtree across
                # many arguments to evade byte/node budgets.
                seen.remove(object_id)
            canonical.sort(key=lambda item: item[0])
            return size, max_depth, nodes, canonical

        if isinstance(value, (list, tuple, set, frozenset)):
            object_id = id(value)
            if object_id in seen:
                return 0, depth, 1, {"__cycle__": "sequence"}
            seen.add(object_id)
            try:
                size = 0
                nodes = 1
                max_depth = depth
                canonical_items: list[Any] = []
                for child in value:
                    child_size, child_depth, child_nodes, child_value = (
                        ToolCallBudget._measure_and_canonicalise(
                            child, depth=depth + 1, seen=seen
                        )
                    )
                    size += child_size
                    nodes += child_nodes
                    max_depth = max(max_depth, child_depth)
                    canonical_items.append(child_value)
            finally:
                seen.remove(object_id)
            if isinstance(value, (set, frozenset)):
                canonical_items.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
            return size, max_depth, nodes, canonical_items

        # Do not invoke an untrusted object's __str__ or __repr__ merely to
        # enforce a security boundary.
        marker = {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}
        return 0, depth, 1, marker

    @classmethod
    def _fingerprint(cls, tool_name: str, canonical_params: Any) -> str:
        payload = json.dumps(
            [tool_name.casefold(), canonical_params],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _violation(
        self, reason: str, code: str, tool_name: str, ctx: SessionContext
    ) -> ShieldResult:
        ctx.metadata["tool_budget_violation"] = {
            "tool_name": tool_name,
            "reason_code": code,
        }
        if self.on_violation == "warn":
            warnings.warn(f"[AgentGuard ToolCallBudget] {reason}", stacklevel=4)
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=False, reason=reason, reason_code=code)

    async def scan_tool_call(
        self, tool_name: str, params: dict[str, Any], ctx: SessionContext
    ) -> ShieldResult:
        size, depth, nodes, canonical = self._measure_and_canonicalise(params)
        if self.max_argument_bytes is not None and size > self.max_argument_bytes:
            return self._violation(
                f"Tool arguments exceed {self.max_argument_bytes} bytes",
                "TOOL_ARGUMENT_SIZE_EXCEEDED",
                tool_name,
                ctx,
            )
        if self.max_argument_depth is not None and depth > self.max_argument_depth:
            return self._violation(
                f"Tool arguments exceed nesting depth {self.max_argument_depth}",
                "TOOL_ARGUMENT_DEPTH_EXCEEDED",
                tool_name,
                ctx,
            )
        if self.max_argument_nodes is not None and nodes > self.max_argument_nodes:
            return self._violation(
                f"Tool arguments exceed {self.max_argument_nodes} structural nodes",
                "TOOL_ARGUMENT_NODE_LIMIT_EXCEEDED",
                tool_name,
                ctx,
            )

        fingerprint = self._fingerprint(tool_name, canonical)
        normalised_name = tool_name.casefold()
        with self._lock:
            raw_state = ctx.metadata.setdefault(
                self.state_key,
                {
                    "total": 0,
                    "by_tool": {},
                    "last_fingerprint": None,
                    "consecutive_identical": 0,
                },
            )
            if not isinstance(raw_state, dict):
                return self._violation(
                    "Tool budget session state is invalid",
                    "TOOL_BUDGET_STATE_INVALID",
                    tool_name,
                    ctx,
                )

            by_tool = raw_state.setdefault("by_tool", {})
            if not isinstance(by_tool, dict):
                return self._violation(
                    "Tool budget session state is invalid",
                    "TOOL_BUDGET_STATE_INVALID",
                    tool_name,
                    ctx,
                )
            total = int(raw_state.get("total", 0)) + 1
            tool_total = int(by_tool.get(normalised_name, 0)) + 1
            distinct = len(set(by_tool) | {normalised_name})
            consecutive = (
                int(raw_state.get("consecutive_identical", 0)) + 1
                if raw_state.get("last_fingerprint") == fingerprint
                else 1
            )

            # Commit attempts before evaluating, so repeated blocked probes are
            # still visible to monitoring and remain bounded.
            raw_state["total"] = total
            by_tool[normalised_name] = tool_total
            raw_state["last_fingerprint"] = fingerprint
            raw_state["consecutive_identical"] = consecutive

            violation: tuple[str, str] | None = None
            if self.max_calls_per_session is not None and total > self.max_calls_per_session:
                violation = (
                    f"Session tool-call budget of {self.max_calls_per_session} exceeded",
                    "TOOL_SESSION_BUDGET_EXCEEDED",
                )
            per_tool_limit = self._tool_limit(tool_name)
            if per_tool_limit is not None and tool_total > per_tool_limit:
                violation = (
                    f"Tool {tool_name!r} call budget of {per_tool_limit} exceeded",
                    "TOOL_BUDGET_EXCEEDED",
                )
            if self.max_distinct_tools is not None and distinct > self.max_distinct_tools:
                violation = (
                    f"Distinct-tool budget of {self.max_distinct_tools} exceeded",
                    "TOOL_DISTINCT_BUDGET_EXCEEDED",
                )
            if (
                self.max_consecutive_identical is not None
                and consecutive > self.max_consecutive_identical
            ):
                violation = (
                    f"Repeated identical tool-call limit of {self.max_consecutive_identical} exceeded",
                    "TOOL_LOOP_DETECTED",
                )

        if violation is not None:
            return self._violation(violation[0], violation[1], tool_name, ctx)
        return ShieldResult(allowed=True)

    def reset(self, ctx: SessionContext) -> None:
        """Clear this shield's counters for a session."""
        with self._lock:
            ctx.metadata.pop(self.state_key, None)
