"""Containment: a kill switch for an agent that has gone wrong.

Every other shield answers one question about one piece of traffic. None of them
can answer "this agent is misbehaving — stop it." That gap is OWASP ASI10 (Rogue
Agents), and it is where the other risk categories terminate: any compromise you
cannot promptly stop becomes a rogue agent holding valid credentials.

:class:`CircuitBreaker` is the in-process half of that control. It trips on
manual command or automatically when denials accumulate, and once tripped it
denies **every** boundary — input, output, tool calls, tool results, memory —
rather than only the one that failed. Partial containment is not containment: an
agent that lost tool access but kept its memory-write path is still acting.

Two properties make it useful rather than decorative:

**It fails closed by scope, not by boundary.** A trip is checked before any other
policy on every hook, so a rogue loop cannot find an unguarded path out.

**It does not trust the agent to stop itself.** Tripping is driven by the guard
pipeline and by application code, never by a model instruction. Cooperative
shutdown trusts the component you are trying to contain.

Scope selects blast radius, narrowest first — ``"session"`` contains one
conversation, ``"user"`` one principal, ``"global"`` the whole process::

    breaker = CircuitBreaker(max_blocks=5, scope="session")
    guard = Guard(shields=[PromptShield(), breaker])
    ...
    breaker.trip(reason="on-call paged", ctx=ctx)   # manual
    breaker.reset(ctx=ctx)                          # deliberate re-enable

This is process-local containment: it stops traffic crossing *this* guard. It is
not a substitute for revoking credentials at the control plane, which is the
authoritative kill switch when an agent holds tokens of its own.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, GuardDecision, ShieldResult
from agentguard.core.session import SessionContext

_GLOBAL_KEY = "__global__"


class CircuitBreakerTripped(Exception):
    """Raised by :meth:`CircuitBreaker.check` when the breaker is open."""

    def __init__(self, reason: str, scope_key: str) -> None:
        self.reason = reason
        self.scope_key = scope_key
        super().__init__(reason)


class CircuitBreaker(BaseShield):
    """Halt all traffic for a session, user, or process after repeated denials.

    Parameters
    ----------
    max_blocks:
        Trip automatically once this many blocks are observed in the window.
        ``None`` disables automatic tripping (manual :meth:`trip` only).
    window_seconds:
        Rolling window for ``max_blocks``. ``None`` counts for the lifetime of
        the scope, which is the stricter reading of "repeated denials".
    scope:
        ``"session"`` (default), ``"user"``, or ``"global"``. ``"user"`` requires
        ``ctx.user_id`` and fails closed without one, matching
        :class:`~agentguard.shields.rate_limit.RateLimit`.
    cooldown_seconds:
        Auto-reset after this long. ``None`` (default) requires an explicit
        :meth:`reset`, so a tripped breaker cannot quietly re-arm a rogue agent.
    trip_on_codes:
        Only count these reason codes toward ``max_blocks``. ``None`` counts all.
        Use this to trip on the signals that indicate an attack rather than on
        ordinary policy noise.
    trip_immediately_on:
        Reason codes that trip on a single occurrence. Some events are not
        "noise that accumulates" — one confirmed rug pull is enough.

    Notes
    -----
    State is instance-level and lock-guarded, so a breaker shared across guards
    contains all of them. Because containment must outlive one request, state is
    deliberately *not* stored in ``SessionContext``: a caller who supplies a
    fresh context per call cannot thereby escape a trip.
    """

    # Containment applies to a request as a whole; the argument-DLP phase must
    # not double-count it.
    scan_tool_arguments_as_input = False

    def __init__(
        self,
        *,
        max_blocks: int | None = 10,
        window_seconds: float | None = 300.0,
        scope: Literal["session", "user", "global"] = "session",
        cooldown_seconds: float | None = None,
        trip_on_codes: tuple[str, ...] | None = None,
        trip_immediately_on: tuple[str, ...] = (
            "TOOL_DEFINITION_CHANGED",
            "TOOL_DEFINITION_POISONED",
        ),
        max_tracked_scopes: int = 10_000,
    ) -> None:
        if scope not in ("session", "user", "global"):
            raise ValueError("scope must be 'session', 'user', or 'global'")
        if max_blocks is not None and (
            isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks < 1
        ):
            raise ValueError("max_blocks must be >= 1 or None")
        for name, value in (
            ("window_seconds", window_seconds),
            ("cooldown_seconds", cooldown_seconds),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            ):
                raise ValueError(f"{name} must be > 0 or None")
        for name, codes in (
            ("trip_on_codes", trip_on_codes),
            ("trip_immediately_on", trip_immediately_on),
        ):
            if codes is None:
                continue
            if isinstance(codes, str) or not all(
                isinstance(code, str) and code for code in codes
            ):
                raise ValueError(f"{name} must be a sequence of non-empty strings")
        if (
            isinstance(max_tracked_scopes, bool)
            or not isinstance(max_tracked_scopes, int)
            or max_tracked_scopes < 1
        ):
            raise ValueError("max_tracked_scopes must be >= 1")

        self.max_blocks = max_blocks
        self.window_seconds = window_seconds
        self.scope = scope
        self.cooldown_seconds = cooldown_seconds
        self.trip_on_codes = tuple(trip_on_codes) if trip_on_codes is not None else None
        self.trip_immediately_on = tuple(trip_immediately_on)
        self.max_tracked_scopes = max_tracked_scopes
        self._lock = threading.Lock()
        self._tripped: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ #
    # Manual control                                                       #
    # ------------------------------------------------------------------ #

    def trip(self, *, reason: str = "manually tripped", ctx: SessionContext | None = None) -> str:
        """Open the breaker for the scope implied by ``ctx``.

        Returns the scope key that was tripped, so an operator can confirm the
        blast radius. A global-scope breaker ignores ``ctx``.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        key = self._scope_key(ctx, for_write=True)
        with self._lock:
            self._evict_if_needed()
            self._tripped[key] = {"reason": reason, "at": time.monotonic()}
        return key

    def reset(self, *, ctx: SessionContext | None = None) -> None:
        """Close the breaker for this scope and clear its denial history."""
        key = self._scope_key(ctx, for_write=True)
        with self._lock:
            self._tripped.pop(key, None)
            self._events.pop(key, None)

    def reset_all(self) -> None:
        """Close every breaker this instance is holding open."""
        with self._lock:
            self._tripped.clear()
            self._events.clear()

    def is_tripped(self, ctx: SessionContext | None = None) -> bool:
        """Whether traffic for this scope is currently contained."""
        try:
            key = self._scope_key(ctx)
        except CircuitBreakerTripped:
            return True
        with self._lock:
            return self._active_trip(key) is not None

    def state(self, ctx: SessionContext | None = None) -> dict[str, Any]:
        """Return ``{"tripped": bool, "reason": str|None, "blocks": int}``."""
        try:
            key = self._scope_key(ctx)
        except CircuitBreakerTripped:
            return {"tripped": True, "reason": "unidentified principal", "blocks": 0}
        with self._lock:
            trip = self._active_trip(key)
            return {
                "tripped": trip is not None,
                "reason": trip["reason"] if trip else None,
                "blocks": len(self._recent_events(key)),
            }

    def check(self, ctx: SessionContext | None = None) -> None:
        """Raise :class:`CircuitBreakerTripped` if contained.

        For application code outside a guard boundary — a scheduler loop, a
        background worker — that should also stop when the breaker opens.
        """
        key = self._scope_key(ctx)
        with self._lock:
            trip = self._active_trip(key)
        if trip is not None:
            raise CircuitBreakerTripped(trip["reason"], key)

    # ------------------------------------------------------------------ #
    # Boundaries — every hook is contained                                 #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_output_preview(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: SessionContext
    ) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_tool_arguments(
        self, tool_name: str, text: str, ctx: SessionContext
    ) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_memory_write(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._verdict(ctx)

    async def scan_memory_read(self, text: str, ctx: SessionContext) -> ShieldResult:
        return self._verdict(ctx)

    async def on_decision(self, decision: GuardDecision, ctx: SessionContext) -> None:
        """Count denials and trip when they accumulate.

        Using the content-free observer hook means the breaker sees blocks made
        by *any* shield, including shields ordered after it, without ever
        receiving the payload that caused them.
        """
        if decision.allowed:
            return
        code = decision.reason_code
        # A trip is itself a block; counting it would be circular.
        if code == "CIRCUIT_BREAKER_OPEN":
            return
        try:
            key = self._scope_key(ctx)
        except CircuitBreakerTripped:
            return
        if code and code in self.trip_immediately_on:
            with self._lock:
                self._evict_if_needed()
                self._tripped.setdefault(
                    key,
                    {"reason": f"critical denial {code}", "at": time.monotonic()},
                )
            return
        if self.max_blocks is None:
            return
        if self.trip_on_codes is not None and code not in self.trip_on_codes:
            return
        with self._lock:
            self._evict_if_needed()
            events = self._events.setdefault(key, [])
            events.append(time.monotonic())
            recent = self._recent_events(key)
            self._events[key] = recent
            if len(recent) >= self.max_blocks:
                self._tripped.setdefault(
                    key,
                    {
                        "reason": (
                            f"{len(recent)} blocked requests"
                            + (
                                f" within {self.window_seconds:g}s"
                                if self.window_seconds is not None
                                else ""
                            )
                        ),
                        "at": time.monotonic(),
                    },
                )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _verdict(self, ctx: SessionContext) -> ShieldResult:
        try:
            key = self._scope_key(ctx)
        except CircuitBreakerTripped as exc:
            return ShieldResult(
                allowed=False, reason=exc.reason, reason_code="CIRCUIT_BREAKER_OPEN"
            )
        with self._lock:
            trip = self._active_trip(key)
        if trip is None:
            return ShieldResult(allowed=True)
        return ShieldResult(
            allowed=False,
            reason=(
                f"Circuit breaker is open for {self.scope} scope "
                f"({trip['reason']}); traffic is contained until it is reset"
            ),
            reason_code="CIRCUIT_BREAKER_OPEN",
        )

    def _scope_key(self, ctx: SessionContext | None, *, for_write: bool = False) -> str:
        if self.scope == "global":
            return _GLOBAL_KEY
        if ctx is None:
            if for_write:
                raise ValueError(f"{self.scope} scope requires a SessionContext")
            raise CircuitBreakerTripped("missing session context", _GLOBAL_KEY)
        if self.scope == "user":
            if not ctx.user_id:
                # Matching RateLimit: an unidentified principal cannot be
                # contained per-user, so it is denied rather than exempted.
                raise CircuitBreakerTripped(
                    "user-scoped containment requires ctx.user_id", _GLOBAL_KEY
                )
            return f"user:{ctx.user_id}"
        return f"session:{ctx.session_id}"

    def _active_trip(self, key: str) -> dict[str, Any] | None:
        """Return the live trip record, honouring cooldown. Caller holds lock."""
        trip = self._tripped.get(key)
        if trip is None:
            return None
        if self.cooldown_seconds is None:
            return trip
        if time.monotonic() - trip["at"] >= self.cooldown_seconds:
            del self._tripped[key]
            self._events.pop(key, None)
            return None
        return trip

    def _recent_events(self, key: str) -> list[float]:
        """Events inside the window. Caller holds lock."""
        events = self._events.get(key, [])
        if self.window_seconds is None:
            return list(events)
        cutoff = time.monotonic() - self.window_seconds
        return [stamp for stamp in events if stamp >= cutoff]

    def _evict_if_needed(self) -> None:
        """Bound tracked scopes. Caller holds lock.

        Tripped scopes are retained in preference to mere history: forgetting a
        trip would silently release containment, while forgetting counters only
        loses progress toward one.
        """
        if len(self._events) > self.max_tracked_scopes:
            for key in list(self._events)[: len(self._events) - self.max_tracked_scopes]:
                if key not in self._tripped:
                    del self._events[key]
        if len(self._tripped) > self.max_tracked_scopes:
            for key in list(self._tripped)[: len(self._tripped) - self.max_tracked_scopes]:
                del self._tripped[key]
