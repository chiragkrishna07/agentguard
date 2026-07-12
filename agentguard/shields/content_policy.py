"""Provider-neutral content policy enforcement.

There is no universal, jurisdiction-independent definition of harmful content.
This module therefore supplies a strict policy boundary and callback contract,
not a misleading one-size-fits-all keyword list. Applications can attach their
chosen moderation model and add deterministic organization-specific rules.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Pattern

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

ContentDirection = Literal["input", "output", "tool_call", "tool_output"]


@dataclass(frozen=True)
class ContentVerdict:
    """Normalized result returned by a content-policy classifier."""

    allowed: bool
    categories: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None
    reason_code: str = "CONTENT_POLICY_VIOLATION"
    modified_text: str | None = None


@dataclass(frozen=True)
class ContentRule:
    """A deterministic organization policy rule.

    ``pattern`` may be a regex string or compiled regex. Rules should target
    high-confidence policy violations; broad semantic safety decisions belong
    in the classifier callback.
    """

    category: str
    pattern: str | Pattern[str]
    directions: tuple[ContentDirection, ...] = (
        "input",
        "output",
        "tool_call",
        "tool_output",
    )
    reason: str | None = None
    flags: int = re.IGNORECASE


ClassifierResult = ContentVerdict | Mapping[str, float | bool]
ContentClassifier = Callable[
    [str, ContentDirection, SessionContext],
    ClassifierResult | Awaitable[ClassifierResult],
]


class ContentPolicyShield(BaseShield):
    """Enforce content safety on user input, agent output, and tool output.

    A classifier can return either :class:`ContentVerdict` or a mapping of
    category names to probabilities/booleans. Mapping categories whose scores
    meet their configured threshold are violations. Deterministic ``rules`` run
    first, making the shield useful for organization-specific prohibitions and
    regulated terms even without a moderation service.

    Classifier failures and malformed responses fail closed by default through
    AgentGuard's normal ``GuardShieldError`` path. Set ``on_error="warn"`` only
    when availability is more important than the content-policy boundary.
    """

    @property
    def requires_buffered_output(self) -> bool:
        return "output" in self.directions

    def __init__(
        self,
        *,
        classifier: ContentClassifier | None = None,
        rules: Sequence[ContentRule | Mapping[str, Any]] | None = None,
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = 0.5,
        directions: Sequence[ContentDirection] = (
            "input",
            "output",
            "tool_call",
            "tool_output",
        ),
        classifier_timeout_seconds: float | None = 10.0,
        on_violation: Literal["block", "warn"] = "block",
        on_error: Literal["block", "warn"] = "block",
    ) -> None:
        if classifier is None and not rules:
            raise ValueError("ContentPolicyShield needs a classifier and/or at least one rule")
        if classifier is not None and not callable(classifier):
            raise ValueError("classifier must be callable or None")
        if isinstance(default_threshold, bool) or not 0 <= default_threshold <= 1:
            raise ValueError("default_threshold must be between 0 and 1")
        if classifier_timeout_seconds is not None and (
            isinstance(classifier_timeout_seconds, bool) or classifier_timeout_seconds <= 0
        ):
            raise ValueError("classifier_timeout_seconds must be > 0 or None")
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'")
        if on_error not in ("block", "warn"):
            raise ValueError("on_error must be 'block' or 'warn'")

        valid_directions: set[str] = {"input", "output", "tool_call", "tool_output"}
        if not directions or not set(directions).issubset(valid_directions):
            raise ValueError("directions must contain input, output, tool_call, and/or tool_output")
        self.classifier = classifier
        self.thresholds = dict(thresholds or {})
        for category, threshold in self.thresholds.items():
            if isinstance(threshold, bool) or not 0 <= threshold <= 1:
                raise ValueError(f"threshold for {category!r} must be between 0 and 1")
        self.default_threshold = default_threshold
        self.directions = frozenset(directions)
        self.classifier_timeout_seconds = classifier_timeout_seconds
        self.on_violation = on_violation
        self.on_error = on_error
        self.rules = tuple(self._coerce_rule(rule) for rule in (rules or ()))

    @staticmethod
    def _coerce_rule(raw: ContentRule | Mapping[str, Any]) -> tuple[ContentRule, Pattern[str]]:
        rule = raw if isinstance(raw, ContentRule) else ContentRule(**raw)
        if not rule.category:
            raise ValueError("content rule category must not be empty")
        if not rule.directions or set(rule.directions) - {
            "input",
            "output",
            "tool_call",
            "tool_output",
        }:
            raise ValueError(f"invalid directions for content rule {rule.category!r}")
        try:
            compiled = (
                rule.pattern
                if isinstance(rule.pattern, Pattern)
                else re.compile(rule.pattern, rule.flags)
            )
        except re.error as exc:
            raise ValueError(f"invalid pattern for content rule {rule.category!r}: {exc}") from exc
        return rule, compiled

    def _from_scores(self, scores: Mapping[str, float | bool]) -> ContentVerdict:
        normalized: dict[str, float] = {}
        triggered: list[str] = []
        for raw_category, raw_score in scores.items():
            category = str(raw_category)
            if isinstance(raw_score, bool):
                score = 1.0 if raw_score else 0.0
            elif isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                score = float(raw_score)
                if not 0 <= score <= 1:
                    raise ValueError(f"classifier score for {category!r} is outside [0, 1]")
            else:
                raise TypeError(f"classifier score for {category!r} must be bool or number")
            normalized[category] = score
            if score >= self.thresholds.get(category, self.default_threshold):
                triggered.append(category)
        return ContentVerdict(
            allowed=not triggered,
            categories=tuple(sorted(triggered)),
            scores=normalized,
        )

    async def _classify(
        self, text: str, direction: ContentDirection, ctx: SessionContext
    ) -> ContentVerdict:
        if self.classifier is None:
            return ContentVerdict(allowed=True)
        if inspect.iscoroutinefunction(self.classifier):
            pending: Any = self.classifier(text, direction, ctx)
        else:
            # A synchronous moderation client must not freeze the agent's event
            # loop. wait_for bounds the caller's wait; as with all Python thread
            # timeouts, the callback itself should also configure I/O timeouts.
            pending = asyncio.to_thread(self.classifier, text, direction, ctx)
        if inspect.isawaitable(pending):
            if self.classifier_timeout_seconds is None:
                result = await pending
            else:
                result = await asyncio.wait_for(pending, timeout=self.classifier_timeout_seconds)
        else:
            result = pending
        # A regular callable may intentionally return an awaitable. Support it
        # while retaining the same timeout boundary.
        if inspect.isawaitable(result):
            if self.classifier_timeout_seconds is None:
                result = await result
            else:
                result = await asyncio.wait_for(result, timeout=self.classifier_timeout_seconds)
        if isinstance(result, ContentVerdict):
            return result
        if isinstance(result, Mapping):
            return self._from_scores(result)
        raise TypeError("content classifier must return ContentVerdict or a score mapping")

    def _record(
        self,
        ctx: SessionContext,
        direction: ContentDirection,
        verdict: ContentVerdict,
    ) -> None:
        # Scores and category labels are safe telemetry; raw content is never
        # retained in the shared session metadata.
        ctx.metadata["content_policy_violation"] = {
            "direction": direction,
            "categories": list(verdict.categories),
            "scores": dict(verdict.scores),
            "reason_code": verdict.reason_code,
        }

    def _enforce(
        self,
        verdict: ContentVerdict,
        direction: ContentDirection,
        ctx: SessionContext,
    ) -> ShieldResult:
        if verdict.allowed:
            return ShieldResult(allowed=True, modified_input=verdict.modified_text)
        self._record(ctx, direction, verdict)
        reason = verdict.reason or (
            "Content policy violation"
            + (f": {', '.join(verdict.categories)}" if verdict.categories else "")
        )
        if self.on_violation == "warn":
            warnings.warn(f"[AgentGuard ContentPolicyShield] {reason}", stacklevel=4)
            return ShieldResult(allowed=True, modified_input=verdict.modified_text)
        return ShieldResult(
            allowed=False,
            reason=reason,
            reason_code=verdict.reason_code,
        )

    async def _scan(
        self, text: str, direction: ContentDirection, ctx: SessionContext
    ) -> ShieldResult:
        if direction not in self.directions:
            return ShieldResult(allowed=True)
        for rule, compiled in self.rules:
            if direction in rule.directions and compiled.search(text):
                verdict = ContentVerdict(
                    allowed=False,
                    categories=(rule.category,),
                    reason=rule.reason,
                    reason_code="CONTENT_RULE_VIOLATION",
                )
                return self._enforce(verdict, direction, ctx)
        try:
            verdict = await self._classify(text, direction, ctx)
        except Exception as exc:
            if self.on_error == "warn":
                warnings.warn(
                    f"[AgentGuard ContentPolicyShield] classifier error: {type(exc).__name__}",
                    stacklevel=4,
                )
                ctx.metadata["content_policy_error"] = type(exc).__name__
                return ShieldResult(allowed=True)
            raise
        return self._enforce(verdict, direction, ctx)

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        return await self._scan(text, "input", ctx)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        return await self._scan(text, "output", ctx)

    async def scan_tool_arguments(
        self, tool_name: str, text: str, ctx: SessionContext
    ) -> ShieldResult:
        return await self._scan(text, "tool_call", ctx)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        return await self._scan(output, "tool_output", ctx)
