import math
import threading
from numbers import Real
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

# Example pricing per 1 million tokens (USD). Provider prices change; production
# deployments should pass an explicitly reviewed ``pricing`` table rather than
# treating package defaults as a billing source of truth.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "llama-3.1-70b": {"input": 0.88, "output": 0.88},
    "llama-3.1-8b": {"input": 0.20, "output": 0.20},
    # Conservative fallback for unknown models
    "unknown": {"input": 10.00, "output": 30.00},
}

# tiktoken is accurate for OpenAI models. For others we apply a safety multiplier
# to avoid underestimating cost (different tokenisers produce different counts).
_NON_OPENAI_MULTIPLIER = 1.3


class CostLimit(BaseShield):
    scan_tool_arguments_as_input = False

    def __init__(
        self,
        max_usd: float,
        per: Literal["session", "global"] = "session",
        on_limit: Literal["block", "warn"] = "block",
        model: str = "gpt-4o",
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if (
            isinstance(max_usd, bool)
            or not isinstance(max_usd, Real)
            or not math.isfinite(float(max_usd))
            or max_usd <= 0
        ):
            raise ValueError("max_usd must be > 0")
        if per not in ("session", "global"):
            raise ValueError("per must be 'session' or 'global'")
        if on_limit not in ("block", "warn"):
            raise ValueError("on_limit must be 'block' or 'warn'")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        self.max_usd: float = max_usd
        self.per: Literal["session", "global"] = per
        self.on_limit: Literal["block", "warn"] = on_limit
        self.model: str = model
        self.pricing: dict[str, dict[str, float]] = {
            **_DEFAULT_PRICING,
            **(pricing or {}),
        }
        for name, rates in self.pricing.items():
            if not isinstance(rates, dict) or not {"input", "output"}.issubset(rates):
                raise ValueError(f"pricing for {name!r} needs input and output rates")
            if any(
                isinstance(rates[key], bool)
                or not isinstance(rates[key], Real)
                or not math.isfinite(float(rates[key]))
                or rates[key] < 0
                for key in ("input", "output")
            ):
                raise ValueError(f"pricing for {name!r} must be non-negative")
        self._global_cost: float = 0.0
        self._encoder: Any | None = None
        # Serialises the check-then-charge in global mode so two concurrent
        # threads can't both slip past the budget on the same tick.
        self._lock: threading.Lock = threading.Lock()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "tiktoken is required for CostLimit. Run: pip install tiktoken"
                ) from exc
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        raw = len(self._get_encoder().encode(text))
        if not self.model.startswith("gpt-"):
            raw = int(raw * _NON_OPENAI_MULTIPLIER)
        return raw

    def _token_cost(self, text: str, direction: Literal["input", "output"]) -> float:
        tokens = self._count_tokens(text)
        key = self.model if self.model in self.pricing else "unknown"
        rate = self.pricing[key][direction]
        return (tokens / 1_000_000) * rate

    def _current(self, ctx: SessionContext) -> float:
        return ctx.cost_usd if self.per == "session" else self._global_cost

    def _add(self, ctx: SessionContext, amount: float) -> None:
        ctx.cost_usd += amount
        self._global_cost += amount

    def remaining_usd(self, ctx: SessionContext) -> float:
        """Return the non-negative remaining budget for this scope."""
        with self._lock:
            return max(0.0, self.max_usd - self._current(ctx))

    def estimate_cost(
        self,
        *,
        input_text: str = "",
        output_text: str = "",
    ) -> float:
        """Estimate cost without mutating budget state."""
        return self._token_cost(input_text, "input") + self._token_cost(output_text, "output")

    def _check_and_charge(
        self,
        ctx: SessionContext,
        cost: float,
        direction: Literal["input", "output"],
    ) -> ShieldResult:
        with self._lock:
            current = self._current(ctx)
            if current + cost > self.max_usd:
                msg = (
                    f"Cost limit ${self.max_usd:.4f} would be exceeded "
                    f"(running: ${current:.4f}, {direction}: ${cost:.6f})"
                )
                ctx.metadata["cost_limit"] = {
                    "limited": True,
                    "direction": direction,
                    "limit_usd": self.max_usd,
                    "current_usd": round(current, 8),
                    "attempted_usd": round(cost, 8),
                }
                if self.on_limit == "block":
                    if direction == "output":
                        # Output policy runs after provider generation, so this
                        # spend has already happened even though delivery is
                        # blocked. Charge it to prevent repeated over-budget
                        # generations from appearing free to the next request.
                        self._add(ctx, cost)
                        ctx.metadata["cost_limit"]["actual_total_usd"] = round(
                            self._current(ctx), 8
                        )
                    return ShieldResult(
                        allowed=False, reason=msg, reason_code="COST_LIMIT_EXCEEDED"
                    )
                import warnings

                warnings.warn(f"[AgentGuard] {msg}", stacklevel=4)

            self._add(ctx, cost)
        return ShieldResult(allowed=True)

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        cost = self._token_cost(text, "input")
        return self._check_and_charge(ctx, cost, "input")

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        cost = self._token_cost(text, "output")
        # This is necessarily post-generation when used as a normal output
        # shield: it prevents an over-budget response from leaving the boundary
        # and stops subsequent requests, but cannot undo provider spend already
        # incurred. Set provider-side max tokens/timeouts as a first-line cap.
        return self._check_and_charge(ctx, cost, "output")

    async def scan_output_preview(
        self, text: str, ctx: SessionContext
    ) -> ShieldResult:
        """Check a cumulative stream prefix without charging it repeatedly."""
        cost = self._token_cost(text, "output")
        with self._lock:
            current = self._current(ctx)
            if current + cost > self.max_usd and self.on_limit == "block":
                return ShieldResult(
                    allowed=False,
                    reason=(
                        f"Cost limit ${self.max_usd:.4f} would be exceeded by "
                        "the streamed output"
                    ),
                    reason_code="COST_LIMIT_EXCEEDED",
                )
        return ShieldResult(allowed=True)
