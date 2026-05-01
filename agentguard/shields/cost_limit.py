from typing import Dict, Literal, Optional

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

# Pricing per 1 million tokens (USD). Updated May 2026.
_DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
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
    def __init__(
        self,
        max_usd: float,
        per: Literal["session", "global"] = "session",
        on_limit: Literal["block", "warn"] = "block",
        model: str = "gpt-4o",
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.max_usd = max_usd
        self.per = per
        self.on_limit = on_limit
        self.model = model
        self.pricing = {**_DEFAULT_PRICING, **(pricing or {})}
        self._global_cost: float = 0.0
        self._encoder = None

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

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        cost = self._token_cost(text, "input")
        current = self._current(ctx)

        if current + cost > self.max_usd:
            msg = (
                f"Cost limit ${self.max_usd:.4f} would be exceeded "
                f"(running: ${current:.4f}, this request: ${cost:.6f})"
            )
            if self.on_limit == "block":
                return ShieldResult(
                    allowed=False, reason=msg, reason_code="COST_LIMIT_EXCEEDED"
                )
            import warnings
            warnings.warn(f"[AgentGuard] {msg}", stacklevel=4)

        self._add(ctx, cost)
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        cost = self._token_cost(text, "output")
        self._add(ctx, cost)
        return ShieldResult(allowed=True)
