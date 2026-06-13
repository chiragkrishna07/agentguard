"""
SecretsShield — detect and neutralise leaked credentials.

Two failure modes matter for an agent:
  1. A secret in *user input* (or a retrieved document) gets forwarded to a
     third-party LLM provider — an exfiltration path that bypasses every other
     control.
  2. A secret the model *emits* in its output (echoed from context, hallucinated
     from training data, or pulled from a tool) leaks to the end user or logs.

SecretsShield scans input, output, and tool output. The default action is to
redact (the agent keeps working, the secret never leaves the boundary); set
``on_detect="block"`` to hard-fail instead.

Patterns are deliberately high-signal (provider-specific prefixes, key formats,
PEM blocks) to keep the false-positive rate low. This is a regex tier — pair it
with a dedicated scanner (gitleaks/trufflehog) for exhaustive coverage.
"""
import re
from re import Pattern
from typing import Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext
from agentguard.shields._spans import merge_spans

# ---------------------------------------------------------------------------
# High-signal credential patterns. Ordered most-specific first.
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: dict[str, str] = {
    "PRIVATE_KEY": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b",
    "GITHUB_PAT": r"\bgithub_pat_[A-Za-z0-9_]{82}\b",
    "ANTHROPIC_KEY": r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
    # OpenAI keys start with sk- but not sk-ant- (that's Anthropic, matched above).
    "OPENAI_KEY": r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "SLACK_TOKEN": r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b",
    "STRIPE_KEY": r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}\b",
    "SENDGRID_KEY": r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
}

_COMPILED: dict[str, Pattern[str]] = {
    name: re.compile(pattern) for name, pattern in _SECRET_PATTERNS.items()
}


class SecretsShield(BaseShield):
    """Detect and redact/block credentials in input, output, and tool output.

    Parameters
    ----------
    on_detect:
        ``"redact"`` (default) replaces each secret with ``[REDACTED_<TYPE>]``.
        ``"mask"`` replaces it with asterisks of the same length.
        ``"block"`` raises a GuardBlockedError on the first secret seen.
    scan_directions:
        Which flows to inspect. Defaults to all three.
    custom_patterns:
        Extra ``{name: regex}`` entries merged on top of the built-ins.
    """

    def __init__(
        self,
        on_detect: Literal["redact", "mask", "block"] = "redact",
        scan_directions: tuple[str, ...] = ("input", "output", "tool_output"),
        custom_patterns: dict[str, str] | None = None,
    ) -> None:
        if on_detect not in ("redact", "mask", "block"):
            raise ValueError("on_detect must be 'redact', 'mask', or 'block'")
        self.on_detect = on_detect
        self.scan_directions = scan_directions
        self._patterns: dict[str, Pattern[str]] = dict(_COMPILED)
        if custom_patterns:
            self._patterns.update(
                {name: re.compile(p) for name, p in custom_patterns.items()}
            )

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def _find(self, text: str) -> list[tuple[int, int, str]]:
        hits: list[tuple[int, int, str]] = []
        for name, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    hits.append((m.start(), m.end(), name))
        # Merge overlaps into their union so no secret character can survive in
        # the gap between two overlapping matches (see shields/_spans.py).
        return merge_spans(hits)

    def _scan(self, text: str) -> ShieldResult:
        hits = self._find(text)
        if not hits:
            return ShieldResult(allowed=True)

        if self.on_detect == "block":
            kinds = ", ".join(sorted({name for _, _, name in hits}))
            return ShieldResult(
                allowed=False,
                reason=f"Secret(s) detected and blocked: {kinds}",
                reason_code="SECRET_DETECTED",
            )

        result = text
        for start, end, name in hits:
            original = result[start:end]
            replacement = (
                "*" * len(original) if self.on_detect == "mask" else f"[REDACTED_{name}]"
            )
            result = result[:start] + replacement + result[end:]

        return ShieldResult(
            allowed=True,
            modified_input=result if result != text else None,
        )

    # ------------------------------------------------------------------ #
    # Shield hooks                                                         #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        if "input" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(text)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        if "output" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(text)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        if "tool_output" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(output)
