"""
PII Redactor shield.

Default (engine="regex"): regex-based, zero extra downloads, covers the most
common entity types (SSN, credit card, email, phone, IBAN, IP address).

Enhanced (engine="presidio"): NER-based via Microsoft Presidio.
Requires: pip install agentguard[presidio]
          python -m spacy download en_core_web_sm
"""
import re
import uuid
from re import Pattern
from typing import Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext
from agentguard.shields._spans import merge_spans

# ---------------------------------------------------------------------------
# Regex patterns — ordered so longer/more-specific patterns match first
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: dict[str, str] = {
    # SSN area numbers never start with 9 (that range is ITIN, matched below).
    "SSN": r"\b(?!9)\d{3}-\d{2}-\d{4}\b",
    # 16-digit (Visa/MC/Discover, 4-4-4-4) or 15-digit Amex (4-6-5). Luhn-checked.
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PHONE_US": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b",
    "IP_ADDRESS": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    # Full IPv6 (8 groups). Compressed (::) forms are not matched to keep FPs low.
    "IPV6_ADDRESS": r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b",
    "MAC_ADDRESS": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
    # US ITIN: 9, then a 70-88/90-92/94-99 group — looks like an SSN but starts with 9.
    "ITIN": r"\b9\d{2}-(?:7\d|8[0-8]|9[0-24-9])-\d{4}\b",
    "DATE_OF_BIRTH": r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
}

_COMPILED: dict[str, Pattern[str]] = {
    entity: re.compile(pattern, re.IGNORECASE)
    for entity, pattern in _REGEX_PATTERNS.items()
}


def _luhn_ok(candidate: str) -> bool:
    """Validate a card number with the Luhn checksum (mod-10)."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class PIIRedactor(BaseShield):
    def __init__(
        self,
        entities: list[str] | None = None,
        mode: Literal["redact", "mask", "tokenize"] = "redact",
        language: str = "en",
        score_threshold: float = 0.6,
        engine: Literal["regex", "presidio"] = "regex",
        redact_output: bool = False,
        scan_tool_output: bool = False,
    ) -> None:
        if mode not in ("redact", "mask", "tokenize"):
            raise ValueError("mode must be 'redact', 'mask', or 'tokenize'")
        if engine not in ("regex", "presidio"):
            raise ValueError("engine must be 'regex' or 'presidio'")

        self.entities = entities  # None → all
        self.mode = mode
        self.language = language
        self.score_threshold = score_threshold
        self.engine = engine
        # Detect+redact PII the model emits (leakage), not just de-tokenize.
        self.redact_output = redact_output
        # Detect+redact PII in retrieved/tool content before it re-enters the agent.
        self.scan_tool_output_flag = scan_tool_output
        self._analyzer = None
        self._anonymizer = None

    # ------------------------------------------------------------------ #
    # Lazy Presidio init                                                   #
    # ------------------------------------------------------------------ #

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:
                raise ImportError(
                    "Presidio is not installed. "
                    "Run: pip install agentguard[presidio] && python -m spacy download en_core_web_sm"
                ) from exc
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def _get_anonymizer(self):
        if self._anonymizer is None:
            from presidio_anonymizer import AnonymizerEngine
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    # ------------------------------------------------------------------ #
    # Regex engine                                                         #
    # ------------------------------------------------------------------ #

    def _regex_find(self, text: str) -> list[tuple[int, int, str]]:
        """Return (start, end, entity_type) spans, overlaps merged, end-first.

        Overlapping spans are merged into their union so we never leave part of
        a sensitive value exposed (e.g. a DATE_OF_BIRTH that starts just before
        an overlapping CREDIT_CARD must not shadow the card's tail).
        """
        targets = self.entities or list(_COMPILED.keys())
        hits: list[tuple[int, int, str]] = []
        for entity in targets:
            pattern = _COMPILED.get(entity)
            if pattern is None:
                continue
            for m in pattern.finditer(text):
                if m.end() <= m.start():  # ignore zero-width matches
                    continue
                # Credit cards are validated with the Luhn checksum so arbitrary
                # 16-digit numbers (order IDs, etc.) aren't redacted as cards.
                if entity == "CREDIT_CARD" and not _luhn_ok(m.group()):
                    continue
                hits.append((m.start(), m.end(), entity))
        return merge_spans(hits)

    def _apply_regex_redaction(self, text: str, ctx: SessionContext) -> str | None:
        hits = self._regex_find(text)
        if not hits:
            return None

        result = text
        for start, end, entity in hits:
            original = result[start:end]
            if self.mode == "tokenize":
                token = f"[AGENTGUARD_{entity}_{uuid.uuid4().hex[:8].upper()}]"
                ctx.store_token(token, original)
                replacement = token
            elif self.mode == "mask":
                replacement = "*" * len(original)
            else:  # redact
                replacement = f"[REDACTED_{entity}]"
            result = result[:start] + replacement + result[end:]

        return result if result != text else None

    # ------------------------------------------------------------------ #
    # Presidio engine                                                      #
    # ------------------------------------------------------------------ #

    def _apply_presidio_redaction(self, text: str, ctx: SessionContext) -> str | None:
        from presidio_anonymizer.entities import OperatorConfig

        analyzer = self._get_analyzer()
        anonymizer = self._get_anonymizer()

        results = analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        if not results:
            return None

        if self.mode == "tokenize":
            modified = text
            for r in sorted(results, key=lambda x: x.start, reverse=True):
                original = text[r.start : r.end]
                token = f"[AGENTGUARD_{r.entity_type}_{uuid.uuid4().hex[:8].upper()}]"
                ctx.store_token(token, original)
                modified = modified[: r.start] + token + modified[r.end :]
            return modified

        elif self.mode == "mask":
            operators = {
                r.entity_type: OperatorConfig(
                    "mask",
                    {"chars_to_mask": len(text[r.start : r.end]), "masking_char": "*", "from_end": False},
                )
                for r in results
            }
        else:  # redact
            operators = {
                r.entity_type: OperatorConfig(
                    "replace", {"new_value": f"[REDACTED_{r.entity_type}]"}
                )
                for r in results
            }

        anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        return anonymized.text if anonymized.text != text else None

    # ------------------------------------------------------------------ #
    # Shield hooks                                                         #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        if self.engine == "presidio":
            modified = self._apply_presidio_redaction(text, ctx)
        else:
            modified = self._apply_regex_redaction(text, ctx)

        return ShieldResult(allowed=True, modified_input=modified)

    def _redact(self, text: str, ctx: SessionContext) -> str | None:
        if self.engine == "presidio":
            return self._apply_presidio_redaction(text, ctx)
        return self._apply_regex_redaction(text, ctx)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        # tokenize mode re-inserts the user's original PII so the agent's reply
        # stays coherent across turns — this takes precedence over redaction.
        if self.mode == "tokenize" and ctx._token_map:
            resolved = ctx.resolve_all_tokens(text)
            return ShieldResult(
                allowed=True,
                modified_input=resolved if resolved != text else None,
            )
        # Otherwise optionally redact PII the model itself emitted (leakage).
        if self.redact_output and self.mode in ("redact", "mask"):
            return ShieldResult(allowed=True, modified_input=self._redact(text, ctx))
        return ShieldResult(allowed=True)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        if not self.scan_tool_output_flag or self.mode == "tokenize":
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=True, modified_input=self._redact(output, ctx))
