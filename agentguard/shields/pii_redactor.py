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
from typing import Dict, List, Literal, Optional, Pattern, Tuple

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

# ---------------------------------------------------------------------------
# Regex patterns — ordered so longer/more-specific patterns match first
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: Dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PHONE_US": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b",
    "IP_ADDRESS": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    "DATE_OF_BIRTH": r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
}

_COMPILED: Dict[str, Pattern[str]] = {
    entity: re.compile(pattern, re.IGNORECASE)
    for entity, pattern in _REGEX_PATTERNS.items()
}


class PIIRedactor(BaseShield):
    def __init__(
        self,
        entities: Optional[List[str]] = None,
        mode: Literal["redact", "mask", "tokenize"] = "redact",
        language: str = "en",
        score_threshold: float = 0.6,
        engine: Literal["regex", "presidio"] = "regex",
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

    def _regex_find(self, text: str) -> List[Tuple[int, int, str]]:
        """Returns list of (start, end, entity_type) sorted by start position."""
        targets = self.entities or list(_COMPILED.keys())
        hits: List[Tuple[int, int, str]] = []
        for entity in targets:
            pattern = _COMPILED.get(entity)
            if pattern is None:
                continue
            for m in pattern.finditer(text):
                hits.append((m.start(), m.end(), entity))
        # Sort descending by start so replacements don't shift offsets
        hits.sort(key=lambda x: x[0], reverse=True)
        return hits

    def _apply_regex_redaction(self, text: str, ctx: SessionContext) -> Optional[str]:
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

    def _apply_presidio_redaction(self, text: str, ctx: SessionContext) -> Optional[str]:
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

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        if self.mode == "tokenize" and ctx._token_map:
            resolved = ctx.resolve_all_tokens(text)
            return ShieldResult(
                allowed=True,
                modified_input=resolved if resolved != text else None,
            )
        return ShieldResult(allowed=True)
