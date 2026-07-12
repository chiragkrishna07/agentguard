"""PII detection and redaction for agent input, output, and tool results.

The zero-dependency regex engine favours structured identifiers and checksum or
context validation. It covers common contact, payment, network, travel,
government, healthcare, tax, and banking identifiers. The optional Presidio
engine adds NER-based detection (``pip install agentguard[presidio]``).
"""

import json
import re
import uuid
from datetime import date
from re import Match, Pattern
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext
from agentguard.shields._spans import merge_spans

# ---------------------------------------------------------------------------
# Regex candidates. Validation below removes ambiguous/invalid candidates.
# Named ``value`` groups retain contextual labels while redacting only the PII.
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: dict[str, str] = {
    "PASSPORT_MRZ": r"(?<![A-Z0-9<])P<[A-Z]{3}[A-Z<]{39}\r?\n[A-Z0-9<]{44}(?![A-Z0-9<])",
    "PASSPORT": r"\b(?:passport|travel[\s_-]*document)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*(?:is\s*)?[:#=-]?\s*(?P<value>[A-Z0-9][A-Z0-9-]{5,13})\b",
    "AADHAAR_MASKED": r"\b(?:aadhaar|uidai)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>(?:X{4}[\s-]?){2}\d{4})\b",
    "AADHAAR_VID": r"\b(?:(?:aadhaar[\s_-]+)?virtual[\s_-]*id|aadhaar[\s_-]*vid)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b",
    "AADHAAR": r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
    "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "VOTER_ID_IN": r"\b(?:voter(?:[\s_-]*id)?|epic)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>[A-Z]{3}\d{7})\b",
    "NINO_UK": r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[ABCEGHJKLMNPRSTWXYZ]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
    "NHS_NUMBER": r"\b(?:nhs(?:[\s_-]*number)?)(?:[\s_-]*(?:no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>\d{3}[\s-]?\d{3}[\s-]?\d{4})\b",
    "SIN_CA": r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)",
    "CPF_BR": r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)",
    "CNPJ_BR": r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)",
    "TFN_AU": r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)",
    "MEDICARE_AU": r"(?<!\d)\d{4}[\s-]?\d{5}[\s-]?\d(?!\d)",
    "EIN_US": r"\b\d{2}-\d{7}\b",
    "BANK_ACCOUNT": r"\b(?:bank[\s_-]*)?account(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]\s*(?P<value>\d(?:[\s-]?\d){5,33})\b",
    "DRIVERS_LICENSE": r"\b(?:driver'?s?[\s_-]*licen[cs]e|driving[\s_-]*licen[cs]e)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>[A-Z0-9][A-Z0-9-]{4,19})\b",
    "NATIONAL_ID": r"\b(?:national[\s_-]*(?:id|identity)(?:[\s_-]*card)?|identity[\s_-]*card)(?:[\s_-]*(?:number|no\.?|#))?[\s_-]*[:#=-]?\s*(?P<value>[A-Z0-9][A-Z0-9-]{4,19})\b",
    "UPI_ID": r"\b[A-Z0-9][A-Z0-9._-]{1,255}@[A-Z][A-Z0-9.-]{1,63}\b",
    # SSN area numbers never start with 9 (that range is ITIN, below).
    "SSN": r"\b(?!9)\d{3}-\d{2}-\d{4}\b",
    "ITIN": r"\b9\d{2}-(?:7\d|8[0-8]|9[0-24-9])-\d{4}\b",
    # Luhn validation makes a broad 13-19 digit payment-card candidate safe.
    "CREDIT_CARD": r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
    "EMAIL": r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63}\b",
    "PHONE_IN": r"(?<!\w)(?:\+91[\s.-]?)?[6-9]\d{4}[\s.-]?\d{5}(?!\d)",
    "PHONE_INTERNATIONAL": r"(?<!\w)\+\d(?:[\s().-]*\d){7,14}(?!\d)",
    "PHONE_US": r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
    "IBAN": r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b",
    "IP_ADDRESS": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    # Full IPv6 form. Compressed forms are omitted to avoid matching times and
    # prose punctuation without bringing in a full IP parser dependency.
    "IPV6_ADDRESS": r"\b(?:[0-9A-F]{1,4}:){7}[0-9A-F]{1,4}\b",
    "MAC_ADDRESS": r"\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b",
    "DATE_OF_BIRTH": r"\b(?:\d{4}[/.-](?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])|(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:18|19|20)\d{2})\b",
}

_COMPILED: dict[str, Pattern[str]] = {
    entity: re.compile(pattern, re.IGNORECASE) for entity, pattern in _REGEX_PATTERNS.items()
}

_CONTEXT_PATTERNS: dict[str, Pattern[str]] = {
    "AADHAAR": re.compile(
        r"\b(?:aadhaar(?:[\s_-]*(?:number|no))?|uidai|unique[\s_-]+identification)\b",
        re.IGNORECASE,
    ),
    "PAN": re.compile(
        r"\b(?:pan(?:[\s_-]*(?:number|no))?|permanent[\s_-]+account[\s_-]+number)\b",
        re.IGNORECASE,
    ),
    "SIN_CA": re.compile(r"\b(?:sin|social[\s_-]+insurance[\s_-]+number)\b", re.IGNORECASE),
    "CPF_BR": re.compile(r"\b(?:cpf|cadastro\s+de\s+pessoas\s+f[íi]sicas)\b", re.IGNORECASE),
    "CNPJ_BR": re.compile(r"\b(?:cnpj|cadastro\s+nacional)\b", re.IGNORECASE),
    "TFN_AU": re.compile(r"\b(?:tfn|tax[\s_-]+file[\s_-]+number)\b", re.IGNORECASE),
    "MEDICARE_AU": re.compile(r"\bmedicare(?:[\s_-]+(?:card|number))?\b", re.IGNORECASE),
    "PHONE_IN": re.compile(
        r"\b(?:phone|mobile|telephone|tel|contact|whatsapp)(?:[\s_-]*(?:number|no))?\b",
        re.IGNORECASE,
    ),
    "PHONE_US": re.compile(
        r"\b(?:phone|mobile|telephone|tel|contact|cell)(?:[\s_-]*(?:number|no))?\b",
        re.IGNORECASE,
    ),
    "DATE_OF_BIRTH": re.compile(
        r"\b(?:d\.?o\.?b\.?|date[\s_-]+of[\s_-]+birth|birth[\s_-]*date|born[\s_-]+on|birthday)\b",
        re.IGNORECASE,
    ),
    "UPI_ID": re.compile(r"\b(?:upi|vpa|payment[\s_-]+address)\b", re.IGNORECASE),
}

_DOB_PREFIX = re.compile(
    r"(?:d\.?o\.?b\.?|date[\s_-]+of[\s_-]+birth|birth[\s_-]*date|born[\s_-]+on|birthday)"
    r"\s*(?:is\s*)?[:#=-]?\s*$",
    re.IGNORECASE,
)
_DOB_SUFFIX = re.compile(
    r"^\s*(?:\(|\[)?\s*(?:d\.?o\.?b\.?|date[\s_-]+of[\s_-]+birth|birth[\s_-]*date)\b",
    re.IGNORECASE,
)

_PII_SCHEMA_KEY = re.compile(
    r"""(?ix)(?:^|[_\s-])(?:
        passport|travel[_\s-]*document|aadhaar|uidai|pan|gstin|voter|epic|
        nhs|sin|cpf|cnpj|tfn|medicare|ein|bank[_\s-]*account|
        driver'?s?[_\s-]*licen[cs]e|national[_\s-]*(?:id|identity)|
        upi|vpa|ssn|itin|credit[_\s-]*card|email|phone|mobile|telephone|
        iban|ipv?6?|mac|date[_\s-]*of[_\s-]*birth|dob|birth[_\s-]*date
    )(?:$|[_\s-])"""
)

_UPI_PROVIDERS = {
    "airtel",
    "apl",
    "axl",
    "freecharge",
    "hdfcbank",
    "ibl",
    "icici",
    "jio",
    "kotak",
    "okaxis",
    "okhdfcbank",
    "okicici",
    "oksbi",
    "paytm",
    "pingpay",
    "sbi",
    "upi",
    "ybl",
}


def _digits(candidate: str) -> str:
    return "".join(char for char in candidate if char.isdigit())


def _luhn_checksum_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if not digits:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        number = int(char)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        checksum += number
    return checksum % 10 == 0


def _luhn_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    return 13 <= len(digits) <= 19 and _luhn_checksum_ok(digits)


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _aadhaar_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 12 or digits[0] in "01" or len(set(digits)) == 1:
        return False
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(char)]]
    return checksum == 0


def _cpf_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    numbers = [int(char) for char in digits]
    for length in (9, 10):
        total = sum(numbers[index] * (length + 1 - index) for index in range(length))
        check = (total * 10) % 11
        if check == 10:
            check = 0
        if numbers[length] != check:
            return False
    return True


def _cnpj_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    numbers = [int(char) for char in digits]
    for length, weights in (
        (12, (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)),
        (13, (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)),
    ):
        remainder = sum(numbers[index] * weights[index] for index in range(length)) % 11
        check = 0 if remainder < 2 else 11 - remainder
        if numbers[length] != check:
            return False
    return True


def _tfn_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 9:
        return False
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    return sum(int(char) * weight for char, weight in zip(digits, weights)) % 11 == 0


def _nhs_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 10:
        return False
    total = sum(int(digits[index]) * (10 - index) for index in range(9))
    check = 11 - total % 11
    if check == 11:
        check = 0
    return check != 10 and check == int(digits[-1])


def _medicare_au_ok(candidate: str) -> bool:
    digits = _digits(candidate)
    if len(digits) != 10:
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    check = sum(int(digits[index]) * weights[index] for index in range(8)) % 10
    return check == int(digits[8])


def _iban_ok(candidate: str) -> bool:
    compact = re.sub(r"\s+", "", candidate).upper()
    if not 15 <= len(compact) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    remainder = 0
    for char in numeric:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1


def _valid_birth_date(candidate: str) -> bool:
    parts = re.split(r"[/.-]", candidate)
    try:
        if len(parts[0]) == 4:
            candidates = [(int(parts[0]), int(parts[1]), int(parts[2]))]
        else:
            year = int(parts[2])
            candidates = [
                (year, int(parts[0]), int(parts[1])),
                (year, int(parts[1]), int(parts[0])),
            ]
        for year, month, day_number in candidates:
            if 1800 <= year <= date.today().year:
                date(year, month, day_number)
                return True
    except ValueError:
        pass
    return False


def _near_context(text: str, start: int, end: int, entity: str) -> bool:
    pattern = _CONTEXT_PATTERNS.get(entity)
    if pattern is None:
        return False
    if entity == "DATE_OF_BIRTH":
        prefix = text[max(0, start - 64) : start]
        suffix = text[end : min(len(text), end + 24)]
        return _DOB_PREFIX.search(prefix) is not None or _DOB_SUFFIX.search(suffix) is not None
    # Preceding labels are most common; a small suffix also covers forms such as
    # ``046-454-286 (SIN)``.
    window = text[max(0, start - 48) : min(len(text), end + 20)]
    return pattern.search(window) is not None


def _candidate_span(match: Match[str]) -> tuple[int, int]:
    group: str | int = "value" if "value" in match.re.groupindex else 0
    return match.span(group)


def _candidate_value(match: Match[str]) -> str:
    group: str | int = "value" if "value" in match.re.groupindex else 0
    return match.group(group)


def _candidate_allowed(entity: str, value: str, text: str, start: int, end: int) -> bool:
    if entity == "CREDIT_CARD":
        return _luhn_ok(value)
    if entity == "AADHAAR":
        return _aadhaar_ok(value) or _near_context(text, start, end, entity)
    if entity == "PAN":
        compact = value.upper()
        return compact[3] in "PCHFATBLJG" or _near_context(text, start, end, entity)
    if entity == "PASSPORT":
        compact = value.replace("-", "")
        return 6 <= len(compact) <= 12 and any(char.isdigit() for char in compact)
    if entity == "SIN_CA":
        return _luhn_checksum_ok(value) and _near_context(text, start, end, entity)
    if entity == "CPF_BR":
        return _cpf_ok(value) and ("." in value or _near_context(text, start, end, entity))
    if entity == "CNPJ_BR":
        return _cnpj_ok(value) and ("/" in value or _near_context(text, start, end, entity))
    if entity == "TFN_AU":
        return _tfn_ok(value) and _near_context(text, start, end, entity)
    if entity == "NHS_NUMBER":
        return _nhs_ok(value)
    if entity == "MEDICARE_AU":
        return _medicare_au_ok(value) and _near_context(text, start, end, entity)
    if entity == "IBAN":
        return _iban_ok(value)
    if entity == "DATE_OF_BIRTH":
        return _valid_birth_date(value) and _near_context(text, start, end, entity)
    if entity == "PHONE_IN":
        return value.lstrip().startswith("+91") or _near_context(text, start, end, entity)
    if entity == "PHONE_US":
        compact = value.lstrip()
        formatted = bool(re.search(r"[().-]", value)) or len(value.split()) >= 3
        return (
            compact.startswith(("+1", "1-", "1.", "1 "))
            or formatted
            or _near_context(text, start, end, entity)
        )
    if entity == "PHONE_INTERNATIONAL":
        return 8 <= len(_digits(value)) <= 15
    if entity == "UPI_ID":
        provider = value.rsplit("@", 1)[-1].lower()
        return provider in _UPI_PROVIDERS or _near_context(text, start, end, entity)
    return True


class PIIRedactor(BaseShield):
    """Detect and transform personally identifiable information.

    Regex rule IDs can be selected with ``entities`` (for example,
    ``["PASSPORT", "AADHAAR", "PAN", "EMAIL"]``). Successful scans set
    ``ctx.metadata["pii_detected"]`` and accumulate ``pii_rule_ids`` and
    ``pii_detection_count`` for audit and HumanGate use.

    ``tool_argument_policy`` is ``"off"`` by default because many legitimate
    booking, healthcare, and financial tools require PII. Applications can
    choose ``"block"`` for strict DLP or ``"redact"`` for structure-preserving
    sanitization through :meth:`Guard.scan_tool_arguments`.
    """

    needs_structured_context: bool = True

    @property
    def requires_buffered_output(self) -> bool:
        # Cards, addresses, dates, and other PII may contain whitespace. Token
        # resolution likewise must complete before output is released.
        return self.redact_output or self.mode == "tokenize"

    def select_structured_context_key(self, key_path: tuple[str, ...]) -> str | None:
        for key in reversed(key_path):
            normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            if _PII_SCHEMA_KEY.search(normalized):
                return key
        return super().select_structured_context_key(key_path)

    def __init__(
        self,
        entities: list[str] | None = None,
        mode: Literal["redact", "mask", "tokenize"] = "redact",
        language: str = "en",
        score_threshold: float = 0.6,
        engine: Literal["regex", "presidio"] = "regex",
        redact_output: bool = False,
        scan_tool_output: bool = False,
        max_tokenized_values: int = 256,
        clear_resolved_tokens: bool = True,
        tool_argument_policy: Literal["off", "block", "redact"] = "off",
    ) -> None:
        if mode not in ("redact", "mask", "tokenize"):
            raise ValueError("mode must be 'redact', 'mask', or 'tokenize'")
        if engine not in ("regex", "presidio"):
            raise ValueError("engine must be 'regex' or 'presidio'")
        if isinstance(max_tokenized_values, bool) or max_tokenized_values < 1:
            raise ValueError("max_tokenized_values must be a positive integer")
        if tool_argument_policy not in ("off", "block", "redact"):
            raise ValueError("tool_argument_policy must be 'off', 'block', or 'redact'")

        self.entities = entities  # None means all available rules.
        self.mode = mode
        self.language = language
        self.score_threshold = score_threshold
        self.engine = engine
        self.redact_output = redact_output
        self.scan_tool_output_flag = scan_tool_output
        self.max_tokenized_values = max_tokenized_values
        self.clear_resolved_tokens = clear_resolved_tokens
        self.tool_argument_policy = tool_argument_policy
        self._analyzer = None
        self._anonymizer = None

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record_metadata(ctx: SessionContext, rule_ids: list[str]) -> None:
        if not rule_ids:
            return
        combined = set(rule_ids)
        previous = ctx.metadata.get("pii_rule_ids", [])
        if isinstance(previous, (list, tuple, set)):
            combined.update(str(value) for value in previous)
        previous_count = ctx.metadata.get("pii_detection_count", 0)
        if not isinstance(previous_count, int):
            previous_count = 0
        ctx.metadata["pii_detected"] = True
        ctx.metadata["pii_rule_ids"] = sorted(combined)
        ctx.metadata["pii_detection_count"] = previous_count + len(rule_ids)

    # ------------------------------------------------------------------ #
    # Lazy Presidio init                                                  #
    # ------------------------------------------------------------------ #

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:
                raise ImportError(
                    "Presidio is not installed. Run: pip install agentguard[presidio] "
                    "&& python -m spacy download en_core_web_sm"
                ) from exc
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def _get_anonymizer(self):
        if self._anonymizer is None:
            from presidio_anonymizer import AnonymizerEngine

            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    # ------------------------------------------------------------------ #
    # Regex engine                                                        #
    # ------------------------------------------------------------------ #

    def _regex_find(self, text: str, context_hint: str | None = None) -> list[tuple[int, int, str]]:
        """Return validated, merged ``(start, end, rule_id)`` spans."""

        # A JSON property name supplies useful, tightly-scoped context for an
        # otherwise ambiguous scalar (``{"passport_number": "A1234567"}``).
        prefix = ""
        if context_hint:
            normalised_hint = re.sub(r"[_-]+", " ", context_hint)[:80]
            prefix = f"{normalised_hint}: "
        scan_text = prefix + text
        offset = len(prefix)

        targets = self.entities or list(_COMPILED.keys())
        hits: list[tuple[int, int, str]] = []
        for entity in targets:
            pattern = _COMPILED.get(entity)
            if pattern is None:
                continue
            for match in pattern.finditer(scan_text):
                start, end = _candidate_span(match)
                # Only redact spans wholly inside the scalar value, never the
                # synthetic context or a JSON property name.
                if start < offset or end <= start or end > len(scan_text):
                    continue
                value = _candidate_value(match)
                if not _candidate_allowed(entity, value, scan_text, start, end):
                    continue
                hits.append((start - offset, end - offset, entity))
        return merge_spans(hits)

    def _store_tokenized_value(self, ctx: SessionContext, token: str, original: str) -> None:
        pii_tokens = [key for key in ctx._token_map if key.startswith("[AGENTGUARD_")]
        evicted = 0
        while len(pii_tokens) >= self.max_tokenized_values:
            oldest = pii_tokens.pop(0)
            ctx._token_map.pop(oldest, None)
            evicted += 1
        if evicted:
            previous = ctx.metadata.get("pii_token_evictions", 0)
            if not isinstance(previous, int):
                previous = 0
            ctx.metadata["pii_token_evictions"] = previous + evicted
        ctx.store_token(token, original)

    def clear_tokenized_values(self, ctx: SessionContext) -> int:
        """Remove raw PII retained for token resolution at session teardown.

        Returns the number of values removed. Resolved values are cleared
        automatically by default; this method covers unused/in-flight tokens.
        """

        keys = [key for key in ctx._token_map if key.startswith("[AGENTGUARD_")]
        for key in keys:
            ctx._token_map.pop(key, None)
        return len(keys)

    def _apply_regex_redaction(
        self, text: str, ctx: SessionContext, context_hint: str | None = None
    ) -> str | None:
        hits = self._regex_find(text, context_hint=context_hint)
        if not hits:
            return None

        self._record_metadata(ctx, [entity for _, _, entity in hits])
        result = text
        for start, end, entity in hits:
            original = result[start:end]
            if self.mode == "tokenize":
                token = f"[AGENTGUARD_{entity}_{uuid.uuid4().hex.upper()}]"
                self._store_tokenized_value(ctx, token, original)
                replacement = token
            elif self.mode == "mask":
                replacement = "*" * len(original)
            else:
                replacement = f"[REDACTED_{entity}]"
            result = result[:start] + replacement + result[end:]

        return result if result != text else None

    # ------------------------------------------------------------------ #
    # Presidio engine                                                     #
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

        self._record_metadata(ctx, [result.entity_type for result in results])
        if self.mode == "tokenize":
            modified = text
            for result in sorted(results, key=lambda item: item.start, reverse=True):
                original = text[result.start : result.end]
                token = f"[AGENTGUARD_{result.entity_type}_{uuid.uuid4().hex.upper()}]"
                self._store_tokenized_value(ctx, token, original)
                modified = modified[: result.start] + token + modified[result.end :]
            return modified

        if self.mode == "mask":
            operators = {
                result.entity_type: OperatorConfig(
                    "mask",
                    {
                        "chars_to_mask": len(text[result.start : result.end]),
                        "masking_char": "*",
                        "from_end": False,
                    },
                )
                for result in results
            }
        else:
            operators = {
                result.entity_type: OperatorConfig(
                    "replace", {"new_value": f"[REDACTED_{result.entity_type}]"}
                )
                for result in results
            }

        anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        return anonymized.text if anonymized.text != text else None

    # ------------------------------------------------------------------ #
    # Shield hooks                                                        #
    # ------------------------------------------------------------------ #

    def _redact_scalar(
        self, text: str, ctx: SessionContext, context_hint: str | None = None
    ) -> str | None:
        if self.engine == "presidio":
            return self._apply_presidio_redaction(text, ctx)
        return self._apply_regex_redaction(text, ctx, context_hint=context_hint)

    def _redact_json_value(
        self,
        value: Any,
        ctx: SessionContext,
        context_hint: str | None = None,
    ) -> tuple[Any, bool]:
        if isinstance(value, dict):
            modified = False
            result: dict[str, Any] = {}
            for key, child in value.items():
                transformed, child_modified = self._redact_json_value(
                    child, ctx, context_hint=str(key)
                )
                result[key] = transformed
                modified = modified or child_modified
            return result, modified
        if isinstance(value, list):
            modified = False
            result_list: list[Any] = []
            for child in value:
                transformed, child_modified = self._redact_json_value(
                    child, ctx, context_hint=context_hint
                )
                result_list.append(transformed)
                modified = modified or child_modified
            return result_list, modified
        if isinstance(value, str):
            transformed = self._redact_scalar(value, ctx, context_hint=context_hint)
            return (transformed, True) if transformed is not None else (value, False)
        # bool is an int subclass, so preserve it before handling JSON numbers.
        if value is None or isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            original = str(value)
            transformed = self._redact_scalar(original, ctx, context_hint=context_hint)
            # Sensitive numeric values become strings containing a redaction,
            # mask, or reversible token. Non-sensitive numbers retain type.
            return (transformed, True) if transformed is not None else (value, False)
        return value, False

    def _try_redact_json(self, text: str, ctx: SessionContext) -> tuple[bool, str | None]:
        stripped = text.strip()
        if not stripped:
            return False, None
        if stripped[0] not in '{["-0123456789' and stripped not in {
            "true",
            "false",
            "null",
        }:
            return False, None

        def reject_nonstandard_number(value: str) -> None:
            raise ValueError(f"Non-standard JSON number: {value}")

        try:
            parsed = json.loads(stripped, parse_constant=reject_nonstandard_number)
        except RecursionError as exc:
            # Never fall back to regex-splicing a valid-but-overdeep structured
            # payload: that could corrupt JSON or leave a deeply nested value.
            raise ValueError("JSON nesting exceeds the safe parser depth") from exc
        except (json.JSONDecodeError, ValueError):
            return False, None

        transformed, modified = self._redact_json_value(parsed, ctx)
        if not modified:
            return True, None
        return True, json.dumps(
            transformed, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        modified = self._redact(text, ctx)
        return ShieldResult(allowed=True, modified_input=modified)

    def _redact(self, text: str, ctx: SessionContext) -> str | None:
        was_json, modified = self._try_redact_json(text, ctx)
        if was_json:
            return modified
        return self._redact_scalar(text, ctx)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        # Tokenization re-inserts a user's original PII so the response remains
        # coherent. This intentionally takes precedence over output redaction.
        if self.mode == "tokenize" and ctx._token_map:
            resolved_tokens = [
                token
                for token in ctx._token_map
                if token.startswith("[AGENTGUARD_") and token in text
            ]
            resolved = ctx.resolve_all_tokens(text)
            if self.clear_resolved_tokens and resolved != text:
                for token in resolved_tokens:
                    ctx._token_map.pop(token, None)
            return ShieldResult(
                allowed=True,
                modified_input=resolved if resolved != text else None,
            )
        if self.redact_output and self.mode in ("redact", "mask"):
            return ShieldResult(allowed=True, modified_input=self._redact(text, ctx))
        return ShieldResult(allowed=True)

    async def scan_tool_arguments(
        self, tool_name: str, text: str, ctx: SessionContext
    ) -> ShieldResult:
        if self.tool_argument_policy == "off":
            return ShieldResult(allowed=True)

        if self.engine == "presidio":
            analyzer_results = self._get_analyzer().analyze(
                text=text,
                language=self.language,
                entities=self.entities,
                score_threshold=self.score_threshold,
            )
            hits = merge_spans(
                [(result.start, result.end, result.entity_type) for result in analyzer_results]
            )
        else:
            hits = self._regex_find(text)
        if not hits:
            return ShieldResult(allowed=True)

        self._record_metadata(ctx, [entity for _, _, entity in hits])
        ctx.metadata["pii_tool_argument_detected"] = True
        if self.tool_argument_policy == "block":
            kinds = ", ".join(sorted({entity for _, _, entity in hits}))
            return ShieldResult(
                allowed=False,
                reason=f"PII detected in tool arguments: {kinds}",
                reason_code="PII_IN_TOOL_ARGUMENTS",
            )

        result = text
        for start, end, entity in hits:
            result = result[:start] + f"[REDACTED_{entity}]" + result[end:]
        return ShieldResult(allowed=True, modified_input=result if result != text else None)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        if not self.scan_tool_output_flag or self.mode == "tokenize":
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=True, modified_input=self._redact(output, ctx))
