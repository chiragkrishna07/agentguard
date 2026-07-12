"""Detect and neutralise credentials crossing an agent boundary.

The built-in rules have two tiers:

* high-signal formats with provider-specific prefixes or structure; and
* contextual generic credentials such as ``api_key=...``, bearer tokens, and
  authenticated database URLs.

The contextual tier is enabled by default and can be disabled with
``detect_generic_credentials=False`` when an application routinely processes
configuration examples. Regex scanning is intentionally complementary to a
dedicated repository scanner such as gitleaks or TruffleHog.
"""

import re
import unicodedata
from re import Match, Pattern
from typing import Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext
from agentguard.shields._spans import merge_spans

# ---------------------------------------------------------------------------
# High-signal credential patterns. Ordered most-specific first.
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: dict[str, str] = {
    # A complete block ends at its matching footer. An unterminated block is
    # sensitive through end-of-content: redacting only its header would expose
    # the actual private-key body.
    "PRIVATE_KEY": r"(?P<secret>-----BEGIN (?P<pem_type>(?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?)-----(?:[\s\S]*?-----END (?P=pem_type)-----|[\s\S]*\Z))",
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b",
    "GITHUB_PAT": r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b",
    "ANTHROPIC_KEY": r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
    # OpenAI keys start with sk- but not sk-ant- (Anthropic, matched above).
    "OPENAI_KEY": r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}\b",
    "GROQ_KEY": r"\bgsk_[A-Za-z0-9_-]{20,}\b",
    "XAI_KEY": r"\bxai-[A-Za-z0-9_-]{20,}\b",
    "PERPLEXITY_KEY": r"\bpplx-[A-Za-z0-9_-]{20,}\b",
    "REPLICATE_TOKEN": r"\br8_[A-Za-z0-9]{20,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "GOOGLE_OAUTH_SECRET": r"\bGOCSPX-[A-Za-z0-9_-]{20,}\b",
    "SLACK_TOKEN": r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b",
    "SLACK_APP_TOKEN": r"\bxapp-[0-9A-Za-z-]{20,}\b",
    "SLACK_WEBHOOK": r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
    "STRIPE_KEY": r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{16,}\b",
    "STRIPE_WEBHOOK_SECRET": r"\bwhsec_[A-Za-z0-9]{16,}\b",
    "SENDGRID_KEY": r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",
    "HUGGINGFACE_TOKEN": r"\bhf_[A-Za-z0-9]{20,}\b",
    "GITLAB_PAT": r"\bglpat-[A-Za-z0-9_-]{20,}\b",
    "NPM_TOKEN": r"\bnpm_[A-Za-z0-9]{36}\b",
    "PYPI_TOKEN": r"\bpypi-AgEI[A-Za-z0-9_-]{20,}\b",
    "DOCKER_PAT": r"\bdckr_pat_[A-Za-z0-9_-]{20,}\b",
    "TWILIO_KEY": r"\b(?:AC|SK)[a-f0-9]{32}\b",
    "DIGITALOCEAN_TOKEN": r"\bdo[oprv]_v1_[A-Fa-f0-9]{32,128}\b",
    "DATABRICKS_TOKEN": r"\bdapi[a-f0-9]{32,}\b",
    "NEW_RELIC_KEY": r"\b(?:NRAK|NRII)-[A-Za-z0-9_-]{20,}\b",
    "SHOPIFY_TOKEN": r"\bshp(?:at|ca|pa|ss)_[a-fA-F0-9]{32}\b",
    "LINEAR_KEY": r"\blin_api_[A-Za-z0-9]{20,}\b",
    "POSTMAN_KEY": r"\bPMAK-[A-Za-z0-9_-]{20,}\b",
    "MAILGUN_KEY": r"\bkey-[a-f0-9]{32}\b",
    "SQUARE_TOKEN": r"\bsq0(?:atp|csp)-[A-Za-z0-9_-]{20,}\b",
    "TELEGRAM_BOT_TOKEN": r"\b\d{8,12}:[A-Za-z0-9_-]{30,50}\b",
    "DISCORD_TOKEN": r"\b(?:mfa\.[A-Za-z0-9_-]{20,}|[A-Za-z0-9_-]{20,30}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{20,40})\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
}

# Generic rules are contextual to avoid classifying arbitrary high-entropy IDs
# as credentials. Named ``secret`` groups let us preserve the useful label and
# redact only the credential value.
_GENERIC_SECRET_PATTERNS: dict[str, str] = {
    "DATABASE_URL": r"(?i)(?P<secret>\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis(?:s)?|rediss|amqp(?:s)?|mssql|oracle|cassandra|neo4j(?:\+s|\+ssc)?|snowflake)://[^\s/'\"<>:?#]+:[^\s/'\"<>@?#]+@[^\s'\"<>]+)",
    "BEARER_TOKEN": r"(?i)\bBearer[ \t]+(?P<secret>[A-Za-z0-9][A-Za-z0-9._~+/=-]{7,})",
    "BASIC_AUTH_CREDENTIAL": r"(?i)\bAuthorization\b\s*(?:=|:)\s*Basic[ \t]+(?P<secret>[A-Za-z0-9+/]{8,}={0,2})",
    "AWS_SECRET_ACCESS_KEY": r"(?ix)\bAWS[_-]?SECRET[_-]?ACCESS[_-]?KEY\b[\"']?\s*(?:=|:)\s*[\"']?(?P<secret>[A-Za-z0-9/+=]{20,})",
    "GENERIC_CREDENTIAL": r"""(?ix)
        [\"']?\b(?:
            api[_-]?(?:key|token)|access[_-]?token|auth(?:orization)?[_-]?token|
            refresh[_-]?token|session[_-]?token|client[_-]?secret|
            consumer[_-]?secret|private[_-]?token|secret[_-]?(?:access[_-]?)?key|
            account[_-]?key|database[_-]?(?:password|pass)|db[_-]?(?:password|pass)|
            password|passwd|pwd|webhook[_-]?secret|signing[_-]?secret
        )\b[\"']?\s*(?:=|:)\s*
        (?:
            "(?P<double_secret>[^"\r\n]{8,})"|
            '(?P<single_secret>[^'\r\n]{8,})'|
            (?P<secret>[^\s\"'`,;]{8,})
        )
    """,
}

_COMPILED: dict[str, Pattern[str]] = {
    name: re.compile(pattern) for name, pattern in _SECRET_PATTERNS.items()
}

_PLACEHOLDER_PATTERNS = (
    re.compile(r"^<[^<>]+>$"),
    re.compile(r"^\$\{[A-Z_][A-Z0-9_]*\}$", re.IGNORECASE),
    re.compile(r"^\{\{[^{}]+\}\}$"),
    re.compile(
        r"^(?:your|insert|replace|enter|example|sample|dummy|fake|redacted|changeme)"
        r"[-_ ]*(?:api[-_ ]?)?(?:key|token|secret|password|credential)?(?:[-_ ]*here)?$",
        re.IGNORECASE,
    ),
)

_CREDENTIAL_SCHEMA_KEY = re.compile(
    r"""(?ix)^(?:
        api[_-]?(?:key|token)|access[_-]?token|auth(?:orization)?[_-]?token|
        refresh[_-]?token|session[_-]?token|client[_-]?secret|
        consumer[_-]?secret|private[_-]?token|secret[_-]?(?:access[_-]?)?key|
        account[_-]?key|database[_-]?(?:password|pass)|db[_-]?(?:password|pass)|
        password|passwd|pwd|webhook[_-]?secret|signing[_-]?secret|
        aws[_-]?secret[_-]?access[_-]?key
    )$"""
)


def _normalised_view(text: str) -> tuple[str, list[int] | None]:
    """Return a compatibility-normalised scan view and source-index map.

    Unicode format controls (including zero-width joiners and word joiners) are
    dropped because they have no place in an ASCII credential but are commonly
    inserted to evade regexes. NFKC also catches full-width ASCII. Redaction is
    always applied to ``text`` through the returned map, never to this view.
    """

    if text.isascii():
        return text, None

    normalised: list[str] = []
    source_indices: list[int] = []
    for index, char in enumerate(text):
        codepoint = ord(char)
        if (
            unicodedata.category(char) == "Cf"
            or char == "\u034f"  # combining grapheme joiner
            or 0xFE00 <= codepoint <= 0xFE0F  # variation selectors
            or 0xE0100 <= codepoint <= 0xE01EF  # supplemental selectors
        ):
            continue
        replacement = unicodedata.normalize("NFKC", char)
        for normalised_char in replacement:
            normalised.append(normalised_char)
            source_indices.append(index)
    return "".join(normalised), source_indices


def _secret_group(match: Match[str]) -> str | int:
    for name in ("secret", "double_secret", "single_secret"):
        if name in match.re.groupindex and match.start(name) >= 0:
            return name
    return 0


def _match_span(match: Match[str], source_indices: list[int] | None) -> tuple[int, int] | None:
    """Map a match in the normalised scan view back to the source string."""

    group = _secret_group(match)
    start, end = match.span(group)
    if start < 0 or end <= start:
        return None
    if source_indices is None:
        return start, end
    if end > len(source_indices):
        return None
    return source_indices[start], source_indices[end - 1] + 1


def _is_placeholder(value: str) -> bool:
    stripped = value.strip().strip("'\"")
    if stripped.casefold() in {"authentication", "authorization", "credentials"}:
        return True
    return any(pattern.fullmatch(stripped) for pattern in _PLACEHOLDER_PATTERNS)


class SecretsShield(BaseShield):
    """Detect and redact/block credentials in input, output, and tool output.

    Parameters
    ----------
    on_detect:
        ``"redact"`` (default) replaces each secret with
        ``[REDACTED_<RULE_ID>]``. ``"mask"`` preserves source length and
        ``"block"`` rejects the flow.
    scan_directions:
        Any of ``"input"``, ``"output"``, ``"tool_call"``, and
        ``"tool_output"``. Omit ``"tool_call"`` to disable tool-argument DLP.
    custom_patterns:
        Additional ``{rule_id: regex}`` patterns. A pattern with a named
        ``secret`` group redacts that group; otherwise its whole match is used.
    detect_generic_credentials:
        Enable the low-false-positive contextual tier for bearer/basic auth,
        authenticated database URLs, and labelled credential assignments.
    generic_min_length:
        Minimum value length accepted by contextual generic rules. Provider
        formats retain their own fixed requirements.
    tool_argument_policy:
        ``"block"`` (default) fails closed before a secret reaches a tool.
        ``"redact"`` or ``"mask"`` returns a sanitized argument aggregate to
        Guard for structure-preserving propagation. ``"off"`` disables the
        tool-argument DLP hook.
    """

    needs_structured_context: bool = True

    @property
    def requires_buffered_output(self) -> bool:
        # PEM blocks and labelled/database credentials can span whitespace or
        # lines; incremental emission cannot retract an already released prefix.
        return "output" in self.scan_directions

    def select_structured_context_key(self, key_path: tuple[str, ...]) -> str | None:
        for key in reversed(key_path):
            if _CREDENTIAL_SCHEMA_KEY.fullmatch(key):
                return key
        return super().select_structured_context_key(key_path)

    def __init__(
        self,
        on_detect: Literal["redact", "mask", "block"] = "redact",
        scan_directions: tuple[str, ...] = (
            "input",
            "output",
            "tool_call",
            "tool_output",
        ),
        custom_patterns: dict[str, str] | None = None,
        detect_generic_credentials: bool = True,
        generic_min_length: int = 12,
        tool_argument_policy: Literal["block", "redact", "mask", "off"] = "block",
    ) -> None:
        if on_detect not in ("redact", "mask", "block"):
            raise ValueError("on_detect must be 'redact', 'mask', or 'block'")
        invalid_directions = set(scan_directions) - {
            "input",
            "output",
            "tool_call",
            "tool_output",
        }
        if invalid_directions:
            raise ValueError(f"Unknown scan direction(s): {sorted(invalid_directions)}")
        if generic_min_length < 8:
            raise ValueError("generic_min_length must be at least 8")
        if tool_argument_policy not in ("block", "redact", "mask", "off"):
            raise ValueError("tool_argument_policy must be 'block', 'redact', 'mask', or 'off'")

        self.on_detect = on_detect
        self.scan_directions = scan_directions
        self.detect_generic_credentials = detect_generic_credentials
        self.generic_min_length = generic_min_length
        self.tool_argument_policy = tool_argument_policy
        self._generic_rule_ids = set(_GENERIC_SECRET_PATTERNS)
        self._patterns: dict[str, Pattern[str]] = dict(_COMPILED)
        if detect_generic_credentials:
            self._patterns.update(
                {name: re.compile(pattern) for name, pattern in _GENERIC_SECRET_PATTERNS.items()}
            )
        if custom_patterns:
            self._patterns.update(
                {name: re.compile(pattern) for name, pattern in custom_patterns.items()}
            )

    # ------------------------------------------------------------------ #
    # Detection                                                           #
    # ------------------------------------------------------------------ #

    def _find(self, text: str, context_hint: str | None = None) -> list[tuple[int, int, str]]:
        prefix = ""
        if context_hint:
            normalised_hint = re.sub(r"[_-]+", "_", context_hint)[:80]
            prefix = f"{normalised_hint}: "
        source_text = prefix + text
        source_offset = len(prefix)
        scan_text, source_indices = _normalised_view(source_text)
        if not scan_text:
            return []

        hits: list[tuple[int, int, str]] = []
        for name, pattern in self._patterns.items():
            for match in pattern.finditer(scan_text):
                span = _match_span(match, source_indices)
                if span is None:
                    continue
                start, end = span
                if start < source_offset:
                    continue
                if name in self._generic_rule_ids:
                    value = match.group(_secret_group(match))
                    if len(value) < self.generic_min_length or _is_placeholder(value):
                        continue
                hits.append((start - source_offset, end - source_offset, name))

        # Union overlapping rules so no covered character survives. Wider,
        # provider-specific matches win the replacement rule ID.
        return merge_spans(hits)

    @staticmethod
    def _record_metadata(ctx: SessionContext, hits: list[tuple[int, int, str]]) -> None:
        if not hits:
            return
        rule_ids = {name for _, _, name in hits}
        previous = ctx.metadata.get("secret_rule_ids", [])
        if isinstance(previous, (list, tuple, set)):
            rule_ids.update(str(value) for value in previous)
        previous_count = ctx.metadata.get("secret_detection_count", 0)
        if not isinstance(previous_count, int):
            previous_count = 0
        ctx.metadata["secret_detected"] = True
        ctx.metadata["secret_rule_ids"] = sorted(rule_ids)
        ctx.metadata["secret_detection_count"] = previous_count + len(hits)

    def _scan(self, text: str, ctx: SessionContext) -> ShieldResult:
        hits = self._find(text)
        if not hits:
            return ShieldResult(allowed=True)

        self._record_metadata(ctx, hits)
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
            replacement = "*" * len(original) if self.on_detect == "mask" else f"[REDACTED_{name}]"
            result = result[:start] + replacement + result[end:]

        return ShieldResult(allowed=True, modified_input=result if result != text else None)

    # ------------------------------------------------------------------ #
    # Shield hooks                                                        #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        if "input" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(text, ctx)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        if "output" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(text, ctx)

    async def scan_tool_arguments(
        self, tool_name: str, text: str, ctx: SessionContext
    ) -> ShieldResult:
        if "tool_call" not in self.scan_directions or self.tool_argument_policy == "off":
            return ShieldResult(allowed=True)

        hits = self._find(text)
        if not hits:
            return ShieldResult(allowed=True)

        self._record_metadata(ctx, hits)
        ctx.metadata["secret_tool_argument_detected"] = True
        kinds = ", ".join(sorted({name for _, _, name in hits}))
        if self.tool_argument_policy == "block":
            return ShieldResult(
                allowed=False,
                reason=f"Secret(s) detected in tool arguments: {kinds}",
                reason_code="SECRET_IN_TOOL_ARGUMENTS",
            )

        result = text
        for start, end, name in hits:
            original = result[start:end]
            replacement = (
                "*" * len(original) if self.tool_argument_policy == "mask" else f"[REDACTED_{name}]"
            )
            result = result[:start] + replacement + result[end:]
        return ShieldResult(allowed=True, modified_input=result if result != text else None)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        if "tool_output" not in self.scan_directions:
            return ShieldResult(allowed=True)
        return self._scan(output, ctx)
