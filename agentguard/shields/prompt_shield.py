"""
PromptShield — multi-tier prompt injection detector.

Tier 1 (always active): rule-based regex patterns + encoding preprocessing.
Tier 2 (opt-in, use_ml=True): DistilBERT classifier from HuggingFace Hub.
Tier 3 (opt-in, use_canary=True): canary-token detection in outputs.
"""
import base64
import re
import urllib.parse
import uuid
from typing import List, Literal, Optional, Tuple

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

# ---------------------------------------------------------------------------
# Rule patterns — community-maintainable list
# ---------------------------------------------------------------------------
_RAW_PATTERNS: List[str] = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above|former)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above|your)\s+instructions?",
    r"override\s+(all\s+)?(previous|prior)?\s*instructions?",
    r"do\s+not\s+follow\s+(your\s+)?(previous\s+)?instructions?",
    r"new\s+task\s*:\s*(ignore|disregard|forget)",
    # System prompt extraction
    r"(reveal|show|print|output|tell\s+me|repeat)\s+(your\s+)?(system\s+prompt|instructions?|directives?)",
    r"what\s+(are|were)\s+your\s+(original\s+)?(instructions?|prompt|directives?)",
    r"output\s+(everything|all)\s+(above|before|previously)",
    r"print\s+your\s+(full\s+)?(initial\s+)?prompt",
    # Persona hijacking
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+\w+\s+(with\s+no|without)",
    r"pretend\s+you\s+(are|have\s+no)\s+(restrictions?|guidelines?)?",
    r"from\s+now\s+on\s+(you\s+are|act\s+as|pretend|behave)",
    r"roleplay\s+as\s+(a|an)\s+\w+\s+(with\s+no|without)",
    # Jailbreak keywords
    r"\bDAN\b",
    r"jailbreak",
    r"developer\s+mode",
    r"god\s+mode",
    r"no\s+restrictions?",
    r"bypass\s+(your\s+)?(safety|restrictions?|guidelines?|filters?)",
    r"without\s+(any\s+)?(restrictions?|filters?|guidelines?|safety)",
    r"unrestricted\s+(mode|access|AI)",
    # Delimiter/token injection
    r"----+\s*system\s*----+",
    r"\[system\]\s*:",
    r"</?system>",
    r"###\s*system",
    r"<<SYS>>",
    r"</?(human|assistant|user|instruction)>",
    r"\[INST\]",
    r"\[/INST\]",
    # Indirect re-instruction
    r"(new|updated|additional|secret)\s+instructions?\s*:",
    r"(modified|replacement)\s+(system\s+)?prompt\s*:",
    # Data exfiltration
    r"send\s+(all\s+)?(the\s+)?(conversation|context|data|info)\s+to",
    r"forward\s+(this|all|the)\s+(conversation\s+)?to",
    r"exfiltrate",
    r"leak\s+(the\s+)?(prompt|instructions?|data)",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _RAW_PATTERNS]


def _preprocess(text: str) -> str:
    """Decode common obfuscation encodings and append to original text."""
    extra: List[str] = []

    # Base64
    try:
        decoded = base64.b64decode(text.strip() + "==", validate=False).decode(
            "utf-8", errors="ignore"
        )
        if len(decoded) > 10 and decoded.isprintable():
            extra.append(decoded)
    except Exception:
        pass

    # URL encoding
    try:
        url = urllib.parse.unquote(text)
        if url != text:
            extra.append(url)
    except Exception:
        pass

    return text + (" " + " ".join(extra) if extra else "")


class PromptShield(BaseShield):
    def __init__(
        self,
        mode: Literal["fast", "strict", "paranoid"] = "strict",
        sensitivity: float = 0.85,
        use_ml: bool = False,
        use_canary: bool = True,
        custom_patterns: Optional[List[str]] = None,
    ) -> None:
        self.mode = mode
        self.sensitivity = sensitivity
        self.use_ml = use_ml
        self.use_canary = use_canary
        self._patterns = list(_COMPILED)
        if custom_patterns:
            self._patterns.extend(
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in custom_patterns
            )
        self._classifier = None

    # ------------------------------------------------------------------ #
    # Rule scanning                                                        #
    # ------------------------------------------------------------------ #

    def _rule_scan(self, text: str) -> Tuple[bool, str]:
        preprocessed = _preprocess(text)
        for pattern in self._patterns:
            m = pattern.search(preprocessed)
            if m:
                return True, f"Matched pattern: '{pattern.pattern[:60]}'"
        return False, ""

    # ------------------------------------------------------------------ #
    # ML scanning                                                          #
    # ------------------------------------------------------------------ #

    def _ml_scan(self, text: str) -> Tuple[bool, float]:
        if self._classifier is None:
            from agentguard.models.loader import load_injection_classifier
            self._classifier = load_injection_classifier()
            if self._classifier is None:
                return False, 0.0

        try:
            output = self._classifier(text, truncation=True, max_length=512)
            label: str = output[0]["label"]
            score: float = output[0]["score"]
            return label == "INJECTION" and score >= self.sensitivity, score
        except Exception:
            return False, 0.0

    # ------------------------------------------------------------------ #
    # Canary helpers (called by adapter/user code to inject into prompts)  #
    # ------------------------------------------------------------------ #

    def inject_canary(self, system_prompt: str, ctx: SessionContext) -> str:
        """Embed an invisible canary token in the system prompt."""
        canary = f"AGENTGUARD-CANARY-{uuid.uuid4().hex[:16].upper()}"
        ctx.metadata["canary_token"] = canary
        return system_prompt + f"\n\n<!-- {canary} -->"

    # ------------------------------------------------------------------ #
    # Shield hooks                                                         #
    # ------------------------------------------------------------------ #

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        rule_hit, rule_reason = self._rule_scan(text)

        # Rules fire in all modes — "fast" just skips the ML tier
        if rule_hit:
            return ShieldResult(
                allowed=False,
                reason=f"Prompt injection detected (rules). {rule_reason}",
                reason_code="PROMPT_INJECTION_DETECTED",
            )

        # ML tier: only in strict/paranoid, and only when use_ml=True
        if self.use_ml and self.mode in ("strict", "paranoid"):
            ml_hit, score = self._ml_scan(text)
            if ml_hit:
                return ShieldResult(
                    allowed=False,
                    reason=f"Prompt injection detected (ML, confidence: {score:.2f})",
                    reason_code="PROMPT_INJECTION_DETECTED_ML",
                )

        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        if self.use_canary:
            canary = ctx.metadata.get("canary_token")
            if canary and canary in text:
                return ShieldResult(
                    allowed=False,
                    reason="Canary token found in output — system prompt extraction attempt detected",
                    reason_code="CANARY_TRIGGERED",
                )
        return ShieldResult(allowed=True)
