import uuid
from dataclasses import dataclass, field
from typing import Any

# Per-call provenance for an inbound peer message. ``Guard.scan_agent_message``
# writes these for the duration of one scan and restores whatever was there
# before, so a shield can read the envelope without every hook signature
# growing a parameter. Kept here because both the guard and the shields already
# depend on this module; putting them in either would create an import cycle.
AGENT_SENDER_KEY = "agentguard.agent_sender"
AGENT_ENVELOPE_KEY = "agentguard.agent_envelope"


@dataclass
class SessionContext:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    cost_usd: float = 0.0
    request_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    _token_map: dict[str, str] = field(default_factory=dict, repr=False)

    def store_token(self, token: str, original: str) -> None:
        self._token_map[token] = original

    def resolve_token(self, token: str) -> str | None:
        return self._token_map.get(token)

    def resolve_all_tokens(self, text: str) -> str:
        for token, original in self._token_map.items():
            text = text.replace(token, original)
        return text
