import uuid
from dataclasses import dataclass, field
from typing import Any


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
