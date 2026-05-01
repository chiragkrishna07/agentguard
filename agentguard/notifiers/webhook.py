import hashlib
import hmac
import json
from typing import Optional

from agentguard.notifiers.base import BaseNotifier


class WebhookNotifier(BaseNotifier):
    """Posts a signed JSON payload to any HTTP endpoint.

    If a secret is provided, the request includes an
    ``X-AgentGuard-Signature: sha256=<hex>`` header for verification.
    """

    def __init__(self, url: str, secret: Optional[str] = None) -> None:
        self.url = url
        self.secret = secret

    async def notify(self, gate_id: str, context: dict) -> None:
        import httpx

        payload = json.dumps({"gate_id": gate_id, **context}, default=str).encode()
        headers = {"Content-Type": "application/json"}

        if self.secret:
            sig = hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()
            headers["X-AgentGuard-Signature"] = f"sha256={sig}"

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.url, content=payload, headers=headers)
            response.raise_for_status()
