import hashlib
import hmac
import json
import math
import time
from numbers import Real
from typing import Literal

from agentguard.notifiers._endpoint import validate_notifier_url
from agentguard.notifiers.base import BaseNotifier


class WebhookNotifier(BaseNotifier):
    """Posts a signed JSON payload to any HTTP endpoint.

    If a secret is provided, the request includes a timestamp and an HMAC over
    ``<timestamp>.<raw-body>``. Receivers should reject stale timestamps before
    comparing the signature with a constant-time function; this prevents replay
    as well as payload tampering.
    """

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        *,
        timeout_seconds: float = 10.0,
        max_payload_bytes: int = 65_536,
        allow_insecure_http: bool = False,
        signature_version: Literal["v1", "legacy"] = "v1",
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be > 0")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes < 1
        ):
            raise ValueError("max_payload_bytes must be >= 1")
        if secret is not None and not isinstance(secret, str):
            raise ValueError("webhook signing secret must be a string or None")
        if secret is not None and len(secret.encode("utf-8")) < 16:
            raise ValueError("webhook signing secret must be at least 16 bytes")
        if signature_version not in ("v1", "legacy"):
            raise ValueError("signature_version must be 'v1' or 'legacy'")
        self.url: str = validate_notifier_url(url, allow_insecure_http=allow_insecure_http)
        self.secret: str | None = secret
        self.timeout_seconds: float = timeout_seconds
        self.max_payload_bytes: int = max_payload_bytes
        self.signature_version: Literal["v1", "legacy"] = signature_version

    async def notify(self, gate_id: str, context: dict) -> None:
        import httpx

        serialization_failed = False
        try:
            payload = json.dumps(
                {"gate_id": gate_id, **context}, default=str, separators=(",", ":")
            ).encode()
        except (TypeError, ValueError, RecursionError):
            serialization_failed = True
            payload = b""
        if serialization_failed:
            # Raise outside the except block so the sanitized error has no raw
            # exception context for loggers to traverse.
            raise RuntimeError("approval webhook payload serialization failed")
        if len(payload) > self.max_payload_bytes:
            raise RuntimeError("approval webhook payload exceeds configured size limit")
        headers = {"Content-Type": "application/json"}

        if self.secret:
            if self.signature_version == "legacy":
                signed_payload = payload
                prefix = "sha256"
            else:
                timestamp = str(int(time.time()))
                headers["X-AgentGuard-Timestamp"] = timestamp
                signed_payload = timestamp.encode("ascii") + b"." + payload
                prefix = "v1"
            sig = hmac.new(self.secret.encode(), signed_payload, hashlib.sha256).hexdigest()
            headers["X-AgentGuard-Signature"] = f"{prefix}={sig}"

        delivery_failed = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.url, content=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError:
            # Webhook paths frequently contain credentials. Never allow an HTTP
            # client's URL-bearing exception string to flow into guard logs.
            delivery_failed = True
        if delivery_failed:
            raise RuntimeError("approval webhook delivery failed")
