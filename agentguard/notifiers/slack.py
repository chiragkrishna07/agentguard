import html
import math
from numbers import Real

from agentguard.notifiers._endpoint import validate_notifier_url
from agentguard.notifiers.base import BaseNotifier


class SlackNotifier(BaseNotifier):
    """Posts an approval request to a Slack incoming webhook.

    Requires: pip install httpx (included in agentguard core deps)
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout_seconds: float = 10.0,
        max_value_chars: int = 500,
        allow_insecure_http: bool = False,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be > 0")
        if (
            isinstance(max_value_chars, bool)
            or not isinstance(max_value_chars, int)
            or max_value_chars < 1
        ):
            raise ValueError("max_value_chars must be >= 1")
        self.webhook_url: str = validate_notifier_url(
            webhook_url, allow_insecure_http=allow_insecure_http
        )
        self.timeout_seconds: float = timeout_seconds
        self.max_value_chars: int = max_value_chars

    def _safe(self, value: object) -> str:
        text = str(value)
        if len(text) > self.max_value_chars:
            text = text[: self.max_value_chars] + "…"
        # Slack mrkdwn uses these characters for links and mentions. Escaping
        # prevents agent-controlled tool names from forging approval UI.
        return html.escape(text, quote=False)

    async def notify(self, gate_id: str, context: dict) -> None:
        import httpx

        lines = [
            "*[AgentGuard] HumanGate — Approval Required*",
            f"Gate ID: `{self._safe(gate_id)}`",
        ]
        for key, value in context.items():
            lines.append(f"*{self._safe(key)}*: {self._safe(value)}")

        payload = {"text": "\n".join(lines)}
        delivery_failed = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            delivery_failed = True
        if delivery_failed:
            raise RuntimeError("Slack approval notification failed")
