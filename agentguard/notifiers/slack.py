from agentguard.notifiers.base import BaseNotifier


class SlackNotifier(BaseNotifier):
    """Posts an approval request to a Slack incoming webhook.

    Requires: pip install httpx (included in agentguard core deps)
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    async def notify(self, gate_id: str, context: dict) -> None:
        import httpx

        lines = [f"*[AgentGuard] HumanGate — Approval Required*", f"Gate ID: `{gate_id}`"]
        for key, value in context.items():
            lines.append(f"*{key}*: {value}")

        payload = {"text": "\n".join(lines)}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
