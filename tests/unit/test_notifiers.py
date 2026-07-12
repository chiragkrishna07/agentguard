import hashlib
import hmac
import json

import httpx
import pytest

from agentguard.notifiers.slack import SlackNotifier
from agentguard.notifiers.webhook import WebhookNotifier


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeClient:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs, self.kwargs))
        return FakeResponse()


class FailingClient(FakeClient):
    async def post(self, url, **kwargs):
        request = httpx.Request("POST", url)
        raise httpx.ConnectError(f"failed for sensitive URL {url}", request=request)


@pytest.mark.parametrize(
    "factory",
    [
        lambda url: WebhookNotifier(url),
        lambda url: SlackNotifier(url),
    ],
)
def test_notifier_requires_safe_url(factory):
    with pytest.raises(ValueError):
        factory("http://example.com/hook")
    with pytest.raises(ValueError):
        factory("file:///tmp/hook")
    with pytest.raises(ValueError):
        factory("https://user:pass@example.com/hook")
    with pytest.raises(ValueError):
        factory("https://example.com/line\nbreak")


@pytest.mark.asyncio
async def test_webhook_v1_signature_binds_timestamp_and_body(monkeypatch):
    FakeClient.calls.clear()
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    notifier = WebhookNotifier(
        "https://hooks.example.com/approve", secret="a-secure-secret-123"
    )
    await notifier.notify("gate-1", {"tool": "send"})
    _, kwargs, _ = FakeClient.calls[0]
    timestamp = kwargs["headers"]["X-AgentGuard-Timestamp"]
    body = kwargs["content"]
    expected = hmac.new(
        b"a-secure-secret-123",
        timestamp.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    assert kwargs["headers"]["X-AgentGuard-Signature"] == f"v1={expected}"
    assert json.loads(body)["gate_id"] == "gate-1"


@pytest.mark.asyncio
async def test_slack_escapes_agent_controlled_markup(monkeypatch):
    FakeClient.calls.clear()
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    notifier = SlackNotifier("https://hooks.slack.test/services/example")
    await notifier.notify("gate-1", {"tool": "<https://evil.test|Approve>"})
    _, kwargs, _ = FakeClient.calls[0]
    assert "<https://evil.test" not in kwargs["json"]["text"]
    assert "&lt;https://evil.test" in kwargs["json"]["text"]


def test_short_webhook_secret_rejected():
    with pytest.raises(ValueError):
        WebhookNotifier("https://example.com/hook", secret="short")


@pytest.mark.asyncio
async def test_webhook_payload_is_bounded():
    notifier = WebhookNotifier(
        "https://example.com/hook",
        max_payload_bytes=20,
    )
    with pytest.raises(RuntimeError, match="size limit"):
        await notifier.notify("gate", {"large": "x" * 100})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "notifier",
    [
        WebhookNotifier("https://example.com/sensitive-hook-token"),
        SlackNotifier("https://example.com/sensitive-hook-token"),
    ],
)
async def test_http_failures_do_not_chain_url_bearing_exceptions(monkeypatch, notifier):
    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    with pytest.raises(RuntimeError) as exc_info:
        await notifier.notify("gate", {})
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "sensitive-hook-token" not in str(exc_info.value)
