import asyncio

import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.human_gate import HumanGate


class CapturingNotifier:
    def __init__(self):
        self.gate = None
        self.context = None
        self.decision = True

    async def notify(self, gate_id, context):
        self.context = context
        if self.decision:
            await self.gate.approve(gate_id)
        else:
            await self.gate.deny(gate_id)


@pytest.mark.asyncio
async def test_tool_gate_is_case_insensitive_and_hides_param_values_by_default():
    notifier = CapturingNotifier()
    gate = HumanGate(
        triggers=["tool_call:SEND_*"], notifier=notifier, timeout_seconds=1
    )
    notifier.gate = gate
    result = await gate.scan_tool_call(
        "send_email",
        {"api_key": "top-secret", "to": "person@example.com"},
        SessionContext(),
    )
    assert result.allowed
    encoded = repr(notifier.context)
    assert "top-secret" not in encoded
    assert "person@example.com" not in encoded
    assert "param_keys" not in notifier.context
    assert "params_fingerprint" in notifier.context
    assert notifier.context["session_id"].startswith("hmac:")


@pytest.mark.asyncio
async def test_explicit_sanitizer_controls_notification_payload():
    notifier = CapturingNotifier()
    gate = HumanGate(
        triggers=["tool_call:*"],
        notifier=notifier,
        timeout_seconds=1,
        param_sanitizer=lambda params: {"amount": params["amount"], "token": "[REDACTED]"},
    )
    notifier.gate = gate
    assert (
        await gate.scan_tool_call(
            "pay", {"amount": 5, "token": "secret"}, SessionContext()
        )
    ).allowed
    assert notifier.context["sanitized_params"] == {"amount": 5, "token": "[REDACTED]"}


@pytest.mark.asyncio
async def test_raw_schema_and_identity_require_explicit_opt_in():
    notifier = CapturingNotifier()
    gate = HumanGate(
        triggers=["tool_call:*"],
        notifier=notifier,
        timeout_seconds=1,
        identity_mode="raw",
        include_param_keys=True,
    )
    notifier.gate = gate
    ctx = SessionContext(session_id="session-raw")
    assert (await gate.scan_tool_call("read", {"safe_key": "value"}, ctx)).allowed
    assert notifier.context["session_id"] == "session-raw"
    assert notifier.context["param_keys"] == ["safe_key"]


@pytest.mark.asyncio
async def test_denial_and_unknown_gate_ids():
    notifier = CapturingNotifier()
    notifier.decision = False
    gate = HumanGate(triggers=["tool_call:*"], notifier=notifier, timeout_seconds=1)
    notifier.gate = gate
    result = await gate.scan_tool_call("delete", {}, SessionContext())
    assert not result.allowed
    assert await gate.approve("unknown") is False
    assert await gate.deny("unknown") is False
    assert gate._decisions == {}


@pytest.mark.asyncio
async def test_timeout_fails_closed():
    class SilentNotifier:
        async def notify(self, gate_id, context):
            pass

    gate = HumanGate(
        triggers=["tool_call:*"],
        notifier=SilentNotifier(),
        timeout_seconds=0.001,
    )
    result = await gate.scan_tool_call("write", {}, SessionContext())
    assert not result.allowed


def test_invalid_configuration_fails_fast():
    with pytest.raises(ValueError):
        HumanGate(triggers=[])
    with pytest.raises(ValueError):
        HumanGate(triggers=["cost_exceeds:nope"])
    with pytest.raises(ValueError):
        HumanGate(triggers=["tool_call:"])
    with pytest.raises(ValueError):
        HumanGate(triggers=["pii_detected"], timeout_seconds=0)
    with pytest.raises(ValueError):
        HumanGate(triggers=["pii_detected"], param_sanitizer="redact")
    with pytest.raises(ValueError):
        HumanGate(triggers=["pii_detected"], max_pending_gates=0)
    with pytest.raises(ValueError):
        HumanGate(triggers=["pii_detected"], identity_mode="plain")


@pytest.mark.asyncio
async def test_gate_ids_have_at_least_128_bits_of_unpredictable_material():
    captured = []

    class DenyingNotifier:
        async def notify(self, gate_id, context):
            captured.append(gate_id)
            await gate.deny(gate_id)

    gate = HumanGate(
        triggers=["tool_call:*"], notifier=DenyingNotifier(), timeout_seconds=1
    )
    assert not (await gate.scan_tool_call("write", {}, SessionContext())).allowed
    token = captured[0].removeprefix("gate-")
    assert len(token) >= 22


@pytest.mark.asyncio
async def test_pending_gate_limit_fails_closed_without_notifying_again():
    class BlockingNotifier:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.ids = []

        async def notify(self, gate_id, context):
            self.ids.append(gate_id)
            self.started.set()
            await self.release.wait()

    notifier = BlockingNotifier()
    gate = HumanGate(
        triggers=["tool_call:*"],
        notifier=notifier,
        timeout_seconds=1,
        max_pending_gates=1,
    )
    first = asyncio.create_task(gate.scan_tool_call("write", {}, SessionContext()))
    await notifier.started.wait()
    second = await gate.scan_tool_call("write", {}, SessionContext())
    assert not second.allowed
    assert len(notifier.ids) == 1
    await gate.deny(notifier.ids[0])
    notifier.release.set()
    assert not (await first).allowed
