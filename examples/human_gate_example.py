"""
AgentGuard HumanGate — demo of async human-in-the-loop approval.

The agent attempts to call a 'send_email' tool.
HumanGate intercepts and waits for approval via CLI.

Run
---
    python examples/human_gate_example.py
"""
import asyncio

from agentguard import Guard, HumanGate
from agentguard.core.exceptions import GuardBlockedError
from agentguard.notifiers.cli import CLINotifier


gate = HumanGate(
    triggers=["tool_call:send_*", "tool_call:delete_*"],
    notifier=CLINotifier(),
    timeout_seconds=30,
    on_timeout="block",
)

guard = Guard(shields=[gate])


async def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to}: {subject}"


guarded_send = guard.scan_tool_call  # direct tool scan demo


async def auto_approve_after_delay(gate_id: str, delay: float = 2.0) -> None:
    """Simulates a human approving after reading the request."""
    await asyncio.sleep(delay)
    print(f"\n  [Simulated human] Approving gate {gate_id}")
    await gate.approve(gate_id)


async def main() -> None:
    print("\nHumanGate Demo")
    print("The agent will try to send an email. You have 30s to approve.\n")

    # In a real scenario, approval comes from a Slack message, webhook, or UI.
    # Here we auto-approve after 2 seconds to demonstrate the flow.

    from agentguard.core.session import SessionContext
    ctx = SessionContext()

    # Patch: capture the gate_id when notify is called
    original_notify = gate.notifier.notify
    captured_ids: list = []

    async def capturing_notify(gate_id: str, context: dict) -> None:
        captured_ids.append(gate_id)
        await original_notify(gate_id, context)

    gate.notifier.notify = capturing_notify

    async def run_tool():
        await guard.scan_tool_call("send_email", {"to": "ceo@company.com", "subject": "Q4 Report"}, ctx)
        result = await send_email(to="ceo@company.com", subject="Q4 Report", body="...")
        print(f"\nTool result: {result}")

    # Start auto-approver and tool execution concurrently
    await asyncio.gather(
        run_tool(),
        asyncio.create_task(
            _auto_approve(captured_ids, gate)
        ),
    )


async def _auto_approve(captured_ids: list, gate: HumanGate) -> None:
    while not captured_ids:
        await asyncio.sleep(0.1)
    gate_id = captured_ids[0]
    await asyncio.sleep(2)
    print(f"\n  [Simulated human] Approving gate {gate_id}")
    await gate.approve(gate_id)


if __name__ == "__main__":
    asyncio.run(main())
