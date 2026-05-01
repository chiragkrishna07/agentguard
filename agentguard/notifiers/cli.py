from agentguard.notifiers.base import BaseNotifier


class CLINotifier(BaseNotifier):
    """Prints approval request to stdout. Useful during development."""

    async def notify(self, gate_id: str, context: dict) -> None:
        print("\n" + "=" * 60)
        print("[AgentGuard] HumanGate — Approval Required")
        print(f"  Gate ID : {gate_id}")
        for key, value in context.items():
            print(f"  {key:<14}: {value}")
        print(f"\n  To approve: await gate.approve('{gate_id}')")
        print(f"  To deny:    await gate.deny('{gate_id}')")
        print("=" * 60 + "\n")
