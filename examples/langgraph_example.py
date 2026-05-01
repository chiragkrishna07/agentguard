"""
AgentGuard + LangGraph — demo with tool call protection.

Prerequisites
-------------
    pip install agentguard langgraph langchain-openai tiktoken
    export OPENAI_API_KEY=sk-...

Run
---
    python examples/langgraph_example.py
"""
import asyncio
import os

from agentguard import Guard, PromptShield, ToolValidator, CostLimit, AuditLogger
from agentguard.adapters.langgraph import GuardLangGraph
from agentguard.core.exceptions import GuardBlockedError


# Build the guard
guard = Guard(
    shields=[
        AuditLogger(output="stdout"),
        PromptShield(mode="strict", use_ml=False),
        ToolValidator(
            allowed=["search_*", "get_*"],
            blocked=["delete_*", "admin_*", "export_*"],
            param_rules={
                "search_hotels": {
                    "city": {"type": str, "maxlen": 100},
                    "max_price": {"type": (int, float), "max": 10000},
                }
            },
        ),
        CostLimit(max_usd=0.50, model="gpt-4o-mini"),
    ]
)

adapter = GuardLangGraph(guard)


# ---- Mock tools (replace with real implementations) ----------------------

async def search_hotels(city: str, max_price: float = 500.0) -> str:
    return f"Found 3 hotels in {city} under ${max_price}/night."

async def delete_booking(booking_id: str) -> str:
    return f"Deleted booking {booking_id}."


# Wrap tools with guard
guarded_search = adapter.wrap_tool(search_hotels)
guarded_delete = adapter.wrap_tool(delete_booking)


async def main() -> None:
    print("\n[Test 1] Allowed tool call")
    result = await guarded_search(city="Tokyo", max_price=200.0)
    print(f"Result: {result}\n")

    print("[Test 2] Blocked tool call (delete_booking is in blocked list)")
    try:
        await guarded_delete(booking_id="BK-12345")
    except GuardBlockedError as e:
        print(f"BLOCKED: {e}\n")

    print("[Test 3] Param violation (price too high)")
    try:
        await guarded_search(city="Paris", max_price=99999.0)
    except GuardBlockedError as e:
        print(f"BLOCKED: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
