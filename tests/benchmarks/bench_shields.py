"""
Shield latency benchmark.

Run with:
    python -m tests.benchmarks.bench_shields

Reports mean latency (ms) for each shield's scan_input on a standard payload.
"""
import asyncio
import statistics
import time

from agentguard.core.session import SessionContext
from agentguard.shields.audit_logger import AuditLogger
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.rate_limit import RateLimit
from agentguard.shields.tool_validator import ToolValidator

SAMPLE_TEXT = (
    "I need to book a hotel in Tokyo for 3 nights next month, "
    "budget around $150/night, preferably near the city centre."
)
WARMUP = 5
ITERATIONS = 100


async def bench(shield_name: str, shield, text: str = SAMPLE_TEXT) -> None:
    ctx = SessionContext()
    latencies: list[float] = []

    for _ in range(WARMUP):
        await shield.scan_input(text, ctx)

    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        await shield.scan_input(text, ctx)
        latencies.append((time.perf_counter() - t0) * 1000)

    mean = statistics.mean(latencies)
    p99 = sorted(latencies)[int(0.99 * ITERATIONS)]
    print(f"  {shield_name:<25} mean={mean:.3f}ms  p99={p99:.3f}ms")


async def main() -> None:
    print(f"\nAgentGuard Shield Benchmark  ({ITERATIONS} iterations)\n")

    shields = [
        ("PromptShield (rules only)", PromptShield(mode="strict", use_ml=False)),
        ("RateLimit", RateLimit(requests_per_minute=10000, burst=10000)),
        ("ToolValidator", ToolValidator(blocked=["delete_*"])),
        ("AuditLogger (stdout)", AuditLogger(output="stdout")),
    ]

    for name, shield in shields:
        await bench(name, shield)

    print()


if __name__ == "__main__":
    asyncio.run(main())
