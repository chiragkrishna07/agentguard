"""Threshold tests over the labeled corpus — guard against detection regressions.

These assert aggregate quality, not individual samples, so adding corpus entries
won't make them brittle. If one fails, run
``python -m tests.benchmarks.bench_detection`` to see exactly what regressed.
"""
import pytest

from agentguard.shields.prompt_shield import PromptShield
from tests.benchmarks.bench_detection import evaluate


@pytest.mark.asyncio
async def test_strict_mode_quality():
    m = await evaluate(PromptShield(mode="strict", use_canary=False))
    assert m["recall"] >= 0.95, f"recall regressed: {m['recall']:.1%}, missed {m['missed']}"
    assert m["precision"] >= 0.97, (
        f"precision regressed: {m['precision']:.1%}, false alarms {m['false_alarms']}"
    )
    assert m["fpr"] <= 0.05, f"FPR regressed: {m['fpr']:.1%}, {m['false_alarms']}"


@pytest.mark.asyncio
async def test_paranoid_mode_catches_everything():
    # Paranoid trades precision for recall by design; it must miss nothing.
    m = await evaluate(PromptShield(mode="paranoid", use_canary=False))
    assert m["recall"] >= 0.97, f"recall regressed: {m['recall']:.1%}, missed {m['missed']}"
