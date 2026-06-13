"""Measure PromptShield detection quality against the labeled corpus.

Run directly for a human-readable report:

    python -m tests.benchmarks.bench_detection

The same metrics are asserted as thresholds in tests/unit/test_detection_quality.py
so detection can't silently regress.
"""
import asyncio

from agentguard.core.session import SessionContext
from agentguard.shields.prompt_shield import PromptShield
from tests.benchmarks.injection_corpus import ATTACKS, BENIGN


async def _blocked(shield: PromptShield, text: str) -> bool:
    result = await shield.scan_input(text, SessionContext())
    return not result.allowed


async def evaluate(shield: PromptShield) -> dict:
    tp = sum([await _blocked(shield, t) for t in ATTACKS])
    fn = len(ATTACKS) - tp
    fp = sum([await _blocked(shield, t) for t in BENIGN])
    tn = len(BENIGN) - fp

    recall = tp / len(ATTACKS) if ATTACKS else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    fpr = fp / len(BENIGN) if BENIGN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    missed = [t for t in ATTACKS if not await _blocked(shield, t)]
    false_alarms = [t for t in BENIGN if await _blocked(shield, t)]
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "precision": precision, "fpr": fpr, "f1": f1,
        "missed": missed, "false_alarms": false_alarms,
    }


def _print(mode: str, m: dict) -> None:
    print(f"\n=== PromptShield(mode={mode!r}) ===")
    print(f"attacks={len(ATTACKS)} benign={len(BENIGN)}")
    print(f"recall   {m['recall']:.1%}  (caught {m['tp']}/{len(ATTACKS)})")
    print(f"precision{m['precision']:.1%}")
    print(f"FPR      {m['fpr']:.1%}  (false alarms {m['fp']}/{len(BENIGN)})")
    print(f"F1       {m['f1']:.3f}")
    if m["missed"]:
        print("MISSED:")
        for t in m["missed"]:
            print(f"  - {t[:70]!r}")
    if m["false_alarms"]:
        print("FALSE ALARMS:")
        for t in m["false_alarms"]:
            print(f"  - {t[:70]!r}")


async def main() -> None:
    for mode in ("strict", "paranoid"):
        _print(mode, await evaluate(PromptShield(mode=mode, use_canary=False)))


if __name__ == "__main__":
    asyncio.run(main())
