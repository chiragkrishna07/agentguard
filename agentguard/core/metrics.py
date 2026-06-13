"""Lightweight, thread-safe counters for production monitoring.

A `Guard` owns one `GuardMetrics`. It records how much traffic each pipeline
saw and every block (keyed by reason code and by the shield that raised it),
so operators can wire `guard.stats()` into a dashboard or alert without bolting
on external instrumentation.
"""
import threading
from dataclasses import dataclass, field


@dataclass
class GuardMetrics:
    inputs_scanned: int = 0
    outputs_scanned: int = 0
    tool_calls_scanned: int = 0
    tool_outputs_scanned: int = 0
    blocked: int = 0
    blocks_by_code: dict[str, int] = field(default_factory=dict)
    blocks_by_shield: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_scan(self, kind: str) -> None:
        with self._lock:
            setattr(self, f"{kind}_scanned", getattr(self, f"{kind}_scanned") + 1)

    def record_block(self, shield_name: str, reason_code: str) -> None:
        with self._lock:
            self.blocked += 1
            self.blocks_by_code[reason_code] = self.blocks_by_code.get(reason_code, 0) + 1
            self.blocks_by_shield[shield_name] = (
                self.blocks_by_shield.get(shield_name, 0) + 1
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "inputs_scanned": self.inputs_scanned,
                "outputs_scanned": self.outputs_scanned,
                "tool_calls_scanned": self.tool_calls_scanned,
                "tool_outputs_scanned": self.tool_outputs_scanned,
                "blocked": self.blocked,
                "blocks_by_code": dict(self.blocks_by_code),
                "blocks_by_shield": dict(self.blocks_by_shield),
            }

    def reset(self) -> None:
        with self._lock:
            self.inputs_scanned = 0
            self.outputs_scanned = 0
            self.tool_calls_scanned = 0
            self.tool_outputs_scanned = 0
            self.blocked = 0
            self.blocks_by_code.clear()
            self.blocks_by_shield.clear()
