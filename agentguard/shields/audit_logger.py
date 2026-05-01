import hashlib
import json
import logging
import time
from typing import Literal, Optional

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class AuditLogger(BaseShield):
    """Structured JSON audit log for all agent I/O and tool calls.

    Never logs raw input or output text — only SHA-256 hashes and lengths,
    so sensitive content never appears in log files.
    """

    def __init__(
        self,
        output: Literal["stdout", "file"] = "stdout",
        path: str = "./agentguard_audit.log",
        include_input_hash: bool = True,
    ) -> None:
        self.output = output
        self.path = path
        self.include_input_hash = include_input_hash
        self._logger: Optional[logging.Logger] = None

    def _get_logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(f"agentguard.audit.{id(self)}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler: logging.Handler = (
                    logging.FileHandler(self.path, encoding="utf-8")
                    if self.output == "file"
                    else logging.StreamHandler()
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            self._logger = logger
        return self._logger

    def _emit(self, event: str, **fields: object) -> None:
        record = {"event": event, "ts": round(time.time(), 3), **fields}
        self._get_logger().info(json.dumps(record, default=str))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        self._emit(
            "input_scan",
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            input_hash=self._hash(text) if self.include_input_hash else None,
            input_length=len(text),
            cost_so_far_usd=round(ctx.cost_usd, 6),
            request_count=ctx.request_count,
        )
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        self._emit(
            "output_scan",
            session_id=ctx.session_id,
            output_hash=self._hash(text) if self.include_input_hash else None,
            output_length=len(text),
            cost_total_usd=round(ctx.cost_usd, 6),
        )
        return ShieldResult(allowed=True)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: SessionContext
    ) -> ShieldResult:
        self._emit(
            "tool_call",
            session_id=ctx.session_id,
            tool_name=tool_name,
            param_keys=sorted(params.keys()),
            cost_so_far_usd=round(ctx.cost_usd, 6),
        )
        return ShieldResult(allowed=True)
