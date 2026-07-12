import hashlib
import hmac
import json
import logging
import os
import secrets as stdlib_secrets
import time
from typing import Literal

from agentguard.core.base_shield import BaseShield, GuardDecision, ShieldResult
from agentguard.core.session import SessionContext


class AuditLogger(BaseShield):
    """Structured JSON audit log for all agent I/O and tool calls.

    Raw input/output text is never logged.  Secure defaults use keyed HMAC
    fingerprints for content and pseudonymize session/user identifiers, making
    low-entropy values resistant to offline dictionary attacks.  Supply the
    same ``hmac_key`` to multiple processes only when cross-process correlation
    is required.

    Set ``identity_mode="raw"``, ``schema_mode="raw"``, or
    ``fingerprint_mode="sha256"`` only for compatibility with legacy audit
    consumers; those settings reduce privacy.
    """

    scan_tool_arguments_as_input = False

    def __init__(
        self,
        output: Literal["stdout", "file"] = "stdout",
        path: str = "./agentguard_audit.log",
        include_input_hash: bool = True,
        *,
        identity_mode: Literal["hmac", "omit", "raw"] = "hmac",
        fingerprint_mode: Literal["hmac", "omit", "sha256"] = "hmac",
        schema_mode: Literal["hmac", "omit", "raw"] = "hmac",
        hmac_key: bytes | str | None = None,
    ) -> None:
        if output not in ("stdout", "file"):
            raise ValueError("output must be 'stdout' or 'file'")
        if identity_mode not in ("hmac", "omit", "raw"):
            raise ValueError("identity_mode must be 'hmac', 'omit', or 'raw'")
        if fingerprint_mode not in ("hmac", "omit", "sha256"):
            raise ValueError("fingerprint_mode must be 'hmac', 'omit', or 'sha256'")
        if schema_mode not in ("hmac", "omit", "raw"):
            raise ValueError("schema_mode must be 'hmac', 'omit', or 'raw'")
        if isinstance(hmac_key, str):
            hmac_key = hmac_key.encode("utf-8")
        if hmac_key is not None and len(hmac_key) < 16:
            raise ValueError("hmac_key must contain at least 16 bytes")
        self.output = output
        self.path = path
        self.include_input_hash = include_input_hash
        self.identity_mode = identity_mode
        self.fingerprint_mode = fingerprint_mode
        self.schema_mode = schema_mode
        self._hmac_key = hmac_key or stdlib_secrets.token_bytes(32)
        self._logger_name = f"agentguard.audit.{stdlib_secrets.token_hex(16)}"
        self._logger: logging.Logger | None = None

    def _get_logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(self._logger_name)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                if self.output == "file":
                    # Atomically create only a new file with owner-only access.
                    # Existing operator-managed files keep their permissions.
                    try:
                        fd = os.open(
                            self.path,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                        )
                    except FileExistsError:
                        pass
                    else:
                        os.close(fd)
                    handler: logging.Handler = logging.FileHandler(self.path, encoding="utf-8")
                else:
                    handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            self._logger = logger
        return self._logger

    def _emit(self, event: str, **fields: object) -> None:
        record = {"event": event, "ts": round(time.time(), 3), **fields}
        self._get_logger().info(json.dumps(record, default=str))

    def _hmac(self, text: str, domain: str) -> str:
        message = f"agentguard:{domain}\0{text}".encode()
        return hmac.new(self._hmac_key, message, hashlib.sha256).hexdigest()[:32]

    def _fingerprint(self, text: str) -> str | None:
        if not self.include_input_hash or self.fingerprint_mode == "omit":
            return None
        if self.fingerprint_mode == "sha256":
            # Explicit legacy compatibility path (unkeyed and truncated).
            return hashlib.sha256(text.encode()).hexdigest()[:16]
        return self._hmac(text, "content")

    def _identity(self, value: str | None, kind: str) -> str | None:
        if value is None or self.identity_mode == "omit":
            return None
        if self.identity_mode == "raw":
            return value
        return f"hmac:{self._hmac(value, kind)}"

    def _identity_fields(
        self, ctx: SessionContext, *, include_user: bool = False
    ) -> dict[str, str]:
        fields = {}
        session_id = self._identity(ctx.session_id, "session_id")
        user_id = self._identity(ctx.user_id, "user_id") if include_user else None
        if session_id is not None:
            fields["session_id"] = session_id
        if user_id is not None:
            fields["user_id"] = user_id
        return fields

    def _fingerprint_field(self, name: str, text: str) -> dict[str, str]:
        fingerprint = self._fingerprint(text)
        return {name: fingerprint} if fingerprint is not None else {}

    def _schema(self, value: str, kind: str) -> str | None:
        if self.schema_mode == "omit":
            return None
        if self.schema_mode == "raw":
            return value[:128]
        return f"hmac:{self._hmac(value, kind)}"

    def _safe_param_keys(self, params: dict) -> list[str]:
        keys = []
        for key in params:
            if isinstance(key, (str, int, float, bool)) or key is None:
                rendered = str(key)
            else:
                rendered = f"<{key.__class__.__name__}>"
            protected = self._schema(rendered, "param_key")
            if protected is not None:
                keys.append(protected)
        return sorted(keys)[:100]

    def _tool_field(self, tool_name: str) -> dict[str, str]:
        protected = self._schema(tool_name, "tool_name")
        return {"tool_name": protected} if protected is not None else {}

    async def scan_input(self, text: str, ctx: SessionContext) -> ShieldResult:
        self._emit(
            "input_scan",
            **self._identity_fields(ctx, include_user=True),
            **self._fingerprint_field("input_hash", text),
            input_length=len(text),
            cost_so_far_usd=round(ctx.cost_usd, 6),
            request_count=ctx.request_count,
        )
        return ShieldResult(allowed=True)

    async def scan_output(self, text: str, ctx: SessionContext) -> ShieldResult:
        self._emit(
            "output_scan",
            **self._identity_fields(ctx),
            **self._fingerprint_field("output_hash", text),
            output_length=len(text),
            cost_total_usd=round(ctx.cost_usd, 6),
        )
        return ShieldResult(allowed=True)

    async def scan_output_preview(
        self, text: str, ctx: SessionContext
    ) -> ShieldResult:
        # Only the final output and a blocking preview decision belong in the
        # audit trail; cumulative prefixes would duplicate events/fingerprints.
        return ShieldResult(allowed=True)

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: SessionContext
    ) -> ShieldResult:
        self._emit(
            "tool_call",
            **self._identity_fields(ctx),
            **self._tool_field(tool_name),
            param_keys=self._safe_param_keys(params),
            cost_so_far_usd=round(ctx.cost_usd, 6),
        )
        return ShieldResult(allowed=True)

    async def scan_tool_output(
        self, tool_name: str, output: str, ctx: SessionContext
    ) -> ShieldResult:
        self._emit(
            "tool_output",
            **self._identity_fields(ctx),
            **self._tool_field(tool_name),
            **self._fingerprint_field("output_hash", output),
            output_length=len(output),
            indirect_injection_flagged=bool(ctx.metadata.get("indirect_injection_detected")),
        )
        return ShieldResult(allowed=True)

    async def on_decision(self, decision: GuardDecision, ctx: SessionContext) -> None:
        """Record blocks made anywhere in the pipeline without raw content."""
        if decision.allowed:
            return
        self._emit(
            "guard_blocked",
            **self._identity_fields(ctx, include_user=True),
            flow=decision.flow,
            blocking_shield=decision.shield_name,
            reason_code=decision.reason_code,
            **self._tool_field(decision.tool_name) if decision.tool_name else {},
        )
