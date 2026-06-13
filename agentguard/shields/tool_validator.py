import fnmatch
import re
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext


class ToolValidator(BaseShield):
    """Validates tool calls by name pattern and parameter rules.

    Parameters
    ----------
    allowed:
        Glob patterns for permitted tool names. None means all are permitted
        (unless matched by `blocked`).
    blocked:
        Glob patterns for denied tool names. Evaluated before `allowed`.
    param_rules:
        Per-tool parameter constraints. Each rule is a dict with optional keys:
        ``type``, ``required`` (bool), ``max``, ``min``, ``maxlen``,
        ``pattern`` (regex string).
    on_violation:
        "block" (default) raises GuardBlockedError.
        "warn" logs a warning and allows the call.

    Notes
    -----
    Tool names are matched case-insensitively (most dispatchers are), so
    ``blocked=["delete_*"]`` also stops ``DELETE_FILE``. Numeric ``min``/``max``
    rules coerce numeric strings (LLM tool args are often strings) and fail
    closed on non-numeric values, and ``maxlen``/``pattern`` apply to the value's
    string form — so a rule can't be dodged by changing the argument's type.
    """

    def __init__(
        self,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        param_rules: dict[str, dict[str, Any]] | None = None,
        on_violation: Literal["block", "warn"] = "block",
    ) -> None:
        self.allowed = allowed
        self.blocked = blocked or []
        self.param_rules = param_rules or {}
        self.on_violation = on_violation

    def _name_check(self, tool_name: str) -> tuple[bool, str]:
        name = tool_name.lower()
        for pat in self.blocked:
            if fnmatch.fnmatchcase(name, pat.lower()):
                return False, f"Tool '{tool_name}' matches blocked pattern '{pat}'"

        if self.allowed is not None:
            for pat in self.allowed:
                if fnmatch.fnmatchcase(name, pat.lower()):
                    return True, ""
            return False, f"Tool '{tool_name}' is not in the allowed list"

        return True, ""

    @staticmethod
    def _as_number(value: Any) -> float | None:
        # bool is an int subclass — don't treat True/False as numeric.
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _param_check(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, str]:
        rules = self.param_rules.get(tool_name, {})
        for param, rule in rules.items():
            value = params.get(param)
            if value is None:
                if rule.get("required"):
                    return False, f"Required param '{param}' is missing"
                continue

            expected_type = rule.get("type")
            if expected_type is not None and not isinstance(value, expected_type):
                return (
                    False,
                    f"Param '{param}': expected {expected_type.__name__}, got {type(value).__name__}",
                )

            # Numeric bounds — coerce numeric strings, fail closed on non-numbers.
            if "max" in rule or "min" in rule:
                num = self._as_number(value)
                if num is None:
                    return False, f"Param '{param}' must be numeric, got {type(value).__name__}"
                if "max" in rule and num > rule["max"]:
                    return False, f"Param '{param}' value {num} exceeds max {rule['max']}"
                if "min" in rule and num < rule["min"]:
                    return False, f"Param '{param}' value {num} is below min {rule['min']}"

            # Length / pattern apply to the string form so a non-str can't dodge them.
            sval = value if isinstance(value, str) else str(value)
            if "maxlen" in rule and len(sval) > rule["maxlen"]:
                return False, f"Param '{param}' length {len(sval)} exceeds maxlen {rule['maxlen']}"

            if "pattern" in rule and not re.fullmatch(rule["pattern"], sval):
                return False, f"Param '{param}' does not match required pattern"

        return True, ""

    async def scan_tool_call(
        self, tool_name: str, params: dict[str, Any], ctx: SessionContext
    ) -> ShieldResult:
        name_ok, name_reason = self._name_check(tool_name)
        if not name_ok:
            if self.on_violation == "warn":
                import warnings
                warnings.warn(f"[AgentGuard ToolValidator] {name_reason}", stacklevel=4)
                return ShieldResult(allowed=True)
            return ShieldResult(allowed=False, reason=name_reason, reason_code="TOOL_NOT_ALLOWED")

        params_ok, params_reason = self._param_check(tool_name, params)
        if not params_ok:
            if self.on_violation == "warn":
                import warnings
                warnings.warn(f"[AgentGuard ToolValidator] {params_reason}", stacklevel=4)
                return ShieldResult(allowed=True)
            return ShieldResult(
                allowed=False, reason=params_reason, reason_code="TOOL_PARAM_INVALID"
            )

        return ShieldResult(allowed=True)
