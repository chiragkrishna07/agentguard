import fnmatch
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

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
        type, max, min, maxlen, pattern (regex string).
    on_violation:
        "block" (default) raises GuardBlockedError.
        "warn" logs a warning and allows the call.
    """

    def __init__(
        self,
        allowed: Optional[List[str]] = None,
        blocked: Optional[List[str]] = None,
        param_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        on_violation: Literal["block", "warn"] = "block",
    ) -> None:
        self.allowed = allowed
        self.blocked = blocked or []
        self.param_rules = param_rules or {}
        self.on_violation = on_violation

    def _name_check(self, tool_name: str) -> Tuple[bool, str]:
        for pat in self.blocked:
            if fnmatch.fnmatch(tool_name, pat):
                return False, f"Tool '{tool_name}' matches blocked pattern '{pat}'"

        if self.allowed is not None:
            for pat in self.allowed:
                if fnmatch.fnmatch(tool_name, pat):
                    return True, ""
            return False, f"Tool '{tool_name}' is not in the allowed list"

        return True, ""

    def _param_check(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        rules = self.param_rules.get(tool_name, {})
        for param, rule in rules.items():
            value = params.get(param)
            if value is None:
                continue

            expected_type = rule.get("type")
            if expected_type is not None and not isinstance(value, expected_type):
                return (
                    False,
                    f"Param '{param}': expected {expected_type.__name__}, got {type(value).__name__}",
                )

            if "max" in rule and isinstance(value, (int, float)) and value > rule["max"]:
                return False, f"Param '{param}' value {value} exceeds max {rule['max']}"

            if "min" in rule and isinstance(value, (int, float)) and value < rule["min"]:
                return False, f"Param '{param}' value {value} is below min {rule['min']}"

            if "maxlen" in rule and isinstance(value, str) and len(value) > rule["maxlen"]:
                return False, f"Param '{param}' length {len(value)} exceeds maxlen {rule['maxlen']}"

            if "pattern" in rule and isinstance(value, str):
                if not re.fullmatch(rule["pattern"], value):
                    return False, f"Param '{param}' does not match required pattern"

        return True, ""

    async def scan_tool_call(
        self, tool_name: str, params: Dict[str, Any], ctx: SessionContext
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
