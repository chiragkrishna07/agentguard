import fnmatch
import inspect
import json
import math
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping
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
        Per-tool parameter constraints. Tool keys are case-insensitive globs,
        so a ``"*"`` rule can establish defaults and exact tools can override
        them. Parameter names may be dotted mapping paths. Each rule accepts:
        ``type``, ``required``, ``nullable``, ``max``, ``min``, ``maxlen``,
        ``minlen``, ``pattern``, ``choices``/``enum``, and ``predicate``.
    allow_extra_params:
        When false, reject parameters not declared by the matched rules.
    require_user_id:
        Require an authenticated principal in ``ctx.user_id`` before any tool
        can run.
    authorize:
        Optional sync/async application authorization callback receiving
        ``(tool_name, params, ctx)``. It may return ``True``/``None`` to allow,
        ``False`` or a reason string to deny, or a ``ShieldResult``.
    validators:
        Optional per-tool sync/async validators keyed by tool-name glob. Each
        receives ``(params, ctx)`` and follows the same return convention as
        ``authorize``. This is the extension point for Pydantic, JSON Schema,
        ABAC/RBAC, transaction policy, or domain-specific invariants.
    on_violation:
        "block" (default) raises GuardBlockedError.
        "warn" logs a warning and allows the call.

    Notes
    -----
    Tool names are matched case-insensitively (most dispatchers are), so
    ``blocked=["delete_*"]`` also stops ``DELETE_FILE``. Numeric ``min``/``max``
    rules coerce numeric strings (LLM tool args are often strings) and fail
    closed on non-numeric values, and ``maxlen``/``pattern`` apply to the value's
        stable JSON form — so a rule can't be dodged by changing the argument's
        type. NaN and infinity fail numeric constraints closed.
    """

    def __init__(
        self,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        param_rules: dict[str, dict[str, Any]] | None = None,
        on_violation: Literal["block", "warn"] = "block",
        *,
        allow_extra_params: bool = True,
        require_user_id: bool = False,
        authorize: Callable[
            [str, dict[str, Any], SessionContext],
            bool | str | ShieldResult | None | Awaitable[bool | str | ShieldResult | None],
        ]
        | None = None,
        validators: Mapping[
            str,
            Callable[
                [dict[str, Any], SessionContext],
                bool | str | ShieldResult | None | Awaitable[bool | str | ShieldResult | None],
            ],
        ]
        | None = None,
        max_tool_name_chars: int = 256,
        max_params: int | None = 100,
        max_total_chars: int | None = 65_536,
    ) -> None:
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'")
        if isinstance(max_tool_name_chars, bool) or max_tool_name_chars < 1:
            raise ValueError("max_tool_name_chars must be >= 1")
        if max_params is not None and (isinstance(max_params, bool) or max_params < 1):
            raise ValueError("max_params must be >= 1 or None")
        if max_total_chars is not None and (
            isinstance(max_total_chars, bool) or max_total_chars < 1
        ):
            raise ValueError("max_total_chars must be >= 1 or None")
        if isinstance(allowed, str):
            allowed = [allowed]
        if isinstance(blocked, str):
            blocked = [blocked]
        if allowed is not None and any(not isinstance(item, str) or not item for item in allowed):
            raise ValueError("allowed patterns must be non-empty strings")
        if blocked is not None and any(not isinstance(item, str) or not item for item in blocked):
            raise ValueError("blocked patterns must be non-empty strings")
        if authorize is not None and not callable(authorize):
            raise ValueError("authorize must be callable or None")
        if any(
            not pattern or not callable(validator)
            for pattern, validator in (validators or {}).items()
        ):
            raise ValueError("validators must map non-empty patterns to callables")
        self.allowed = allowed
        self.blocked = blocked or []
        self.param_rules = param_rules or {}
        self.on_violation = on_violation
        self.allow_extra_params = allow_extra_params
        self.require_user_id = require_user_id
        self.authorize = authorize
        self.validators = dict(validators or {})
        self.max_tool_name_chars = max_tool_name_chars
        self.max_params = max_params
        self.max_total_chars = max_total_chars
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for tool_pattern, rules in self.param_rules.items():
            if not isinstance(rules, Mapping):
                raise ValueError(f"param rules for {tool_pattern!r} must be a mapping")
            for param, rule in rules.items():
                if not isinstance(rule, Mapping):
                    raise ValueError(f"param rule {tool_pattern!r}.{param!r} must be a mapping")
                if "pattern" in rule:
                    try:
                        re.compile(rule["pattern"])
                    except (TypeError, re.error) as exc:
                        raise ValueError(
                            f"invalid regex for {tool_pattern!r}.{param!r}: {exc}"
                        ) from exc
                if "predicate" in rule and not callable(rule["predicate"]):
                    raise ValueError(f"predicate for {tool_pattern!r}.{param!r} must be callable")
                expected = rule.get("type")
                expected_items = expected if isinstance(expected, tuple) else (expected,)
                if expected is not None and (
                    not expected_items or any(not isinstance(item, type) for item in expected_items)
                ):
                    raise ValueError(
                        f"type for {tool_pattern!r}.{param!r} must be a type or tuple of types"
                    )
                for bound in ("min", "max", "minlen", "maxlen"):
                    if bound in rule and (
                        isinstance(rule[bound], bool) or not isinstance(rule[bound], (int, float))
                    ):
                        raise ValueError(f"{bound} for {tool_pattern!r}.{param!r} must be numeric")
                if "minlen" in rule and rule["minlen"] < 0:
                    raise ValueError(f"minlen for {tool_pattern!r}.{param!r} must be non-negative")
                if "maxlen" in rule and rule["maxlen"] < 0:
                    raise ValueError(f"maxlen for {tool_pattern!r}.{param!r} must be non-negative")
                if "min" in rule and "max" in rule and rule["min"] > rule["max"]:
                    raise ValueError(f"min exceeds max for {tool_pattern!r}.{param!r}")
                if "minlen" in rule and "maxlen" in rule and rule["minlen"] > rule["maxlen"]:
                    raise ValueError(f"minlen exceeds maxlen for {tool_pattern!r}.{param!r}")

    def _name_check(self, tool_name: str) -> tuple[bool, str]:
        if not isinstance(tool_name, str) or not tool_name:
            return False, "Tool name must be a non-empty string"
        if len(tool_name) > self.max_tool_name_chars:
            return False, f"Tool name exceeds {self.max_tool_name_chars} characters"
        if any(ord(char) < 32 or ord(char) == 127 for char in tool_name):
            return False, "Tool name contains control characters"
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
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        if isinstance(value, str):
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None
        return None

    @staticmethod
    def _type_name(expected_type: Any) -> str:
        if isinstance(expected_type, tuple):
            return " or ".join(getattr(item, "__name__", repr(item)) for item in expected_type)
        return getattr(expected_type, "__name__", repr(expected_type))

    @staticmethod
    def _stable_text(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError):
            return f"<{type(value).__module__}.{type(value).__qualname__}>"

    def _rules_for(self, tool_name: str) -> dict[str, Mapping[str, Any]]:
        name = tool_name.casefold()
        merged: dict[str, Mapping[str, Any]] = {}
        exact: list[Mapping[str, Any]] = []
        for pattern, rules in self.param_rules.items():
            if pattern.casefold() == name:
                exact.append(rules)
            elif fnmatch.fnmatchcase(name, pattern.casefold()):
                merged.update(rules)
        for exact_rules in exact:
            merged.update(exact_rules)
        return merged

    @staticmethod
    def _get_path(params: Mapping[str, Any], path: str) -> tuple[bool, Any, bool]:
        """Resolve a dotted rule and report literal/nested ambiguity.

        A literal dotted key remains supported, but accepting it when the same
        path also exists as a nested mapping lets a validator approve one value
        while the tool consumes the other. Such dual representations are
        rejected fail-closed.
        """
        literal_present = path in params
        current: Any = params
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                return (True, params[path], False) if literal_present else (False, None, False)
            current = current[component]
        if literal_present and "." in path:
            return True, None, True
        return True, params[path] if literal_present else current, False

    @classmethod
    def _unexpected_param_paths(
        cls,
        params: Mapping[str, Any],
        declared: set[str],
        prefix: tuple[str, ...] = (),
    ) -> list[str]:
        """Return undeclared paths for a recursively closed argument schema.

        A rule for a leaf/container (for example ``filters``) owns that entire
        value when it has no more-specific descendants. Once dotted child rules
        exist (``payment.amount``), that branch is closed recursively.
        """
        unexpected: list[str] = []
        for raw_key, value in params.items():
            if not isinstance(raw_key, str):
                unexpected.append(".".join((*prefix, f"<{type(raw_key).__name__}>")))
                continue
            path_parts = (*prefix, raw_key)
            path = ".".join(path_parts)
            descendants = {item for item in declared if item.startswith(path + ".")}
            exact = path in declared
            if not exact and not descendants:
                unexpected.append(path)
                continue
            if descendants:
                if isinstance(value, Mapping):
                    unexpected.extend(
                        cls._unexpected_param_paths(value, declared, path_parts)
                    )
                elif not exact:
                    unexpected.append(path)
        return unexpected

    async def _param_check(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(params, dict):
            return False, "Tool params must be a dictionary"
        if self.max_params is not None and len(params) > self.max_params:
            return False, f"Tool call has {len(params)} params; max is {self.max_params}"
        if self.max_total_chars is not None:
            serialised = self._stable_text(params)
            if len(serialised) > self.max_total_chars:
                return False, f"Tool params exceed {self.max_total_chars} characters"

        rules = self._rules_for(tool_name)
        if not self.allow_extra_params:
            extras = sorted(self._unexpected_param_paths(params, set(rules)))
            if extras:
                return False, f"Unexpected params: {', '.join(extras[:10])}"

        for param, rule in rules.items():
            present, value, ambiguous = self._get_path(params, param)
            if ambiguous:
                return False, f"Param '{param}' has ambiguous literal and nested values"
            if not present:
                if rule.get("required"):
                    return False, f"Required param '{param}' is missing"
                continue
            if value is None:
                if rule.get("nullable"):
                    continue
                if rule.get("required") or rule.get("type") is not None:
                    return False, f"Param '{param}' may not be null"
                continue

            expected_type = rule.get("type")
            bool_mismatch = (
                isinstance(value, bool)
                and expected_type is not None
                and expected_type is not bool
                and not (isinstance(expected_type, tuple) and bool in expected_type)
            )
            if expected_type is not None and (
                bool_mismatch or not isinstance(value, expected_type)
            ):
                return (
                    False,
                    f"Param '{param}': expected {self._type_name(expected_type)}, "
                    f"got {type(value).__name__}",
                )

            # Numeric bounds — coerce numeric strings, fail closed on non-numbers.
            if "max" in rule or "min" in rule:
                num = self._as_number(value)
                if num is None:
                    return False, f"Param '{param}' must be numeric, got {type(value).__name__}"
                if "max" in rule and num > rule["max"]:
                    return False, f"Param '{param}' exceeds max {rule['max']}"
                if "min" in rule and num < rule["min"]:
                    return False, f"Param '{param}' is below min {rule['min']}"

            # Length / pattern apply to the string form so a non-str can't dodge them.
            sval = value if isinstance(value, str) else self._stable_text(value)
            if "maxlen" in rule and len(sval) > rule["maxlen"]:
                return False, f"Param '{param}' length {len(sval)} exceeds maxlen {rule['maxlen']}"
            if "minlen" in rule and len(sval) < rule["minlen"]:
                return False, f"Param '{param}' length {len(sval)} is below minlen {rule['minlen']}"

            if "pattern" in rule and not re.fullmatch(rule["pattern"], sval):
                return False, f"Param '{param}' does not match required pattern"

            choices = rule.get("choices", rule.get("enum"))
            if choices is not None and value not in choices:
                return False, f"Param '{param}' is not one of the allowed choices"

            predicate = rule.get("predicate")
            if predicate is not None:
                outcome = predicate(value)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                if isinstance(outcome, str):
                    return False, outcome
                if not outcome:
                    return False, f"Param '{param}' failed its custom predicate"

        return True, ""

    @staticmethod
    def _policy_result(
        outcome: bool | str | ShieldResult | None,
        default_reason: str,
    ) -> tuple[bool, str, str]:
        if isinstance(outcome, ShieldResult):
            return (
                outcome.allowed,
                outcome.reason or default_reason,
                outcome.reason_code or "TOOL_AUTHORIZATION_DENIED",
            )
        if isinstance(outcome, str):
            return False, outcome, "TOOL_AUTHORIZATION_DENIED"
        if outcome is False:
            return False, default_reason, "TOOL_AUTHORIZATION_DENIED"
        if outcome is True or outcome is None:
            return True, "", ""
        raise TypeError("authorization callbacks must return bool, str, ShieldResult, or None")

    def _handle_violation(
        self,
        reason: str,
        code: str,
        tool_name: str,
        ctx: SessionContext,
    ) -> ShieldResult:
        ctx.metadata["tool_policy_violation"] = {
            "tool_name": tool_name,
            "reason_code": code,
        }
        if self.on_violation == "warn":
            warnings.warn(f"[AgentGuard ToolValidator] {reason}", stacklevel=4)
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=False, reason=reason, reason_code=code)

    async def scan_tool_call(
        self, tool_name: str, params: dict[str, Any], ctx: SessionContext
    ) -> ShieldResult:
        name_ok, name_reason = self._name_check(tool_name)
        if not name_ok:
            return self._handle_violation(name_reason, "TOOL_NOT_ALLOWED", tool_name, ctx)

        if self.require_user_id and not ctx.user_id:
            return self._handle_violation(
                "An authenticated user_id is required for tool execution",
                "TOOL_IDENTITY_REQUIRED",
                tool_name,
                ctx,
            )

        params_ok, params_reason = await self._param_check(tool_name, params)
        if not params_ok:
            return self._handle_violation(params_reason, "TOOL_PARAM_INVALID", tool_name, ctx)

        if self.authorize is not None:
            outcome = self.authorize(tool_name, params, ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            allowed, reason, code = self._policy_result(
                outcome, f"Authorization denied for tool {tool_name!r}"
            )
            if not allowed:
                return self._handle_violation(reason, code, tool_name, ctx)

        for pattern, validator in self.validators.items():
            if not fnmatch.fnmatchcase(tool_name.casefold(), pattern.casefold()):
                continue
            outcome = validator(params, ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            allowed, reason, code = self._policy_result(
                outcome, f"Policy validation denied tool {tool_name!r}"
            )
            if not allowed:
                return self._handle_violation(reason, code, tool_name, ctx)

        return ShieldResult(allowed=True)
