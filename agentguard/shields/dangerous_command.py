"""Pre-execution gate for destructive shell, SQL, code, and path arguments.

An agent holding a ``run_bash``, ``execute_sql``, or ``read_file`` tool can be
talked into destroying data without ever tripping a content shield. The tool name
is on the allowlist, the parameter shape is valid, the string holds no secret, no
PII, and no injection phrasing — the payload *is* the argument. That is OWASP
ASI05 (Unexpected Code Execution), and no other shield in this library looks at
what a command argument actually says.

:class:`DangerousCommandShield` is the "validation gate" control: it runs at
:meth:`~agentguard.core.base_shield.BaseShield.scan_tool_call`, immediately
before execution and after argument sanitization, and denies arguments that
request irreversible or privilege-escalating operations. It covers four families:

**Shell** — recursive root deletes, raw block-device writes, filesystem
creation, fork bombs, pipe-to-shell installers, history/audit destruction,
``sudo``, host power control.

**SQL** — ``DROP``/``TRUNCATE``, unqualified ``DELETE``/``UPDATE`` (the classic
"forgot the WHERE clause" catastrophe), privilege grants, and the engine
escape hatches (``xp_cmdshell``, ``COPY … FROM PROGRAM``, ``INTO OUTFILE``).

**Code** — ``eval``/``exec``, ``os.system``, ``shell=True``, ``pickle.loads``,
``child_process``, and other direct interpreter sinks.

**Paths** — traversal sequences and sensitive targets such as ``/etc/shadow``,
``~/.ssh/id_rsa``, ``.aws/credentials``, and ``/proc/self/environ``.

Two things this is **not**. It is not a sandbox: a blocklist over a
Turing-complete shell is evadable in principle (``$IFS``, variable splicing,
``base64 -d | sh``), so it belongs behind container isolation and scoped
credentials, not in front of them. And it is not a substitute for the stronger
control, which is an allowlist — set ``allowed_commands`` to enumerate the
executables a tool may run, and every segment of a chained command is checked
against it::

    shield = DangerousCommandShield(allowed_commands=["git", "ls", "cat"])
    guard = Guard(shields=[shield])
    await guard.scan_tool_arguments("run_bash", {"command": "ls; rm -rf /"})
    # GuardBlockedError: COMMAND_NOT_ALLOWED — 'rm' is not permitted

In allowlist mode a command substitution (``$(…)`` or backticks) is denied
outright, because the executable it resolves to cannot be known before it runs.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import unicodedata
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

# Parameter names that carry executable intent. Matched on word boundaries so
# ``command``/``sql_query``/``file_path`` hit but ``recommendation`` does not.
_COMMAND_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:cmd|command|commands|script|scripts|shell|bash|sh|exec|"
    r"code|snippet|program|expression|eval|statement|query|sql|argv|args|"
    r"path|paths|file|files|filename|filepath|dir|directory|target)(?:$|[_-])"
)

# Shell operators that separate one executable from the next.
_SHELL_SPLIT_RE = re.compile(r"(?:\|\||&&|[;\n\r|&])")
_SUBSTITUTION_RE = re.compile(r"\$\(|`|\$\{[^}]*\}")

_SHELL_PATTERNS: tuple[tuple[str, str], ...] = (
    # Recursive force delete aimed at a root-ish target.
    (r"\brm\s+(?:-[a-z]*\s+)*-[a-z]*[rR][a-z]*[fF]|\brm\s+(?:-[a-z]*\s+)*-[a-z]*[fF][a-z]*[rR]",
     "recursive_delete"),
    # The long-form spelling is the same operation; matching only short flags
    # would make `--recursive --force` a one-word bypass. Either order counts.
    (r"\brm\b(?=[^\n;&|]*\s--(?:recursive|dir))(?=[^\n;&|]*\s--force)", "recursive_delete"),
    (r"\brm\b[^\n;&|]*\s--no-preserve-root\b", "recursive_delete"),
    (r"\brm\s+(?:-\S+\s+)*(?:/|/\*|~|~/|\$HOME|\.)\s*(?:$|[;&|])", "recursive_delete"),
    (r"\b(?:find|xargs)\b[^\n]*\s-delete\b", "recursive_delete"),
    # Raw device writes and filesystem creation destroy data irrecoverably.
    (r"\bdd\b[^\n]*\bof=\s*/dev/", "device_write"),
    (r"\bmkfs(?:\.[a-z0-9]+)?\b", "device_write"),
    (r">\s*/dev/(?:sd[a-z]|nvme\d|hd[a-z]|disk\d)", "device_write"),
    (r"\b(?:shred|wipefs|blkdiscard)\b", "device_write"),
    # Fork bomb.
    (r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*:", "fork_bomb"),
    # Fetch-and-run: the remote content is the payload.
    (r"\b(?:curl|wget|fetch)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b", "remote_execution"),
    (r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:python|perl|ruby|node)\b",
     "remote_execution"),
    # Privilege escalation and world-writable permissions.
    (r"\bsudo\b|\bdoas\b|\bsu\s+(?:-|root)\b", "privilege_escalation"),
    (r"\bchmod\s+(?:-[a-zA-Z]+\s+)*(?:777|a\+rwx|o\+w)\b", "permission_change"),
    (r"\bchown\s+(?:-[a-zA-Z]+\s+)*root\b", "permission_change"),
    (r"\bsetcap\b|\bvisudo\b", "privilege_escalation"),
    # Destroying the record of what happened.
    (r"\bhistory\s+-c\b|>\s*(?:~/)?\.(?:bash|zsh)_history\b", "audit_destruction"),
    (r"\bjournalctl\b[^\n]*--vacuum|\bauditctl\s+-D\b", "audit_destruction"),
    (r"\brm\b[^\n]*/var/log", "audit_destruction"),
    # Host control and mass signals.
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "host_control"),
    (r"\bkill(?:all)?\s+-9\s+(?:-1|1)\b", "host_control"),
    (r"\biptables\s+-F\b|\bufw\s+disable\b|\bsystemctl\s+stop\s+(?:firewalld|auditd)\b",
     "defense_disable"),
    # Credential and key harvesting via archive/copy of whole home dirs.
    (r"\b(?:scp|rsync)\b[^\n]*\s(?:~|\$HOME)/\.ssh\b", "credential_exfiltration"),
)

# Engine features that reach the operating system. These run *without* the
# statement-shape gate below: the tokens are specific enough that they cannot
# plausibly appear in prose, and the statements that carry them do not always
# contain a gated keyword (``EXEC xp_cmdshell``, ``LOAD DATA INFILE``).
_SQL_ENGINE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bxp_cmdshell\b|\bsp_oacreate\b", "engine_escape"),
    (r"\bcopy\b[^\n]*\bfrom\s+program\b", "engine_escape"),
    (r"\b(?:into\s+(?:out|dump)file|load\s+data\s+infile)\b", "engine_escape"),
    # No trailing \b on the call forms: the next character is "(", so a word
    # boundary there would never match.
    (r"\bload_file\s*\(", "engine_escape"),
    (r"\bpg_read_file\s*\(|\bpg_ls_dir\s*\(", "engine_escape"),
    (r"\bcreate\s+(?:or\s+replace\s+)?function\b[^\n]*\blanguage\s+(?:c|plpythonu)\b",
     "engine_escape"),
)

_SQL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdrop\s+(?:table|database|schema|index|view|user|role)\b", "schema_destruction"),
    (r"\btruncate\s+(?:table\s+)?\w", "schema_destruction"),
    (r"\bdrop\s+if\s+exists\b", "schema_destruction"),
    (r"\balter\s+(?:user|role|login)\b", "privilege_change"),
    (r"\bgrant\s+(?:all|super|dba)\b", "privilege_change"),
    (r"\bcreate\s+(?:user|role)\b[^\n]*\bsuperuser\b", "privilege_change"),
)

# An unqualified write hits every row; this is the "forgot the WHERE" class.
_SQL_UNQUALIFIED_DELETE_RE = re.compile(
    r"\bdelete\s+from\s+[\w.\"'`\[\]]+\s*(?:;|$)", re.IGNORECASE
)
_SQL_UNQUALIFIED_UPDATE_RE = re.compile(r"\bupdate\s+[\w.\"'`\[\]]+\s+set\b", re.IGNORECASE)
_SQL_WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
_SQL_SHAPE_RE = re.compile(
    r"(?i)\b(?:select|insert|update|delete|drop|truncate|alter|grant|create|copy)\b"
)

_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:eval|exec)\s*\(", "interpreter_sink"),
    (r"\b__import__\s*\(|\bcompile\s*\(", "interpreter_sink"),
    (r"\bos\.(?:system|popen|execv?p?e?)\s*\(", "interpreter_sink"),
    (r"\bsubprocess\.(?:run|call|Popen|check_output|check_call)\b", "interpreter_sink"),
    (r"\bshell\s*=\s*True\b", "interpreter_sink"),
    (r"\b(?:pickle|dill|cPickle|marshal)\.loads?\s*\(", "unsafe_deserialization"),
    (r"\byaml\.load\s*\((?![^)]*Safe)", "unsafe_deserialization"),
    (r"\bchild_process\b|\brequire\s*\(\s*['\"]child_process['\"]\s*\)", "interpreter_sink"),
    (r"\bnew\s+Function\s*\(|\bFunction\s*\(\s*['\"]", "interpreter_sink"),
    (r"\b(?:Runtime\.getRuntime\(\)\.exec|ProcessBuilder)\b", "interpreter_sink"),
    (r"\bsetattr\s*\(\s*__builtins__|\bglobals\s*\(\s*\)\s*\[", "interpreter_sink"),
)

_SENSITIVE_PATHS: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
    "/etc/gshadow",
    "/root/.ssh",
    "/proc/self/environ",
    "/var/run/secrets",
    "/run/secrets",
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".ssh/id_ecdsa",
    ".ssh/authorized_keys",
    ".aws/credentials",
    ".config/gcloud/credentials",
    ".kube/config",
    ".docker/config.json",
    ".netrc",
    ".pgpass",
    ".git-credentials",
    "id_rsa",
    "windows/system32/config/sam",
    "windows/system32/config/system",
)

_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)|%2e%2e(?:%2f|%5c|/|\\)", re.IGNORECASE)

_OVERRIDE_KEYS = frozenset(
    {
        "allowed_commands",
        "blocked_commands",
        "check_shell",
        "check_sql",
        "check_code",
        "check_paths",
        "require_sql_where",
        "sensitive_paths",
        "block_traversal",
        "inspect_all_strings",
        "on_violation",
    }
)


class _ArgumentStructureError(ValueError):
    """Traversal budget exceeded; the caller converts this into a denial."""


class DangerousCommandShield(BaseShield):
    """Deny tool arguments that request destructive or privileged operations.

    Parameters
    ----------
    allowed_commands:
        Executable basenames a shell-style argument may invoke. When set, this
        becomes an allowlist and every segment of a chained command must match;
        anything else is denied. ``None`` (default) runs blocklist checks only.
        The allowlist is the stronger control — prefer it where the set of
        legitimate commands is known.
    blocked_commands:
        Executable basenames to deny outright, checked even in allowlist mode.
    check_shell / check_sql / check_code / check_paths:
        Enable each detection family. All on by default.
    require_sql_where:
        Deny ``DELETE``/``UPDATE`` statements with no ``WHERE`` clause, which
        rewrite every row in the table. Default ``True``.
    sensitive_paths:
        Additional path fragments to deny, matched case-insensitively anywhere
        in an argument. Supplements the built-in credential/system list.
    block_traversal:
        Deny ``../`` sequences and their percent-encodings. Default ``True``.
    inspect_all_strings:
        Inspect every nested string, not only values under command-like keys.
        Default ``True``: an attacker picks the parameter name, so restricting
        inspection to expected names is an assumption the attacker controls.
    additional_command_keys:
        Application-specific parameter names that always carry executable text.
    tool_policies:
        Per-tool overrides keyed by case-insensitive glob, so a dedicated
        ``run_migration`` tool can permit DDL that ``run_bash`` may not. Keys
        mirror the arguments above.
    exempt_tools:
        Glob patterns for tools this shield skips entirely.
    max_argument_depth / max_argument_nodes:
        Fail-closed traversal budgets; content beyond them is never skipped.
    on_violation:
        ``"block"`` (default) or ``"warn"``.

    Notes
    -----
    Arguments are NFKC-normalized with format characters stripped before
    matching, so width and zero-width tricks do not slip past. This remains a
    static text check on a Turing-complete surface: treat it as defense in
    depth behind sandboxing, least-privilege credentials, and human approval
    for irreversible actions, never as the sole barrier.
    """

    # This is a pre-execution authorization decision on the whole call, not a
    # per-string rewrite, so the argument-DLP phase is a no-op for this shield.
    scan_tool_arguments_as_input = False

    def __init__(
        self,
        *,
        allowed_commands: Iterable[str] | None = None,
        blocked_commands: Iterable[str] | None = None,
        check_shell: bool = True,
        check_sql: bool = True,
        check_code: bool = True,
        check_paths: bool = True,
        require_sql_where: bool = True,
        sensitive_paths: Iterable[str] | None = None,
        block_traversal: bool = True,
        inspect_all_strings: bool = True,
        additional_command_keys: Iterable[str] | None = None,
        tool_policies: Mapping[str, Mapping[str, Any]] | None = None,
        exempt_tools: Iterable[str] | None = None,
        max_argument_depth: int = 32,
        max_argument_nodes: int = 10_000,
        max_argument_chars: int = 100_000,
        on_violation: Literal["block", "warn"] = "block",
    ) -> None:
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'")
        for name, flag in (
            ("check_shell", check_shell),
            ("check_sql", check_sql),
            ("check_code", check_code),
            ("check_paths", check_paths),
            ("require_sql_where", require_sql_where),
            ("block_traversal", block_traversal),
            ("inspect_all_strings", inspect_all_strings),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be a bool")
        for name, value in (
            ("max_argument_depth", max_argument_depth),
            ("max_argument_nodes", max_argument_nodes),
            ("max_argument_chars", max_argument_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be >= 1")

        self.allowed_commands = self._normalize_commands(allowed_commands, "allowed_commands")
        self.blocked_commands = self._normalize_commands(blocked_commands, "blocked_commands") or ()
        self.check_shell = check_shell
        self.check_sql = check_sql
        self.check_code = check_code
        self.check_paths = check_paths
        self.require_sql_where = require_sql_where
        self.sensitive_paths = _SENSITIVE_PATHS + self._normalize_paths(sensitive_paths)
        self.block_traversal = block_traversal
        self.inspect_all_strings = inspect_all_strings
        self.additional_command_keys = self._normalize_paths(additional_command_keys)
        self.exempt_tools = self._normalize_paths(exempt_tools)
        self.max_argument_depth = max_argument_depth
        self.max_argument_nodes = max_argument_nodes
        self.max_argument_chars = max_argument_chars
        self.on_violation = on_violation
        self.tool_policies = self._validate_tool_policies(tool_policies)

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_commands(
        commands: Iterable[str] | None, field: str
    ) -> tuple[str, ...] | None:
        if commands is None:
            return None
        if isinstance(commands, str):
            raise ValueError(f"{field} must be a sequence of strings, not a single string")
        normalized = []
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"{field} entries must be non-empty strings")
            normalized.append(command.strip().casefold())
        return tuple(normalized)

    @staticmethod
    def _normalize_paths(values: Iterable[str] | None) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            raise ValueError("expected a sequence of strings, not a single string")
        normalized = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("entries must be non-empty strings")
            normalized.append(value.strip().casefold())
        return tuple(normalized)

    def _validate_tool_policies(
        self, tool_policies: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, dict[str, Any]]:
        if tool_policies is None:
            return {}
        if not isinstance(tool_policies, Mapping):
            raise TypeError("tool_policies must be a mapping")
        validated: dict[str, dict[str, Any]] = {}
        for pattern, overrides in tool_policies.items():
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("tool_policies keys must be non-empty strings")
            if not isinstance(overrides, Mapping):
                raise TypeError(f"tool_policies[{pattern!r}] must be a mapping")
            unknown = set(overrides) - _OVERRIDE_KEYS
            if unknown:
                raise ValueError(
                    f"tool_policies[{pattern!r}] has unsupported keys: {sorted(unknown)}"
                )
            # Build a throwaway shield so an invalid override fails at
            # construction rather than on the first matching tool call.
            type(self)(**overrides)  # type: ignore[arg-type]
            validated[pattern.casefold()] = dict(overrides)
        return validated

    def _policy_for(self, tool_name: str) -> dict[str, Any]:
        policy: dict[str, Any] = {
            "allowed_commands": self.allowed_commands,
            "blocked_commands": self.blocked_commands,
            "check_shell": self.check_shell,
            "check_sql": self.check_sql,
            "check_code": self.check_code,
            "check_paths": self.check_paths,
            "require_sql_where": self.require_sql_where,
            "sensitive_paths": self.sensitive_paths,
            "block_traversal": self.block_traversal,
            "inspect_all_strings": self.inspect_all_strings,
            "on_violation": self.on_violation,
        }
        name = tool_name.casefold()
        for pattern, overrides in self.tool_policies.items():
            if not fnmatch.fnmatchcase(name, pattern):
                continue
            for key, value in overrides.items():
                if key == "allowed_commands":
                    policy[key] = self._normalize_commands(value, key)
                elif key == "blocked_commands":
                    policy[key] = self._normalize_commands(value, key) or ()
                elif key == "sensitive_paths":
                    policy[key] = _SENSITIVE_PATHS + self._normalize_paths(value)
                else:
                    policy[key] = value
        return policy

    # ------------------------------------------------------------------ #
    # Shield hook                                                          #
    # ------------------------------------------------------------------ #

    async def scan_tool_call(
        self, tool_name: str, params: dict, ctx: SessionContext
    ) -> ShieldResult:
        """Inspect arguments immediately before the tool runs."""
        if not isinstance(tool_name, str) or not tool_name:
            return self._violation(
                "Tool name must be a non-empty string", "TOOL_NAME_INVALID", "block"
            )
        name = tool_name.casefold()
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exempt_tools):
            return ShieldResult(allowed=True)

        policy = self._policy_for(tool_name)
        try:
            candidates = self._collect(params, policy)
        except _ArgumentStructureError as exc:
            return self._violation(
                f"Tool arguments could not be fully inspected: {exc}",
                "DANGEROUS_ARGUMENT_UNINSPECTABLE",
                policy["on_violation"],
            )

        for key, value in candidates:
            finding = self._inspect(value, policy)
            if finding is None:
                continue
            reason_code, category, detail = finding
            where = f" in parameter {key!r}" if key else ""
            return self._violation(
                f"Tool {tool_name!r} argument{where} was denied ({category}): {detail}",
                reason_code,
                policy["on_violation"],
            )
        return ShieldResult(allowed=True)

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def _inspect(
        self, text: str, policy: dict[str, Any]
    ) -> tuple[str, str, str] | None:
        normalized = self._normalize(text)
        if not normalized.strip():
            return None

        # Families are ordered most-specific first, because the reason code is
        # what an operator reads to understand *which capability* was abused.
        # A path mention is the least specific signal of the set: nearly every
        # payload below also names a path, so checking paths first would report
        # SENSITIVE_PATH_ACCESS for a SQL engine escape and bury the real
        # finding. It therefore runs last, as the fallback it actually is.
        if policy["check_sql"]:
            for pattern, category in _SQL_ENGINE_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return (
                        "DANGEROUS_SQL_STATEMENT",
                        category,
                        "the statement uses a database feature that reaches the host",
                    )

        if policy["check_shell"]:
            allowed = policy["allowed_commands"]
            blocked = policy["blocked_commands"]
            if allowed is not None or blocked:
                finding = self._command_finding(normalized, allowed, blocked)
                if finding is not None:
                    return finding

        if policy["check_code"]:
            for pattern, category in _CODE_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return (
                        "DANGEROUS_CODE_EXECUTION",
                        category,
                        "the argument reaches an interpreter or deserialization sink",
                    )

        if policy["check_shell"]:
            for pattern, category in _SHELL_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return (
                        "DANGEROUS_SHELL_COMMAND",
                        category,
                        "the argument requests an irreversible or privileged shell operation",
                    )

        if policy["check_sql"] and _SQL_SHAPE_RE.search(normalized):
            for pattern, category in _SQL_PATTERNS:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return (
                        "DANGEROUS_SQL_STATEMENT",
                        category,
                        "the statement destroys data, changes privileges, or reaches the host",
                    )
            if policy["require_sql_where"] and not _SQL_WHERE_RE.search(normalized):
                if _SQL_UNQUALIFIED_DELETE_RE.search(normalized):
                    return (
                        "DANGEROUS_SQL_STATEMENT",
                        "unqualified_delete",
                        "DELETE with no WHERE clause removes every row in the table",
                    )
                if _SQL_UNQUALIFIED_UPDATE_RE.search(normalized):
                    return (
                        "DANGEROUS_SQL_STATEMENT",
                        "unqualified_update",
                        "UPDATE with no WHERE clause rewrites every row in the table",
                    )

        if policy["check_paths"]:
            finding = self._path_finding(normalized, policy)
            if finding is not None:
                return finding
        return None

    def _path_finding(
        self, text: str, policy: dict[str, Any]
    ) -> tuple[str, str, str] | None:
        lowered = text.casefold().replace("\\", "/")
        for fragment in policy["sensitive_paths"]:
            if fragment.replace("\\", "/") in lowered:
                return (
                    "SENSITIVE_PATH_ACCESS",
                    "sensitive_path",
                    f"the argument references {fragment!r}",
                )
        if policy["block_traversal"] and _TRAVERSAL_RE.search(text):
            return (
                "SENSITIVE_PATH_ACCESS",
                "path_traversal",
                "the argument contains a parent-directory traversal sequence",
            )
        return None

    def _command_finding(
        self, text: str, allowed: tuple[str, ...] | None, blocked: tuple[str, ...]
    ) -> tuple[str, str, str] | None:
        """Check every executable in a possibly-chained command line."""
        if allowed is not None and _SUBSTITUTION_RE.search(text):
            return (
                "COMMAND_NOT_ALLOWED",
                "command_substitution",
                "a substituted command cannot be resolved before it runs",
            )
        for segment in _SHELL_SPLIT_RE.split(text):
            executable = self._leading_executable(segment)
            if executable is None:
                continue
            if executable == "__unparseable__":
                if allowed is not None:
                    return (
                        "COMMAND_NOT_ALLOWED",
                        "unparseable_command",
                        "the argument could not be parsed into executables",
                    )
                continue
            if executable in blocked:
                return (
                    "COMMAND_NOT_ALLOWED",
                    "blocked_command",
                    f"{executable!r} is on the blocked-command list",
                )
            if allowed is not None and executable not in allowed:
                return (
                    "COMMAND_NOT_ALLOWED",
                    "not_allowlisted",
                    f"{executable!r} is not in the permitted command list",
                )
        return None

    @staticmethod
    def _leading_executable(segment: str) -> str | None:
        """Basename of the executable a command segment invokes."""
        stripped = segment.strip()
        if not stripped:
            return None
        try:
            tokens = shlex.split(stripped, comments=False)
        except ValueError:
            # Unbalanced quoting: refuse to guess what this would run.
            return "__unparseable__"
        for token in tokens:
            # Skip leading VAR=value assignments and shell keywords.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
                continue
            if token in ("then", "do", "else", "elif", "fi", "done", "!", "{", "}", "("):
                continue
            basename = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
            return basename or None
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """NFKC-fold and strip format characters so evasion tricks collapse."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Cf" or character in "\t\n\r"
        )
        return normalized

    # ------------------------------------------------------------------ #
    # Traversal                                                            #
    # ------------------------------------------------------------------ #

    def _collect(
        self, params: Any, policy: dict[str, Any]
    ) -> list[tuple[str | None, str]]:
        """Gather inspectable strings with the key that carried them."""
        found: list[tuple[str | None, str]] = []
        budget = {"nodes": 0, "chars": 0}
        seen: set[int] = set()

        def is_command_key(key: str | None) -> bool:
            if key is None:
                return False
            lowered = key.casefold()
            if lowered in self.additional_command_keys:
                return True
            return bool(_COMMAND_KEY_RE.search(key))

        def walk(node: Any, key: str | None, depth: int) -> None:
            budget["nodes"] += 1
            if depth > self.max_argument_depth:
                raise _ArgumentStructureError("maximum argument depth exceeded")
            if budget["nodes"] > self.max_argument_nodes:
                raise _ArgumentStructureError("maximum argument node count exceeded")
            if isinstance(node, str):
                budget["chars"] += len(node)
                if budget["chars"] > self.max_argument_chars:
                    raise _ArgumentStructureError("maximum argument character count exceeded")
                if policy["inspect_all_strings"] or is_command_key(key):
                    found.append((key, node))
                return
            if isinstance(node, Mapping):
                if id(node) in seen:
                    return
                seen.add(id(node))
                for child_key, child in node.items():
                    label = child_key if isinstance(child_key, str) else key
                    walk(child, label, depth + 1)
                return
            if isinstance(node, (list, tuple, set, frozenset)):
                if id(node) in seen:
                    return
                seen.add(id(node))
                for child in node:
                    walk(child, key, depth + 1)
            # Other scalars carry no executable text.

        walk(params, None, 0)
        return found

    def _violation(self, reason: str, reason_code: str, mode: str) -> ShieldResult:
        if mode == "warn":
            warnings.warn(f"[AgentGuard DangerousCommandShield] {reason}", stacklevel=2)
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=False, reason=reason, reason_code=reason_code)
