"""Outbound network policy enforcement for agent tool calls.

This shield is deliberately applied at the tool boundary.  Prompt filtering is
not an SSRF or egress control: an agent may produce a syntactically valid URL
that still points at instance metadata, loopback, or an internal service.
``NetworkPolicyShield`` validates URL-bearing tool arguments immediately before
execution and can apply a different egress allowlist to each tool.

The shield does not perform network I/O by default.  Applications that need DNS
resolution checks can provide ``host_resolver``; the actual HTTP client should
still pin the validated address (or enforce the same policy at the network
layer) to avoid DNS rebinding between validation and connection.
"""

from __future__ import annotations

import fnmatch
import inspect
import ipaddress
import re
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, Literal
from urllib.parse import SplitResult, urlsplit

from agentguard.core.base_shield import BaseShield, ShieldResult
from agentguard.core.session import SessionContext

HostResolver = Callable[[str], Iterable[str] | Awaitable[Iterable[str]]]

_EMBEDDED_URL_RE = re.compile(
    r"(?i)(?:(?:[a-z][a-z0-9+.-]*):/{2}|(?:file|data|javascript):)[^\s<>\"']+"
)
_URL_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:url|uri|endpoint|webhook|callback|redirect|host|hostname|"
    r"ip|ip_address|link|href|proxy|dsn)(?:$|[_-])"
)
_AMBIGUOUS_URL_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:destination|address|remote|source|server|target|origin)(?:$|[_-])"
)
_TRAILING_PUNCTUATION = ".,;!?)]}"
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")


class _NetworkStructureError(ValueError):
    pass


class NetworkPolicyShield(BaseShield):
    """Validate outbound URLs found in tool-call parameters.

    Parameters
    ----------
    allowed_schemes:
        Schemes allowed globally.  HTTPS-only is the safe default.
    allowed_hosts:
        Optional host allowlist. Entries may be exact names or glob patterns
        such as ``"*.example.com"``. ``None`` allows any public host.
    blocked_hosts:
        Host denylist, evaluated before the allowlist.
    allow_private_networks:
        Permit loopback, private, link-local, reserved and other non-global IP
        destinations. Disabled by default to prevent SSRF.
    allow_userinfo:
        Permit ``user:password@host`` URL authority components. Disabled by
        default because credentials are easily exfiltrated through URLs.
    block_unqualified_hosts:
        Reject single-label hosts such as ``intranet``. Enabled by default.
    inspect_all_strings:
        Inspect explicit URLs embedded anywhere in nested parameters. URL-like
        parameter names are always inspected, even for schemeless values.
    max_argument_depth / max_argument_nodes:
        Fail-closed traversal budgets. Content beyond them is never skipped.
    additional_url_keys:
        Application-specific parameter names that always contain destinations.
    tool_policies:
        Per-tool overrides keyed by case-insensitive glob. Supported override
        keys mirror the policy arguments above (except ``tool_policies`` and
        ``host_resolver``). This enables least-privilege egress per tool.
    host_resolver:
        Optional sync or async callback returning all resolved IP strings. Every
        address is checked. Resolver errors fail closed.
    """

    def __init__(
        self,
        *,
        allowed_schemes: Iterable[str] = ("https",),
        allowed_hosts: Iterable[str] | None = None,
        blocked_hosts: Iterable[str] | None = None,
        allow_private_networks: bool = False,
        allow_userinfo: bool = False,
        block_unqualified_hosts: bool = True,
        inspect_all_strings: bool = True,
        max_urls_per_call: int = 20,
        max_url_length: int = 2_048,
        max_argument_depth: int = 32,
        max_argument_nodes: int = 10_000,
        additional_url_keys: Iterable[str] | None = None,
        tool_policies: Mapping[str, Mapping[str, Any]] | None = None,
        host_resolver: HostResolver | None = None,
        on_violation: Literal["block", "warn"] = "block",
    ) -> None:
        raw_schemes = (allowed_schemes,) if isinstance(allowed_schemes, str) else allowed_schemes
        schemes = tuple(str(s).lower().rstrip(":") for s in raw_schemes)
        if not schemes or any(not re.fullmatch(r"[a-z][a-z0-9+.-]*", s) for s in schemes):
            raise ValueError("allowed_schemes must contain at least one scheme")
        if (
            isinstance(max_urls_per_call, bool)
            or not isinstance(max_urls_per_call, int)
            or max_urls_per_call < 1
        ):
            raise ValueError("max_urls_per_call must be >= 1")
        if (
            isinstance(max_url_length, bool)
            or not isinstance(max_url_length, int)
            or max_url_length < 1
        ):
            raise ValueError("max_url_length must be >= 1")
        for name, value in {
            "max_argument_depth": max_argument_depth,
            "max_argument_nodes": max_argument_nodes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be >= 1")
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'")
        if host_resolver is not None and not callable(host_resolver):
            raise ValueError("host_resolver must be callable or None")
        for name, value in {
            "allow_private_networks": allow_private_networks,
            "allow_userinfo": allow_userinfo,
            "block_unqualified_hosts": block_unqualified_hosts,
            "inspect_all_strings": inspect_all_strings,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")

        self.allowed_schemes = schemes
        self.allowed_hosts = self._normalise_patterns(allowed_hosts)
        self.blocked_hosts = self._normalise_patterns(blocked_hosts) or ()
        self.allow_private_networks = allow_private_networks
        self.allow_userinfo = allow_userinfo
        self.block_unqualified_hosts = block_unqualified_hosts
        self.inspect_all_strings = inspect_all_strings
        self.max_urls_per_call = max_urls_per_call
        self.max_url_length = max_url_length
        self.max_argument_depth = max_argument_depth
        self.max_argument_nodes = max_argument_nodes
        raw_url_keys = (
            (additional_url_keys,)
            if isinstance(additional_url_keys, str)
            else additional_url_keys or ()
        )
        self.additional_url_keys = frozenset(
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).casefold()
            for key in raw_url_keys
        )
        if any(not key for key in self.additional_url_keys):
            raise ValueError("additional_url_keys must contain non-empty names")
        self.tool_policies = {
            str(pattern): dict(policy) for pattern, policy in (tool_policies or {}).items()
        }
        self.host_resolver = host_resolver
        self.on_violation = on_violation
        self._validate_tool_policies()

    @staticmethod
    def _normalise_patterns(values: Iterable[str] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        raw_values = (values,) if isinstance(values, str) else values
        patterns = tuple(str(value).rstrip(".").lower() for value in raw_values)
        if any(not pattern or "://" in pattern or "/" in pattern for pattern in patterns):
            raise ValueError("host patterns must be hostnames/globs, not URLs")
        return patterns

    def _validate_tool_policies(self) -> None:
        supported = {
            "allowed_schemes",
            "allowed_hosts",
            "blocked_hosts",
            "allow_private_networks",
            "allow_userinfo",
            "block_unqualified_hosts",
            "inspect_all_strings",
            "max_urls_per_call",
            "max_url_length",
            "max_argument_depth",
            "max_argument_nodes",
        }
        for pattern, policy in self.tool_policies.items():
            if not pattern:
                raise ValueError("tool policy patterns must not be empty")
            unknown = set(policy) - supported
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported policy keys for {pattern!r}: {names}")
            if "allowed_schemes" in policy:
                raw = policy["allowed_schemes"]
                raw = (raw,) if isinstance(raw, str) else raw
                normalized_schemes = tuple(str(item).lower().rstrip(":") for item in raw)
                if not normalized_schemes or any(
                    not re.fullmatch(r"[a-z][a-z0-9+.-]*", item) for item in normalized_schemes
                ):
                    raise ValueError(f"allowed_schemes for {pattern!r} must not be empty")
                policy["allowed_schemes"] = normalized_schemes
            for key in ("allowed_hosts", "blocked_hosts"):
                if key in policy:
                    policy[key] = self._normalise_patterns(policy[key])
            for key in (
                "max_urls_per_call",
                "max_url_length",
                "max_argument_depth",
                "max_argument_nodes",
            ):
                if key in policy and (
                    isinstance(policy[key], bool)
                    or not isinstance(policy[key], int)
                    or policy[key] < 1
                ):
                    raise ValueError(f"{key} for {pattern!r} must be >= 1")
            for key in (
                "allow_private_networks",
                "allow_userinfo",
                "block_unqualified_hosts",
                "inspect_all_strings",
            ):
                if key in policy and not isinstance(policy[key], bool):
                    raise ValueError(f"{key} for {pattern!r} must be boolean")

    def _policy_for(self, tool_name: str) -> dict[str, Any]:
        policy: dict[str, Any] = {
            "allowed_schemes": self.allowed_schemes,
            "allowed_hosts": self.allowed_hosts,
            "blocked_hosts": self.blocked_hosts,
            "allow_private_networks": self.allow_private_networks,
            "allow_userinfo": self.allow_userinfo,
            "block_unqualified_hosts": self.block_unqualified_hosts,
            "inspect_all_strings": self.inspect_all_strings,
            "max_urls_per_call": self.max_urls_per_call,
            "max_url_length": self.max_url_length,
            "max_argument_depth": self.max_argument_depth,
            "max_argument_nodes": self.max_argument_nodes,
        }
        name = tool_name.casefold()
        # Multiple matching policies intentionally compose in declaration
        # order; a later, more-specific entry can override a broad default.
        for pattern, override in self.tool_policies.items():
            if fnmatch.fnmatchcase(name, pattern.casefold()):
                policy.update(override)

        raw_schemes = policy["allowed_schemes"]
        raw_schemes = (raw_schemes,) if isinstance(raw_schemes, str) else raw_schemes
        policy["allowed_schemes"] = tuple(str(s).lower().rstrip(":") for s in raw_schemes)
        policy["allowed_hosts"] = self._normalise_patterns(policy["allowed_hosts"])
        policy["blocked_hosts"] = self._normalise_patterns(policy["blocked_hosts"]) or ()
        return policy

    @staticmethod
    def _path_key(path: tuple[str, ...]) -> str | None:
        for component in reversed(path):
            if not component.isdigit():
                return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", component).casefold()
        return None

    @staticmethod
    def _looks_like_network_destination(text: str) -> bool:
        value = text.strip()
        if not value:
            return False
        if _EMBEDDED_URL_RE.search(value) or value.startswith("//"):
            return True
        if any(char.isspace() for char in value):
            return False
        authority = re.split(r"[/#?]", value, maxsplit=1)[0]
        host = authority.rsplit("@", 1)[-1].strip("[]")
        if host.count(":") == 1:
            host = host.split(":", 1)[0]
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            pass
        return (
            host.casefold() == "localhost"
            or "." in host
            or value.casefold().startswith(("intranet/", "metadata/"))
        )

    def _is_url_key(self, path: tuple[str, ...], text: str) -> bool:
        if not path:
            return False
        key = self._path_key(path)
        if key is None:
            return False
        if key in self.additional_url_keys or _URL_KEY_RE.search(key):
            return True
        return bool(
            _AMBIGUOUS_URL_KEY_RE.search(key)
            and self._looks_like_network_destination(text)
        )

    def _iter_strings(
        self,
        value: Any,
        path: tuple[str, ...] = (),
        *,
        depth: int = 0,
        seen: set[int] | None = None,
        nodes: list[int] | None = None,
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> Iterable[tuple[tuple[str, ...], str]]:
        max_depth = self.max_argument_depth if max_depth is None else max_depth
        max_nodes = self.max_argument_nodes if max_nodes is None else max_nodes
        nodes = [0] if nodes is None else nodes
        nodes[0] += 1
        if depth > max_depth:
            raise _NetworkStructureError(
                f"tool arguments exceed network inspection depth {max_depth}"
            )
        if nodes[0] > max_nodes:
            raise _NetworkStructureError(
                f"tool arguments exceed network inspection node limit {max_nodes}"
            )
        if isinstance(value, str):
            yield path, value
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            return

        seen = seen or set()
        if isinstance(value, Mapping):
            object_id = id(value)
            if object_id in seen:
                raise _NetworkStructureError("cyclic tool arguments are not supported")
            seen.add(object_id)
            try:
                for key, child in value.items():
                    yield from self._iter_strings(
                        child,
                        path + (str(key),),
                        depth=depth + 1,
                        seen=seen,
                        nodes=nodes,
                        max_depth=max_depth,
                        max_nodes=max_nodes,
                    )
            finally:
                seen.remove(object_id)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            object_id = id(value)
            if object_id in seen:
                raise _NetworkStructureError("cyclic tool arguments are not supported")
            seen.add(object_id)
            try:
                for index, child in enumerate(value):
                    yield from self._iter_strings(
                        child,
                        path + (str(index),),
                        depth=depth + 1,
                        seen=seen,
                        nodes=nodes,
                        max_depth=max_depth,
                        max_nodes=max_nodes,
                    )
            finally:
                seen.remove(object_id)

    @staticmethod
    def _candidate_urls(text: str, direct: bool) -> list[str]:
        if direct:
            # A URL-designated field is one destination, not free-form text.
            # Validate the complete value so parser-confusion suffixes cannot
            # hide outside the regex match consumed by this shield.
            return [text.strip()] if text.strip() else []
        candidates = [
            m.group(0).rstrip(_TRAILING_PUNCTUATION) for m in _EMBEDDED_URL_RE.finditer(text)
        ]
        return candidates

    @staticmethod
    def _parse_url(candidate: str) -> SplitResult:
        if candidate.startswith("//"):
            return urlsplit("https:" + candidate)
        parsed = urlsplit(candidate)
        if not parsed.scheme:
            # URL-valued parameters commonly omit a scheme. Treat them as
            # HTTPS destinations instead of silently skipping validation.
            parsed = urlsplit("https://" + candidate)
        return parsed

    @staticmethod
    def _canonical_host(raw_host: str) -> str:
        if "%" in raw_host:
            # Encoded host delimiters and IPv6 scope identifiers are unsafe at
            # a general outbound boundary and are easy to interpret differently
            # across HTTP clients.
            raise ValueError("encoded or scoped hosts are not allowed")
        host = raw_host.rstrip(".").casefold()
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("host is not valid IDNA") from exc
        if not host or any(ord(c) < 33 for c in host):
            raise ValueError("host is empty or contains control characters")
        return host

    @staticmethod
    def _legacy_numeric_ip(host: str) -> ipaddress.IPv4Address | None:
        """Recognise non-canonical IPv4 forms accepted by some clients."""
        try:
            if host.lower().startswith("0x"):
                return ipaddress.IPv4Address(int(host, 16))
            if host.isdigit():
                return ipaddress.IPv4Address(int(host, 10))
        except (ValueError, ipaddress.AddressValueError):
            return None

        labels = host.split(".")
        if not labels or not all(
            re.fullmatch(r"(?:0x[0-9a-f]+|0[0-7]*|[0-9]+)", p, re.I) for p in labels
        ):
            return None
        # Ambiguous short/octal/hex dotted forms vary between URL clients.
        # Resolve the common forms when possible; otherwise reject them later
        # as a suspicious numeric hostname.
        try:
            numbers = [
                int(
                    p,
                    16
                    if p.lower().startswith("0x")
                    else 8
                    if len(p) > 1 and p.startswith("0")
                    else 10,
                )
                for p in labels
            ]
            if len(numbers) == 2 and numbers[0] <= 255 and numbers[1] <= 0xFFFFFF:
                return ipaddress.IPv4Address((numbers[0] << 24) | numbers[1])
            if (
                len(numbers) == 3
                and numbers[0] <= 255
                and numbers[1] <= 255
                and numbers[2] <= 0xFFFF
            ):
                return ipaddress.IPv4Address((numbers[0] << 24) | (numbers[1] << 16) | numbers[2])
            if len(numbers) == 4 and all(n <= 255 for n in numbers):
                return ipaddress.IPv4Address(bytes(numbers))
        except (ValueError, ipaddress.AddressValueError):
            return None
        return None

    @classmethod
    def _as_ip(cls, host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        try:
            return ipaddress.ip_address(host)
        except ValueError:
            return cls._legacy_numeric_ip(host)

    @staticmethod
    def _host_matches(host: str, patterns: Iterable[str]) -> bool:
        return any(fnmatch.fnmatchcase(host, pattern) for pattern in patterns)

    def _check_host(self, host: str, policy: Mapping[str, Any]) -> str | None:
        blocked_hosts = policy["blocked_hosts"]
        allowed_hosts = policy["allowed_hosts"]
        if self._host_matches(host, blocked_hosts):
            return "destination host is denied"
        if allowed_hosts is not None and not self._host_matches(host, allowed_hosts):
            return "destination host is outside the allowlist"

        address = self._as_ip(host)
        if address is not None:
            if not policy["allow_private_networks"] and not address.is_global:
                return f"destination address {address} is not globally routable"
            return None

        if all(part.isdigit() or part.lower().startswith("0x") for part in host.split(".")):
            return "ambiguous numeric destination host is not allowed"
        if not policy["allow_private_networks"]:
            if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
                return "local destination host is not allowed"
            if policy["block_unqualified_hosts"] and "." not in host:
                return "unqualified destination host is not allowed"
        return None

    async def _validate_url(self, candidate: str, policy: Mapping[str, Any]) -> str | None:
        if len(candidate) > int(policy["max_url_length"]):
            return "destination URL exceeds the configured length limit"
        if "\\" in candidate or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in candidate
        ):
            return "destination URL contains unsafe whitespace, controls, or backslashes"
        try:
            parsed = self._parse_url(candidate)
            # Accessing port performs urllib's range and syntax validation.
            _ = parsed.port
        except ValueError:
            return "destination URL is malformed"

        scheme = parsed.scheme.casefold()
        if scheme not in policy["allowed_schemes"]:
            return f"URL scheme {scheme!r} is not allowed"
        if (parsed.username is not None or parsed.password is not None) and not policy[
            "allow_userinfo"
        ]:
            return "credentials in destination URLs are not allowed"
        if parsed.hostname is None:
            return "destination URL has no host"

        try:
            host = self._canonical_host(parsed.hostname)
        except ValueError as exc:
            return str(exc)
        reason = self._check_host(host, policy)
        if reason is not None:
            return reason

        if self.host_resolver is not None:
            try:
                resolved = self.host_resolver(host)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                addresses = list(resolved)
            except Exception:
                return "destination host resolution failed"
            if not addresses:
                return "destination host did not resolve"
            for raw_address in addresses:
                try:
                    address = ipaddress.ip_address(str(raw_address))
                except ValueError:
                    return "host resolver returned an invalid address"
                if not policy["allow_private_networks"] and not address.is_global:
                    return f"destination resolves to non-global address {address}"
        return None

    def _violation(
        self,
        reason: str,
        tool_name: str,
        ctx: SessionContext,
        *,
        code: str = "NETWORK_POLICY_VIOLATION",
    ) -> ShieldResult:
        # Do not retain the raw URL: it may contain query-string credentials.
        ctx.metadata["network_policy_violation"] = {
            "tool_name": tool_name,
            "reason_code": code,
        }
        if self.on_violation == "warn":
            warnings.warn(f"[AgentGuard NetworkPolicyShield] {reason}", stacklevel=4)
            return ShieldResult(allowed=True)
        return ShieldResult(allowed=False, reason=reason, reason_code=code)

    async def scan_tool_call(
        self, tool_name: str, params: dict[str, Any], ctx: SessionContext
    ) -> ShieldResult:
        policy = self._policy_for(tool_name)
        candidates: list[str] = []
        try:
            strings = self._iter_strings(
                params,
                max_depth=int(policy["max_argument_depth"]),
                max_nodes=int(policy["max_argument_nodes"]),
            )
            for path, text in strings:
                direct = self._is_url_key(path, text)
                if not direct and not policy["inspect_all_strings"]:
                    continue
                candidates.extend(self._candidate_urls(text, direct))
                if len(candidates) > int(policy["max_urls_per_call"]):
                    return self._violation(
                        "tool call contains too many outbound URL candidates",
                        tool_name,
                        ctx,
                        code="NETWORK_URL_LIMIT_EXCEEDED",
                    )
        except _NetworkStructureError as exc:
            return self._violation(
                str(exc),
                tool_name,
                ctx,
                code="NETWORK_ARGUMENT_STRUCTURE_INVALID",
            )

        for candidate in candidates:
            reason = await self._validate_url(candidate, policy)
            if reason is not None:
                return self._violation(reason, tool_name, ctx)
        return ShieldResult(allowed=True)
