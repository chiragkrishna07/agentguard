import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.network_policy import NetworkPolicyShield


async def scan(shield, params, tool="fetch"):
    return await shield.scan_tool_call(tool, params, SessionContext())


@pytest.mark.asyncio
async def test_allows_public_https_by_default():
    result = await scan(NetworkPolicyShield(), {"url": "https://example.com/path"})
    assert result.allowed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "file:///etc/passwd",
        "gopher://example.com/_payload",
        "javascript:alert(1)",
    ],
)
async def test_blocks_non_https_schemes(url):
    result = await scan(NetworkPolicyShield(), {"target_url": url})
    assert not result.allowed
    assert result.reason_code == "NETWORK_POLICY_VIOLATION"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/admin",
        "https://[::1]/admin",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.4/internal",
        "https://2130706433/admin",
        "https://0x7f000001/admin",
        "https://127.1/admin",
        "https://localhost/admin",
        "https://metadata.internal/token",
        "https://intranet/path",
    ],
)
async def test_blocks_ssrf_and_internal_destinations(url):
    result = await scan(NetworkPolicyShield(), {"url": url})
    assert not result.allowed, url


@pytest.mark.asyncio
async def test_blocks_credentials_in_url_without_echoing_secret():
    result = await scan(
        NetworkPolicyShield(), {"url": "https://alice:topsecret@example.com/path"}
    )
    assert not result.allowed
    assert "topsecret" not in (result.reason or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path with-space",
        "https://example.com\\@127.0.0.1/admin",
        "https://example.com\n@127.0.0.1/admin",
        "prefix https://example.com/path",
    ],
)
async def test_url_fields_reject_parser_confusion_and_free_form_values(url):
    result = await scan(NetworkPolicyShield(), {"callback_url": url})
    assert not result.allowed


@pytest.mark.asyncio
async def test_scans_explicit_urls_nested_in_non_url_parameters():
    result = await scan(
        NetworkPolicyShield(),
        {"payload": {"links": ["read https://127.0.0.1/private next"]}},
    )
    assert not result.allowed


@pytest.mark.asyncio
async def test_nested_url_key_keeps_context_through_sequences():
    shield = NetworkPolicyShield()
    result = await scan(shield, {"url": [["127.0.0.1/admin"]]})
    assert not result.allowed


@pytest.mark.asyncio
async def test_ambiguous_destination_key_only_treats_network_like_values_as_urls():
    shield = NetworkPolicyShield()
    assert (await scan(shield, {"destination": "Delhi"})).allowed
    assert (await scan(shield, {"address": "12 Main Street"})).allowed
    assert (await scan(shield, {"target": "Mumbai"})).allowed
    assert not (await scan(shield, {"destination": "127.0.0.1/admin"})).allowed
    assert not (await scan(shield, {"remote": "metadata.internal/token"})).allowed
    assert not (await scan(shield, {"target": "169.254.169.254/latest/meta-data"})).allowed
    assert not (await scan(shield, {"origin": "localhost:8000"})).allowed


@pytest.mark.asyncio
async def test_ip_and_ip_address_fields_are_always_destinations():
    shield = NetworkPolicyShield()
    assert not (await scan(shield, {"ip": "127.0.0.1"})).allowed
    assert not (await scan(shield, {"source_ip_address": "10.0.0.1"})).allowed


@pytest.mark.asyncio
async def test_network_traversal_bounds_fail_closed_instead_of_skipping_deep_url():
    value = "https://127.0.0.1/private"
    for _ in range(22):
        value = {"wrapper": value}
    shield = NetworkPolicyShield(max_argument_depth=20)
    result = await scan(shield, value)
    assert not result.allowed
    assert result.reason_code == "NETWORK_ARGUMENT_STRUCTURE_INVALID"


@pytest.mark.asyncio
async def test_network_cycle_fails_closed():
    cyclic = {}
    cyclic["self"] = cyclic
    result = await scan(NetworkPolicyShield(), cyclic)
    assert not result.allowed
    assert result.reason_code == "NETWORK_ARGUMENT_STRUCTURE_INVALID"


@pytest.mark.asyncio
async def test_host_allowlist_is_boundary_safe():
    shield = NetworkPolicyShield(allowed_hosts=["api.example.com", "*.trusted.test"])
    assert (await scan(shield, {"url": "https://api.example.com/v1"})).allowed
    assert (await scan(shield, {"url": "https://x.trusted.test/v1"})).allowed
    assert not (await scan(shield, {"url": "https://api.example.com.evil.test"})).allowed


@pytest.mark.asyncio
async def test_string_scheme_host_config_and_camel_case_url_key():
    shield = NetworkPolicyShield(
        allowed_schemes="https",
        allowed_hosts="example.com",
    )
    assert (await scan(shield, {"callbackUrl": "example.com/hook"})).allowed
    assert not (await scan(shield, {"callbackUrl": "evil.test/hook"})).allowed


@pytest.mark.asyncio
async def test_per_tool_policy_enforces_least_privilege_egress():
    shield = NetworkPolicyShield(
        allowed_hosts=["shared.example.com"],
        tool_policies={
            "weather_*": {"allowed_hosts": ["weather.example.com"]},
            "weather_admin": {"allowed_hosts": ["admin.weather.example.com"]},
        },
    )
    assert (
        await scan(shield, {"endpoint": "weather.example.com/v1"}, "weather_lookup")
    ).allowed
    assert not (
        await scan(shield, {"url": "https://shared.example.com"}, "weather_lookup")
    ).allowed
    assert (
        await scan(shield, {"url": "https://admin.weather.example.com"}, "weather_admin")
    ).allowed


@pytest.mark.asyncio
async def test_optional_resolver_blocks_public_name_resolving_private():
    shield = NetworkPolicyShield(host_resolver=lambda _host: ["127.0.0.1"])
    result = await scan(shield, {"url": "https://example.com"})
    assert not result.allowed
    assert "non-global" in (result.reason or "")


@pytest.mark.asyncio
async def test_url_count_limit_bounds_work():
    shield = NetworkPolicyShield(max_urls_per_call=1)
    result = await scan(
        shield,
        {"text": "https://example.com/a and https://example.org/b"},
    )
    assert not result.allowed
    assert result.reason_code == "NETWORK_URL_LIMIT_EXCEEDED"


def test_rejects_unknown_per_tool_policy_option():
    with pytest.raises(ValueError, match="unsupported policy"):
        NetworkPolicyShield(tool_policies={"fetch": {"surprise": True}})

    with pytest.raises(ValueError, match="must be >= 1"):
        NetworkPolicyShield(tool_policies={"fetch": {"max_urls_per_call": 0}})

    with pytest.raises(ValueError, match="host patterns"):
        NetworkPolicyShield(allowed_hosts=["https://example.com"])

    with pytest.raises(ValueError, match="allowed_schemes"):
        NetworkPolicyShield(allowed_schemes="https://")
