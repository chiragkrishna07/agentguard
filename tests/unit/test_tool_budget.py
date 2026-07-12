import pytest

from agentguard.core.session import SessionContext
from agentguard.shields.tool_budget import ToolCallBudget


@pytest.mark.asyncio
async def test_total_session_budget_blocks_runaway_agent():
    shield = ToolCallBudget(
        max_calls_per_session=2,
        max_calls_per_tool=None,
        max_consecutive_identical=None,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("a", {"n": 1}, ctx)).allowed
    assert (await shield.scan_tool_call("b", {"n": 2}, ctx)).allowed
    result = await shield.scan_tool_call("c", {"n": 3}, ctx)
    assert not result.allowed
    assert result.reason_code == "TOOL_SESSION_BUDGET_EXCEEDED"


@pytest.mark.asyncio
async def test_per_tool_glob_budgets():
    shield = ToolCallBudget(
        max_calls_per_session=None,
        max_calls_per_tool={"read_*": 2, "write_*": 1},
        max_consecutive_identical=None,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("READ_FILE", {"p": "a"}, ctx)).allowed
    assert (await shield.scan_tool_call("read_file", {"p": "b"}, ctx)).allowed
    assert not (await shield.scan_tool_call("read_file", {"p": "c"}, ctx)).allowed
    assert (await shield.scan_tool_call("write_file", {"p": "a"}, SessionContext())).allowed


@pytest.mark.asyncio
async def test_detects_identical_call_loop_with_order_independent_dict_fingerprint():
    shield = ToolCallBudget(
        max_calls_per_session=None,
        max_calls_per_tool=None,
        max_consecutive_identical=2,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("search", {"q": "x", "n": 1}, ctx)).allowed
    assert (await shield.scan_tool_call("SEARCH", {"n": 1, "q": "x"}, ctx)).allowed
    result = await shield.scan_tool_call("search", {"q": "x", "n": 1}, ctx)
    assert not result.allowed
    assert result.reason_code == "TOOL_LOOP_DETECTED"


@pytest.mark.asyncio
async def test_different_call_resets_consecutive_counter():
    shield = ToolCallBudget(
        max_calls_per_session=None,
        max_calls_per_tool=None,
        max_consecutive_identical=1,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("search", {"q": "x"}, ctx)).allowed
    assert (await shield.scan_tool_call("search", {"q": "y"}, ctx)).allowed


@pytest.mark.asyncio
async def test_distinct_tool_budget():
    shield = ToolCallBudget(
        max_calls_per_session=None,
        max_calls_per_tool=None,
        max_distinct_tools=1,
        max_consecutive_identical=None,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("one", {}, ctx)).allowed
    result = await shield.scan_tool_call("two", {}, ctx)
    assert not result.allowed
    assert result.reason_code == "TOOL_DISTINCT_BUDGET_EXCEEDED"


@pytest.mark.asyncio
async def test_argument_size_depth_and_node_limits():
    ctx = SessionContext()
    size = ToolCallBudget(max_argument_bytes=3)
    assert not (await size.scan_tool_call("x", {"v": "abcd"}, ctx)).allowed

    depth = ToolCallBudget(max_argument_depth=2)
    result = await depth.scan_tool_call("x", {"a": {"b": {"c": 1}}}, SessionContext())
    assert not result.allowed
    assert result.reason_code == "TOOL_ARGUMENT_DEPTH_EXCEEDED"

    nodes = ToolCallBudget(max_argument_nodes=3)
    result = await nodes.scan_tool_call("x", {"v": [1, 2, 3]}, SessionContext())
    assert not result.allowed
    assert result.reason_code == "TOOL_ARGUMENT_NODE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_cycles_are_bounded_and_do_not_crash():
    cyclic = {}
    cyclic["self"] = cyclic
    result = await ToolCallBudget().scan_tool_call("x", cyclic, SessionContext())
    assert result.allowed


@pytest.mark.asyncio
async def test_shared_substructures_are_charged_for_each_occurrence():
    shared = {"values": ["x"] * 10}
    params = {"first": shared, "second": shared}

    result = await ToolCallBudget(max_argument_nodes=20).scan_tool_call(
        "x", params, SessionContext()
    )

    assert not result.allowed
    assert result.reason_code == "TOOL_ARGUMENT_NODE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_reset_clears_session_counters():
    shield = ToolCallBudget(
        max_calls_per_session=1,
        max_calls_per_tool=None,
        max_consecutive_identical=None,
    )
    ctx = SessionContext()
    assert (await shield.scan_tool_call("x", {}, ctx)).allowed
    assert not (await shield.scan_tool_call("x", {"n": 2}, ctx)).allowed
    shield.reset(ctx)
    assert (await shield.scan_tool_call("x", {}, ctx)).allowed


def test_invalid_limits_fail_fast():
    with pytest.raises(ValueError):
        ToolCallBudget(max_calls_per_session=0)
    with pytest.raises(ValueError):
        ToolCallBudget(max_calls_per_tool={"*": 0})
    with pytest.raises(ValueError):
        ToolCallBudget(max_argument_nodes="many")
