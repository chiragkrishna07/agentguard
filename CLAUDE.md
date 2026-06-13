# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AgentGuard is "Helmet.js for AI Agents" — lightweight, framework-agnostic security middleware for production AI agents, distributed as the `agentguard` PyPI package. Core install is dependency-minimal on purpose; heavier capabilities (Presidio NER, DistilBERT ML) are optional extras.

## Commands

```bash
pip install -e ".[dev]"          # dev setup (also: [presidio], [ml], [all])

pytest tests/unit/               # run unit tests (asyncio_mode=auto, no @pytest.mark.asyncio needed)
pytest tests/unit/test_guard.py::test_name -v   # single test
pytest --cov=agentguard          # with coverage

ruff check agentguard/ tests/    # lint (CI gates on this)
mypy agentguard/ --ignore-missing-imports   # type check (CI gates on this)

python -m tests.benchmarks.bench_shields     # shield latency benchmark
python training/train_injection_classifier.py  # retrain the ML injection classifier
```

CI (`.github/workflows/ci.yml`) runs ruff → mypy → pytest on Python 3.10/3.11/3.12. Only `tests/unit/` runs in CI. Match the existing ruff rules (`E,F,I,N,W,UP`, line-length 100, E501 ignored).

## Architecture

The system is a **shield pipeline**. Everything flows through ordered lists of shields, and the orchestration lives entirely in `core/guard.py`.

- **`BaseShield`** (`core/base_shield.py`) — abstract base with four async hooks every shield may override: `scan_input`, `scan_output`, `scan_tool_call`, and `scan_tool_output`. Each returns a `ShieldResult(allowed, modified_input, reason, reason_code)`. A shield only overrides the hooks relevant to it (e.g. `ToolValidator` only implements `scan_tool_call`). `scan_tool_output` is the **indirect-injection chokepoint** — it inspects content a tool *returns* (web pages, emails, RAG docs) before it re-enters the agent; `GuardedTool` calls it automatically after the wrapped tool runs.

- **`Guard`** (`core/guard.py`) — holds `shields: List[BaseShield]` and runs three pipelines: `_scan_input`, `_scan_output`, `scan_tool_call`. Key invariants when editing the pipeline:
  - Shields run **in list order**; `modified_input` from one shield becomes the input to the next (this is how PIIRedactor's redacted text feeds downstream).
  - `result.allowed == False` raises `GuardBlockedError` (carries `reason_code` + `shield_name`).
  - **Fail-closed**: any unexpected exception inside a shield is wrapped in `GuardShieldError` and propagates — it does *not* silently pass through. Preserve this when adding error handling.
  - Entry points: `@guard.protect` (async fns), `@guard.protect_sync` (sync, wraps via `asyncio.run`), or explicit `guard.run(...)`. The protected function's first positional arg is always the query string; a `SessionContext` can be threaded via the `_guard_ctx` kwarg.
  - Every block goes through `Guard._raise_block`, which records the block in `self.metrics` (`core/metrics.py`, thread-safe `GuardMetrics`) before raising. `guard.stats()` returns a snapshot. When adding a new block site, route it through `_raise_block` so metrics stay complete.

- **`SessionContext`** (`core/session.py`) — per-session state passed to every shield: `session_id`, `cost_usd`, `request_count`, `metadata`, plus a `_token_map` for PII tokenize/de-tokenize round-tripping (`store_token`/`resolve_token`/`resolve_all_tokens`). Shields that accumulate state (cost, rate) read/write this rather than holding global state, except when `per="global"`.

- **Shields** (`shields/`) — each is a `BaseShield` subclass: `PromptShield` (injection detection, rule tiers + optional ML/canary), `PIIRedactor` (regex default / Presidio optional), `CostLimit` (tiktoken token counting + kill switch), `RateLimit` (token bucket), `ToolValidator` (glob allow/block + param rules), `HumanGate` (async approval via notifiers), `AuditLogger` (hashed JSON trail — never logs raw input/output).

- **`GuardedTool`** (`tools.py`) — wraps any sync/async callable so calling it runs `guard.scan_tool_call(name, kwargs, ctx)` before the real call. This is the bridge between tool execution and the `scan_tool_call` hook.

- **Adapters** (`adapters/`) — `GuardLangGraph`, `GuardOpenAI`, `GuardCrewAI` wrap framework-native nodes/clients/tools to inject the guard. Keep these thin: they should delegate to `Guard`/`GuardedTool`, not reimplement scanning.

- **Optional-dependency loading** — heavy deps are lazy-imported inside the shield/loader that needs them, never at module top level. `PIIRedactor` lazy-inits Presidio; `models/loader.py` lazy-loads the HuggingFace DistilBERT pipeline and returns `None` (warning, not crash) if `transformers` is absent or the model can't be fetched. Preserve this pattern — importing `agentguard` must work with only the core deps installed.

## Conventions

- All shield hooks are `async`. New shields subclass `BaseShield` and override only the relevant hook(s).
- To signal a block, return `ShieldResult(allowed=False, reason=..., reason_code=...)` — do not raise directly; the `Guard` pipeline converts it to `GuardBlockedError`.
- To transform text (redaction, etc.), return `ShieldResult(allowed=True, modified_input=...)`.
- Notifiers (`notifiers/`) used by `HumanGate` are async-only (`asyncio.Event` based): `CLINotifier`, `SlackNotifier`, `WebhookNotifier`.
