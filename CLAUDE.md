# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AgentGuard is "Helmet.js for AI Agents" — lightweight, framework-agnostic security middleware for production AI agents, distributed as the `pyagentguard` PyPI package (import name: `agentguard`). Core install is dependency-minimal on purpose; heavier capabilities (Presidio NER, DistilBERT ML) are optional extras.

## Commands

```bash
pip install -e ".[dev]"          # dev setup (also: [presidio], [ml], [all])

pytest tests/unit/               # run unit tests (asyncio_mode=auto, no @pytest.mark.asyncio needed)
pytest tests/unit/test_guard.py::test_name -v   # single test
pytest --cov=agentguard          # with coverage

ruff check agentguard/ tests/    # lint (CI gates on this)
mypy agentguard/ --ignore-missing-imports   # type check (CI gates on this)

python -m tests.benchmarks.bench_shields     # shield latency benchmark
python -m tests.benchmarks.bench_detection   # injection detection quality (recall/precision/FPR)
python training/train_injection_classifier.py  # retrain the ML injection classifier
```

CI (`.github/workflows/ci.yml`) runs ruff → mypy → pytest on Python 3.10/3.11/3.12. Only `tests/unit/` runs in CI. Match the existing ruff rules (`E,F,I,N,W,UP`, line-length 100, E501 ignored).

## Architecture

The system is a **shield pipeline**. Everything flows through ordered lists of shields, and the orchestration lives entirely in `core/guard.py`.

- **`BaseShield`** (`core/base_shield.py`) — async hooks include `scan_input`, `scan_output`, `scan_tool_arguments`, `scan_tool_call`, `scan_tool_output`, and the content-free `on_decision` observer. Each scanning hook returns `ShieldResult(allowed, modified_input, reason, reason_code)`. `scan_tool_arguments` is the DLP/sanitization phase for model-generated arguments; `scan_tool_call` is the authorization/policy phase. `scan_tool_output` is the **indirect-injection chokepoint** for web pages, email, RAG, and other returned data.

- **`Guard`** (`core/guard.py`) — holds `shields: List[BaseShield]` and exposes type-preserving `scan_input`, `scan_output`, `scan_tool_arguments`, `scan_tool_call`, and `scan_tool_output` pipelines. Key invariants when editing the pipeline:
  - Shields run **in list order**; `modified_input` from one shield becomes the input to the next (this is how PIIRedactor's redacted text feeds downstream).
  - `result.allowed == False` raises `GuardBlockedError` (carries `reason_code` + `shield_name`).
  - **Fail-closed**: any unexpected exception inside a shield is wrapped in `GuardShieldError` and propagates — it does *not* silently pass through. Preserve this when adding error handling.
  - Entry points: `@guard.protect` (async fns), `@guard.protect_sync` (sync, wraps via `asyncio.run`), or explicit `guard.run(...)`. Inputs and outputs may be JSON-like structures; non-string scalars and container types must be preserved. A `SessionContext` can be threaded via `_guard_ctx`.
  - Structured traversal must remain bounded and cycle-aware. Schema keys/numeric scalars are visible for block decisions, but a shield must never rewrite a key or silently change a scalar type.
  - Every block goes through `Guard._raise_block`, which records the block in `self.metrics` (`core/metrics.py`, thread-safe `GuardMetrics`) before raising. `guard.stats()` returns a snapshot. When adding a new block site, route it through `_raise_block` so metrics stay complete.

- **`SessionContext`** (`core/session.py`) — per-session state passed to every shield: `session_id`, `cost_usd`, `request_count`, `metadata`, plus a `_token_map` for PII tokenize/de-tokenize round-tripping (`store_token`/`resolve_token`/`resolve_all_tokens`). Shields that accumulate state (cost, rate) read/write this rather than holding global state, except when `per="global"`.

- **Shields** (`shields/`) — content/DLP (`PromptShield`, `SecretsShield`, `PIIRedactor`, `ContentPolicyShield`), tool/egress (`ToolValidator`, `NetworkPolicyShield`, `ToolCallBudget`, `HumanGate`), resources (`SizeLimit`, `RateLimit`, `CostLimit`), and privacy-preserving audit (`AuditLogger`).

- **`GuardedTool`** (`tools.py`) — binds positional/keyword/variadic arguments, propagates `guard.scan_tool_arguments(...)` rewrites into the real invocation, validates policy before execution, awaits awaitable results, and scans structured tool output.

- **Adapters** (`adapters/`) — `GuardLangGraph`, `GuardOpenAI`, `GuardCrewAI` wrap framework-native nodes/clients/tools to inject the guard. Keep these thin: they should delegate to `Guard`/`GuardedTool`, not reimplement scanning.

- **Optional-dependency loading** — heavy deps are lazy-imported inside the shield/loader that needs them, never at module top level. `PIIRedactor` lazy-inits Presidio; `models/loader.py` lazy-loads the HuggingFace DistilBERT pipeline and returns `None` (warning, not crash) if `transformers` is absent or the model can't be fetched. Preserve this pattern — importing `agentguard` must work with only the core deps installed.

## Conventions

- All shield hooks are `async`. New shields subclass `BaseShield` and override only the relevant hook(s).
- To signal a block, return `ShieldResult(allowed=False, reason=..., reason_code=...)` — do not raise directly; the `Guard` pipeline converts it to `GuardBlockedError`.
- To transform text (redaction, etc.), return `ShieldResult(allowed=True, modified_input=...)`.
- Notifiers (`notifiers/`) used by `HumanGate` are async-only (`asyncio.Event` based): `CLINotifier`, `SlackNotifier`, `WebhookNotifier`.
