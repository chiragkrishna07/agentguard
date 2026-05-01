# Changelog

All notable changes to AgentGuard are documented here.

## [0.1.0] — Unreleased

### Added
- `Guard` class with `@guard.protect` (async) and `@guard.protect_sync` decorators
- `Guard.run()` explicit run method
- `SessionContext` for per-session state (cost, PII token map, metadata)
- **`PromptShield`** — rule-based injection detection (40+ patterns) + canary tokens + optional ML tier
- **`PIIRedactor`** — regex engine (SSN, email, credit card, phone, IBAN, IP) + optional Presidio engine; redact/mask/tokenize modes
- **`CostLimit`** — token-counting cost budget with session/global scope; supports all major model families
- **`RateLimit`** — token-bucket rate limiting per session or globally
- **`ToolValidator`** — glob-pattern tool allowlist/blocklist + per-tool parameter validation
- **`HumanGate`** — async human-in-the-loop approval with `CLINotifier`, `SlackNotifier`, `WebhookNotifier`
- **`AuditLogger`** — structured JSON audit trail (hashes only, never raw text)
- **`GuardedTool`** — wrapper for tool functions to run through `ToolValidator` + `HumanGate`
- Framework adapters: `GuardLangGraph`, `GuardOpenAI`, `GuardCrewAI`
- DistilBERT injection classifier training script (`training/train_injection_classifier.py`)
- Full unit test suite with 50+ tests
- Shield latency benchmark (`tests/benchmarks/bench_shields.py`)
- GitHub Actions CI (Python 3.10, 3.11, 3.12)
