# Changelog

All notable changes to AgentGuard are documented here.

## [0.4.0] — Unreleased

### Added
- **Presets** (`agentguard.presets`): `minimal()`, `recommended()`, `paranoid()`
  factory functions returning curated, production-ready shield stacks so teams
  don't have to hand-tune seven shields on day one.
- **PEP 561 `py.typed`** marker — the package now ships its type information, so
  downstream code type-checks against AgentGuard's hints.
- New runnable example `examples/indirect_injection_and_secrets.py`.

### Fixed
- **`HumanGate` under `protect_sync`** now raises `HumanGateSyncError` with a
  clear message instead of silently hanging until timeout — the approval can
  never arrive in a per-call event loop. Implemented via a decoupled
  `BaseShield.requires_async` marker. The docstring no longer over-promises.
- Example files are now lint-clean and covered by CI.

## [0.3.0] — Unreleased

### Added
- **Unicode evasion hardening** — `PromptShield` now normalises input before
  matching: strips zero-width/invisible separators, applies NFKC folding
  (fullwidth, mathematical/styled letters), and maps common Cyrillic/Greek
  homoglyphs back to Latin. Defeats `ig​nore`, `Ｉｇｎｏｒｅ`, `𝐢𝐠𝐧𝐨𝐫𝐞`,
  soft-hyphen splits, and `ignоre` (Cyrillic o) bypasses.
- **`Guard.stats()`** — thread-safe scan/block counters (`inputs_scanned`,
  `outputs_scanned`, `tool_calls_scanned`, `tool_outputs_scanned`, `blocked`,
  `blocks_by_code`, `blocks_by_shield`) for production monitoring. Accessible
  via `guard.metrics` with `reset()`.

## [0.2.0] — Unreleased

### Added
- **Indirect prompt injection defense** — new `scan_tool_output` shield hook,
  wired through `GuardedTool`, so content returned by tools and retrieval steps
  is inspected for hidden instructions before it re-enters the agent. This is
  the vector behind real-world incidents (EchoLeak, RAG poisoning). `PromptShield`
  gains `inspect_tool_output` and `on_indirect` (`"block"` | `"neutralize"`).
- **`SecretsShield`** — detects and redacts/blocks credentials (AWS keys, GitHub
  tokens, OpenAI/Anthropic keys, Google API keys, Slack/Stripe/SendGrid tokens,
  JWTs, PEM private keys) across input, output, and tool output. Prevents secrets
  leaving the trust boundary to a third-party LLM and stops secret leakage in
  responses.
- **Output-side PII redaction** — `PIIRedactor(redact_output=True)` detects and
  redacts PII the model emits, and `scan_tool_output=True` sanitises PII in
  retrieved content. tokenize mode still de-tokenizes for multi-turn coherence.
- `AuditLogger` now records `tool_output` events (hash + length +
  indirect-injection flag).

### Changed
- **`PromptShield`** injection rules are now split into strong (single hit
  blocks) and weak (lone jailbreak buzzwords) tiers. Weak signals only block
  when corroborated (2+) or in `paranoid` mode, sharply cutting false positives
  on benign text and retrieved documents while preserving detection of real
  attacks. Added several strong patterns (`forget everything above`, `act as if
  you have no restrictions`, …).

## [0.1.1] — Unreleased

### Fixed
- **`PIIRedactor`**: overlapping regex matches (e.g. an email whose host is all
  digits also matching `PHONE_US`) no longer corrupt the output or leak PII
  fragments. Spans are now resolved to a non-overlapping set before redaction.

### Changed
- **`RateLimit`** and **`CostLimit`** now guard their state mutations with a
  lock, keeping them correct when a single shield instance is shared across OS
  threads (threaded servers, `protect_sync` from multiple threads).
- Modernised type hints across the package (`dict`/`list`/`X | None`) and made
  the codebase lint-clean under current `ruff`.

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
