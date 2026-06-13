# Changelog

All notable changes to AgentGuard are documented here.

## [0.6.0] — Unreleased

### Added
- **`SizeLimit`** shield — bounds input / output / tool-output length (chars) to
  blunt denial-of-wallet and context-stuffing; `on_exceed="block"` or
  `"truncate"`. Added to the `recommended()` preset.
- **`GuardOpenAI` validates model-requested tool calls** — when the model
  returns `tool_calls`, each name + parsed arguments is run through
  `scan_tool_call` (ToolValidator / HumanGate) before the caller executes them.

### Changed
- **Credit-card detection now requires a valid Luhn checksum**, so arbitrary
  16-digit numbers (order IDs, etc.) are no longer redacted as cards — a large
  false-positive reduction.

## [0.5.0] — Unreleased

Hardening pass driven by an internal adversarial audit; every item ships with a
regression test (`tests/unit/test_redteam_regressions.py`).

### Fixed (security)
- **PII/secret redaction could leak part of an overlapping value.** Overlapping
  spans are now merged into their **union** (shared `shields/_spans.py`) and
  labelled by the widest match, so e.g. a `DATE_OF_BIRTH` overlapping a
  `CREDIT_CARD` can no longer expose the card's tail. (Supersedes the partial
  0.1.1 fix.)
- **`ToolValidator` bypasses closed:**
  - `required: True` param rules are now enforced — an omitted parameter no
    longer skips its constraints.
  - Numeric `min`/`max` coerce numeric strings and **fail closed** on
    non-numeric values, so `{"amount": "5000"}` can't dodge a cap of 100.
  - `maxlen`/`pattern` apply to the value's string form (a list/dict can't slip
    past), and tool-name matching is now **case-insensitive** (`delete_*` also
    blocks `DELETE_FILE`).
- **`PromptShield` obfuscation bypasses closed:** base64 payloads embedded
  mid-sentence are now decoded and scanned, and a compacted/leetspeak-folded
  matching surface catches `1gn0re`, `i.g.n.o.r.e`, and `ignore-all-previous`
  style separators — with no false positives on benign text.

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
