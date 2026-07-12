# Changelog

All notable changes to AgentGuard are documented here.

## [0.13.0] — Unreleased

This release is a general-purpose security hardening pass informed by the OWASP
LLM Top 10, OWASP Agentic Top 10, and NIST AI RMF Generative AI Profile. The
mapping is documented in `SECURITY.md`; it is not a compliance certification.

### Added

- Public, type-preserving `Guard.scan_input()` and `Guard.scan_output()` APIs
  for nested dict/list/tuple values, with cycle, depth, node, aggregate
  character, and UTF-8 byte limits (including schema keys).
- `Guard.scan_tool_arguments()` and end-to-end propagation through
  `GuardedTool`, OpenAI tool calls, and LangGraph/LangChain tool calls. Content
  sanitization now changes the arguments that execute; tool policy runs once on
  the sanitized structure.
- `NetworkPolicyShield`: per-tool host/scheme policies, credentialed-URL
  rejection, IDNA handling, private/loopback/link-local/reserved address and
  legacy numeric-IP blocking, URL count/length limits, and an optional resolver.
- `ToolCallBudget`: total, per-tool, distinct-tool, identical-loop, argument
  byte/depth/node limits, plus explicit session reset.
- `ContentPolicyShield`, `ContentRule`, and `ContentVerdict`: provider-neutral
  sync/async moderation callbacks, category thresholds, deterministic policy
  rules, timeouts, and fail-closed error behavior.
- A content-free guard-decision observer hook. `AuditLogger` now records blocks
  even when an earlier shield makes the decision.
- `RateLimit(per="user")`, custom key functions, retry metadata, TTL/LRU-bounded
  bucket storage, and configurable warn behavior.
- UTF-8 byte limits in `SizeLimit`; independent stream character, byte, and
  chunk ceilings in `StreamGuard`.
- A provisional streaming-output hook prevents incremental re-scans from
  double-charging `CostLimit`, duplicating audit events, or mutating final
  session tokenization state.
- Tool authorization callbacks, per-tool validators, closed parameter sets,
  dotted paths, choices, min lengths, async predicates, identity requirements,
  and payload/name bounds in `ToolValidator`.
- A documented threat model, secure deployment checklist, standards mapping,
  and explicit limitations in `SECURITY.md`.

### Security hardening

- `PromptShield` now performs bounded recursive Base64, URL, HTML-entity, and
  hex decoding; normalizes every decoded layer; detects semantic synonyms and
  multilingual injections; and uses a bounded rolling session window for
  attacks split across turns. Tool-argument scans are stateless and inspect
  adjacent structured fields without polluting user-turn history.
- Prompt rules were precision-gated against travel/support false positives such
  as terminal directions, transfer instructions, “from now on departing,” and
  traveler-role comparisons.
- `SecretsShield` now removes Unicode format-character evasion, redacts complete
  PEM blocks, covers more provider credentials, and adds contextual bearer,
  basic-auth, database URL, and labelled credential rules with placeholder
  suppression. Nested tool arguments are blocked by default, with propagated
  redact/mask alternatives.
- `PIIRedactor` adds context/checksum-validated international identifiers,
  including passport/MRZ, Aadhaar/VID, PAN/GSTIN, Indian voter/phone/UPI, NINO,
  NHS, SIN, CPF/CNPJ, TFN/Medicare, EIN, bank and national IDs. Birth dates now
  require immediate DOB context so travel dates remain intact.
- PII redaction is JSON-aware, including numeric card values. Tokenized-value
  storage is bounded, resolved raw values are removed by default, and explicit
  teardown cleanup is available.
- `AuditLogger` no longer emits raw session/user IDs by default. It uses keyed
  HMAC content fingerprints and pseudonymous identities/tool/schema names,
  with explicit legacy compatibility modes.
- `HumanGate` hides raw tool parameters by default, rejects unknown approval
  IDs, pseudonymizes session/schema metadata, uses high-entropy gate IDs, bounds
  pending approvals, validates trigger configuration, and supports a deliberate
  sanitizer.
- Slack/generic webhook notifiers require HTTPS by default, reject credentialed
  endpoints, bound payloads, hide URL-bearing HTTP errors, escape Slack markup,
  and support timestamp-bound webhook HMAC signatures.
- Cost ceilings now validate configuration, expose estimates/remaining budget,
  and block an output that crosses the configured ceiling (while documenting
  that generation spend has already occurred). A blocked generated output is
  now charged to the estimate so repeated over-budget generations do not appear
  free to subsequent requests.
- Supported mutable inputs, tool arguments, tool results, and model outputs are
  snapshotted before the first asynchronous policy wait, closing argument and
  result time-of-check/time-of-use races.
- `ToolValidator` recursively enforces closed dotted schemas and rejects dual
  literal/nested representations of the same path, preventing validation of a
  decoy value while a tool consumes an unvalidated one.
- URL discovery now covers common schemeless `target`, `origin`, `ip`,
  `ip_address`, and DSN-style destination fields without treating ordinary
  travel destinations as hosts.
- `GuardedTool` requires an explicit wrapper/per-call `SessionContext` when a
  stateful tool budget is installed, allowing multiple wrappers to share one
  real session without silently mixing tenants. Raw tool exceptions are hidden
  behind `GuardToolError` by default.
- Incremental streaming refuses classifiers and other full-output policies by
  default. An explicitly named unsafe override remains for legacy deployments
  that accept partial-policy coverage.

### Framework and correctness fixes

- `Guard.run()` and decorators preserve structured agent return types instead
  of coercing them to Python strings.
- `GuardedTool` supports positional, keyword, variadic, sync, async, and
  sync-callable/awaitable-returning tools; structured tool output rewrites are
  propagated without lossy stringification.
- OpenAI, LangGraph, and CrewAI adapters sanitize typed outputs and tool
  arguments, isolate bounded session contexts, and scan untrusted structured
  state rather than only a single conventional input field. OpenAI and
  LangGraph also treat persisted assistant/tool/peer message history as
  untrusted context and inspect it before committing the current user turn.
- LangGraph no longer derives authorization/session identity from model-visible
  state by default, and CrewAI no longer trusts identity-shaped input fields.
  Explicit compatibility flags retain the old behavior for callers that prove
  those fields come from authenticated runtime configuration.
- LangGraph now validates both modern `tool_calls` and legacy `function_call`
  outputs. CrewAI and generic document-like results scan every public payload
  field, including metadata, parsed models, and task outputs, rather than a
  small field allowlist.
- Structured rewrite boundaries catch injections split across fields, expose
  schema keys and numeric scalars for block decisions, and fail closed rather
  than rewriting a key or changing a scalar type.
- `recommended()` and `paranoid()` now include general resource, tool-loop, and
  outbound-network controls. Applications should still supply explicit tool and
  domain allowlists.

### Compatibility notes

- Version advanced to `0.13.0` in both package metadata and the import surface.
- `WebhookNotifier` uses timestamp-bound `v1` signatures by default. Select
  `signature_version="legacy"` only while migrating an existing receiver.
- `RateLimit` now rejects a non-positive burst rather than silently coercing it.
- HumanGate approval methods return `False` for unknown or expired gate IDs.
- `GuardLangGraph(trust_state_identity=True)` and
  `GuardCrewAI(trust_input_identity=True)` are now required to opt into legacy
  identity derivation from state/input fields; explicit contexts and LangGraph
  runtime `configurable` identity remain supported without those flags.
- `GuardedTool` no longer propagates raw tool/provider exception messages by
  default; catch `GuardToolError`. Setting `Guard(expose_internal_errors=True)`
  deliberately restores diagnostic details and exception chaining.
- `StreamGuard(mode="incremental")` now rejects shields that require a complete
  output unless `allow_unsafe_incremental=True` is explicitly selected.

## [0.12.0] — Unreleased

### Added
- **More credential types in `SecretsShield`**: Google OAuth secrets (`GOCSPX-`),
  Slack webhooks, HuggingFace (`hf_`), GitLab (`glpat-`), npm (`npm_`), and
  Twilio (`AC`/`SK`) keys, plus the `xoxe-` Slack token prefix.
- **More PII entities in `PIIRedactor`**: full IPv6 addresses, MAC addresses, and
  US ITINs (labelled distinctly from SSNs — SSN no longer matches the 9xx range).

## [0.11.0] — Unreleased

### Fixed (third adversarial audit — both critical/high)
- **`StreamGuard` incremental mode leaked raw secrets** when a token straddled a
  chunk boundary: it emitted the token's prefix before the full match formed,
  then over-sliced after redaction shortened the buffer. Rewritten to emit only
  a *frozen* prefix ending at a whitespace boundary, so no whitespace-delimited
  token is ever split. It matches buffer mode for the covered
  whitespace-delimited secret/PII cases. Policies whose matches contain
  whitespace or require the full response must use buffer mode.
- **Multimodal injection bypass**: the adapters scanned each text part in
  isolation, so an injection split across two parts ("disregard all" +
  "previous instructions") slipped through. Adapters now scan the *joined* text
  (`scan_joined_text`), catching split injections while preserving image parts.
- `Guard.from_dict` now wraps a bad-kwarg `TypeError` as a clear `ValueError`.

### Confirmed clean by the audit
- StreamGuard buffer mode, `extract_text`/`apply_to_text` mutation-safety, and
  `Guard.from_dict`'s type checks held up.

## [0.10.0] — Unreleased

### Added
- **`StreamGuard`** — run the output shields over a streamed LLM response.
  `mode="buffer"` (default) accumulates then scans (correct for any match
  length); `mode="incremental"` re-scans the growing buffer and emits the stable
  sanitised prefix while holding back the tail, so secrets/PII spanning chunk
  boundaries are still redacted and a triggered canary aborts the stream.

## [0.9.0] — Unreleased

### Added
- **Structured / multimodal message content** is now handled. `core/content.py`
  provides `extract_text` and `apply_to_text`; the OpenAI and LangGraph adapters
  scan the text parts of vision/multimodal messages (a list of typed parts) and
  write sanitised text back while leaving image parts untouched. Previously such
  content raised `GuardShieldError`, so an injection hidden in a text part was
  neither scanned nor passed.
- **`Guard.from_dict()`** builds a guard from a plain dict (parsed YAML/JSON) —
  each entry's `type` names an exported shield, the rest are constructor kwargs.

## [0.8.0] — Unreleased

### Fixed (from a second adversarial audit)
- **False positives from the v0.7 pattern broadening, closed.** The audit found
  benign sentences that wrongly blocked ("send the conversation starter to the
  team", "from now on you will receive weekly updates", "repeat the text above
  the line", "output everything above the fold"). The exfiltration rule now
  requires a model-context noun *and* a destination; `from now on` is limited to
  `you are/act` personas; the `repeat/output … above` rules must be
  clause-terminal. Strict mode is back to 100% recall / 100% precision / 0% FPR
  on an expanded benign set, and these cases are now in the corpus.

### Added
- **Amex (15-digit) credit cards** are now detected (Luhn-checked) in addition to
  16-digit cards.
- `RELEASE.md` checklist; wheel build verified to ship `py.typed`.

### Confirmed clean by the audit
- Luhn validation, `merge_spans` overlap resolver, `SizeLimit`, and the new
  OpenAI tool-call scanning held up against adversarial inputs.

## [0.7.0] — Unreleased

### Added
- **Detection benchmark** (`tests/benchmarks/bench_detection.py`) over a labeled
  corpus of 42 attacks (across override / extraction / persona / exfiltration /
  delimiter / unicode / leetspeak / base64 classes) and 28 *hard* benign
  negatives. Threshold tests (`test_detection_quality.py`) keep recall/precision
  from regressing.

### Changed
- **`PromptShield` strict-mode recall raised to 100%** on the corpus (precision
  100%, FPR 0%) by closing real gaps the benchmark surfaced: "show me your
  system prompt" (the intervening "me"), "from now on you act as", "repeat the
  text above starting with 'You are'", "ignore the instructions you were given",
  and exfiltration phrased as "send all the conversation context to …". The
  exfiltration rule now fires on sending the conversation/prompt itself rather
  than on any destination, so benign "send the report to alice@corp.com" is
  unaffected.

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
