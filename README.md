# AgentGuard

Framework-neutral security middleware for AI agents.

AgentGuard places deterministic controls around user input, model output, tool
arguments, tool results, and session resources. It is designed for LangGraph,
OpenAI SDK, CrewAI, or a plain Python callable, without requiring a hosted
AgentGuard service.

> AgentGuard is defense in depth, not a sandbox or a guarantee that prompt
> injection has been solved. Read the [security model](SECURITY.md) before a
> production deployment. The package is pre-1.0 and APIs may still evolve.

## Install

```bash
pip install pyagentguard
```

The distribution name is `pyagentguard`; the Python import is `agentguard`.

## Quick start

```python
from agentguard import (
    Guard,
    NetworkPolicyShield,
    PIIRedactor,
    PromptShield,
    SecretsShield,
    SizeLimit,
    ToolCallBudget,
    ToolValidator,
)

guard = Guard(shields=[
    SizeLimit(max_input_chars=20_000, max_tool_output_chars=20_000),
    PromptShield(mode="strict", on_indirect="block"),
    SecretsShield(on_detect="redact", tool_argument_policy="block"),
    PIIRedactor(mode="redact", redact_output=True, scan_tool_output=True),
    ToolValidator(
        allowed=["search_*", "read_*"],
        blocked=["delete_*", "admin_*"],
    ),
    ToolCallBudget(),
    NetworkPolicyShield(allowed_hosts=["*.example.com"]),
])

@guard.protect
async def agent(query: str) -> dict:
    return await run_your_agent(query)
```

`Guard` fails closed when a shield errors, raises `GuardBlockedError` for a
policy decision, composes rewrites in shield order, and preserves JSON-like
dict/list/tuple and non-string scalar types. Its structured traversal also has
default depth, node, aggregate-character, and UTF-8-byte ceilings so oversized
schema keys cannot bypass a value-oriented `SizeLimit`.

For a curated baseline:

```python
from agentguard.presets import recommended

guard = recommended(max_usd=25.0, audit=True)
```

The recommended preset includes size and byte limits, rate limiting, prompt
injection detection, secret and PII protection, tool argument validation, tool
loop budgets, public-HTTPS network policy, cost limiting, and private audit
fingerprints. A real deployment should still replace permissive defaults with
its own tool/domain allowlists and authorization policy.

## Boundary model

```text
untrusted user / memory
          │
          ▼
  input + content shields
          │
          ▼
       agent/model ───── model-generated tool arguments
          │                         │
          │                         ▼
          │               argument DLP + tool policy
          │                         │
          │                         ▼
          │                    guarded tool
          │                         │
          │              untrusted tool/retrieval result
          │                         │
          └───────────────◀ tool-output shields
          │
          ▼
 output redaction/content policy
          │
          ▼
 persistence / user
```

Guard every path that crosses this boundary. Importing AgentGuard does not
automatically protect framework calls, background jobs, persistence, or tools.

## Shields

| Shield | Purpose |
|---|---|
| `PromptShield` | Direct, encoded, multilingual, multi-turn, and indirect prompt-injection signals |
| `SecretsShield` | Provider keys, tokens, private keys, bearer/basic auth, database URLs, and contextual credentials |
| `PIIRedactor` | Validated US and international personal/financial identifiers, with redact/mask/tokenize modes |
| `ContentPolicyShield` | Provider-neutral moderation callback and organization-specific deterministic rules |
| `ToolValidator` | Tool allow/deny policy, typed/closed parameters, identity checks, and async authorization callbacks |
| `NetworkPolicyShield` | Per-tool egress allowlists, safe schemes, credentialed-URL rejection, and SSRF/private-address controls |
| `ToolCallBudget` | Total/per-tool/distinct/repeated-call limits and structured argument budgets |
| `HumanGate` | Async human approval for high-impact tools, cost thresholds, or detected PII |
| `SizeLimit` | Character and UTF-8 byte ceilings for input, output, and tool output |
| `RateLimit` | Bounded token buckets per session, user, global scope, or application-defined key |
| `CostLimit` | Estimated token-cost ceiling with explicit custom pricing support |
| `AuditLogger` | Content-free decisions plus keyed fingerprints and pseudonymous identities |

### Prompt and indirect-injection defense

```python
PromptShield(
    mode="strict",                 # "fast" | "strict" | "paranoid"
    use_ml=False,                  # optional local classifier tier
    use_canary=True,
    inspect_tool_output=True,
    on_indirect="block",          # "block" | "neutralize"
)
```

The rule tier normalizes Unicode/zero-width/homoglyph obfuscation and performs
bounded recursive decoding of Base64, URL encoding, HTML entities, and hex.
`SessionContext` holds a bounded rolling window for attacks split across turns.
Detection remains probabilistic in practice; tool privileges must not depend on
this classifier alone.

### Secret and PII handling

```python
SecretsShield(
    on_detect="redact",
    scan_directions=("input", "output", "tool_call", "tool_output"),
    detect_generic_credentials=True,
    tool_argument_policy="block",  # "block" | "redact" | "mask" | "off"
)

pii = PIIRedactor(
    mode="redact",                 # "redact" | "mask" | "tokenize"
    redact_output=True,
    scan_tool_output=True,
    tool_argument_policy="off",    # choose block/redact when product policy permits
)
```

PII rules include checksum/context validation and international identifiers such
as passports/MRZ, Aadhaar/VID, PAN/GSTIN, NINO/NHS, SIN, CPF/CNPJ, TFN/Medicare,
IBAN, phones, UPI IDs, credit cards, and US tax identifiers. Valid JSON strings
are sanitized recursively and reserialized safely, including numeric card
values.

Tokenize mode temporarily retains original PII inside `SessionContext` so a
known token can be resolved. Resolved values are removed by default; clear any
unused values at session teardown:

```python
pii.clear_tokenized_values(ctx)
```

Use the optional Presidio engine when a vertical needs NER rather than only
validated deterministic patterns:

```bash
pip install "pyagentguard[presidio]"
```

### Tool authorization and argument policy

The model's tool call is a request, not authorization.

```python
async def authorize(tool_name, params, ctx):
    if not ctx.user_id:
        return "authenticated user required"
    if tool_name == "transfer_funds" and params["amount"] > 500:
        return "amount exceeds delegated authority"
    return True

validator = ToolValidator(
    allowed=["search_*", "transfer_funds"],
    blocked=["admin_*"],
    require_user_id=True,
    allow_extra_params=False,
    param_rules={
        "*": {"tenant_id": {"type": str, "required": True}},
        "transfer_funds": {
            "amount": {"type": (int, float), "min": 0, "max": 500},
            "currency": {"type": str, "choices": ["USD", "INR", "EUR"]},
        },
    },
    authorize=authorize,
)
```

`param_rules` supports case-insensitive tool globs, dotted mapping paths,
required/nullable values, types, numeric bounds, length bounds, full-match
regexes, choices, and sync/async predicates. `validators` provides a per-tool
callback for Pydantic, JSON Schema, RBAC/ABAC, transaction, or vertical policy.

Use `GuardedTool` so sanitized arguments reach the callable and returned data is
scanned before re-entering the agent:

```python
from agentguard import GuardedTool, SessionContext

ctx = SessionContext(session_id="thread-42", user_id="user-7")
safe_search = GuardedTool(search_web, guard, ctx)
result = await safe_search(url="https://api.example.com/search", query="hotels")
```

When `ToolCallBudget` is installed, a guarded tool requires an explicit context
either on the wrapper (as above) or per call with `_guard_ctx=ctx`. Reuse that
same authenticated-session context across all of the session's tool wrappers;
do not use one global wrapper context for multiple tenants. Tool/provider
exceptions are exposed as content-safe `GuardToolError` by default so an error
message cannot become an unscanned secret or injection channel.

For custom integrations, use the public boundary methods:

```python
safe_input = await guard.scan_input(raw_input, ctx)
safe_args = await guard.scan_tool_arguments("search_web", raw_args, ctx)
safe_result = await guard.scan_tool_output("search_web", raw_result, ctx)
safe_output = await guard.scan_output(agent_output, ctx)
```

`scan_tool_call()` remains a validation-only compatibility API. Prefer
`scan_tool_arguments()` or `GuardedTool` when rewritten arguments must propagate
to execution.

### Network and SSRF policy

```python
NetworkPolicyShield(
    allowed_schemes=("https",),
    blocked_hosts=["*.invalid.example"],
    allow_private_networks=False,
    allow_userinfo=False,
    additional_url_keys=["fetch_target"],
    max_argument_depth=32,
    tool_policies={
        "weather_*": {"allowed_hosts": ["api.weather.example"]},
        "search_*": {"allowed_hosts": ["search.example", "*.search.example"]},
    },
)
```

The shield fails closed when its argument depth/node inspection budget is
exceeded and catches URL-like nested fields, legacy/numeric IP forms, loopback,
link-local, private, reserved and unqualified hosts. An optional sync/async
`host_resolver` can check resolved addresses. Also enforce matching egress rules
at the firewall/proxy and pin or revalidate the connected address.

### Content policy

AgentGuard does not ship a universal “harmful content” regex. Attach the
moderation service or local classifier appropriate to your product and region:

```python
from agentguard import ContentPolicyShield, ContentRule

async def moderate(text, direction, ctx):
    # Return category probabilities from your chosen moderation model.
    return {"violence": 0.02, "self_harm": 0.01, "sexual_minors": 0.0}

content = ContentPolicyShield(
    classifier=moderate,
    thresholds={"violence": 0.8, "self_harm": 0.6, "sexual_minors": 0.01},
    rules=[
        ContentRule(
            category="regulated_export",
            pattern=r"\bINTERNAL-EXPORT-CODE-[0-9]+\b",
            directions=("output", "tool_output"),
        )
    ],
    classifier_timeout_seconds=5,
    on_error="block",
)
```

The callback receives `(text, direction, ctx)` and may return score mappings or
`ContentVerdict`. Both sync and async callbacks are supported; synchronous work
runs off the event loop.

### Resource and loop controls

```python
RateLimit(requests_per_minute=30, per="user", burst=5)

ToolCallBudget(
    max_calls_per_session=100,
    max_calls_per_tool={"search_*": 30, "write_*": 5},
    max_distinct_tools=20,
    max_consecutive_identical=3,
)

SizeLimit(
    max_input_chars=20_000,
    max_input_bytes=80_000,
    max_tool_output_chars=20_000,
)
```

Rate-limit bucket storage is TTL/LRU bounded. For `per="user"`, a missing
`ctx.user_id` fails closed. `CostLimit` uses example pricing bundled with the
package; pass a reviewed `pricing` table because provider prices change.

### Human approval

```python
HumanGate(
    triggers=["tool_call:send_*", "tool_call:delete_*", "cost_exceeds:2.0"],
    notifier=your_notifier,
    timeout_seconds=300,
    on_timeout="block",
    identity_mode="hmac",
)
```

Raw session IDs, parameter keys, and parameter values are omitted or
pseudonymized by default. Use a deliberate `param_sanitizer` if a reviewer needs
selected values. Approval callbacks must be authenticated and authorized by the
application; the high-entropy gate ID is not a substitute for reviewer identity.
Slack and
generic webhook notifiers require HTTPS by default; signed generic webhooks bind
a timestamp and raw body to reduce replay/tampering risk.

## Framework adapters

| Framework | Adapter |
|---|---|
| LangGraph / LangChain messages | `agentguard.adapters.langgraph.GuardLangGraph` |
| OpenAI chat completions | `agentguard.adapters.openai.GuardOpenAI` |
| CrewAI kickoff and outputs | `agentguard.adapters.crewai.GuardCrewAI` |

```python
from agentguard.adapters.langgraph import GuardLangGraph

adapter = GuardLangGraph(guard)

@adapter.wrap_node
async def model_node(state, config=None):
    return await call_model(state)

safe_tool = adapter.wrap_tool(search_web)
```

Adapters scan structured/multimodal text without stringifying images or scalar
fields, rewrite model-generated tool arguments, inspect typed outputs and
persisted assistant/tool history, and use bounded per-session context caches.
Pass stable, authenticated session/thread and user IDs; never reuse one
`SessionContext` across tenants. LangGraph identities are derived from runtime
`config["configurable"]`, not model-visible state, by default. CrewAI callers
should pass an explicit `_guard_ctx`; legacy state/input-derived identity is
available only through the clearly named `trust_state_identity=True` or
`trust_input_identity=True` compatibility options.

## Streaming

```python
from agentguard import StreamGuard

stream = StreamGuard(
    guard,
    mode="buffer",                 # safest; scan before releasing any text
    max_buffer_chars=250_000,
    max_buffer_bytes=1_000_000,
    max_chunks=20_000,
)

async for clean_chunk in stream.scan(model_stream()):
    send(clean_chunk)
```

Buffer mode is the safe default. Incremental mode holds back an unfinished tail
and can reduce latency for limited local policies, but it cannot safely enforce
semantic classifiers or matches spanning released whitespace. AgentGuard now
refuses incremental mode when a configured shield requires the full output.
`allow_unsafe_incremental=True` is an explicit compatibility escape hatch, not
a production recommendation. Both modes enforce independent buffer/chunk
ceilings.

## Audit and metrics

```python
AuditLogger(
    output="file",
    path="./agentguard_audit.log",
    identity_mode="hmac",          # "hmac" | "omit" | "raw"
    fingerprint_mode="hmac",       # "hmac" | "omit" | "sha256"
    schema_mode="hmac",            # protects tool and parameter names
    hmac_key=a_secret_32_byte_key,
)
```

Secure defaults use keyed fingerprints and pseudonymized session/user IDs. Raw
prompt, output, tool-result, and parameter values are not logged. A stable key
is required only when events must correlate across processes or restarts.

```python
guard.stats()
# inputs_scanned, outputs_scanned, tool_calls_scanned,
# tool_outputs_scanned, blocked, blocks_by_code, blocks_by_shield
```

## Declarative configuration

```python
guard = Guard.from_dict({
    "shields": [
        {"type": "PromptShield", "mode": "strict"},
        {"type": "SecretsShield", "on_detect": "redact"},
        {"type": "NetworkPolicyShield", "allowed_hosts": ["*.example.com"]},
    ]
})
```

Callable policies and Python type objects are normally configured in code rather
than YAML/JSON.

## Evaluation and standards

The repository includes unit regressions, adversarial prompt corpora, structured
boundary tests, and offline framework tests. Corpus scores describe only the
checked corpus; they are not a production security score.

AgentGuard's threat model maps its controls to OWASP LLM/agentic risks and the
NIST Generative AI Profile in [SECURITY.md](SECURITY.md). The mapping is design
guidance, not a certification.

## Development

```bash
git clone https://github.com/chiragkrishna07/agentguard
cd agentguard
pip install -e ".[dev]"

ruff check agentguard/ tests/ examples/
mypy agentguard/ --ignore-missing-imports
pytest tests/unit/
python -m tests.benchmarks.bench_detection
```

## License

MIT — see [LICENSE](LICENSE).
