# AgentGuard security policy and threat model

AgentGuard is defense-in-depth middleware for AI applications. It reduces risk
at input, model-output, tool-call, and tool-output boundaries; it is not a
sandbox, an identity provider, or proof that an agent is secure.

## Supported versions

Before 1.0, security fixes are made on the latest published minor release and
the `main` branch. Older `0.x` minors may receive a fix when a clean backport is
practical, but users should plan to upgrade to the newest release.

| Version | Support |
|---|---|
| Latest published minor | Security fixes |
| `main` / next release | Security fixes under development |
| Older minors | Best-effort only |

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Email
`chiragkrishna1732@gmail.com` with:

- the affected version and configuration;
- a minimal reproduction with synthetic data;
- expected and observed behavior;
- impact and any suggested remediation.

Do not send real credentials, personal data, or production prompts. Reports are
acknowledged and triaged on a best-effort basis. A disclosure date and credit
will be coordinated with the reporter after impact and a remediation path are
understood.

In scope includes bypasses, unsafe rewrites, authorization or isolation flaws,
resource-exhaustion issues, sensitive-data leakage, insecure defaults, and
vulnerabilities in AgentGuard's adapters, notifiers, or documented optional
integrations.

## Trust boundaries

AgentGuard treats these as untrusted unless the application proves otherwise:

- user, developer, and peer-agent messages;
- retrieved documents, email, web pages, files, OCR, and RAG results;
- model output, including structured tool arguments;
- tool responses and remote API payloads;
- persisted conversation or agent memory that can contain untrusted content.

AgentGuard configuration, shield code, policy callbacks, signing keys, and the
host process are trusted. A compromised Python process can bypass or modify the
guard and is outside this library's protection boundary.

## Security invariants

- Unexpected shield failures are fail-closed at a `Guard` boundary.
- Text rewrites compose in declared shield order.
- JSON-like dict/list/tuple results retain their container and scalar types.
- Supported mutable boundary values are snapshotted before asynchronous policy
  waits so later caller/tool mutation cannot change an approved operation.
- A blocked tool call is decided before the wrapped callable executes.
- Tool output is inspected before it re-enters the agent when tools are wrapped.
- Audit decisions contain no raw prompt, output, tool-result, or tool-parameter
  values. Configured tool and parameter names may be recorded as schema.
- Session state must be isolated by an application-controlled session/thread ID.
- Raw guarded-tool exception messages do not cross the boundary unless internal
  error exposure is explicitly enabled for diagnostics.

These invariants apply only to data that actually traverses a guarded entry
point. Importing AgentGuard does not automatically protect an application.

## Important limitations

- Prompt injection is not a solved classification problem. Rules, normalization,
  and moderation models can have false positives and false negatives. Use least
  privilege and deterministic authorization even when `PromptShield` passes.
- `ContentPolicyShield` requires an organization-selected classifier and/or
  rules. AgentGuard intentionally does not pretend that a universal regex list
  is adequate for self-harm, violence, sexual safety, malware, medical, legal,
  financial, or jurisdiction-specific policy.
- `NetworkPolicyShield` validates destinations before execution, but application
  clients or infrastructure must pin/revalidate resolved addresses to prevent
  DNS-rebinding/time-of-check-to-time-of-use attacks. Enforce egress policy at
  the network layer too.
- `CostLimit` estimates token cost. Provider pricing and tokenizers change, and
  output checks occur after generation. Set provider-side output/token/time
  limits and supply a reviewed pricing table.
- AgentGuard cannot validate factual accuracy, eliminate model hallucinations,
  secure a compromised dependency, or sandbox arbitrary code execution.
- In-process session caches and counters are bounded safety controls, not an
  authoritative distributed quota store. Derive session/user IDs from trusted
  authenticated runtime configuration and enforce tenant quotas in durable,
  concurrency-safe infrastructure when they are security boundaries.
- Images, audio, archives, encrypted content, and proprietary binary formats
  require preprocessing or a dedicated multimodal security service before
  their extracted text can be scanned.

## Secure deployment checklist

1. Guard every ingress and egress path, including streaming, retries, background
   tasks, framework nodes, tool arguments, tool results, and persistence.
2. Reuse a `SessionContext` only within one authenticated session. Never share a
   context across tenants; clear sensitive tokenization state at session end.
3. Use an explicit `ToolValidator.allowed` list, closed argument rules, an
   authorization callback, and `HumanGate` for high-impact actions. The model's
   request is not authorization.
4. Configure `NetworkPolicyShield` with per-tool host allowlists and enforce
   matching outbound firewall/proxy rules.
5. Apply rate, size, cost, tool-call, recursion, and provider-side time/token
   budgets. Alert on repeated blocks and loop detections.
6. Use `ContentPolicyShield` with a classifier appropriate to the product,
   region, audience, and vertical. Establish escalation and safe-response paths.
7. Keep raw secrets out of prompts and tool schemas. Prefer short-lived,
   least-privilege credentials injected by trusted execution code after policy
   approval.
8. Protect audit destinations, restrict access, set retention limits, and avoid
   enabling raw identity or argument logging in production.
   Authenticate and authorize every HumanGate approval callback; never treat a
   gate ID alone as reviewer identity.
9. Lock dependencies, verify release artifacts, run SCA/secret scanning, and
   retest policies whenever models, prompts, tools, or retrieval sources change.
10. Red-team the complete deployed system. Unit results for AgentGuard alone are
    not a security certification for an application.

## Industry guidance mapping

The control design is informed by the
[OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/),
the
[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/),
and the
[NIST AI RMF Generative AI Profile (NIST AI 600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence).
This is a control mapping, not an OWASP or NIST certification.

| Risk area | AgentGuard controls | Required outside AgentGuard |
|---|---|---|
| Prompt/goal hijacking and context poisoning | `PromptShield`, recursive decoding, multi-turn session signals, tool-output scanning, adapter boundary scanning | Prompt/data provenance, least privilege, adversarial evaluation, memory governance |
| Sensitive information disclosure | `SecretsShield`, `PIIRedactor`, structured redaction, privacy-preserving audit | Data minimization, encryption, retention/access controls, secrets manager |
| Improper output handling | Type-preserving scans, `ToolValidator`, policy callbacks, `ContentPolicyShield` | Parameterized downstream APIs, output encoding, application schema validation |
| Tool misuse, excessive agency, privilege abuse | Tool allow/deny rules, closed parameters, identity/authorization callbacks, `HumanGate`, `ToolCallBudget` | Real IAM, scoped credentials, transaction controls, sandboxing |
| SSRF and unintended egress | `NetworkPolicyShield`, per-tool hosts/schemes, private-address blocking | Firewall/proxy allowlists, address pinning, DNS controls |
| Unbounded consumption and cascading loops | `SizeLimit`, `RateLimit`, `CostLimit`, `ToolCallBudget`, bounded adapter/session caches | Provider quotas/timeouts, job controls, circuit breakers, capacity monitoring |
| Human trust exploitation and unsafe content | `ContentPolicyShield`, approval gates, decision audit | UX that distinguishes suggestions from approved actions, trained reviewers, incident response |
| Agentic supply chain, RCE, rogue agents | Partial: allowlisted/versioned tool names and pre-execution policy can reduce exposure | Signed/pinned components, code sandbox, workload isolation, runtime detection and revocation |
