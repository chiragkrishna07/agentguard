# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: chiragkrishna1732@gmail.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. If the vulnerability is confirmed,
a patch will be released within 7 days and you will be credited in the release notes
(unless you prefer to remain anonymous).

## Scope

In-scope:
- Bypass of any AgentGuard shield
- False-negative rate issues that could mislead users about their security posture
- Dependency vulnerabilities in core dependencies

Out-of-scope:
- Issues in optional dependencies (`[ml]`, `[presidio]`)
- The quality of the ML classifier's detection rate (expected to improve over time)
- Social engineering
