"""
AgentGuard — Security middleware for production AI agents.

Quick start
-----------
from agentguard import Guard, PromptShield, PIIRedactor, CostLimit

guard = Guard(shields=[PromptShield(), PIIRedactor(), CostLimit(max_usd=5.0)])

@guard.protect
async def my_agent(query: str) -> str:
    ...
"""

from agentguard.core.base_shield import BaseShield, GuardDecision, ShieldResult
from agentguard.core.exceptions import (
    AgentGuardError,
    GuardBlockedError,
    GuardShieldError,
    GuardToolError,
    HumanGateSyncError,
    HumanGateTimeoutError,
)
from agentguard.core.guard import Guard
from agentguard.core.session import SessionContext
from agentguard.shields.audit_logger import AuditLogger
from agentguard.shields.content_policy import ContentPolicyShield, ContentRule, ContentVerdict
from agentguard.shields.cost_limit import CostLimit
from agentguard.shields.human_gate import HumanGate
from agentguard.shields.network_policy import NetworkPolicyShield
from agentguard.shields.pii_redactor import PIIRedactor
from agentguard.shields.prompt_shield import PromptShield
from agentguard.shields.rate_limit import RateLimit
from agentguard.shields.secrets import SecretsShield
from agentguard.shields.size_limit import SizeLimit
from agentguard.shields.tool_budget import ToolCallBudget
from agentguard.shields.tool_validator import ToolValidator
from agentguard.streaming import StreamGuard
from agentguard.tools import GuardedTool

__version__ = "0.13.0"

__all__ = [
    # Core
    "Guard",
    "SessionContext",
    "BaseShield",
    "ShieldResult",
    "GuardDecision",
    # Shields
    "PromptShield",
    "PIIRedactor",
    "CostLimit",
    "RateLimit",
    "ToolValidator",
    "HumanGate",
    "AuditLogger",
    "SecretsShield",
    "SizeLimit",
    "NetworkPolicyShield",
    "ToolCallBudget",
    "ContentPolicyShield",
    "ContentRule",
    "ContentVerdict",
    # Streaming
    "StreamGuard",
    # Tools
    "GuardedTool",
    # Exceptions
    "AgentGuardError",
    "GuardBlockedError",
    "GuardShieldError",
    "GuardToolError",
    "HumanGateSyncError",
    "HumanGateTimeoutError",
]
