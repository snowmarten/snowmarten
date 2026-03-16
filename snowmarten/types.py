from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    ALLOW    = "allow"
    BLOCK    = "block"
    SANITIZE = "sanitize"
    ALERT    = "alert"


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    PROMPT_INJECTION        = "prompt_injection"
    INDIRECT_INJECTION      = "indirect_prompt_injection"
    JAILBREAK               = "jailbreak"
    TOOL_ABUSE              = "tool_abuse"
    DATA_EXFILTRATION       = "data_exfiltration"
    PRIVILEGE_ESCALATION    = "privilege_escalation"
    UNKNOWN                 = "unknown"


@dataclass
class ThreatEvent:
    threat_type:  ThreatType
    severity:     Severity
    confidence:   float           # 0.0 – 1.0
    action_taken: Action
    detail:       str
    raw_payload:  Optional[str] = None


@dataclass
class PolicyDecision:
    action:    Action
    reason:    str
    policy:    str
    risk_score: float = 0.0


@dataclass
class ToolCallRequest:
    name:    str
    params:  dict[str, Any]
    context: Optional[str] = None


@dataclass
class AgentContext:
    session_id:         str
    user_input:         str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    pending_tool_call:  Optional[ToolCallRequest] = None
    role:               str = "user"
    metadata:           dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityResult:
    output:          str
    safe:            bool
    risk_score:      float
    threats_detected: list[ThreatEvent] = field(default_factory=list)
    blocked_actions: list[PolicyDecision] = field(default_factory=list)
    audit_trace:     list[dict[str, Any]] = field(default_factory=list)
