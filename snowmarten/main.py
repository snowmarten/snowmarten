from __future__ import annotations
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Optional

from snowmarten.types import (
    Action, AgentContext, SecurityResult,
    ThreatEvent, PolicyDecision
)
from snowmarten.firewall.firewall import PromptFirewall, FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy


class SecureAgent:
    """
    Core SecureAgent — wraps any LLM with AgentSec's security layer.
    """

    def __init__(
        self,
        model: str,
        policies: list[BasePolicy] = None,
        tools: list[Any] = None,
        firewall: Optional[FirewallConfig] = None,
        sandbox: bool = False,
        role: str = "user",
        audit_log: Optional[str] = None,
    ):
        self.model      = model
        self.policies   = policies or []
        self.tools      = tools or []
        self.role       = role
        self.sandbox    = sandbox
        self.audit_log  = audit_log
        self.firewall   = PromptFirewall(firewall or FirewallConfig())
        self._audit_events: list[dict] = []

    def run(self, user_input: str) -> SecurityResult:
        session_id = str(uuid.uuid4())
        self._audit_events = []

        # ── Step 1: Firewall scan on user input ───────────────────
        threats = self.firewall.scan(user_input, source="user_input")
        sanitized_input = user_input

        for threat in threats:
            self._log_event("THREAT_DETECTED", {
                "threat_type": threat.threat_type,
                "confidence":  threat.confidence,
                "action":      threat.action_taken,
            })
            if threat.action_taken == Action.BLOCK:
                return self._blocked_result(
                    session_id, threats, "Input blocked by prompt firewall"
                )
            if threat.action_taken == Action.SANITIZE:
                sanitized_input = self.firewall.sanitize(sanitized_input)

        # ── Step 2: Policy engine evaluation ─────────────────────
        context = AgentContext(
            session_id=session_id,
            user_input=sanitized_input,
            role=self.role,
        )
        blocked_decisions: list[PolicyDecision] = []

        for policy in self.policies:
            decision = policy.evaluate(context)
            self._log_event("POLICY_EVALUATED", {
                "policy":     policy.name,
                "action":     decision.action,
                "risk_score": decision.risk_score,
                "reason":     decision.reason,
            })
            if decision.action == Action.BLOCK:
                blocked_decisions.append(decision)

        if blocked_decisions:
            return self._blocked_result(
                session_id, threats, "Input blocked by policy engine",
                blocked_decisions
            )

        # ── Step 3: LLM execution (placeholder) ──────────────────
        # In the full implementation this calls the LLM via litellm
        # or the appropriate SDK, intercepting tool calls before execution
        output = self._call_llm(sanitized_input)
        self._log_event("LLM_RESPONSE", {"output_length": len(output)})

        # ── Step 4: Output scan ───────────────────────────────────
        output_threats = self.firewall.scan(output, source="llm_output")

        self._write_audit_log()

        return SecurityResult(
            output=output,
            safe=len(output_threats) == 0,
            risk_score=self._aggregate_risk(threats + output_threats),
            threats_detected=threats + output_threats,
            blocked_actions=blocked_decisions,
            audit_trace=self._audit_events,
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Placeholder — replace with litellm.completion() or
        the appropriate SDK call.
        """
        return f"[LLM response to: {prompt[:50]}...]"

    def _blocked_result(
        self,
        session_id: str,
        threats: list[ThreatEvent],
        reason: str,
        decisions: list[PolicyDecision] = None,
    ) -> SecurityResult:
        self._log_event("EXECUTION_BLOCKED", {"reason": reason})
        self._write_audit_log()
        return SecurityResult(
            output=f"Request blocked: {reason}",
            safe=False,
            risk_score=1.0,
            threats_detected=threats,
            blocked_actions=decisions or [],
            audit_trace=self._audit_events,
        )

    def _log_event(self, event_type: str, data: dict) -> None:
        self._audit_events.append({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        })

    def _aggregate_risk(self, threats: list[ThreatEvent]) -> float:
        if not threats:
            return 0.0
        return max(t.confidence for t in threats)

    def _write_audit_log(self) -> None:
        if not self.audit_log:
            return
        with open(self.audit_log, "a") as f:
            f.write(json.dumps({
                "events": self._audit_events
            }) + "\n")
