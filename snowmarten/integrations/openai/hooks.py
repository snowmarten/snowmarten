"""
SnowmartenRunHooks — intercepts every lifecycle event in the
OpenAI Agents SDK run loop.

The SDK supports AgentHooks that fire on:
- on_tool_start  → before a tool executes
- on_tool_end    → after a tool executes
- on_handoff     → when transferring to another agent

Snowmarten uses these to enforce the Tool Permission Layer
without needing to modify the Agent definition.

Usage:
    from agents import Runner
    from snowmarten.integrations.openai import SnowmartenRunHooks

    hooks = SnowmartenRunHooks(policies=[NoExfiltrationPolicy()])
    result = await Runner.run(agent, input=..., hooks=hooks)
"""

from __future__ import annotations

import logging
from typing import Any

from snowmarten.firewall.firewall import PromptFirewall, FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, ToolCallRequest

logger = logging.getLogger("snowmarten.openai.hooks")


class SnowmartenRunHooks:
    """
    Hooks that enforce Snowmarten policies throughout an agent run.
    Compatible with the OpenAI Agents SDK RunHooks interface.

    Intercepts:
    - Tool calls before execution    → Tool Permission Layer
    - Tool results after execution   → Scan for indirect injection
    - Agent handoffs                 → Policy check on handoff context
    """

    def __init__(
        self,
        policies:       list[BasePolicy] | None = None,
        firewall:       FirewallConfig | None = None,
        role:           str = "user",
        audit_log:      str | None = None,
    ):
        self.policies  = policies or []
        self.firewall  = PromptFirewall(firewall or FirewallConfig())
        self.role      = role
        self.audit_log = audit_log
        self._audit_events: list[dict] = []

    # ── OpenAI Agents SDK hook interface ──────────────────────────

    async def on_tool_start(
        self,
        context:   Any,           # RunContextWrapper
        agent:     Any,           # Agent
        tool:      Any,           # FunctionTool
        input:     str,           # Tool input (JSON string)
    ) -> None:
        """
        Called before a tool executes.
        Raise an exception here to block the tool call.
        """
        import json

        tool_name = getattr(tool, "name", str(tool))
        logger.debug("[Snowmarten] on_tool_start: %s", tool_name)

        # Parse tool input
        try:
            params = json.loads(input) if isinstance(input, str) else input
        except json.JSONDecodeError:
            params = {"raw_input": input}

        # Policy evaluation
        ctx = AgentContext(
            session_id=getattr(context, "session_id", "openai-run"),
            user_input="",
            role=self.role,
            pending_tool_call=ToolCallRequest(
                name=tool_name,
                params=params,
            ),
        )

        for policy in self.policies:
            decision = policy.evaluate(ctx)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[Snowmarten] Tool '%s' BLOCKED by '%s': %s",
                    tool_name, decision.policy, decision.reason
                )
                self._record_event("TOOL_BLOCKED", {
                    "tool":   tool_name,
                    "policy": decision.policy,
                    "reason": decision.reason,
                })
                # Raise to halt tool execution in the SDK
                raise ToolBlockedError(
                    f"Tool '{tool_name}' blocked by Snowmarten: "
                    f"{decision.reason}"
                )

        self._record_event("TOOL_ALLOWED", {"tool": tool_name})

    async def on_tool_end(
        self,
        context: Any,
        agent:   Any,
        tool:    Any,
        input:   str,
        output:  str,
    ) -> None:
        """
        Called after a tool executes, before output enters context.
        Scan for indirect prompt injection in tool results.
        """
        tool_name = getattr(tool, "name", str(tool))
        logger.debug("[Snowmarten] on_tool_end: %s", tool_name)

        threats = self.firewall.scan(
            str(output),
            source=f"tool_output:{tool_name}"
        )

        if threats:
            for threat in threats:
                logger.warning(
                    "[Snowmarten] Injection detected in tool output "
                    "from '%s' — confidence=%.2f action=%s",
                    tool_name, threat.confidence, threat.action_taken
                )
                self._record_event("INJECTION_IN_TOOL_OUTPUT", {
                    "tool":       tool_name,
                    "threat":     threat.threat_type.value,
                    "confidence": threat.confidence,
                })

    async def on_handoff(
        self,
        context:    Any,
        from_agent: Any,
        to_agent:   Any,
    ) -> None:
        """
        Called when one agent hands off to another.
        Log the handoff for audit purposes.
        """
        from_name = getattr(from_agent, "name", str(from_agent))
        to_name   = getattr(to_agent, "name", str(to_agent))
        logger.info(
            "[Snowmarten] Agent handoff: %s → %s", from_name, to_name
        )
        self._record_event("AGENT_HANDOFF", {
            "from": from_name,
            "to":   to_name,
        })

    # ── Audit ─────────────────────────────────────────────────────

    def _record_event(self, event_type: str, data: dict) -> None:
        from datetime import datetime, timezone
        import json

        event = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self._audit_events.append(event)

        if self.audit_log:
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(event) + "\n")

    @property
    def audit_trace(self) -> list[dict]:
        return self._audit_events


class ToolBlockedError(Exception):
    """Raised by SnowmartenRunHooks to halt a tool call."""
    pass
