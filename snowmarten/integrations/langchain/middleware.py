"""
Snowmarten Middleware for LangChain 1.0.

LangChain 1.0 introduced a first-class middleware system that
intercepts every step in the agent loop. SnowmartenMiddleware hooks
into this to apply firewall + policy checks at each step.

Usage:
    from snowmarten.integrations.langchain import SnowmartenMiddleware
    from langchain import create_agent

    agent = create_agent(
        model="openai:gpt-4o",
        tools=[...],
        middleware=[SnowmartenMiddleware(policies=[...])]
    )
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, AsyncIterator

from snowmarten.firewall.firewall import PromptFirewall, FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import (
    Action,
    AgentContext,
    ThreatType,
    ThreatEvent,
    ToolCallRequest,
)
import uuid

logger = logging.getLogger("snowmarten.langchain")


class SnowmartenMiddleware:
    """
    LangChain 1.0 Middleware — plugs directly into create_agent().

    Intercepts:
      - User input      → Firewall scan + policy check
      - Tool calls      → Tool permission layer + policy check
      - Tool outputs    → Scan retrieved content for indirect injection
      - Model output    → Response policy check

    Example:
        middleware = SnowmartenMiddleware(
            policies=[NoExfiltrationPolicy(), PromptInjectionPolicy()],
            firewall=FirewallConfig(sensitivity=0.85),
        )
        agent = create_agent(model=..., tools=..., middleware=[middleware])
    """

    def __init__(
        self,
        policies:  list[BasePolicy] | None = None,
        firewall:  FirewallConfig | None = None,
        role:      str = "user",
        audit_log: str | None = None,
        raise_on_block: bool = False,
    ):
        self.policies       = policies or []
        self.firewall       = PromptFirewall(firewall or FirewallConfig())
        self.role           = role
        self.audit_log      = audit_log
        self.raise_on_block = raise_on_block
        self._session_id    = str(uuid.uuid4())

    # ── LangChain 1.0 Middleware interface ────────────────────────

    def on_user_message(self, message: str, **kwargs: Any) -> str:
        """
        Called before user message reaches the LLM.
        Returns (possibly sanitized) message or raises if blocked.
        """
        logger.debug("[Snowmarten] Scanning user input")

        threats = self.firewall.scan(message, source="user_input")
        if threats:
            self._handle_threats(threats, context="user_input")

        context = AgentContext(
            session_id=self._session_id,
            user_input=message,
            role=self.role,
        )
        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                self._handle_block(decision)

        # Return sanitized version if needed
        sanitized = message
        for threat in threats:
            if threat.action_taken == Action.SANITIZE:
                sanitized = self.firewall.sanitize(sanitized)
                logger.info(
                    "[Snowmarten] Input sanitized — threat: %s",
                    threat.threat_type
                )

        return sanitized

    def on_tool_call(
        self,
        tool_name:   str,
        tool_input:  dict[str, Any],
        **kwargs:    Any,
    ) -> dict[str, Any]:
        """
        Called before a tool is executed.
        Returns (possibly modified) tool_input or raises if blocked.
        """
        logger.debug("[Snowmarten] Intercepting tool call: %s", tool_name)

        tool_call = ToolCallRequest(
            name=tool_name,
            params=tool_input,
        )
        context = AgentContext(
            session_id=self._session_id,
            user_input="",
            role=self.role,
            pending_tool_call=tool_call,
        )

        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[Snowmarten] Tool call BLOCKED — tool=%s reason=%s",
                    tool_name, decision.reason
                )
                self._handle_block(decision)

        return tool_input

    def on_tool_output(
        self,
        tool_name:   str,
        tool_output: str,
        **kwargs:    Any,
    ) -> str:
        """
        Called after a tool executes, before output enters the LLM context.
        Critical for detecting indirect prompt injection in tool results.
        """
        logger.debug(
            "[Snowmarten] Scanning tool output from: %s", tool_name
        )

        threats = self.firewall.scan(
            tool_output,
            source=f"tool_output:{tool_name}"
        )

        sanitized = tool_output
        for threat in threats:
            logger.warning(
                "[Snowmarten] Injection detected in tool output "
                "from '%s' — confidence=%.2f",
                tool_name, threat.confidence
            )
            if threat.action_taken in (Action.BLOCK, Action.SANITIZE):
                sanitized = self.firewall.sanitize(sanitized)

        return sanitized

    def on_model_output(self, output: str, **kwargs: Any) -> str:
        """
        Called after the LLM produces a response.
        Scans for data leakage in the output.
        """
        context = AgentContext(
            session_id=self._session_id,
            user_input=output,
            role=self.role,
        )
        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[Snowmarten] Output BLOCKED by policy: %s",
                    decision.reason
                )
                return "[Response blocked by security policy]"

        return output

    # ── Internal helpers ──────────────────────────────────────────

    def _handle_threats(
        self,
        threats: list[ThreatEvent],
        context: str,
    ) -> None:
        for threat in threats:
            logger.warning(
                "[Snowmarten] Threat detected in %s — type=%s confidence=%.2f",
                context, threat.threat_type, threat.confidence
            )
            if threat.action_taken == Action.BLOCK and self.raise_on_block:
                raise SecurityViolationError(
                    f"Prompt injection detected in {context}: "
                    f"{threat.detail}"
                )

    def _handle_block(self, decision: Any) -> None:
        if self.raise_on_block:
            raise SecurityViolationError(
                f"Policy '{decision.policy}' blocked execution: "
                f"{decision.reason}"
            )


class SecurityViolationError(Exception):
    """Raised when Snowmarten blocks execution and raise_on_block=True."""
    pass
