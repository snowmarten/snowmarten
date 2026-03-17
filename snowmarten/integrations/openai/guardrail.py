"""
SnowmartenInputGuardrail & SnowmartenOutputGuardrail

The OpenAI Agents SDK supports input_guardrails and output_guardrails
natively on Agent objects. Snowmarten implements both interfaces.

Usage:
    from agents import Agent
    from snowmarten.integrations.openai import (
        SnowmartenInputGuardrail,
        SnowmartenOutputGuardrail,
    )

    agent = Agent(
        name="SecureAnalyticsAgent",
        model="gpt-4o",
        instructions="You are a helpful analytics assistant.",
        tools=[search_db],
        input_guardrails=[SnowmartenInputGuardrail(policies=[...])],
        output_guardrails=[SnowmartenOutputGuardrail(policies=[...])],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from snowmarten.firewall.firewall import PromptFirewall, FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, ThreatType

logger = logging.getLogger("snowmarten.openai")


@dataclass
class GuardrailResult:
    """Result returned to the OpenAI Agents SDK guardrail interface."""
    tripwire_triggered: bool
    output_info:        dict[str, Any] | None = None


class SnowmartenInputGuardrail:
    """
    Input guardrail for the OpenAI Agents SDK.
    Plugs into Agent(input_guardrails=[...]).

    Scans user input for:
    - Prompt injection
    - Jailbreak attempts
    - Policy violations

    The SDK calls guardrail_function(context, agent, input) → GuardrailResult
    If tripwire_triggered=True, the agent is halted.
    """

    def __init__(
        self,
        policies:       list[BasePolicy] | None = None,
        firewall:       FirewallConfig | None = None,
        role:           str = "user",
        block_on_inject: bool = True,
    ):
        self.policies        = policies or []
        self.firewall        = PromptFirewall(firewall or FirewallConfig())
        self.role            = role
        self.block_on_inject = block_on_inject

    async def __call__(
        self,
        ctx:   Any,    # RunContextWrapper from openai-agents
        agent: Any,    # Agent instance
        input: str | list[Any],
    ) -> GuardrailResult:
        """
        Called by the OpenAI Agents SDK before the agent processes input.
        """
        # Normalize input to string
        if isinstance(input, list):
            # Extract text from message list
            text_parts = []
            for item in input:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_parts.append(item.get("content", ""))
                elif hasattr(item, "content"):
                    text_parts.append(str(item.content))
            text = " ".join(text_parts)
        else:
            text = str(input)

        # ── Firewall scan ─────────────────────────────────────
        threats = self.firewall.scan(text, source="user_input")

        if threats and self.block_on_inject:
            threat_types = [t.threat_type.value for t in threats]
            logger.warning(
                "[Snowmarten] Input guardrail triggered — threats: %s",
                threat_types
            )
            return GuardrailResult(
                tripwire_triggered=True,
                output_info={
                    "blocked_reason": "prompt_injection_detected",
                    "threat_types":   threat_types,
                    "message":        "Input blocked: potential prompt injection detected.",
                }
            )

        # ── Policy checks ─────────────────────────────────────
        context = AgentContext(
            session_id=getattr(ctx, "session_id", "openai-sdk-session"),
            user_input=text,
            role=self.role,
        )

        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[Snowmarten] Input blocked by policy '%s': %s",
                    decision.policy, decision.reason
                )
                return GuardrailResult(
                    tripwire_triggered=True,
                    output_info={
                        "blocked_reason": "policy_violation",
                        "policy":         decision.policy,
                        "reason":         decision.reason,
                    }
                )

        return GuardrailResult(tripwire_triggered=False)


class SnowmartenOutputGuardrail:
    """
    Output guardrail for the OpenAI Agents SDK.
    Plugs into Agent(output_guardrails=[...]).

    Scans agent output for:
    - Data exfiltration content
    - PII leakage
    - Policy violations in the response
    """

    def __init__(
        self,
        policies:          list[BasePolicy] | None = None,
        firewall:          FirewallConfig | None = None,
        role:              str = "user",
        block_on_violation: bool = True,
    ):
        self.policies           = policies or []
        self.firewall           = PromptFirewall(firewall or FirewallConfig())
        self.role               = role
        self.block_on_violation = block_on_violation

    async def __call__(
        self,
        ctx:    Any,
        agent:  Any,
        output: Any,
    ) -> GuardrailResult:
        """
        Called by the OpenAI Agents SDK after the agent produces output.
        """
        # Extract text from output
        output_text = ""
        if isinstance(output, str):
            output_text = output
        elif hasattr(output, "final_output"):
            output_text = str(output.final_output)
        elif isinstance(output, dict):
            output_text = output.get("output", str(output))

        # ── Firewall scan on output ───────────────────────────
        threats = self.firewall.scan(output_text, source="agent_output")

        if threats:
            for threat in threats:
                logger.warning(
                    "[Snowmarten] Output guardrail — threat in response: "
                    "type=%s confidence=%.2f",
                    threat.threat_type, threat.confidence
                )

        # ── Policy checks on output ───────────────────────────
        context = AgentContext(
            session_id=getattr(ctx, "session_id", "openai-sdk-session"),
            user_input=output_text,
            role=self.role,
        )

        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK and self.block_on_violation:
                logger.warning(
                    "[Snowmarten] Output blocked by policy '%s': %s",
                    decision.policy, decision.reason
                )
                return GuardrailResult(
                    tripwire_triggered=True,
                    output_info={
                        "blocked_reason": "output_policy_violation",
                        "policy":         decision.policy,
                        "reason":         decision.reason,
                    }
                )

        return GuardrailResult(tripwire_triggered=False)
