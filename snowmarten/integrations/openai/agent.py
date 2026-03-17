"""
SecureOpenAIAgent — high-level wrapper around the OpenAI Agents SDK.
Combines guardrails + hooks into a single clean API.
"""

from __future__ import annotations

import logging
from typing import Any

from snowmarten.firewall.firewall import FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.integrations.openai.guardrail import (
    SnowmartenInputGuardrail,
    SnowmartenOutputGuardrail,
)
from snowmarten.integrations.openai.hooks import SnowmartenRunHooks

logger = logging.getLogger("snowmarten.openai.agent")


class SecureOpenAIAgent:
    """
    Wraps an OpenAI Agents SDK Agent with full Snowmarten protection.

    Example:
        from agents import Agent, function_tool
        from snowmarten.integrations.openai import SecureOpenAIAgent

        @function_tool
        def search_db(query: str) -> str:
            \"\"\"Search the database.\"\"\"
            return db.search(query)

        secure_agent = SecureOpenAIAgent(
            name="AnalyticsAgent",
            model="gpt-4o",
            instructions="You are a helpful analytics assistant.",
            tools=[search_db],
            policies=[NoExfiltrationPolicy()],
            firewall=FirewallConfig(sensitivity=0.85),
        )

        result = await secure_agent.run("Find customers from BC")
        print(result.final_output)
        print(result.audit_trace)
    """

    def __init__(
        self,
        name:           str,
        model:          str,
        instructions:   str,
        tools:          list[Any] | None = None,
        policies:       list[BasePolicy] | None = None,
        firewall:       FirewallConfig | None = None,
        role:           str = "user",
        audit_log:      str | None = None,
        # Pass-through kwargs to Agent
        **agent_kwargs: Any,
    ):
        from agents import Agent

        self.policies  = policies or []
        self._firewall = firewall or FirewallConfig()
        self._role     = role
        self._hooks    = SnowmartenRunHooks(
            policies=policies,
            firewall=firewall,
            role=role,
            audit_log=audit_log,
        )

        # Build guardrails
        input_guardrails  = [
            SnowmartenInputGuardrail(
                policies=policies,
                firewall=firewall,
                role=role,
            )
        ]
        output_guardrails = [
            SnowmartenOutputGuardrail(
                policies=policies,
                firewall=firewall,
                role=role,
            )
        ]

        # Create the underlying Agent with guardrails
        self._agent = Agent(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools or [],
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            **agent_kwargs,
        )

        logger.info(
            "[Snowmarten] SecureOpenAIAgent '%s' initialized with %d policies",
            name, len(self.policies)
        )

    async def run(
        self,
        input:     str | list[Any],
        **kwargs:  Any,
    ) -> Any:
        """Run the agent with full security layer active."""
        from agents import Runner

        result = await Runner.run(
            self._agent,
            input=input,
            hooks=self._hooks,   # ← Snowmarten hooks intercept every tool call
            **kwargs,
        )

        # Attach audit trace to result for inspection
        result.audit_trace = self._hooks.audit_trace
        return result

    def run_sync(self, input: str | list[Any], **kwargs: Any) -> Any:
        """Synchronous run wrapper."""
        import asyncio
        return asyncio.run(self.run(input, **kwargs))

    @property
    def audit_trace(self) -> list[dict]:
        return self._hooks.audit_trace
