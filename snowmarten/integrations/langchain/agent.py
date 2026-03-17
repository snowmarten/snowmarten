"""
SecureLangChainAgent — high-level wrapper.
Combines middleware + tool wrapping into a single clean API.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from snowmarten.firewall.firewall import FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.integrations.langchain.middleware import SnowmartenMiddleware
from snowmarten.integrations.langchain.tool_wrapper import secure_lc_tool
from snowmarten.types import SecurityResult


class SecureLangChainAgent:
    """
    Drop-in secure wrapper for any LangChain 1.0 agent.

    Example:
        from langchain import create_agent
        from snowmarten.integrations.langchain import SecureLangChainAgent

        base_agent = create_agent(
            model="openai:gpt-4o",
            tools=[search_db, send_email],
        )

        secure_agent = SecureLangChainAgent(
            agent=base_agent,
            policies=[NoExfiltrationPolicy(), PromptInjectionPolicy()],
            secure_tools=True,
        )

        result = secure_agent.invoke("Find all customers from BC")
    """

    def __init__(
        self,
        agent:          Any,                      # LangChain agent/executor
        policies:       list[BasePolicy] | None = None,
        firewall:       FirewallConfig | None = None,
        role:           str = "user",
        secure_tools:   bool = True,              # Auto-wrap agent tools
        audit_log:      str | None = None,
        raise_on_block: bool = False,
    ):
        self._agent     = agent
        self.middleware = SnowmartenMiddleware(
            policies=policies,
            firewall=firewall,
            role=role,
            audit_log=audit_log,
            raise_on_block=raise_on_block,
        )

        # Auto-wrap tools if the agent exposes them
        if secure_tools and hasattr(agent, "tools"):
            agent.tools = [
                secure_lc_tool(t, policies=policies)
                for t in agent.tools
            ]

    def invoke(self, input: str | dict, **kwargs: Any) -> SecurityResult:
        """Synchronous invocation with full security layer."""
        # Pre-process input
        raw_input = input if isinstance(input, str) else input.get("input", "")
        safe_input = self.middleware.on_user_message(raw_input)

        # Replace input with sanitized version
        if isinstance(input, dict):
            input["input"] = safe_input
        else:
            input = safe_input

        # Run the base agent
        result = self._agent.invoke(input, **kwargs)

        # Post-process output
        output_text = result if isinstance(result, str) \
            else result.get("output", str(result))
        safe_output = self.middleware.on_model_output(output_text)

        return SecurityResult(
            output=safe_output,
            safe=True,
            risk_score=0.0,
        )

    async def ainvoke(self, input: str | dict, **kwargs: Any) -> SecurityResult:
        """Async invocation."""
        raw_input = input if isinstance(input, str) else input.get("input", "")
        safe_input = self.middleware.on_user_message(raw_input)

        if isinstance(input, dict):
            input["input"] = safe_input
        else:
            input = safe_input

        result = await self._agent.ainvoke(input, **kwargs)
        output_text = result if isinstance(result, str) \
            else result.get("output", str(result))
        safe_output = self.middleware.on_model_output(output_text)

        return SecurityResult(output=safe_output, safe=True, risk_score=0.0)

    def stream(self, input: str, **kwargs: Any) -> Iterator[Any]:
        """Streaming with security checks on each chunk."""
        safe_input = self.middleware.on_user_message(input)
        for chunk in self._agent.stream(safe_input, **kwargs):
            yield chunk
