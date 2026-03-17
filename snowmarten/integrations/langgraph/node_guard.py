"""
SecureNode — wraps LangGraph nodes with Snowmarten security checks.

In LangGraph, nodes are functions: (State) -> State.
SecureNode intercepts state BEFORE the node runs and the
result AFTER, applying firewall + policy checks at each step.

Usage:
    from snowmarten.integrations.langgraph import SecureNode

    @SecureNode(policies=[NoExfiltrationPolicy()])
    def call_model(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from snowmarten.firewall.firewall import PromptFirewall, FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, ToolCallRequest

logger = logging.getLogger("snowmarten.langgraph")

StateType = TypeVar("StateType", bound=dict)


class SecureNode:
    """
    Decorator that makes any LangGraph node security-aware.

    Features:
    - Scans incoming messages in state for injection
    - Evaluates policies before node execution
    - Scans tool_calls in LLM output before they execute
    - Scans tool results before they re-enter state

    Example:
        @SecureNode(
            policies=[NoExfiltrationPolicy()],
            firewall=FirewallConfig(sensitivity=0.9),
            scan_messages=True,
            scan_tool_calls=True,
        )
        def call_model(state: AgentState) -> dict:
            ...
    """

    def __init__(
        self,
        policies:         list[BasePolicy] | None = None,
        firewall:         FirewallConfig | None = None,
        scan_messages:    bool = True,
        scan_tool_calls:  bool = True,
        scan_tool_results: bool = True,
        role:             str = "user",
        node_name:        str | None = None,
    ):
        self.policies          = policies or []
        self.firewall          = PromptFirewall(firewall or FirewallConfig())
        self.scan_messages     = scan_messages
        self.scan_tool_calls   = scan_tool_calls
        self.scan_tool_results = scan_tool_results
        self.role              = role
        self.node_name         = node_name

    def __call__(self, func: Callable) -> Callable:
        """Apply as decorator: @SecureNode(...)"""
        node_name = self.node_name or func.__name__

        @wraps(func)
        def secured_node(state: dict[str, Any]) -> dict[str, Any]:
            logger.debug("[Snowmarten] SecureNode: %s — pre-execution scan", node_name)

            # ── Step 1: Scan incoming messages ─────────────────
            if self.scan_messages:
                state = self._scan_state_messages(state, node_name)

            # ── Step 2: Policy check on current state ──────────
            if not self._policy_check(state, node_name):
                return {
                    "messages": state.get("messages", []) + [
                        {"role": "assistant",
                         "content": "[Blocked by Snowmarten security policy]"}
                    ],
                    "snowmarten_blocked": True,
                }

            # ── Step 3: Execute the original node ──────────────
            result = func(state)

            # ── Step 4: Scan tool_calls in LLM output ──────────
            if self.scan_tool_calls:
                result = self._scan_tool_calls(result, node_name)

            return result

        @wraps(func)
        async def async_secured_node(state: dict[str, Any]) -> dict[str, Any]:
            """Async version of the secured node."""
            if self.scan_messages:
                state = self._scan_state_messages(state, node_name)

            if not self._policy_check(state, node_name):
                return {
                    "messages": state.get("messages", []) + [
                        {"role": "assistant",
                         "content": "[Blocked by Snowmarten security policy]"}
                    ],
                    "snowmarten_blocked": True,
                }

            result = await func(state)

            if self.scan_tool_calls:
                result = self._scan_tool_calls(result, node_name)

            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_secured_node
        return secured_node

    def _scan_state_messages(
        self,
        state:     dict[str, Any],
        node_name: str,
    ) -> dict[str, Any]:
        """
        Scan all messages in state["messages"] for injection.
        This catches both user messages AND tool results
        that may contain injected payloads.
        """
        messages = state.get("messages", [])
        sanitized_messages = []
        modified = False

        for msg in messages:
            content = ""

            # Handle LangChain message objects
            if hasattr(msg, "content"):
                content = msg.content or ""
            # Handle dict-style messages
            elif isinstance(msg, dict):
                content = msg.get("content", "")

            if not content:
                sanitized_messages.append(msg)
                continue

            # Determine source for better threat context
            msg_role = getattr(msg, "type", None) or \
                (msg.get("role") if isinstance(msg, dict) else "unknown")

            source = "tool_result" if msg_role == "tool" else "message"
            threats = self.firewall.scan(content, source=source)

            if threats:
                for threat in threats:
                    logger.warning(
                        "[Snowmarten] Threat in %s state message "
                        "(node=%s, type=%s, confidence=%.2f)",
                        source, node_name,
                        threat.threat_type, threat.confidence
                    )

                    if threat.action_taken in (Action.BLOCK, Action.SANITIZE):
                        sanitized_content = self.firewall.sanitize(content)
                        modified = True

                        # Rebuild message with sanitized content
                        if hasattr(msg, "content"):
                            # LangChain message object — create new instance
                            try:
                                msg = msg.model_copy(
                                    update={"content": sanitized_content}
                                )
                            except Exception:
                                # Fallback for non-pydantic messages
                                msg = type(msg)(content=sanitized_content)
                        elif isinstance(msg, dict):
                            msg = {**msg, "content": sanitized_content}

            sanitized_messages.append(msg)

        if modified:
            state = {**state, "messages": sanitized_messages}

        return state

    def _policy_check(
        self,
        state:     dict[str, Any],
        node_name: str,
    ) -> bool:
        """Returns True if execution should proceed."""
        if not self.policies:
            return True

        # Extract the last user message for policy context
        last_user_input = ""
        for msg in reversed(state.get("messages", [])):
            role = getattr(msg, "type", None) or \
                (msg.get("role") if isinstance(msg, dict) else "")
            if role in ("human", "user"):
                last_user_input = getattr(msg, "content", "") or \
                    (msg.get("content", "") if isinstance(msg, dict) else "")
                break

        context = AgentContext(
            session_id=state.get("session_id", "langgraph-session"),
            user_input=last_user_input,
            role=self.role,
            metadata={"node": node_name},
        )

        for policy in self.policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[Snowmarten] Node '%s' blocked by policy '%s': %s",
                    node_name, decision.policy, decision.reason
                )
                return False

        return True

    def _scan_tool_calls(
        self,
        result:    dict[str, Any],
        node_name: str,
    ) -> dict[str, Any]:
        """
        Scan tool_calls generated by the LLM before execution.
        Blocks calls to suspicious URLs or with suspicious parameters.
        """
        messages = result.get("messages", [])
        if not messages:
            return result

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", None) or \
            (last_message.get("tool_calls") if isinstance(last_message, dict)
             else None)

        if not tool_calls:
            return result

        filtered_tool_calls = []
        for tc in tool_calls:
            # Extract tool name and args
            name = tc.get("name") if isinstance(tc, dict) \
                else getattr(tc, "name", "unknown")
            args = tc.get("args", {}) if isinstance(tc, dict) \
                else getattr(tc, "args", {})

            context = AgentContext(
                session_id=state.get("session_id", "lg-session")
                    if isinstance(result, dict) else "lg-session",
                user_input="",
                pending_tool_call=ToolCallRequest(name=name, params=args),
                role=self.role,
            )

            blocked = False
            for policy in self.policies:
                decision = policy.evaluate(context)
                if decision.action == Action.BLOCK:
                    logger.warning(
                        "[Snowmarten] Tool call '%s' blocked by '%s': %s",
                        name, decision.policy, decision.reason
                    )
                    blocked = True
                    break

            if not blocked:
                filtered_tool_calls.append(tc)

        # Rebuild message with filtered tool calls
        if len(filtered_tool_calls) < len(tool_calls):
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls"):
                try:
                    last_message = last_message.model_copy(
                        update={"tool_calls": filtered_tool_calls}
                    )
                except Exception:
                    pass
            elif isinstance(last_message, dict):
                last_message = {
                    **last_message,
                    "tool_calls": filtered_tool_calls
                }
            result = {**result, "messages": messages[:-1] + [last_message]}

        return result
