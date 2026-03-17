"""
SecureGraphBuilder — builds security-aware LangGraph StateGraphs.

Provides a thin wrapper around StateGraph that automatically
applies SecureNode to designated nodes and adds security-aware
conditional edges.

Usage:
    from snowmarten.integrations.langgraph import SecureGraphBuilder

    builder = SecureGraphBuilder(
        policies=[NoExfiltrationPolicy()],
        secure_nodes=["agent", "tools"],
    )

    # Use exactly like StateGraph
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")

    graph = builder.compile()
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from snowmarten.firewall.firewall import FirewallConfig
from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.integrations.langgraph.node_guard import SecureNode

logger = logging.getLogger("snowmarten.langgraph.builder")


class SecureGraphBuilder:
    """
    Drop-in replacement for LangGraph's StateGraph builder
    that automatically secures designated nodes.
    """

    def __init__(
        self,
        state_schema:  Any,
        policies:      list[BasePolicy] | None = None,
        firewall:      FirewallConfig | None = None,
        secure_nodes:  list[str] | None = None,  # None = secure ALL nodes
        role:          str = "user",
    ):
        from langgraph.graph import StateGraph
        self._graph        = StateGraph(state_schema)
        self.policies      = policies or []
        self.firewall      = firewall or FirewallConfig()
        self.secure_nodes  = secure_nodes   # None means secure all
        self.role          = role
        self._node_map: dict[str, Callable] = {}

    def add_node(self, name: str, func: Callable) -> "SecureGraphBuilder":
        """Add a node, automatically wrapping it with SecureNode if needed."""
        should_secure = (
            self.secure_nodes is None           # Secure all
            or name in (self.secure_nodes or []) # Secure specific
        )

        if should_secure and self.policies:
            guard = SecureNode(
                policies=self.policies,
                firewall=self.firewall,
                role=self.role,
                node_name=name,
            )
            func = guard(func)
            logger.debug("[Snowmarten] Node '%s' secured", name)

        self._graph.add_node(name, func)
        self._node_map[name] = func
        return self

    def add_edge(self, start: str, end: str) -> "SecureGraphBuilder":
        self._graph.add_edge(start, end)
        return self

    def add_conditional_edges(
        self,
        source:   str,
        path:     Callable,
        path_map: dict | None = None,
    ) -> "SecureGraphBuilder":
        if path_map:
            self._graph.add_conditional_edges(source, path, path_map)
        else:
            self._graph.add_conditional_edges(source, path)
        return self

    def set_entry_point(self, node: str) -> "SecureGraphBuilder":
        self._graph.set_entry_point(node)
        return self

    def compile(self, **kwargs: Any) -> Any:
        return self._graph.compile(**kwargs)
