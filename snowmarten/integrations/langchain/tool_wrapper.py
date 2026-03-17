"""
SecureToolWrapper — wraps any LangChain tool with AgentSec's
Tool Permission Layer. Works with both @tool decorated functions
and BaseTool subclasses.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Type

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, ToolCallRequest

logger = logging.getLogger("agentsec.langchain.tool")


def secure_lc_tool(
    tool_func:       Any,                      # LangChain @tool or BaseTool
    allowed_roles:   list[str] | None = None,
    rate_limit:      dict | None = None,       # {"calls": 10, "per_seconds": 60}
    param_schema:    dict | None = None,       # JSON Schema for param validation
    policies:        list[BasePolicy] | None = None,
):
    """
    Wraps a LangChain tool with AgentSec permission checks.

    Example:
        from langchain_core.tools import tool
        from agentsec.integrations.langchain import secure_lc_tool

        @tool
        def search_db(query: str) -> str:
            \"\"\"Search the customer database.\"\"\"
            return db.search(query)

        secure_search = secure_lc_tool(
            search_db,
            allowed_roles=["analyst", "admin"],
            rate_limit={"calls": 10, "per_seconds": 60},
        )
    """
    from snowmarten.tool_layer.rate_limiter import RateLimiter
    from snowmarten.tool_layer.validator import ParamValidator

    _rate_limiter  = RateLimiter(rate_limit) if rate_limit else None
    _validator     = ParamValidator(param_schema) if param_schema else None
    _allowed_roles = set(allowed_roles or [])
    _policies      = policies or []

    # Wrap the tool's _run / invoke method
    original_invoke = tool_func.invoke if hasattr(tool_func, "invoke") \
        else tool_func

    def secured_invoke(
        input: dict[str, Any] | str,
        role:  str = "user",
        **kwargs: Any,
    ) -> Any:
        # Normalize input
        params = input if isinstance(input, dict) else {"input": input}
        tool_name = getattr(tool_func, "name", str(tool_func))

        # 1. Role check
        if _allowed_roles and role not in _allowed_roles:
            logger.warning(
                "[AgentSec] Tool '%s' denied — role '%s' not in %s",
                tool_name, role, _allowed_roles
            )
            return f"[BLOCKED] Insufficient permissions to use '{tool_name}'"

        # 2. Parameter validation
        if _validator:
            valid, error = _validator.validate(params)
            if not valid:
                logger.warning(
                    "[AgentSec] Tool '%s' param validation failed: %s",
                    tool_name, error
                )
                return f"[BLOCKED] Invalid parameters: {error}"

        # 3. Rate limit check
        if _rate_limiter:
            allowed, wait = _rate_limiter.check(tool_name)
            if not allowed:
                logger.warning(
                    "[AgentSec] Tool '%s' rate limited — retry in %.1fs",
                    tool_name, wait
                )
                return f"[BLOCKED] Rate limit exceeded. Retry in {wait:.1f}s"

        # 4. Policy checks
        context = AgentContext(
            session_id="lc-session",
            user_input="",
            pending_tool_call=ToolCallRequest(name=tool_name, params=params),
            role=role,
        )
        for policy in _policies:
            decision = policy.evaluate(context)
            if decision.action == Action.BLOCK:
                logger.warning(
                    "[AgentSec] Tool '%s' blocked by policy '%s': %s",
                    tool_name, decision.policy, decision.reason
                )
                return f"[BLOCKED] {decision.reason}"

        # 5. Execute original tool
        return original_invoke(input, **kwargs)

    # Preserve tool metadata
    if hasattr(tool_func, "name"):
        secured_invoke.__name__ = tool_func.name
    if hasattr(tool_func, "description"):
        secured_invoke.__doc__ = tool_func.description

    # Return a wrapped version
    if hasattr(tool_func, "invoke"):
        tool_func.invoke = secured_invoke
        return tool_func

    return secured_invoke
