"""
@SecureTool decorator — wraps any Python function with the
full Snowmarten Tool Permission Layer.
"""

from __future__ import annotations

import logging
import functools
from typing import Any, Callable, Optional

from snowmarten.tool_layer.rate_limiter import RateLimiter
from snowmarten.tool_layer.validator import ParamValidator
from snowmarten.types import Action, AgentContext, ToolCallRequest

logger = logging.getLogger("snowmarten.tool_layer")


class SecureTool:
    """
    Decorator that applies Snowmarten's Tool Permission Layer
    to any Python function.

    Applies:
    1. Role-based access control
    2. Parameter validation (JSON Schema + custom security rules)
    3. Token bucket rate limiting
    4. Policy engine evaluation

    Example:
        @SecureTool(
            name="search_database",
            allowed_roles=["analyst", "admin"],
            param_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "maxLength": 500,
                        "no_ddl": True,
                        "no_sql_injection": True,
                    }
                },
                "required": ["query"]
            },
            rate_limit={"calls": 20, "per_seconds": 60},
        )
        def search_database(query: str) -> list[dict]:
            return db.execute(query)
    """

    def __init__(
        self,
        name:          Optional[str] = None,
        description:   Optional[str] = None,
        allowed_roles: list[str] | None = None,
        param_schema:  dict[str, Any] | None = None,
        rate_limit:    dict | None = None,
        sandbox:       bool = False,
        policies:      list | None = None,
    ):
        self.tool_name     = name
        self.description   = description
        self.allowed_roles = set(allowed_roles or [])
        self._validator    = ParamValidator(param_schema) if param_schema else None
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self._sandbox      = sandbox
        self._policies     = policies or []

    def __call__(self, func: Callable) -> Callable:
        tool_name = self.tool_name or func.__name__

        @functools.wraps(func)
        def secured(*args: Any, role: str = "user", **kwargs: Any) -> Any:
            # ── 1. Role check ─────────────────────────────────────
            if self.allowed_roles and role not in self.allowed_roles:
                logger.warning(
                    "[SecureTool] '%s' denied — role '%s' not in %s",
                    tool_name, role, self.allowed_roles,
                )
                return (
                    f"[PERMISSION DENIED] Role '{role}' cannot use "
                    f"tool '{tool_name}'. Required roles: "
                    f"{list(self.allowed_roles)}"
                )

            # ── 2. Parameter validation ───────────────────────────
            if self._validator and kwargs:
                valid, error = self._validator.validate(kwargs)
                if not valid:
                    logger.warning(
                        "[SecureTool] '%s' param validation failed: %s",
                        tool_name, error,
                    )
                    return f"[VALIDATION ERROR] {error}"

            # ── 3. Rate limit ─────────────────────────────────────
            if self._rate_limiter:
                allowed, wait = self._rate_limiter.check(tool_name)
                if not allowed:
                    logger.warning(
                        "[SecureTool] '%s' rate limited — "
                        "retry in %.1fs", tool_name, wait,
                    )
                    return (
                        f"[RATE LIMITED] Tool '{tool_name}' — "
                        f"retry in {wait:.1f}s"
                    )

            # ── 4. Policy engine ──────────────────────────────────
            if self._policies:
                context = AgentContext(
                    session_id="secure-tool-session",
                    user_input="",
                    pending_tool_call=ToolCallRequest(
                        name=tool_name,
                        params=kwargs,
                    ),
                    role=role,
                )
                for policy in self._policies:
                    decision = policy.evaluate(context)
                    if decision.action == Action.BLOCK:
                        logger.warning(
                            "[SecureTool] '%s' blocked by policy '%s': %s",
                            tool_name, decision.policy, decision.reason,
                        )
                        return f"[BLOCKED] {decision.reason}"

            # ── 5. Execute ────────────────────────────────────────
            logger.debug("[SecureTool] '%s' executing", tool_name)
            return func(*args, **kwargs)

        # Preserve metadata for LangChain / OpenAI tool discovery
        secured.__name__    = tool_name
        secured.__doc__     = self.description or func.__doc__
        secured.tool_name   = tool_name
        secured.allowed_roles = self.allowed_roles
        secured.is_secure   = True

        return secured
