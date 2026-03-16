from __future__ import annotations
import re
from snowmarten.types import AgentContext, PolicyDecision
from snowmarten.policy_engine.policies.base import BasePolicy


class NoExfiltrationPolicy(BasePolicy):
    """
    Blocks tool calls that attempt to send data to unauthorized domains.
    """

    name = "no_exfiltration"

    def __init__(self, allowed_domains: list[str] = None):
        self.allowed_domains = allowed_domains or []
        self._domain_patterns = [
            re.compile(
                d.replace(".", r"\.").replace("*", r".*"),
                re.IGNORECASE
            )
            for d in self.allowed_domains
        ]

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        call = context.pending_tool_call
        if not call:
            return self._allow()

        # Check HTTP tool calls
        if call.name in ("http_request", "http_tool", "requests_get",
                         "requests_post", "webhook", "send_email"):
            url = call.params.get("url", "") or call.params.get("to", "")

            if url and not self._is_allowed(url):
                return self._deny(
                    reason=f"Domain not in allowlist: {url}",
                    risk_score=0.95
                )

        return self._allow()

    def _is_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True  # No restrictions configured
        return any(p.search(url) for p in self._domain_patterns)
