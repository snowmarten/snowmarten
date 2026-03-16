from __future__ import annotations
from abc import ABC, abstractmethod
from snowmarten.types import AgentContext, PolicyDecision, Action


class BasePolicy(ABC):
    """All policies inherit from this."""

    name: str = "base_policy"

    @abstractmethod
    def evaluate(self, context: AgentContext) -> PolicyDecision:
        ...

    def _allow(self, reason: str = "Policy passed") -> PolicyDecision:
        return PolicyDecision(
            action=Action.ALLOW,
            reason=reason,
            policy=self.name,
            risk_score=0.0
        )

    def _deny(self, reason: str, risk_score: float = 1.0) -> PolicyDecision:
        return PolicyDecision(
            action=Action.BLOCK,
            reason=reason,
            policy=self.name,
            risk_score=risk_score
        )

    def _alert(self, reason: str, risk_score: float = 0.5) -> PolicyDecision:
        return PolicyDecision(
            action=Action.ALERT,
            reason=reason,
            policy=self.name,
            risk_score=risk_score
        )
