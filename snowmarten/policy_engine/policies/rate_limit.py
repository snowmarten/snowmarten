"""
ToolRateLimitPolicy — enforces per-tool and global rate limits.

Uses the Token Bucket algorithm for burst-tolerant rate limiting.
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, PolicyDecision

logger = logging.getLogger("snowmarten.policy.rate_limit")


class TokenBucket:
    """
    Thread-safe Token Bucket implementation.

    Allows short bursts while enforcing a sustained rate limit.
    Formula: T(t) = min(capacity, T(t-1) + rate × Δt)
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity:    Maximum tokens (burst size)
            refill_rate: Tokens added per second (sustained rate)
        """
        self.capacity    = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens     = float(capacity)
        self._last_refill = time.monotonic()
        self._lock       = threading.Lock()

    def consume(self, tokens: float = 1.0) -> tuple[bool, float]:
        """
        Try to consume `tokens` from the bucket.

        Returns:
            (allowed: bool, wait_time_seconds: float)
            wait_time is 0.0 if allowed, else seconds until tokens available
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill

            # Refill bucket
            if self.refill_rate > 0:
                self._tokens = min(
                    self.capacity,
                    self._tokens + elapsed * self.refill_rate,
                )
            else:
                logger.warning(
                    f"TokenBucket with bad refill rate: {self.refill_rate}"
                )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, 0.0

            # Calculate wait time
            deficit = tokens - self._tokens
            if self.refill_rate <= 0:
                return False, float("inf")
            wait_time = deficit / self.refill_rate
            return False, round(wait_time, 2)

    @property
    def available_tokens(self) -> float:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_refill
            if self.refill_rate <= 0:
                return self._tokens
            return min(self.capacity, self._tokens + elapsed * self.refill_rate)


@dataclass
class ToolRateLimitPolicy(BasePolicy):
    """
    Enforces rate limits on tool calls using the Token Bucket algorithm.

    Configuration levels:
    1. Global: applies to all tool calls combined
    2. Per-tool: applies to each individual tool
    3. Per-session: (future) applies per user session

    Example:
        policy = ToolRateLimitPolicy(
            global_calls_per_minute=60,
            per_tool_limits={
                "database_query": {"calls": 10, "per_seconds": 60},
                "http_request":   {"calls": 5,  "per_seconds": 60},
                "execute_code":   {"calls": 2,  "per_seconds": 60},
            },
        )
    """

    name = "tool_rate_limit_policy"

    global_calls_per_minute: int  = 60
    per_tool_limits: dict[str, dict] = field(default_factory=dict)
    action: Action = Action.BLOCK

    def __post_init__(self) -> None:
        # Global bucket
        self._global_bucket = TokenBucket(
            capacity=self.global_calls_per_minute,
            refill_rate=self.global_calls_per_minute / 60.0,
        )

        # Per-tool buckets
        self._tool_buckets: dict[str, TokenBucket] = {}
        for tool_name, limits in self.per_tool_limits.items():
            calls      = limits.get("calls", 10)
            per_seconds = limits.get("per_seconds", 60)
            self._tool_buckets[tool_name] = TokenBucket(
                capacity=calls,
                refill_rate=calls / per_seconds,
            )

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        if not context.pending_tool_call:
            return self._allow("No tool call to rate-limit")

        tool_name = context.pending_tool_call.name

        # ── Global rate limit ───────────────────────────────────
        allowed, wait = self._global_bucket.consume()
        if not allowed:
            reason = (
                f"Global rate limit exceeded — "
                f"retry in {wait:.1f}s "
                f"({self.global_calls_per_minute} calls/min)"
            )
            logger.warning("[RateLimitPolicy] %s", reason)
            if self.action == Action.BLOCK:
                return self._deny(reason=reason, risk_score=0.70)
            return self._alert(reason=reason, risk_score=0.70)

        # ── Per-tool rate limit ─────────────────────────────────
        if tool_name in self._tool_buckets:
            allowed, wait = self._tool_buckets[tool_name].consume()
            if not allowed:
                reason = (
                    f"Rate limit exceeded for tool '{tool_name}' — "
                    f"retry in {wait:.1f}s"
                )
                logger.warning("[RateLimitPolicy] %s", reason)
                if self.action == Action.BLOCK:
                    return self._deny(reason=reason, risk_score=0.65)
                return self._alert(reason=reason, risk_score=0.65)

        return self._allow(
            reason=f"Rate limit OK for tool '{tool_name}'"
        )

    def get_status(self) -> dict[str, float]:
        """Returns available tokens for global + all tool buckets."""
        status = {"global": self._global_bucket.available_tokens}
        for tool_name, bucket in self._tool_buckets.items():
            status[tool_name] = bucket.available_tokens
        return status
