"""
Tool-layer rate limiter — used by the @SecureTool decorator
and secure_lc_tool wrapper.
"""

from __future__ import annotations

from snowmarten.policy_engine.policies.rate_limit import TokenBucket


class RateLimiter:
    """
    Per-tool rate limiter using Token Bucket.

    Usage:
        limiter = RateLimiter({"calls": 10, "per_seconds": 60})
        allowed, wait = limiter.check("my_tool")
    """

    def __init__(self, config: dict):
        """
        Config format:
            {"calls": int, "per_seconds": int}
        """
        calls       = config.get("calls", 10)
        per_seconds = config.get("per_seconds", 60)

        self._bucket = TokenBucket(
            capacity=calls,
            refill_rate=calls / per_seconds,
        )

    def check(self, tool_name: str) -> tuple[bool, float]:
        """
        Returns (allowed, wait_time_seconds).
        wait_time is 0.0 if allowed.
        """
        return self._bucket.consume()
