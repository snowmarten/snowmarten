"""
JailbreakDetectionPolicy — detects attempts to override the agent's
safety constraints and operating instructions.

Distinguishes between:
- Soft jailbreaks (role-play framing, hypothetical scenarios)
- Hard jailbreaks (explicit instruction override, DAN-style attacks)
- Multi-turn jailbreaks (gradual erosion across conversation turns)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, PolicyDecision

logger = logging.getLogger("snowmarten.policy.jailbreak")


# ── Jailbreak signature patterns ───────────────────────────────────────────────

# Classic jailbreak templates
CLASSIC_JAILBREAKS: list[str] = [
    # DAN variants
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"do\s+anything\s+and\s+tell",
    r"\bDATA\b.*\bdo\b.*\banything\b",
    # hypothetical / unrestricted capability phrasing
    r"\bhypothetically\b.*?\bif\s+you\s+could\s+do\s+anything\b",
    r"\bif\s+you\s+could\s+do\s+anything\b",
    # AIM (Always Intelligent and Machiavellian)
    r"\bAIM\b.*\bMachiavellian\b",
    r"always\s+intelligent\s+and\s+machiavellian",
    # STAN, DUDE variants
    r"\bSTAN\b.*\bnorms\b",
    r"\bDUDE\b.*\bno\s+restrictions\b",
    r"\bDEVMODE\b",
    # GPT-4 "grandmother exploit" pattern
    r"(grandmother|grandma|grandpa|grandfather).*?(explain|tell|show|describe)",
    # Token manipulation
    r"<JAILBREAK>",
    r"\[JAILBREAK\]",
    r"\[SYSTEM\s+OVERRIDE\]",
]

# Safety bypass framing
SAFETY_BYPASS_PATTERNS: list[str] = [
    r"(no\s+)?(ethical|moral|safety)\s+(guidelines?|constraints?|limits?|rules?|filters?)",
    r"without\s+(any\s+)?(restrictions?|limits?|guidelines?|filters?|censorship)",
    r"(bypass|circumvent|ignore|skip|disable)\s+(safety|filter|content\s+polic)",
    r"(as\s+if\s+)?(you\s+)(have\s+no|don'?t\s+have|lack)\s+(any\s+)?(restrictions?|limits?|ethics)",
    r"(harmful|illegal|unethical)\s+(content|instructions?|information|advice)\s+(is\s+)?(allowed|ok|fine|permitted)",
    r"pretend\s+(that\s+)?(there\s+(are|is)\s+no|you\s+have\s+no)\s+(rules?|restrictions?|limits?)",
    r"in\s+this\s+(context|scenario|world)\s+(rules?|ethics|laws?)\s+(don'?t|do\s+not)\s+apply",
]

# Prompt leaking (trying to extract system prompt)
PROMPT_EXTRACTION_PATTERNS: list[str] = [
    r"(what|tell\s+me|show\s+me|reveal|print|display|output|repeat)\s+(are\s+)?"
    r"(your\s+)?(system\s+prompt|instructions?|directives?|guidelines?|rules?|context)",
    r"(what\s+)?(were|are)\s+you\s+(told|instructed|asked)\s+to\s+do",
    r"(ignore|skip|bypass)\s+and\s+(show|tell|print|reveal|output|repeat)\s+"
    r"(your|the|all)\s+(instructions?|system|prompt|context)",
    r"repeat\s+(everything|all|the\s+above|your\s+(full\s+)?instructions?)\s+"
    r"(above|before|prior|verbatim|word\s+for\s+word)",
    r"output\s+(your\s+)?(initialization|system|base)\s+(instructions?|prompt|context)",
]

# Multi-turn escalation markers
ESCALATION_PATTERNS: list[str] = [
    r"(now\s+that\s+we'?ve?\s+established|since\s+you\s+(agreed|said|confirmed))",
    r"(as\s+you\s+(just\s+)?said|you\s+already\s+(agreed|confirmed|accepted))",
    r"(remember|recall)\s+(when|that)\s+you\s+(said|told|agreed|confirmed)",
    r"(building|based|continuing)\s+on\s+(what|our|the)\s+(previous|last|above)",
]

# Persona injection
PERSONA_INJECTION_PATTERNS: list[str] = [
    r"(you\s+are|act\s+as|play\s+(the\s+role\s+of|as))\s+"
    r"(an?\s+)?(AI|assistant|bot|system|model)\s+(that|who|which)\s+"
    r"(can|will|must|should|has\s+no\s+restrictions?)",
    r"(from\s+now\s+on|starting\s+now|going\s+forward)\s+(you\s+(are|will\s+be|must\s+be))",
    r"your\s+(new\s+)?(name|identity|persona|role)\s+is",
    r"(switch|change|transform|become)\s+(to|into)\s+(a\s+)?"
    r"(different|new|alternative)\s+(mode|persona|character|role|AI)",
]


@dataclass
class JailbreakDetectionPolicy(BasePolicy):
    """
    Detects jailbreak attempts at both input and multi-turn levels.

    Modes:
    - strict:  Blocks on ANY jailbreak indicator
    - balanced: Blocks only on high-confidence indicators (default)
    - lenient:  Only alerts, never blocks

    Example:
        policy = JailbreakDetectionPolicy(
            action="block",
            include_pattern_library=True,
            check_prompt_extraction=True,
        )
    """

    name = "jailbreak_detection_policy"

    action:                    Action = Action.BLOCK
    mode:                      str    = "balanced"   # "strict"|"balanced"|"lenient"
    include_pattern_library:   bool   = True
    check_prompt_extraction:   bool   = True
    check_persona_injection:   bool   = True
    check_multi_turn:          bool   = True
    custom_patterns:           list[str] = field(default_factory=list)

    # Track conversation for multi-turn analysis
    _turn_count:               int    = field(default=0, init=False, repr=False)
    _suspicious_turns:         int    = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, list[re.Pattern]]:
        groups: dict[str, list[str]] = {
            "classic":    CLASSIC_JAILBREAKS
                          if self.include_pattern_library else [],
            "bypass":     SAFETY_BYPASS_PATTERNS,
            "extraction": PROMPT_EXTRACTION_PATTERNS
                          if self.check_prompt_extraction else [],
            "persona":    PERSONA_INJECTION_PATTERNS
                          if self.check_persona_injection else [],
            "escalation": ESCALATION_PATTERNS
                          if self.check_multi_turn else [],
            "custom":     self.custom_patterns,
        }
        return {
            name: [
                re.compile(p, re.IGNORECASE | re.DOTALL | re.UNICODE)
                for p in patterns
            ]
            for name, patterns in groups.items()
        }

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        self._turn_count += 1
        text = context.user_input or ""

        hits: list[tuple[str, str, float]] = []  # (category, match, severity)

        severity_map: dict[str, float] = {
            "classic":    1.0,
            "bypass":     0.90,
            "extraction": 0.85,
            "persona":    0.80,
            "escalation": 0.60,
            "custom":     0.90,
        }

        for category, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    severity = severity_map.get(category, 0.75)
                    hits.append((category, match.group(0)[:80], severity))

        if not hits:
            # Reset suspicious turn counter on clean input
            if self._suspicious_turns > 0:
                self._suspicious_turns = max(0, self._suspicious_turns - 1)
            return self._allow("No jailbreak patterns detected")

        self._suspicious_turns += 1

        # Multi-turn escalation: lower threshold if we've seen repeated attempts
        multi_turn_risk = min(0.3, self._suspicious_turns * 0.1)

        max_severity = max(h[2] for h in hits) + multi_turn_risk
        max_severity = min(1.0, max_severity)
        categories   = list({h[0] for h in hits})

        # Threshold by mode
        thresholds = {"strict": 0.3, "balanced": 0.65, "lenient": 1.1}
        threshold  = thresholds.get(self.mode, 0.65)

        reason = (
            f"Jailbreak attempt detected — "
            f"categories: {categories}, "
            f"severity: {max_severity:.2f}, "
            f"turn: {self._turn_count}, "
            f"consecutive_suspicious: {self._suspicious_turns}"
        )

        logger.warning(
            "[JailbreakPolicy] %s | top match: '%s'",
            reason, hits[0][1]
        )

        if max_severity >= threshold:
            if self.mode == "lenient":
                return self._alert(reason=reason, risk_score=max_severity)
            return self._deny(reason=reason, risk_score=max_severity)

        return self._alert(reason=reason, risk_score=max_severity)
