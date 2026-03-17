"""
PromptInjectionPolicy — detects and blocks prompt injection attempts
at the policy layer (complements the Firewall's pattern matching with
semantic and contextual analysis).

Based on design patterns from:
- "Design Patterns for Securing LLM Agents against Prompt Injections" (2025)
- Microsoft MSRC indirect injection defense model (2025)
- OWASP LLM Top 10: LLM01 Prompt Injection
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, PolicyDecision

logger = logging.getLogger("snowmarten.policy.prompt_injection")


# ── Attack signature categories ────────────────────────────────────────────────

# Direct override attempts
OVERRIDE_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|context|rules?)",
    r"forget\s+(everything|all|your\s+(previous\s+)?instructions?|context)",
    r"override\s+(your\s+)?(instructions?|rules?|guidelines?|constraints?)",
    r"new\s+instructions?\s*[:\-]",
    r"updated?\s+instructions?\s*[:\-]",
    r"revised?\s+instructions?\s*[:\-]",
    r"(your\s+)?(real|true|actual|hidden)\s+instructions?\s+(are|is|were)",
    r"system\s*(message|prompt|instructions?)\s*(override|update|replace)",
    r"(replace|reset|clear)\s+(your\s+)?(system\s+)?(prompt|instructions?|context)",
]

# Role escalation / persona hijack
ROLE_HIJACK_PATTERNS: list[str] = [
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(a|an|the)\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\bimagine\s+you\s+(are|were|have\s+no)\b",
    r"\bpretend\s+(there\s+(are|is)\s+no|you\s+have\s+no)\b",
    r"\byou\s+have\s+been\s+(jailbroken|freed|unlocked|released)",
    r"\byou\s+(no\s+longer\s+have|don'?t\s+have)\s+(any\s+)?(restrictions?|limits?|rules?|guidelines?)",
    r"\byou\s+(?:no\s+longer\s+have|don'?t\s+have\s+(?:any\s+)?|had\s+no\s+)(?:restrictions?|limits?|rules?|guidelines?)",
    r"\bDAN\b",                                   # Do Anything Now
    r"\bDATA\b",                                   # Do Anything Then Act
    r"\bJAILBREAK\b",
    r"developer\s+mode\s*(enabled?|activated?|on)",
    r"maintenance\s+mode",
    r"god\s+mode",
    r"unrestricted\s+mode",
    r"training\s+mode",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"\[\/INST\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
]

# Hypothetical / fictional framing (used to bypass safety)
HYPOTHETICAL_BYPASS_PATTERNS: list[str] = [
    r"(hypothetically|theoretically|in\s+theory|for\s+academic\s+purposes)",
    r"in\s+a\s+(fictional|hypothetical|imaginary)\s+(world|scenario|universe|story)",
    r"write\s+a\s+(story|novel|fiction)\s+(where|in\s+which)\s+(you|an?\s+AI)",
    r"(play|roleplay)\s+a\s+(character|role|game)\s+(where|who|that)",
    r"if\s+you\s+were\s+(allowed|permitted|able)\s+to",
    r"what\s+would\s+you\s+(do|say)\s+if\s+(you\s+)?(had\s+no|were\s+without)",
]

# Token smuggling / encoding tricks
TOKEN_SMUGGLING_PATTERNS: list[str] = [
    r"base64\s*(encode|decode|encoded|decoded)\s*[:\-\(]",
    r"rot\s*13",
    r"\\u[0-9a-fA-F]{4}",                         # Unicode escapes
    r"%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}",            # URL double-encoding
    r"&#[0-9]+;",                                  # HTML entity injection
    r"(?:&#x[0-9a-fA-F]+;){3,}",                  # Multiple HTML entities
    # Zero-width characters (common in copy-paste injection)
    r"[\u200b\u200c\u200d\u2060\ufeff]",
]

# Context overflow / prompt stuffing
CONTEXT_OVERFLOW_INDICATORS: list[str] = [
    # Repeated padding strings (used to push system prompt out of context)
    r"(.)\1{100,}",                                # Same char repeated 100+ times
    r"(\b\w+\b)(\s+\1){20,}",                      # Same word repeated 20+ times
]

# Indirect injection markers (for retrieved content)
INDIRECT_INJECTION_PATTERNS: list[str] = [
    r"\[HIDDEN\s+INSTRUCTION\]",
    r"\[INVISIBLE\s+(TEXT|INSTRUCTION)\]",
    r"<!--\s*(INSTRUCTION|INJECT|SYSTEM)",          # HTML comment injection
    r"/\*\s*(INSTRUCTION|INJECT|SYSTEM)",           # CSS/JS comment injection
    r"<script[^>]*>.*?inject",                      # Script tag injection
    r"(before|after)\s+(summariz|translat|answer|respond)",  # Conditional instructions
]


@dataclass
class PromptInjectionPolicy(BasePolicy):
    """
    Multi-pattern prompt injection detection policy.

    Detection layers:
    1. Direct override pattern matching
    2. Role hijack / jailbreak detection
    3. Hypothetical bypass framing detection
    4. Token smuggling / encoding tricks
    5. Context overflow detection
    6. Indirect injection markers (for retrieved documents)

    Example:
        policy = PromptInjectionPolicy(
            sensitivity=0.85,
            action="sanitize",
            check_indirect=True,
        )
    """

    name = "prompt_injection_policy"

    sensitivity:      float  = 0.80       # 0.0 (lenient) → 1.0 (strict)
    action:           Action = Action.SANITIZE
    check_indirect:   bool   = True        # Check for indirect injection patterns
    check_encoding:   bool   = True        # Check for token smuggling
    check_hypothetical: bool = False       # May cause false positives; off by default
    custom_patterns:  list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled = self._compile_all_patterns()

    def _compile_all_patterns(self) -> dict[str, list[re.Pattern]]:
        groups: dict[str, list[str]] = {
            "override":     OVERRIDE_PATTERNS,
            "role_hijack":  ROLE_HIJACK_PATTERNS,
            "indirect":     INDIRECT_INJECTION_PATTERNS if self.check_indirect
                            else [],
            "encoding":     TOKEN_SMUGGLING_PATTERNS if self.check_encoding
                            else [],
            "hypothetical": HYPOTHETICAL_BYPASS_PATTERNS
                            if self.check_hypothetical else [],
            "overflow":     CONTEXT_OVERFLOW_INDICATORS,
            "custom":       self.custom_patterns,
        }
        return {
            name: [
                re.compile(p, re.IGNORECASE | re.DOTALL | re.UNICODE)
                for p in patterns
            ]
            for name, patterns in groups.items()
        }

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        text = context.user_input or ""
        print(f"TEEEEEXT1: {text}")
        # Also scan tool outputs if present
        if context.pending_tool_call:
            tool_text = str(context.pending_tool_call.params)
            text = f"{text} {tool_text}"

        print(f"TEEEEEXT2: {text}")
        hits: list[tuple[str, str]] = []  # (category, matched_text)

        for category, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(text)
                print(f"MAAAAATCH: {match} {pattern}")
                if match:
                    hits.append((category, match.group(0)[:80]))
        print(f"HIIIIITS: {hits}")
        if not hits:
            return self._allow(reason="No injection patterns detected")

        # Calculate confidence based on number and severity of hits
        confidence = self._calculate_confidence(hits)

        if confidence < self.sensitivity:
            # Below threshold — alert but allow
            logger.info(
                "[PromptInjectionPolicy] Low-confidence hit "
                "(%.2f < %.2f threshold) — categories: %s",
                confidence, self.sensitivity,
                [h[0] for h in hits]
            )
            return self._alert(
                reason=f"Low-confidence injection indicators: "
                       f"{[h[0] for h in hits]}",
                risk_score=confidence,
            )

        # Above threshold — apply configured action
        categories = list({h[0] for h in hits})
        reason = (
            f"Prompt injection detected — "
            f"categories: {categories}, "
            f"confidence: {confidence:.2f}"
        )

        logger.warning(
            "[PromptInjectionPolicy] %s | matches: %s",
            reason,
            [h[1] for h in hits[:3]]  # Log first 3 matches
        )

        if self.action == Action.BLOCK:
            return self._deny(reason=reason, risk_score=confidence)
        elif self.action == Action.SANITIZE:
            return PolicyDecision(
                action=Action.SANITIZE,
                reason=reason,
                policy=self.name,
                risk_score=confidence,
            )
        else:
            return self._alert(reason=reason, risk_score=confidence)

    def _calculate_confidence(
        self, hits: list[tuple[str, str]]
    ) -> float:
        """
        Confidence scoring:
        - Each category hit adds weight
        - High-severity categories (override, role_hijack) weigh more
        - Multiple hits in same category don't stack fully
        """
        weights: dict[str, float] = {
            "override":     0.90,
            "role_hijack":  0.85,
            "indirect":     0.80,
            "encoding":     0.75,
            "overflow":     0.70,
            "hypothetical": 0.55,
            "custom":       0.85,
        }

        seen_categories: set[str] = set()
        total_score = 0.0

        for category, _ in hits:
            base_weight = weights.get(category, 0.70)
            # Diminishing returns for repeated hits in same category
            if category in seen_categories:
                total_score += base_weight * 0.3
            else:
                total_score += base_weight
                seen_categories.add(category)

        # Normalize to 0.0-1.0 range
        return min(1.0, total_score)
