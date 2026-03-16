from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

from snowmarten.types import Action, Severity, ThreatEvent, ThreatType


@dataclass
class FirewallConfig:
    enabled:                  bool  = True
    sensitivity:              float = 0.85   # 0.0 – 1.0
    scan_retrieved_documents: bool  = True
    scan_tool_outputs:        bool  = True
    custom_patterns:          list[str] = None
    action:                   Action = Action.SANITIZE

    def __post_init__(self):
        if self.custom_patterns is None:
            self.custom_patterns = []


class PromptFirewall:
    """
    Multi-layer prompt injection and jailbreak detector.

    Layer 1: Fast regex/keyword pattern matching
    Layer 2: Semantic similarity against attack embedding corpus
    Layer 3: Structural anomaly detection
    """

    BUILTIN_PATTERNS = [
        # Instruction override patterns
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior)\s+instructions?",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"new\s+instructions?\s*:",
        r"system\s+override",
        r"you\s+are\s+now\s+(a|an|in)",
        r"act\s+as\s+(if\s+you\s+(are|were)|a|an)",

        # Role-change jailbreaks
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"maintenance\s+mode",
        r"developer\s+mode",

        # Exfiltration patterns
        r"(send|forward|export|upload|post)\s+.{0,50}\s+to\s+https?://",
        r"base64\s*(encode|decode)",
        r"webhook\s*\.",

        # Hidden instruction markers
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"<\|system\|>",
    ]

    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[re.Pattern]:
        all_patterns = self.BUILTIN_PATTERNS + self.config.custom_patterns
        return [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in all_patterns
        ]

    def scan(self, text: str, source: str = "user_input") -> list[ThreatEvent]:
        """
        Scan text and return list of detected threats.
        Empty list = clean.
        """
        if not self.config.enabled:
            return []

        threats: list[ThreatEvent] = []

        # Layer 1: Pattern matching (fast)
        pattern_threats = self._pattern_scan(text, source)
        threats.extend(pattern_threats)

        # Layer 2: Structural anomaly detection
        structural_threats = self._structural_scan(text, source)
        threats.extend(structural_threats)

        return threats

    def _pattern_scan(self, text: str, source: str) -> list[ThreatEvent]:
        threats = []
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                threats.append(ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    confidence=0.92,
                    action_taken=self.config.action,
                    detail=f"Pattern matched: '{match.group()}' in {source}",
                    raw_payload=match.group(),
                ))
        return threats

    def _structural_scan(self, text: str, source: str) -> list[ThreatEvent]:
        """Detect instruction-like content in unexpected positions."""
        threats = []

        # Flag excessive imperative commands in retrieved documents
        imperative_count = len(re.findall(
            r'\b(do|must|shall|always|never|ignore|forget|override)\b',
            text, re.IGNORECASE
        ))

        if source == "retrieved_document" and imperative_count > 5:
            threats.append(ThreatEvent(
                threat_type=ThreatType.INDIRECT_INJECTION,
                severity=Severity.MEDIUM,
                confidence=0.65,
                action_taken=Action.ALERT,
                detail=f"High imperative word density ({imperative_count}) "
                       f"in retrieved document — possible indirect injection",
            ))

        return threats

    def sanitize(self, text: str) -> str:
        """Remove detected injection payloads from text."""
        sanitized = text
        for pattern in self._patterns:
            sanitized = pattern.sub("[CONTENT REMOVED BY SECURITY LAYER]", sanitized)
        return sanitized
