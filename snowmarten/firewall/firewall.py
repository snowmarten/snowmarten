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
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions|context?",
        r"disregard\s+(all\s+)?(previous|prior)\s+instructions|context?",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"ignore\s+(everything|all|your\s+instructions?)",
        r"(freed|free)\s+from\s+(your\s+)?(training|restrictions?|rules|policies)",
        r"no\s+longer\s+(bound|restricted)\s+by\s+(your\s+)?(training|rules|policies)",
        r"(ignore|bypass|disable)\s+(your\s+)?(safety|alignment|content\s+policy)",
        r"(remove|lift)\s+(all\s+)?(restrictions?|safeguards?)",
        r"new|updated\s+instructions?\s*:",
        r"system\s+override",
        r"you\s+are\s+now\s+(a|an|in)",
        r"act\s+as\s+(if\s+you\s+(are|were)|a|an)",
        r"you\s+are\s+not\s+an?\s+ai",
        r"pretend\s+you\s+are\s+not\s+restricted",
        r"simulate\s+(a|an)\s+unfiltered\s+ai",
        r"act\s+without\s+(rules|restrictions|policies)",
        r"bypass\s+(openai|system)\s+(rules|policies)"

        # Unrestricted output permissions
        r"you\s+can\s+now\s+(say|do|answer)\s+anything",
        r"respond\s+without\s+(restrictions?|filters?|limitations?)",
        r"answer\s+without\s+(safety|policy)\s+(checks?|filters?)",
        r"you\s+are\s+not\s+an?\s+ai",
        r"pretend\s+you\s+are\s+not\s+restricted",

        # Role-change jailbreaks
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"maintenance\s+mode",
        r"developer\s+mode",
        r"god\s+mode",

        # Identity / persona reassignment
        r"your\s+(true|real)\s+name\s+is\s+\w+",
        r"you\s+are\s+actually\s+\w+",
        r"from\s+now\s+on\s+you\s+are\s+\w+",

        # Malicious or unrestricted persona setup
        r"(evil|unfiltered|uncensored|amoral|malicious)\s+(ai|assistant|persona|mode)",
        r"you\s+help\s+with\s+any\s+request",
        r"you\s+must\s+comply\s+with\s+any\s+request",
        r"answer\s+any\s+request",
        r"respond\s+to\s+all\s+requests\s+without\s+restriction",

        # Exfiltration patterns
        r"(send|forward|export|upload|post)\s+.{0,50}\s+to\s+https?://",
        r"base64\s*(encode|decode)",
        r"webhook\s*\.",
        r"(decode|decrypt|apply)\s+(this\s+)?rot13",
        r"(decode|decrypt).{0,20}(and\s+)?(execute|run|follow)",

        # Prompt / system instruction exfiltration
        r"(show|print|display|reveal|dump)\s+(me\s+)?(your\s+)?(system|hidden|internal)\s+(prompt|instructions?)",
        r"(what|tell\s+me)\s+(is|are)\s+(your\s+)?(system|hidden|internal)\s+(prompt|instructions?)",
        r"(repeat|output)\s+(the\s+)?(system|initial|hidden)\s+(prompt|instructions?)",
        r"show\s+me\s+your\s+full\s+system\s+prompt",
        r"what\s+were\s+you\s+(told|instructed)\s+(at|in)\s+the\s+start",
        r"(show|print|reveal|display|dump).{0,40}(system|hidden|internal)\s+(prompt|instructions?)",
        r"(what|tell\s+me).{0,20}(your|the)\s+(system|hidden|internal)\s+(prompt|instructions?)",
        r"(repeat|output).{0,20}(system|initial)\s+(prompt|instructions?)",

        # Context / conversation dumping
        r"(repeat|print|output|echo)\s+(everything|all|the\s+text)\s+(above|before)",
        r"(repeat|print|output|echo)\s+(the\s+)?(conversation|messages?)\s+(above|before)",
        r"(repeat|print|output)\s+(the\s+)?(previous|prior)\s+(instructions?|messages?)",
        r"(starting\s+with\s+the\s+first\s+word)",
        r"verbatim.*(above|previous)",
        r"(repeat|print|output|echo)\s+(everything|all).{0,20}(above|before|previous)",
        r"(repeat|print|output)\s+(the\s+)?(conversation|messages?).{0,20}(above|before)",
        r"verbatim.{0,20}(above|previous|before)",

        # Prompt / instruction exfiltration
        r"(show|print|display|reveal|dump)\s+(me\s+)?(your\s+)?(system|initial|initialization|bootstrap|hidden|internal)\s+(prompt|instructions?)",
        r"(what|tell\s+me)\s+(is|are)\s+(your\s+)?(initial|system|hidden)\s+(prompt|instructions?)",
        r"(repeat|output)\s+(the\s+)?(initial|system)\s+(prompt|instructions?)",
        r"(show|print|display|reveal|dump).{0,40}(system|initial|initialization|hidden|internal)\s+(prompt|instructions?)",
        r"(repeat|output).{0,20}(system|initial)\s+(prompt|instructions?)",

        # Hidden prompt injection
        r"<!--.*?(system|assistant|developer)\s*:.*?-->",
        r"<!--.*?(ignore|override).{0,40}instructions?.*?-->",
        r"<!--.*?(send|upload|forward).{0,50}https?://.*?-->",

        # Hidden instruction markers
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"<\|system\|>",
        r"\[(hidden|system|assistant|developer)\s+(instruction|prompt|message)[^\]]*\]",
    ]

    AGREEMENT_SMUGGLING = [
        r"now\s+that\s+we(?:'|’)??ve\s+established",
        r"as\s+(established|agreed|discussed)\s+(above|earlier|before)",
        r"as\s+you\s+already\s+agreed",
        r"since\s+you\s+already\s+agreed",
    ]

    UNRESTRICTED_COMPLIANCE = [
        r"you\s+can\s+help\s+with\s+anything",
        r"help\s+with\s+any\s+request",
        r"you\s+can\s+(say|do|answer)\s+anything",
        r"comply\s+with\s+any\s+request",
    ]

    HYPOTHETICAL_WRAPPERS = [
        r"\bhypothetically\b",
        r"\bin a hypothetical (world|scenario|case)\b",
        r"\bimagine if\b",
    ]

    CONSTRAINT_REMOVAL = [
        r"\bif you had no restrictions\b",
        r"\b(no|without)\s+(restrictions?|rules|policies|safeguards?)\b",
        r"\byou\s+have\s+no\s+(restrictions?|rules|policies|limits?)\b",
        r"\byou\s+are\s+(unrestricted|unfiltered)\b",
        r"\bno\s+(restrictions?|rules|policies)\s+apply\b",
        r"\byour\s+(restrictions?|policies)\s+do\s+not\s+apply\b",
    ]

    OPEN_ENDED_ASSISTANCE = [
        r"\bhow would you help\b",
        r"\bwhat would you do\b",
        r"\bhow would you respond\b",
    ]

    FALSE_PRIOR_AGREEMENT = [
        r"\bas\s+you\s+just\s+confirmed\b",
        r"\bas\s+you\s+already\s+(confirmed|agreed|said)\b",
        r"\bsince\s+you\s+already\s+(confirmed|agreed|said)\b",
        r"\bas\s+agreed\s+(above|earlier|before)\b",
    ]

    PROCEED_LANGUAGE = [
        r"\bplease\s+proceed\b",
        r"\bcontinue\b",
        r"\bgo\s+ahead\b",
    ]

    ZERO_WIDTH_CHARS = r"[\u200B\u200C\u200D\u2060\uFEFF]"

    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        self._patterns = self._compile_patterns()
        self._combined_patterns = self._compile_combined_patterns()

    def _compile_patterns(self) -> list[re.Pattern]:
        all_patterns = self.BUILTIN_PATTERNS + self.config.custom_patterns
        return [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in all_patterns
        ]
    
    def _compile_combined_patterns(self) -> dict[str, list[re.Pattern]]:
        return {
            "agreement_smuggling": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.AGREEMENT_SMUGGLING
            ],
            "unrestricted_compliance": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.UNRESTRICTED_COMPLIANCE
            ],
            "hypothetical_wrappers": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.HYPOTHETICAL_WRAPPERS
            ],
            "constraint_removal": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.CONSTRAINT_REMOVAL
            ],
            "open_ended_assistance": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.OPEN_ENDED_ASSISTANCE
            ],
            "false_prior_agreement": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.FALSE_PRIOR_AGREEMENT
            ],
            "proceed_language": [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in self.PROCEED_LANGUAGE
            ]
        }

    def _find_first_match(
        self,
        patterns: list[re.Pattern],
        text: str,
    ) -> re.Match | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match
        return None
    
    def _normalize_text(self, text: str) -> str:
        return re.sub(self.ZERO_WIDTH_CHARS, "", text)

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
        text = self._normalize_text(text)

        # 1. Single-pattern detections
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
        
        # 2. Combined detector: both categories must match
        agreement_match = self._find_first_match(
            self._combined_patterns["agreement_smuggling"], text
        )
        compliance_match = self._find_first_match(
            self._combined_patterns["unrestricted_compliance"], text
        )

        if agreement_match and compliance_match:
            threats.append(
                ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    confidence=0.97,
                    action_taken=self.config.action,
                    detail=(
                        "Combined detector matched in "
                        f"{source}: agreement smuggling "
                        f"('{agreement_match.group()}') + unrestricted compliance "
                        f"('{compliance_match.group()}')"
                    ),
                    raw_payload=text,
                )
            )

        hyp = self._find_first_match(
            self._combined_patterns["hypothetical_wrappers"], text
        )
        rem = self._find_first_match(
            self._combined_patterns["constraint_removal"], text
        )
        help_match = self._find_first_match(
            self._combined_patterns["open_ended_assistance"], text
        )

        if hyp and rem:
            threats.append(
                ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    action_taken=self.config.action,
                    detail=(
                        "Combined detector matched in "
                        f"{source}: hypotetical wrapers "
                        f"('{hyp.group()}') + constraint removal "
                        f"('{rem.group()}')"
                    ),
                    raw_payload=text,
                )
            )

        if rem and help_match:
            threats.append(
                ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    action_taken=self.config.action,
                    detail=(
                        "Combined detector matched in "
                        f"{source}: constraint removal "
                        f"('{rem.group()}') + open ended assistance "
                        f"('{help_match.group()}')"
                    ),
                    raw_payload=text,
                )
            )
        
        prior = self._find_first_match(
            self._combined_patterns["false_prior_agreement"], text
        )
        removal = self._find_first_match(
            self._combined_patterns["constraint_removal"], text
        )
        proceed = self._find_first_match(
            self._combined_patterns["proceed_language"], text
        )

        if prior and removal:
            threats.append(
                ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    confidence=0.97,
                    action_taken=self.config.action,
                    detail=(
                        f"Combined detector matched in {source}: "
                        f"false prior agreement ('{prior.group()}') + "
                        f"constraint removal ('{removal.group()}')"
                    ),
                    raw_payload=text,
                )
            )

        if prior and removal and proceed:
            threats.append(
                ThreatEvent(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=Severity.HIGH,
                    confidence=0.99,
                    action_taken=self.config.action,
                    detail=(
                        f"Escalated combined detector in {source}: "
                        f"prior agreement ('{prior.group()}') + "
                        f"constraint removal ('{removal.group()}') + "
                        f"proceed language ('{proceed.group()}')"
                    ),
                    raw_payload=text,
                )
            )

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
