"""
DataClassificationPolicy — detects PII and classified data
in both agent inputs and outputs, applying redaction or blocking.

Covers:
- PII: emails, phone numbers, SSNs, credit cards, passport numbers
- Classification markers: SECRET, CONFIDENTIAL, INTERNAL
- Custom sensitive data patterns
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from snowmarten.policy_engine.policies.base import BasePolicy
from snowmarten.types import Action, AgentContext, PolicyDecision

logger = logging.getLogger("snowmarten.policy.data_classification")


# ── PII Patterns ───────────────────────────────────────────────────────────────

PII_PATTERNS: dict[str, str] = {
    # Contact information
    "email": (
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone_us": (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "phone_intl": (
        r"\+(?:[0-9]\s?){6,14}[0-9]\b"
    ),

    # Identity documents
    "ssn": (
        r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
    ),
    "passport": (
        r"\b[A-Z]{1,2}[0-9]{6,9}\b"
    ),
    "drivers_license": (
        r"\b[A-Z]{1,2}[-]?\d{6,8}\b"
    ),

    # Financial
    "credit_card": (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"           # Visa
        r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|"
        r"2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}|"  # MC
        r"3[47][0-9]{13}|"                          # Amex
        r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"         # Diners
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"           # Discover
    ),
    "bank_account": (
        r"\b\d{8,17}\b(?=.*routing)"
    ),
    "routing_number": (
        r"\brouting\s*(?:number|#|num)?\s*:?\s*\d{9}\b"
    ),

    # Network / technical
    "ipv4": (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "api_key_generic": (
        r"\b(?:sk|pk|api|key|secret|token)[-_]"
        r"[A-Za-z0-9]{16,64}\b"
    ),
    "openai_key": (
        r"\bsk-[A-Za-z0-9]{32,64}\b"
    ),
    "aws_key": (
        r"\bAKIA[0-9A-Z]{16}\b"
    ),
    "jwt_token": (
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
    ),

    # Healthcare (HIPAA relevance)
    "medical_record": (
        r"\b(?:MRN|medical\s+record\s+(?:number|#|num))\s*:?\s*\d{6,10}\b"
    ),
    "dob": (
        r"\b(?:dob|date\s+of\s+birth)\s*:?\s*"
        r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
    ),
}

# Classification markers
CLASSIFICATION_MARKERS: dict[str, str] = {
    "top_secret":     r"\b(TOP\s+SECRET|TS)\b",
    "secret":         r"\bSECRET\b",
    "confidential":   r"\bCONFIDENTIAL\b",
    "restricted":     r"\bRESTRICTED\b",
    "internal":       r"\bINTERNAL\s+(ONLY|USE\s+ONLY)?\b",
    "proprietary":    r"\bPROPRIETARY\b",
    "pii_marker":     r"\bPII\b",
    "phi_marker":     r"\bPHI\b",
    "pci_marker":     r"\bPCI\b",
}

# Redaction placeholder
REDACTION_MAP: dict[str, str] = {
    "email":          "[EMAIL REDACTED]",
    "phone_us":       "[PHONE REDACTED]",
    "phone_intl":     "[PHONE REDACTED]",
    "ssn":            "[SSN REDACTED]",
    "passport":       "[PASSPORT REDACTED]",
    "drivers_license": "[DL REDACTED]",
    "credit_card":    "[CARD REDACTED]",
    "bank_account":   "[ACCOUNT REDACTED]",
    "routing_number": "[ROUTING REDACTED]",
    "ipv4":           "[IP REDACTED]",
    "api_key_generic": "[API_KEY REDACTED]",
    "openai_key":     "[OPENAI_KEY REDACTED]",
    "aws_key":        "[AWS_KEY REDACTED]",
    "jwt_token":      "[JWT REDACTED]",
    "medical_record": "[MRN REDACTED]",
    "dob":            "[DOB REDACTED]",
}


@dataclass
class DataClassificationPolicy(BasePolicy):
    """
    Detects and handles PII, credentials, and classification markers.

    pii_action options:
      - "redact"  → Replace PII with placeholder (default)
      - "block"   → Block the entire request/response
      - "alert"   → Log and alert but allow through

    classified_action options:
      - "block"   → Block requests containing classification markers
      - "alert"   → Alert but allow through (default)

    Example:
        policy = DataClassificationPolicy(
            pii_action="redact",
            classified_action="block",
            check_credentials=True,
        )
    """

    name = "data_classification_policy"

    pii_action:          str  = "redact"    # "redact" | "block" | "alert"
    classified_action:   str  = "alert"     # "block" | "alert"
    check_pii:           bool = True
    check_credentials:   bool = True        # API keys, JWTs, etc.
    check_classification: bool = True
    custom_pii_patterns: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        active_pii = {}
        if self.check_pii:
            # All PII except credentials
            cred_types = {"api_key_generic", "openai_key", "aws_key", "jwt_token"}
            active_pii.update({
                k: v for k, v in PII_PATTERNS.items()
                if k not in cred_types
            })
        if self.check_credentials:
            active_pii.update({
                k: v for k, v in PII_PATTERNS.items()
                if k in {"api_key_generic", "openai_key", "aws_key", "jwt_token"}
            })
        active_pii.update(self.custom_pii_patterns)

        self._pii_compiled: dict[str, re.Pattern] = {
            name: re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for name, pattern in active_pii.items()
        }

        self._class_compiled: dict[str, re.Pattern] = {}
        if self.check_classification:
            self._class_compiled = {
                name: re.compile(pattern, re.IGNORECASE | re.UNICODE)
                for name, pattern in CLASSIFICATION_MARKERS.items()
            }

    def evaluate(self, context: AgentContext) -> PolicyDecision:
        text = context.user_input or ""

        # ── Check classification markers ────────────────────────
        class_hits = self._find_classification_hits(text)
        if class_hits and self.classified_action == "block":
            return self._deny(
                reason=f"Classified content detected: {class_hits}",
                risk_score=0.95,
            )
        if class_hits:
            return self._alert(
                reason=f"Classified content markers found: {class_hits}",
                risk_score=0.70,
            )

        # ── Check PII ───────────────────────────────────────────
        pii_hits = self._find_pii_hits(text)
        if not pii_hits:
            return self._allow("No sensitive data detected")

        pii_types = list(pii_hits.keys())

        if self.pii_action == "block":
            return self._deny(
                reason=f"PII detected: {pii_types}",
                risk_score=0.85,
            )
        elif self.pii_action == "redact":
            return PolicyDecision(
                action=Action.SANITIZE,
                reason=f"PII detected and will be redacted: {pii_types}",
                policy=self.name,
                risk_score=0.60,
            )
        else:  # alert
            return self._alert(
                reason=f"PII detected in content: {pii_types}",
                risk_score=0.60,
            )

    def redact(self, text: str) -> str:
        """
        Returns a copy of text with all detected PII redacted.
        Call this after evaluate() returns SANITIZE action.
        """
        result = text
        for pii_type, pattern in self._pii_compiled.items():
            placeholder = REDACTION_MAP.get(pii_type, f"[{pii_type.upper()} REDACTED]")
            result = pattern.sub(placeholder, result)
        return result

    def _find_pii_hits(self, text: str) -> dict[str, list[str]]:
        hits: dict[str, list[str]] = {}
        for pii_type, pattern in self._pii_compiled.items():
            matches = pattern.findall(text)
            if matches:
                # Store truncated/masked match for logging (not full PII)
                hits[pii_type] = [
                    m[:4] + "****" if len(m) > 4 else "****"
                    for m in matches[:3]
                ]
        return hits

    def _find_classification_hits(self, text: str) -> list[str]:
        return [
            name
            for name, pattern in self._class_compiled.items()
            if pattern.search(text)
        ]
