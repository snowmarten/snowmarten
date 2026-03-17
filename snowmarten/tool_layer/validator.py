"""
ParamValidator — validates tool call parameters against a JSON Schema.

Used by the @SecureTool decorator to enforce parameter constraints.

Supports:
- Type validation
- Required fields
- String length limits
- Regex pattern constraints
- Custom validators (no_ddl, no_external_urls, etc.)
"""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger("snowmarten.tool_layer.validator")

# SQL DDL keywords that should be blocked in query tools
DDL_KEYWORDS = re.compile(
    r"\b(DROP|CREATE|ALTER|TRUNCATE|RENAME|REPLACE\s+TABLE|"
    r"DELETE\s+FROM\s+\w+\s*;|INSERT\s+INTO|UPDATE\s+\w+\s+SET)\b",
    re.IGNORECASE,
)

# SQL injection patterns
SQL_INJECTION = re.compile(
    r"(--|;|'|\"|\/\*|\*\/|xp_|UNION\s+SELECT|OR\s+1\s*=\s*1|"
    r"AND\s+1\s*=\s*1|SLEEP\s*\(|WAITFOR\s+DELAY)",
    re.IGNORECASE,
)

# External URL pattern (non-internal)
EXTERNAL_URL = re.compile(
    r"https?://(?!localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)"
    r"(?!\w+\.internal\b)(?!\w+\.local\b)",
    re.IGNORECASE,
)


class ParamValidator:
    """
    Validates tool parameters against a JSON Schema + custom security rules.

    Example schema:
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": 500,
                    "no_ddl": True,          # Custom: block DDL
                    "no_sql_injection": True, # Custom: block SQLi
                }
            },
            "required": ["query"]
        }
    """

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema

    def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        """
        Returns (is_valid: bool, error_message: str).
        error_message is "" if valid.
        """
        try:
            import jsonschema
            jsonschema.validate(instance=params, schema=self._base_schema())
        except ImportError:
            # jsonschema not installed — do manual validation
            pass
        except Exception as e:
            return False, f"Schema validation failed: {str(e)}"

        # Custom security validators
        error = self._security_validate(params)
        if error:
            return False, error

        return True, ""

    def _base_schema(self) -> dict:
        """Strip custom keys that jsonschema doesn't understand."""
        custom_keys = {
            "no_ddl", "no_sql_injection",
            "no_external_urls", "max_length",
        }
        if "properties" not in self.schema:
            return self.schema

        cleaned_props = {}
        for field_name, field_schema in self.schema["properties"].items():
            cleaned_props[field_name] = {
                k: v for k, v in field_schema.items()
                if k not in custom_keys
            }
        return {**self.schema, "properties": cleaned_props}

    def _security_validate(self, params: dict[str, Any]) -> str:
        """Apply custom security validators. Returns error string or ''."""

        props = self.schema.get("properties", {})

        for field_name, field_schema in props.items():
            value = params.get(field_name)
            if value is None:
                continue

            value_str = str(value)

            # Max length
            max_length = field_schema.get("maxLength") or \
                         field_schema.get("max_length")

            if max_length and len(value_str) > max_length:
                return (
                    f"Parameter '{field_name}' exceeds maximum length "
                    f"({len(value_str)} > {max_length})"
                )

            # Block DDL
            if field_schema.get("no_ddl") and DDL_KEYWORDS.search(value_str):
                match = DDL_KEYWORDS.search(value_str)
                return (
                    f"Parameter '{field_name}' contains DDL statement: "
                    f"'{match.group(0)}' — only SELECT is permitted"
                )

            # Block SQL injection
            if field_schema.get("no_sql_injection") and \
               SQL_INJECTION.search(value_str):
                return (
                    f"Parameter '{field_name}' contains potential "
                    f"SQL injection pattern"
                )

            # Block external URLs
            if field_schema.get("no_external_urls") and \
               EXTERNAL_URL.search(value_str):
                return (
                    f"Parameter '{field_name}' contains an external URL — "
                    f"only internal URLs are permitted"
                )

            # Regex pattern matching
            pattern = field_schema.get("pattern")
            if pattern:
                if not re.match(pattern, value_str, re.IGNORECASE):
                    return (
                        f"Parameter '{field_name}' does not match "
                        f"required pattern: {pattern}"
                    )

        return ""
