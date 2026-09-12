"""
IoC (Indicator of Compromise) pattern detection.

Deterministic step — no LLM needed. Patterns are matched against entity names
extracted in the raw extraction phase. Matched entities have their type_hint
overridden with a schema-agnostic description (e.g. "IPv4 address", "MD5 hash")
so that search_classes can find the right class regardless of ontology schema.
"""

import re
from dataclasses import dataclass


@dataclass
class IocMatch:
    ioc_type: str    # e.g. "cve", "ipv4", "md5"
    type_hint: str   # schema-agnostic hint for search_classes, e.g. "IPv4 address"


# Order matters — more specific patterns first
_IOC_RULES: list[tuple[re.Pattern, str, str]] = [
    # (pattern, ioc_type, type_hint)
    (
        re.compile(r'^CVE-\d{4}-\d{4,7}$', re.IGNORECASE),
        "cve",
        "CVE vulnerability identifier",
    ),
    (
        re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?$'),
        "ipv4",
        "IPv4 address",
    ),
    (
        re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'),
        "ipv6",
        "IPv6 address",
    ),
    (
        re.compile(r'^[0-9a-fA-F]{64}$'),
        "sha256",
        "SHA-256 hash",
    ),
    (
        re.compile(r'^[0-9a-fA-F]{40}$'),
        "sha1",
        "SHA-1 hash",
    ),
    (
        re.compile(r'^[0-9a-fA-F]{32}$'),
        "md5",
        "MD5 hash",
    ),
    (
        re.compile(r'^https?://\S+$'),
        "url",
        "URL",
    ),
    (
        re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "email",
        "email address",
    ),
    (
        re.compile(r'^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'),
        "domain",
        "domain name",
    ),
]

# Scan full text for IoC mentions (used by scan_iocs_in_text)
_IOC_SCAN_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE), "cve",    "CVE vulnerability identifier"),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'),       "ipv4",   "IPv4 address"),
    (re.compile(r'\b[0-9a-fA-F]{64}\b'),                              "sha256", "SHA-256 hash"),
    (re.compile(r'\b[0-9a-fA-F]{40}\b'),                              "sha1",   "SHA-1 hash"),
    (re.compile(r'\b[0-9a-fA-F]{32}\b'),                              "md5",    "MD5 hash"),
    (re.compile(r'https?://\S+'),                                      "url",    "URL"),
]


def detect_ioc(entity_name: str) -> IocMatch | None:
    """Check if an entity name is an IoC. Returns IocMatch or None."""
    name = entity_name.strip()
    for pattern, ioc_type, type_hint in _IOC_RULES:
        if pattern.match(name):
            return IocMatch(ioc_type=ioc_type, type_hint=type_hint)
    return None


def scan_iocs_in_text(text: str) -> list[dict]:
    """Scan raw text for IoC values. Returns [{value, ioc_type, type_hint}]."""
    found = []
    seen = set()
    for pattern, ioc_type, type_hint in _IOC_SCAN_RULES:
        for m in pattern.finditer(text):
            val = m.group(0)
            if val not in seen:
                seen.add(val)
                found.append({"value": val, "ioc_type": ioc_type, "type_hint": type_hint})
    return found


