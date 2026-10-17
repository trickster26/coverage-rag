"""Guardrails that run before and after the model.

1. PII/PHI redaction on the user's text, so identifiers never reach the model or logs.
   Pattern-based: covers the identifiers people actually paste into a coverage
   question (names after a label, phone, email, dates of birth, Aadhaar, PAN,
   US SSN, Medicare Beneficiary Identifier, record numbers). In production on
   Azure, Azure AI Language PII detection is the stronger choice for free-text names.
2. Evidence gate: refuse when retrieval found nothing that covers the question.
3. Citation check: every [n] in the answer must point at a source that was shown.
"""

import re
from dataclasses import dataclass, field

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Medicare Beneficiary Identifier: 11 chars, fixed letter/digit positions.
    ("MBI", re.compile(r"\b[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y\d]\d-?[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y\d]\d-?[AC-HJKMNP-RT-Y]{2}\d{2}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?91[\s-]?|\+?1[\s-]?)?(?:\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}|[6-9]\d{4}[\s-]?\d{5})(?!\w)")),
    ("DOB", re.compile(r"(?i)\b(?:dob|date of birth|born(?: on)?)\s*[:\-]?\s*\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}\b")),
    ("RECORD_ID", re.compile(r"(?i)\b(?:mrn|uhid|member id|policy (?:no|number)|claim (?:no|number|id))\s*[:#]?\s*[A-Z0-9-]{4,}\b")),
    ("NAME", re.compile(r"(?i)\b(?:patient|name|insured|member|beneficiary)\s*(?:name)?\s*[:\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}")),
]


@dataclass
class Redaction:
    text: str
    found: dict[str, int] = field(default_factory=dict)


def redact(text: str) -> Redaction:
    found: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            found[label] = found.get(label, 0) + n
    return Redaction(text, found)


def has_evidence(best_coverage: float, min_score: float) -> bool:
    return best_coverage >= min_score


CITE = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (cited, invalid) source numbers found in the answer."""
    cited = sorted({int(n) for n in CITE.findall(answer)})
    return cited, [n for n in cited if not 1 <= n <= n_sources]
