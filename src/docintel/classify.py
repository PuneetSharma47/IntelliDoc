"""Document type classification.

Run before extraction, because the expected field set differs: a receipt has no
GSTIN block and often no due date, so validating a receipt against invoice rules
manufactures false errors. A keyword/prior classifier is enough for this task and
is trivially explainable; the interface is the same one a trained classifier
would expose if the corpus justified one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import DocClass, Document

CUE_WEIGHTS: dict[DocClass, list[tuple[re.Pattern[str], float]]] = {
    DocClass.INVOICE: [
        (re.compile(r"\btax\s+invoice\b", re.I), 0.55),
        (re.compile(r"\binvoice\s*(?:no|number|#)", re.I), 0.35),
        (re.compile(r"\bgstin\b", re.I), 0.15),
        (re.compile(r"\bbill\s*to\b", re.I), 0.12),
    ],
    DocClass.RECEIPT: [
        (re.compile(r"\bcash\s+memo\b|\breceipt\b", re.I), 0.5),
        (re.compile(r"\bthank\s+you\b", re.I), 0.15),
        (re.compile(r"\bchange\b|\bcashier\b", re.I), 0.15),
        (re.compile(r"\bgstin\b", re.I), -0.1),
    ],
    DocClass.CREDIT_NOTE: [
        (re.compile(r"\bcredit\s+note\b", re.I), 0.7),
        (re.compile(r"\brefund\b|\breturn\b", re.I), 0.15),
    ],
}


@dataclass(frozen=True)
class Classification:
    doc_type: DocClass
    confidence: float
    evidence: str


def classify(document: Document) -> Classification:
    text = document.text
    scores: dict[DocClass, float] = {}
    evidence: dict[DocClass, str] = {}
    for doc_class, cues in CUE_WEIGHTS.items():
        total = 0.0
        hits: list[str] = []
        for pattern, weight in cues:
            match = pattern.search(text)
            if match:
                total += weight
                hits.append(match.group(0).strip().lower())
        scores[doc_class] = total
        evidence[doc_class] = ", ".join(sorted(set(hits)))

    best = max(scores, key=lambda c: scores[c])
    raw = scores[best]
    if raw <= 0:
        return Classification(DocClass.UNKNOWN, 0.3, "no cue phrases matched")
    runner = max((v for k, v in scores.items() if k is not best), default=0.0)
    margin = raw - runner
    confidence = min(0.99, 0.5 + 0.45 * min(1.0, raw) + 0.3 * max(0.0, min(0.2, margin)))
    return Classification(best, round(confidence, 3), evidence[best])
