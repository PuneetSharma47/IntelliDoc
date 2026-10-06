"""Extractor interfaces and shared candidate plumbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import Document, FieldName, Provenance

__all__ = ["Candidate", "Extractor", "ExtractionContext", "pick_best"]


@dataclass
class Candidate:
    """A single field hypothesis before arbitration."""

    field: FieldName
    value: Any
    raw: str
    score: float
    provenance: Provenance
    note: str = ""

    def key(self) -> str:
        return f"{self.field.value}={self.value!r}"


@dataclass
class ExtractionContext:
    """Everything an extractor may use; keeps signatures stable as features grow."""

    document: Document
    dayfirst: bool = True
    doc_type: str = "invoice"
    extras: dict[str, Any] = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:  # pragma: no cover
        ...


def pick_best(candidates: list[Candidate]) -> Candidate | None:
    """Return the highest-scoring candidate, breaking ties by document order.

    Deliberately does not average: averaging two disagreeing money values
    produces a third value that appears nowhere in the document, which is worse
    than picking one and flagging the disagreement in confidence scoring.
    """
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (-c.score, c.provenance.page, c.provenance.bbox.y0 if c.provenance.bbox else 0.0))[0]
