"""Typed data model for the extraction pipeline.

Every value that leaves the pipeline is a :class:`FieldValue`, which carries the
predicted (normalised) value *as well as* where it came from. Provenance is not
optional: the reviewer UI, the search index and the evaluation harness all need
the page number and the pixel box, otherwise an extracted number cannot be
verified against the source document.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FieldName(str, Enum):
    DOCUMENT_TYPE = "document_type"
    INVOICE_NUMBER = "invoice_number"
    ISSUE_DATE = "issue_date"
    DUE_DATE = "due_date"
    VENDOR_NAME = "vendor_name"
    VENDOR_GSTIN = "vendor_gstin"
    BUYER_NAME = "buyer_name"
    CURRENCY = "currency"
    SUBTOTAL = "subtotal"
    TAX_AMOUNT = "tax_amount"
    TOTAL_AMOUNT = "total_amount"


#: Fields that are always expected on a commercial invoice, used by validation
#: to raise ``MISSING_FIELD`` instead of failing silently.
REQUIRED_INVOICE_FIELDS: tuple[FieldName, ...] = (
    FieldName.INVOICE_NUMBER,
    FieldName.ISSUE_DATE,
    FieldName.VENDOR_NAME,
    FieldName.TOTAL_AMOUNT,
)

#: Money-valued fields. Kept separate so arithmetic checks and the money-aware
#: tokenizer do not have to re-derive the set from the enum.
MONEY_FIELDS: tuple[FieldName, ...] = (
    FieldName.SUBTOTAL,
    FieldName.TAX_AMOUNT,
    FieldName.TOTAL_AMOUNT,
)


class DocClass(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CREDIT_NOTE = "credit_note"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    """Axis-aligned box in PDF points, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def h_overlap(self, other: BBox) -> float:
        """Horizontal overlap fraction relative to the narrower box."""
        span = min(self.x1, other.x1) - max(self.x0, other.x0)
        if span <= 0:
            return 0.0
        return span / max(1e-6, min(self.width, other.width))

    def v_gap(self, other: BBox) -> float:
        """Vertical distance between boxes (0 when they overlap vertically)."""
        if other.y0 >= self.y1:
            return other.y0 - self.y1
        if self.y0 >= other.y1:
            return self.y0 - other.y1
        return 0.0

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Token(BaseModel):
    """A word with its geometry. The atom of every extractor.

    ``source`` records whether the word came from a real text layer or from OCR,
    and ``ocr_confidence`` is kept because Tesseract confidences are informative
    enough to feed the field-confidence model.
    """

    text: str
    bbox: BBox
    page: int = 0
    ocr_confidence: float | None = None

    # populated by layout.index_tokens(); cheap to carry around, avoids O(n^2)
    line_id: int = -1
    token_id: int = -1

    @property
    def is_numeric(self) -> bool:
        stripped = self.text.replace(",", "").replace(".", "").replace("₹", "")
        return stripped.replace("-", "").isdigit()


class Page(BaseModel):
    number: int
    width: float
    height: float
    tokens: list[Token] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Reading-order text, one line per detected line."""
        lines: dict[int, list[Token]] = {}
        for tok in self.tokens:
            lines.setdefault(tok.line_id, []).append(tok)
        out = []
        for line_id in sorted(lines):
            row = sorted(lines[line_id], key=lambda t: t.bbox.x0)
            out.append(" ".join(t.text for t in row))
        return "\n".join(out)


class SourceKind(str, Enum):
    PDF_TEXT = "pdf_text"       # embedded text layer, lossless
    OCR = "ocr"                 # rasterised page + OCR engine
    SYNTHETIC = "synthetic"     # test fixtures


class Document(BaseModel):
    doc_id: str
    filename: str
    pages: list[Page] = Field(default_factory=list)
    source: SourceKind = SourceKind.PDF_TEXT
    engine: str = "pdftext"
    sha256: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def tokens(self) -> list[Token]:
        return [t for p in self.pages for t in p.tokens]

    def page(self, number: int) -> Page:
        return self.pages[number]

    @property
    def n_tokens(self) -> int:
        return sum(len(p.tokens) for p in self.pages)

    def checksum(self) -> str:
        h = hashlib.sha256()
        h.update(self.text.encode("utf-8"))
        return h.hexdigest()


class Provenance(BaseModel):
    page: int = 0
    bbox: BBox | None = None
    source: str = "rules"          # which extractor produced it
    raw_text: str = ""             # the exact string before normalisation
    rule: str = ""                 # named rule / anchor that matched


class FieldValue(BaseModel):
    field: FieldName
    value: Any
    confidence: float = 0.0
    provenance: Provenance = Field(default_factory=Provenance)
    alternatives: list[Any] = Field(default_factory=list)
    #: per-signal breakdown, persisted so confidence can be audited later
    confidence_parts: dict[str, float] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class Severity(str, Enum):
    BLOCKING = "blocking"   # arithmetic / identity violated
    WARNING = "warning"     # plausible but suspicious
    INFO = "info"


class Issue(BaseModel):
    code: str
    message: str
    severity: Severity = Severity.WARNING
    fields: list[FieldName] = Field(default_factory=list)


class DocStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ReviewReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    VALIDATION_ISSUE = "validation_issue"
    MISSING_FIELD = "missing_field"
    EXTRACTOR_DISAGREEMENT = "extractor_disagreement"
    LOW_OCR_QUALITY = "low_ocr_quality"


class ReviewTask(BaseModel):
    field: FieldName | None = None
    reason: ReviewReason
    priority: float = 0.5
    detail: str = ""


class ExtractionResult(BaseModel):
    document: Document
    doc_type: DocClass = DocClass.UNKNOWN
    doc_type_confidence: float = 0.0
    fields: dict[FieldName, FieldValue] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)
    review_tasks: list[ReviewTask] = Field(default_factory=list)
    status: DocStatus = DocStatus.NEEDS_REVIEW
    extractor: str = "rules_layout"
    latency_ms: float = 0.0

    def value(self, field: FieldName) -> Any:
        fv = self.fields.get(field)
        return None if fv is None else fv.value

    def to_flat(self) -> dict[str, Any]:
        return {f.value: (self.fields[f].value if f in self.fields else None) for f in FieldName}

    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.BLOCKING]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_date(value: Any) -> date | None:
    return value if isinstance(value, date) else None
