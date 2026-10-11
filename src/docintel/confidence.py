"""Field confidence scoring.

A confidence number is only useful if it can be defended, so every field carries
the parts it was built from (:attr:`FieldValue.confidence_parts`):

``base``        score of the winning extraction strategy (label+same-row beats a
                bare pattern match)
``consensus``   bonus/penalty from whether independent candidates agreed
``ocr``         penalty from mean OCR confidence of the supporting tokens
``geometry``    penalty when the value sits far from its label or on a later page
``agreement``   bonus when a second extractor produced the same value
``validation``  penalty applied after arithmetic checks fail

The routed decision (auto-approve / review / reject) is *not* made here; see
:mod:`docintel.review`.
"""

from __future__ import annotations

from .config import Settings
from .normalize import round_money
from .schema import Document, FieldName, FieldValue

#: Fields whose numbers must be exactly right before they are trusted.
STRICT_FIELDS = {
    FieldName.SUBTOTAL,
    FieldName.TAX_AMOUNT,
    FieldName.TOTAL_AMOUNT,
    FieldName.INVOICE_NUMBER,
    FieldName.VENDOR_GSTIN,
}


def _value_key(field: FieldName, value: object) -> str:
    if isinstance(value, float):
        return f"{round_money(value):.2f}"
    return str(value).strip().lower()


def _geometry_penalty(document: Document, value: FieldValue) -> float:
    """Down-weight values found far from their label or on a later page."""
    penalty = 0.0
    if value.provenance.page > 0 and value.field is not FieldName.TOTAL_AMOUNT:
        penalty += 0.05
    box = value.provenance.bbox
    if box is not None and value.provenance.page < len(document.pages):
        page = document.pages[value.provenance.page]
        if page.width and box.x1 > page.width:  # box outside the page: OCR drift
            penalty += 0.10
    if value.provenance.rule == "id_pattern":
        penalty += 0.08
    return penalty


def score_field(
    document: Document,
    field: FieldName,
    candidates: list,  # list[Candidate]
    settings: Settings,
    extra_agreement: bool = False,
) -> FieldValue | None:
    from .extract.base import pick_best

    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda c: -c.score)
    best = pick_best(candidates)
    assert best is not None

    parts: dict[str, float] = {"base": round(float(best.score), 4)}

    same = [c for c in ordered if _value_key(field, c.value) == _value_key(field, best.value)]
    different = [c for c in ordered if _value_key(field, c.value) != _value_key(field, best.value)]

    consensus = 0.0
    if len(same) > 1:
        consensus = 0.04 + 0.02 * (len(same) - 2)
    if different:
        runner_up = different[0]
        gap = best.score - runner_up.score
        if gap < 0.05:
            consensus -= 0.12 if field in STRICT_FIELDS else 0.07
        elif gap < 0.12:
            consensus -= 0.05
    parts["consensus"] = round(consensus, 4)

    ocr = 0.0
    confs = [t.ocr_confidence for t in document.tokens if t.ocr_confidence is not None]
    if confs:
        tokens_conf = [t.ocr_confidence for t in document.tokens if t.ocr_confidence is not None]
        mean_conf = sum(tokens_conf) / len(tokens_conf)
        ocr = -0.15 * max(0.0, 0.85 - mean_conf)
    parts["ocr"] = round(ocr, 4)

    geometry = -_geometry_penalty(document, best)
    parts["geometry"] = round(geometry, 4)

    agreement = settings.extraction.agreement_bonus if extra_agreement else 0.0
    parts["agreement"] = round(agreement, 4)

    confidence = sum(parts.values())
    confidence = max(0.0, min(1.0, confidence))

    # Only near-miss readings are worth showing a reviewer: a header line that
    # scored 0.4 on a vendor field is noise, not an alternative.
    alternatives = []
    for cand in different:
        if best.score - cand.score > 0.15 or len(alternatives) >= 3:
            continue
        if cand.value not in alternatives:
            alternatives.append(cand.value)

    return FieldValue(
        field=field,
        value=best.value,
        confidence=round(confidence, 4),
        provenance=best.provenance,
        alternatives=alternatives,
        confidence_parts=parts,
    )


def score_all(
    document: Document,
    candidates_by_field: dict[FieldName, list],
    settings: Settings,
    agreements: set[FieldName] | None = None,
) -> dict[FieldName, FieldValue]:
    agreements = agreements or set()
    out: dict[FieldName, FieldValue] = {}
    for field, candidates in candidates_by_field.items():
        value = score_field(document, field, candidates, settings, extra_agreement=field in agreements)
        if value is not None:
            out[field] = value
    return out


def apply_validation_penalty(value: FieldValue, amount: float, reason: str) -> None:
    """Bookkeeping helper used by the validation stage."""
    current = value.confidence_parts.get("validation", 0.0)
    value.confidence_parts["validation"] = round(current - amount, 4)
    value.confidence = max(0.0, min(1.0, round(value.confidence - amount, 4)))
