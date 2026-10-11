"""Cross-field validation rules.

Extraction answers "what does this document say"; validation answers "does what
it says make sense". The arithmetic check is the highest-value rule in the whole
pipeline: a total that does not equal subtotal plus tax is either a transcription
error or a document that should not be paid, and both cases need a human.

Rules are intentionally simple and auditable -- an accounts team can read this
module and agree with it, which is not true of a model output.
"""

from __future__ import annotations

from datetime import date, timedelta

from .confidence import apply_validation_penalty
from .config import Settings
from .normalize import (
    GST_SLABS,
    GSTIN_RE,
    gstin_checksum,
    money_equal,
    round_money,
)
from .schema import (
    MONEY_FIELDS,
    REQUIRED_INVOICE_FIELDS,
    DocClass,
    FieldName,
    FieldValue,
    Issue,
    Severity,
)

#: Tolerance when matching an observed tax ratio to a statutory slab, in
#: percentage points. 0.6pp absorbs rounding to the paisa on small invoices.
SLAB_TOLERANCE_PP = 0.6


def _get(fields: dict[FieldName, FieldValue], name: FieldName):
    fv = fields.get(name)
    return None if fv is None else fv.value


def _penalise(fields: dict[FieldName, FieldValue], names: tuple[FieldName, ...], amount: float, reason: str) -> None:
    for name in names:
        if name in fields:
            apply_validation_penalty(fields[name], amount, reason)


def check_arithmetic(
    fields: dict[FieldName, FieldValue], settings: Settings
) -> list[Issue]:
    subtotal = _get(fields, FieldName.SUBTOTAL)
    tax = _get(fields, FieldName.TAX_AMOUNT)
    total = _get(fields, FieldName.TOTAL_AMOUNT)
    issues: list[Issue] = []
    if subtotal is None or tax is None or total is None:
        return issues

    tol = settings.validation.money_tolerance
    expected = round_money(float(subtotal) + float(tax))
    if not money_equal(expected, float(total), tol):
        diff = round_money(float(total) - expected)
        issues.append(
            Issue(
                code="ARITHMETIC_MISMATCH",
                message=(
                    f"subtotal {subtotal:,.2f} + tax {tax:,.2f} = {expected:,.2f} "
                    f"but the document states total {float(total):,.2f} (difference {diff:+,.2f})"
                ),
                severity=Severity.BLOCKING,
                fields=[FieldName.SUBTOTAL, FieldName.TAX_AMOUNT, FieldName.TOTAL_AMOUNT],
            )
        )
        _penalise(fields, (FieldName.SUBTOTAL, FieldName.TAX_AMOUNT, FieldName.TOTAL_AMOUNT), 0.25, "arithmetic")
    return issues


def check_tax_rate(fields: dict[FieldName, FieldValue], settings: Settings) -> list[Issue]:
    subtotal = _get(fields, FieldName.SUBTOTAL)
    tax = _get(fields, FieldName.TAX_AMOUNT)
    if subtotal is None or tax is None or float(subtotal) <= 0:
        return []
    ratio = float(tax) / float(subtotal) * 100.0
    closest = min(GST_SLABS, key=lambda slab: abs(slab - ratio))
    if abs(closest - ratio) <= SLAB_TOLERANCE_PP:
        return []
    if float(tax) == 0.0:
        return [
            Issue(
                code="ZERO_TAX",
                message="no tax was extracted; confirm the supply is exempt rather than a missed field",
                severity=Severity.WARNING,
                fields=[FieldName.TAX_AMOUNT],
            )
        ]
    return [
        Issue(
            code="TAX_RATE_ANOMALY",
            message=(
                f"implied tax rate {ratio:.2f}% does not match a GST slab "
                f"({', '.join(f'{s:g}%' for s in GST_SLABS)})"
            ),
            severity=Severity.WARNING,
            fields=[FieldName.TAX_AMOUNT, FieldName.SUBTOTAL],
        )
    ]


def check_money_completeness(fields: dict[FieldName, FieldValue], doc_type: DocClass) -> list[Issue]:
    """If some money fields were read, the others should have been too.

    A totals block prints subtotal, tax and total together. Extracting one and
    missing another is a layout failure, not a document that lacks the field, so
    it deserves a review task instead of silence.
    """
    present = [name for name in MONEY_FIELDS if name in fields]
    missing = [name for name in MONEY_FIELDS if name not in fields]
    if not present or not missing:
        return []
    if doc_type is DocClass.RECEIPT and FieldName.SUBTOTAL in missing:
        return []
    return [
        Issue(
            code="MONEY_BLOCK_INCOMPLETE",
            message=(
                f"read {', '.join(p.value for p in present)} but not "
                f"{', '.join(m.value for m in missing)}; the totals block was only partly parsed"
            ),
            severity=Severity.WARNING,
            fields=missing,
        )
    ]


def check_amount_signs(fields: dict[FieldName, FieldValue]) -> list[Issue]:
    issues: list[Issue] = []
    for name in MONEY_FIELDS:
        value = _get(fields, name)
        if value is None:
            continue
        if float(value) < 0:
            issues.append(
                Issue(
                    code="NEGATIVE_AMOUNT",
                    message=f"{name.value} is negative ({float(value):,.2f})",
                    severity=Severity.WARNING,
                    fields=[name],
                )
            )
    subtotal = _get(fields, FieldName.SUBTOTAL)
    total = _get(fields, FieldName.TOTAL_AMOUNT)
    if subtotal is not None and total is not None and float(subtotal) > float(total) + 0.01:
        issues.append(
            Issue(
                code="SUBTOTAL_GT_TOTAL",
                message=f"subtotal {float(subtotal):,.2f} exceeds total {float(total):,.2f}",
                severity=Severity.BLOCKING,
                fields=[FieldName.SUBTOTAL, FieldName.TOTAL_AMOUNT],
            )
        )
    return issues


def check_dates(
    fields: dict[FieldName, FieldValue], settings: Settings, today: date | None = None
) -> list[Issue]:
    issues: list[Issue] = []
    today = today or date.today()
    issue_date = _get(fields, FieldName.ISSUE_DATE)
    due_date = _get(fields, FieldName.DUE_DATE)
    if issue_date is not None:
        if issue_date > today + timedelta(days=settings.validation.max_future_days):
            issues.append(
                Issue(
                    code="DATE_IN_FUTURE",
                    message=f"invoice date {issue_date} is in the future",
                    severity=Severity.WARNING,
                    fields=[FieldName.ISSUE_DATE],
                )
            )
        if issue_date < today - timedelta(days=settings.validation.max_age_days):
            issues.append(
                Issue(
                    code="DATE_TOO_OLD",
                    message=f"invoice date {issue_date} is more than {settings.validation.max_age_days} days old",
                    severity=Severity.INFO,
                    fields=[FieldName.ISSUE_DATE],
                )
            )
    if issue_date is not None and due_date is not None and due_date < issue_date:
        issues.append(
            Issue(
                code="DATE_ORDER",
                message=f"due date {due_date} precedes invoice date {issue_date}",
                severity=Severity.WARNING,
                fields=[FieldName.DUE_DATE, FieldName.ISSUE_DATE],
            )
        )
    return issues


def check_gstin(fields: dict[FieldName, FieldValue], settings: Settings) -> list[Issue]:
    gstin = _get(fields, FieldName.VENDOR_GSTIN)
    if not gstin:
        return []
    gstin = str(gstin).upper()
    if not GSTIN_RE.match(gstin):
        return [
            Issue(
                code="GSTIN_FORMAT",
                message=f"'{gstin}' does not match the 15-character GSTIN format",
                severity=Severity.WARNING,
                fields=[FieldName.VENDOR_GSTIN],
            )
        ]
    try:
        expected = gstin_checksum(gstin)
    except Exception:  # noqa: BLE001 - defensive: never fail a document on this
        return []
    if expected != gstin[14]:
        severity = Severity.BLOCKING if settings.validation.require_gstin_checksum else Severity.WARNING
        return [
            Issue(
                code="GSTIN_CHECKSUM",
                message=f"GSTIN check digit should be '{expected}', document shows '{gstin[14]}'",
                severity=severity,
                fields=[FieldName.VENDOR_GSTIN],
            )
        ]
    return []


def check_required(
    fields: dict[FieldName, FieldValue], doc_type: DocClass
) -> list[Issue]:
    issues: list[Issue] = []
    required = REQUIRED_INVOICE_FIELDS
    if doc_type is DocClass.RECEIPT:
        required = (FieldName.INVOICE_NUMBER, FieldName.ISSUE_DATE, FieldName.TOTAL_AMOUNT)
    for name in required:
        if name not in fields or fields[name].value in (None, ""):
            severity = Severity.BLOCKING if name in (FieldName.TOTAL_AMOUNT, FieldName.INVOICE_NUMBER) else Severity.WARNING
            issues.append(
                Issue(
                    code="MISSING_FIELD",
                    message=f"required field '{name.value}' could not be extracted",
                    severity=severity,
                    fields=[name],
                )
            )
    return issues


def check_ocr_quality(document, fields: dict[FieldName, FieldValue]) -> list[Issue]:
    confs = [t.ocr_confidence for t in document.tokens if t.ocr_confidence is not None]
    if not confs:
        return []
    mean = sum(confs) / len(confs)
    if mean >= 0.80:
        return []
    return [
        Issue(
            code="LOW_OCR_QUALITY",
            message=f"mean OCR confidence {mean:.2f} on this document",
            severity=Severity.WARNING,
            fields=[],
        )
    ]


def validate(
    fields: dict[FieldName, FieldValue],
    doc_type: DocClass,
    document,
    settings: Settings,
    today: date | None = None,
) -> list[Issue]:
    """Run every rule and return the ordered issue list."""
    issues: list[Issue] = []
    issues += check_required(fields, doc_type)
    issues += check_arithmetic(fields, settings)
    issues += check_money_completeness(fields, doc_type)
    issues += check_amount_signs(fields)
    issues += check_tax_rate(fields, settings)
    issues += check_dates(fields, settings, today=today)
    issues += check_gstin(fields, settings)
    issues += check_ocr_quality(document, fields)
    order = {Severity.BLOCKING: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(issues, key=lambda i: order[i.severity])
