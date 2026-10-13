"""Human-in-the-loop routing.

The product decision encoded here: never silently accept a value the pipeline is
unsure about, and never send a whole pile of confident fields to a reviewer.
Review tasks are therefore *field-level*, ranked by how much risk they carry, and
capped so a reviewer sees the highest-impact items first.
"""

from __future__ import annotations

from .config import Settings
from .schema import (
    DocStatus,
    ExtractionResult,
    FieldName,
    FieldValue,
    Issue,
    ReviewReason,
    ReviewTask,
    Severity,
)

#: Risk weight per field; a wrong total costs more than a wrong vendor name.
FIELD_RISK = {
    FieldName.TOTAL_AMOUNT: 1.0,
    FieldName.SUBTOTAL: 0.95,
    FieldName.TAX_AMOUNT: 0.95,
    FieldName.INVOICE_NUMBER: 0.9,
    FieldName.VENDOR_GSTIN: 0.85,
    FieldName.ISSUE_DATE: 0.7,
    FieldName.DUE_DATE: 0.55,
    FieldName.VENDOR_NAME: 0.5,
    FieldName.BUYER_NAME: 0.45,
    FieldName.CURRENCY: 0.4,
    FieldName.DOCUMENT_TYPE: 0.3,
}


def field_threshold(name: FieldName, settings: Settings) -> float:
    """Auto-approve threshold for one field (default plus per-field overrides)."""
    return settings.review.field_thresholds.get(name.value, settings.review.field_auto_threshold)


def field_tasks(fields: dict[FieldName, FieldValue], settings: Settings) -> list[ReviewTask]:
    tasks: list[ReviewTask] = []
    for name, value in fields.items():
        threshold = field_threshold(name, settings)
        risk = FIELD_RISK.get(name, 0.5)
        if value.confidence < threshold:
            gap = max(0.0, threshold - value.confidence)
            tasks.append(
                ReviewTask(
                    field=name,
                    reason=ReviewReason.LOW_CONFIDENCE,
                    priority=min(1.0, risk * (0.6 + gap)),
                    detail=f"confidence {value.confidence:.2f} below auto-approve {threshold:.2f}",
                )
            )
        if value.alternatives:
            # Two competing readings are a signal in their own right: a confident
            # winner chosen from disagreeing candidates still deserves a glance.
            tasks.append(
                ReviewTask(
                    field=name,
                    reason=ReviewReason.EXTRACTOR_DISAGREEMENT,
                    priority=min(1.0, risk * 0.8),
                    detail="alternatives: " + ", ".join(str(a) for a in value.alternatives[:3]),
                )
            )
    return tasks


def issue_tasks(issues: list[Issue]) -> list[ReviewTask]:
    tasks: list[ReviewTask] = []
    for issue in issues:
        if issue.severity is Severity.INFO:
            continue
        reason = ReviewReason.VALIDATION_ISSUE
        if issue.code == "MISSING_FIELD":
            reason = ReviewReason.MISSING_FIELD
        elif issue.code == "LOW_OCR_QUALITY":
            reason = ReviewReason.LOW_OCR_QUALITY
        base = 0.95 if issue.severity is Severity.BLOCKING else 0.6
        if issue.fields:
            for name in issue.fields:
                tasks.append(
                    ReviewTask(
                        field=name,
                        reason=reason,
                        priority=min(1.0, base * FIELD_RISK.get(name, 0.5) + 0.25),
                        detail=f"{issue.code}: {issue.message}",
                    )
                )
        else:
            tasks.append(ReviewTask(field=None, reason=reason, priority=base, detail=f"{issue.code}: {issue.message}"))
    return tasks


def decide_status(
    fields: dict[FieldName, FieldValue], issues: list[Issue], settings: Settings
) -> DocStatus:
    if not fields:
        return DocStatus.REJECTED
    blocking = [i for i in issues if i.severity is Severity.BLOCKING]
    if blocking and settings.review.block_on_arithmetic:
        return DocStatus.NEEDS_REVIEW
    if blocking:
        return DocStatus.NEEDS_REVIEW
    weak = [f for f in fields.values() if f.confidence < field_threshold(f.field, settings)]
    if weak:
        return DocStatus.NEEDS_REVIEW
    total = fields.get(FieldName.TOTAL_AMOUNT)
    if total is not None and total.confidence < settings.review.auto_approve_threshold:
        return DocStatus.NEEDS_REVIEW
    return DocStatus.AUTO_APPROVED


def route(result: ExtractionResult, settings: Settings) -> ExtractionResult:
    """Fill ``review_tasks`` and ``status`` on an extraction result."""
    tasks = field_tasks(result.fields, settings) + issue_tasks(result.issues)
    # de-duplicate on (field, reason, detail) keeping the highest priority
    best: dict[tuple, ReviewTask] = {}
    for task in tasks:
        key = (task.field, task.reason, task.detail)
        if key not in best or task.priority > best[key].priority:
            best[key] = task
    tasks = sorted(best.values(), key=lambda t: -t.priority)[: settings.review.max_review_tasks]
    result.review_tasks = tasks
    result.status = decide_status(result.fields, result.issues, settings)
    return result
