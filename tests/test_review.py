
from docintel.config import ReviewConfig, Settings
from docintel.review import decide_status, field_threshold, route
from docintel.schema import (
    DocClass,
    DocStatus,
    Document,
    ExtractionResult,
    FieldName,
    FieldValue,
    Issue,
    ReviewReason,
    Severity,
)


def fv(field, value, confidence):
    return FieldValue(field=field, value=value, confidence=confidence)


def make_result(fields, issues=None):
    return ExtractionResult(
        document=Document(doc_id="d", filename="d.pdf"),
        doc_type=DocClass.INVOICE,
        fields=fields,
        issues=issues or [],
    )


CONFIDENT = {
    FieldName.INVOICE_NUMBER: fv(FieldName.INVOICE_NUMBER, "INV-1", 0.95),
    FieldName.ISSUE_DATE: fv(FieldName.ISSUE_DATE, "2026-03-05", 0.92),
    FieldName.VENDOR_NAME: fv(FieldName.VENDOR_NAME, "Nandini Traders", 0.9),
    FieldName.SUBTOTAL: fv(FieldName.SUBTOTAL, 10000.0, 0.92),
    FieldName.TAX_AMOUNT: fv(FieldName.TAX_AMOUNT, 1800.0, 0.9),
    FieldName.TOTAL_AMOUNT: fv(FieldName.TOTAL_AMOUNT, 11800.0, 0.95),
    FieldName.CURRENCY: fv(FieldName.CURRENCY, "INR", 0.62),
    FieldName.DOCUMENT_TYPE: fv(FieldName.DOCUMENT_TYPE, "invoice", 0.78),
}


def test_confident_document_is_auto_approved():
    result = route(make_result(dict(CONFIDENT)), Settings())
    assert result.status is DocStatus.AUTO_APPROVED
    assert not [t for t in result.review_tasks if t.reason is ReviewReason.LOW_CONFIDENCE]


def test_weak_money_field_goes_to_review():
    fields = dict(CONFIDENT)
    fields[FieldName.TOTAL_AMOUNT] = fv(FieldName.TOTAL_AMOUNT, 11800.0, 0.6)
    result = route(make_result(fields), Settings())
    assert result.status is DocStatus.NEEDS_REVIEW
    flagged = {t.field for t in result.review_tasks}
    assert FieldName.TOTAL_AMOUNT in flagged


def test_blocking_issue_always_routes_to_review():
    issue = Issue(code="ARITHMETIC_MISMATCH", message="does not add up", severity=Severity.BLOCKING,
                  fields=[FieldName.TOTAL_AMOUNT])
    result = route(make_result(dict(CONFIDENT), [issue]), Settings())
    assert result.status is DocStatus.NEEDS_REVIEW
    assert any(t.reason is ReviewReason.VALIDATION_ISSUE for t in result.review_tasks)


def test_per_field_thresholds_protect_low_risk_metadata():
    settings = Settings()
    assert field_threshold(FieldName.CURRENCY, settings) < field_threshold(FieldName.TOTAL_AMOUNT, settings)
    # a defaulted currency at 0.62 must not on its own force a review
    fields = dict(CONFIDENT)
    assert decide_status(fields, [], settings) is DocStatus.AUTO_APPROVED


def test_no_fields_is_rejected():
    assert decide_status({}, [], Settings()) is DocStatus.REJECTED


def test_tasks_are_capped_and_ranked():
    settings = Settings(review=ReviewConfig(max_review_tasks=3))
    fields = {name: fv(name, "x", 0.2) for name in list(CONFIDENT)[:8]}
    result = route(make_result(fields), settings)
    assert len(result.review_tasks) <= 3
    priorities = [t.priority for t in result.review_tasks]
    assert priorities == sorted(priorities, reverse=True)


def test_alternatives_become_disagreement_tasks():
    fields = dict(CONFIDENT)
    fields[FieldName.TOTAL_AMOUNT] = FieldValue(
        field=FieldName.TOTAL_AMOUNT, value=11800.0, confidence=0.9, alternatives=[11900.0]
    )
    result = route(make_result(fields), Settings())
    assert any(t.reason is ReviewReason.EXTRACTOR_DISAGREEMENT for t in result.review_tasks)
