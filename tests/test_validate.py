from datetime import date

from docintel.config import Settings, ValidationConfig
from docintel.schema import DocClass, FieldName, FieldValue, Severity
from docintel.validate import validate


def fv(field, value, confidence=0.95):
    return FieldValue(field=field, value=value, confidence=confidence)


def base_fields(**overrides):
    fields = {
        FieldName.INVOICE_NUMBER: fv(FieldName.INVOICE_NUMBER, "INV-1048"),
        FieldName.ISSUE_DATE: fv(FieldName.ISSUE_DATE, date(2026, 3, 5)),
        FieldName.VENDOR_NAME: fv(FieldName.VENDOR_NAME, "Nandini Traders Pvt Ltd"),
        FieldName.VENDOR_GSTIN: fv(FieldName.VENDOR_GSTIN, "27AAPFU0939F1ZV"),
        FieldName.SUBTOTAL: fv(FieldName.SUBTOTAL, 10000.0),
        FieldName.TAX_AMOUNT: fv(FieldName.TAX_AMOUNT, 1800.0),
        FieldName.TOTAL_AMOUNT: fv(FieldName.TOTAL_AMOUNT, 11800.0),
    }
    fields.update(overrides)
    return fields


class FakeDoc:
    tokens: list = []


def run(fields, settings=None, doc_type=DocClass.INVOICE, today=date(2026, 3, 10)):
    settings = settings or Settings()
    return validate(fields, doc_type, FakeDoc(), settings, today=today)


def test_consistent_invoice_has_no_blocking_issue():
    issues = run(base_fields())
    assert not [i for i in issues if i.severity is Severity.BLOCKING]


def test_arithmetic_mismatch_is_detected_and_penalised():
    fields = base_fields(**{FieldName.TOTAL_AMOUNT: fv(FieldName.TOTAL_AMOUNT, 11900.0)})
    issues = run(fields)
    mismatch = [i for i in issues if i.code == "ARITHMETIC_MISMATCH"]
    assert mismatch and mismatch[0].severity is Severity.BLOCKING
    assert fields[FieldName.TOTAL_AMOUNT].confidence < 0.8
    assert "difference +100.00" in mismatch[0].message


def test_missing_required_field_is_flagged():
    fields = base_fields()
    fields.pop(FieldName.TOTAL_AMOUNT)
    codes = [i.code for i in run(fields)]
    assert "MISSING_FIELD" in codes


def test_partial_totals_block_is_flagged():
    fields = base_fields()
    fields.pop(FieldName.SUBTOTAL)
    issues = run(fields)
    assert any(i.code == "MONEY_BLOCK_INCOMPLETE" for i in issues)


def test_tax_rate_anomaly():
    fields = base_fields(
        **{
            FieldName.SUBTOTAL: fv(FieldName.SUBTOTAL, 10000.0),
            FieldName.TAX_AMOUNT: fv(FieldName.TAX_AMOUNT, 999.0),
            FieldName.TOTAL_AMOUNT: fv(FieldName.TOTAL_AMOUNT, 10999.0),
        }
    )
    codes = [i.code for i in run(fields)]
    assert "TAX_RATE_ANOMALY" in codes


def test_due_date_before_issue_date():
    fields = base_fields(
        **{FieldName.DUE_DATE: fv(FieldName.DUE_DATE, date(2026, 3, 1))}
    )
    codes = [i.code for i in run(fields)]
    assert "DATE_ORDER" in codes


def test_future_dated_invoice():
    fields = base_fields(**{FieldName.ISSUE_DATE: fv(FieldName.ISSUE_DATE, date(2026, 12, 1))})
    codes = [i.code for i in run(fields)]
    assert "DATE_IN_FUTURE" in codes


def test_gstin_checksum_is_a_warning_by_default_and_blocking_when_required():
    fields = base_fields(**{FieldName.VENDOR_GSTIN: fv(FieldName.VENDOR_GSTIN, "27AAPFU0939F1ZX")})
    default_issue = [i for i in run(fields) if i.code == "GSTIN_CHECKSUM"][0]
    assert default_issue.severity is Severity.WARNING

    strict = Settings(validation=ValidationConfig(require_gstin_checksum=True))
    strict_issue = [i for i in run(base_fields(**{FieldName.VENDOR_GSTIN: fv(FieldName.VENDOR_GSTIN, "27AAPFU0939F1ZX")}), strict) if i.code == "GSTIN_CHECKSUM"][0]
    assert strict_issue.severity is Severity.BLOCKING


def test_receipt_does_not_require_due_date():
    fields = {
        FieldName.INVOICE_NUMBER: fv(FieldName.INVOICE_NUMBER, "RCPT-1"),
        FieldName.ISSUE_DATE: fv(FieldName.ISSUE_DATE, date(2026, 3, 5)),
        FieldName.VENDOR_NAME: fv(FieldName.VENDOR_NAME, "Sunrise Electricals"),
        FieldName.TOTAL_AMOUNT: fv(FieldName.TOTAL_AMOUNT, 500.0),
    }
    codes = [i.code for i in run(fields, doc_type=DocClass.RECEIPT)]
    assert "MISSING_FIELD" not in codes
