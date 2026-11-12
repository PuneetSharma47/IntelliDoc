from docintel.evaluate import EvalReport, FieldScore, token_f1, values_match
from docintel.schema import FieldName


def test_values_match_money_tolerance():
    assert values_match(11800.0, 11800.0, FieldName.TOTAL_AMOUNT)
    assert values_match(11800.004, 11800.0, FieldName.TOTAL_AMOUNT)
    assert not values_match(11800.5, 11800.0, FieldName.TOTAL_AMOUNT)


def test_values_match_is_format_insensitive():
    assert values_match("inv-1048", "INV-1048", FieldName.INVOICE_NUMBER)
    assert values_match("Nandini Traders Pvt Ltd", "nandini  traders pvt ltd.", FieldName.VENDOR_NAME)
    assert values_match("05/03/2026", "2026-03-05", FieldName.ISSUE_DATE)
    assert not values_match(None, "INV-1", FieldName.INVOICE_NUMBER)


def test_token_f1_partial_credit():
    assert token_f1("Nandini Traders Pvt Ltd", "Nandini Traders Pvt Ltd") == 1.0
    score = token_f1("Nandini Traders", "Nandini Traders Pvt Ltd")
    assert 0.6 < score < 1.0
    assert token_f1("Aurora Logistics", "Nandini Traders") == 0.0
    assert token_f1(None, "x") == 0.0


def test_field_score_metrics():
    score = FieldScore(field="total_amount", tp=8, fp=2, fn=2, scored=10)
    assert score.precision == 0.8
    assert score.recall == 0.8
    assert round(score.f1, 4) == 0.8
    assert score.exact_match == 0.8


def test_report_derived_metrics():
    report = EvalReport(extractor="rules_layout", split="test", noise_cer=0.0, n_docs=10)
    report.fields = {
        "total_amount": FieldScore(field="total_amount", tp=9, fp=1, fn=1, scored=10),
        "vendor_name": FieldScore(field="vendor_name", tp=10, fp=0, fn=0, scored=10),
    }
    report.auto_approved = 5
    report.auto_approved_correct = 5
    assert report.micro_recall == 19 / 20
    assert report.straight_through_accuracy == 1.0
    payload = report.to_dict()
    assert payload["micro_recall"] == 0.95
    assert "fields" in payload
