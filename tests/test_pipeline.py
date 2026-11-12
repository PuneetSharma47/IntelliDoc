"""End-to-end: file -> extraction -> validation -> routing -> store -> search."""


import pytest

from docintel.pipeline import Pipeline
from docintel.schema import DocStatus, FieldName
from docintel.search import Searcher
from docintel.store import Store


def test_pipeline_extracts_known_values(fixture_pdf, settings):
    path, labels = fixture_pdf
    pipeline = Pipeline(settings)
    outcome = pipeline.process_path(path)
    result = outcome.result
    assert result.fields[FieldName.INVOICE_NUMBER].value == labels["invoice_number"]
    assert result.fields[FieldName.TOTAL_AMOUNT].value == pytest.approx(labels["total_amount"], abs=0.01)
    assert result.fields[FieldName.TOTAL_AMOUNT].provenance.bbox is not None
    assert result.latency_ms >= 0
    assert outcome.engine_note


def test_pipeline_is_deterministic(fixture_pdf, settings):
    path, _ = fixture_pdf
    pipeline = Pipeline(settings)
    first = pipeline.process_path(path).result
    second = pipeline.process_path(path).result
    assert {k: v.value for k, v in first.fields.items()} == {k: v.value for k, v in second.fields.items()}


def test_every_field_has_provenance(fixture_pdf, settings):
    path, _ = fixture_pdf
    result = Pipeline(settings).process_path(path).result
    for name, value in result.fields.items():
        assert 0.0 <= value.confidence <= 1.0
        assert value.provenance.rule, f"{name} missing rule provenance"


def test_flat_extractor_is_available_as_baseline(fixture_pdf, settings):
    path, _ = fixture_pdf
    pipeline = Pipeline(settings)
    result = pipeline.process_path(path, extractor="rules_flat").result
    assert result.extractor == "rules_flat"
    assert FieldName.TOTAL_AMOUNT in result.fields


def test_unknown_extractor_falls_back_with_a_note(settings):
    pipeline = Pipeline(settings)
    extractor, note = pipeline.build_extractor("learned")
    assert extractor is not None
    # no trained model on disk in a clean checkout -> rules, with an explanation
    if note:
        assert "learned model missing" in note


def test_store_roundtrip_and_correction(fixture_pdf, settings):
    path, labels = fixture_pdf
    pipeline = Pipeline(settings)
    store = Store(settings.store.db_path)
    result = pipeline.process_path(path).result
    store.save_result(result, settings_fp=settings.fingerprint())

    doc = store.get_document(result.document.doc_id)
    assert doc is not None and doc["fields"]
    total = next(f for f in doc["fields"] if f["name"] == "total_amount")
    assert total["value"] == pytest.approx(labels["total_amount"], abs=0.01)

    store.apply_correction(result.document.doc_id, "total_amount", 9999.99, reviewer="qa")
    doc = store.get_document(result.document.doc_id)
    total = next(f for f in doc["fields"] if f["name"] == "total_amount")
    assert total["corrected"] == 9999.99
    corrections = store.export_corrections()
    assert corrections[0]["field"] == "total_amount"
    assert corrections[0]["predicted"] == pytest.approx(total["value"], abs=0.01)


def test_search_returns_page_citations(fixture_pdf, settings):
    path, _ = fixture_pdf
    store = Store(settings.store.db_path)
    result = Pipeline(settings).process_path(path).result
    store.save_result(result)
    hits = Searcher(store).search("Nandini")
    assert hits
    assert hits[0].page == 0
    assert "Nandini" in hits[0].snippet
    assert Searcher(store).search("zzz-not-present") == []


def test_statistics_track_review_load(fixture_pdf, settings):
    path, _ = fixture_pdf
    store = Store(settings.store.db_path)
    result = Pipeline(settings).process_path(path).result
    store.save_result(result)
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["fields"] > 0


def test_arithmetic_error_is_flagged_end_to_end(settings):
    from docintel.ingest import ingest_file
    from docintel.synth import SynthDoc, render

    spec = SynthDoc(
        doc_id="broken-total",
        template="classic",
        doc_type="invoice",
        invoice_number="INV-9999",
        issue_date="2026-03-05",
        due_date=None,
        vendor_name="Nandini Traders Pvt Ltd",
        vendor_gstin="27AAPFU0939F1ZV",
        buyer_name="Rakesh Automobiles",
        subtotal=10000.0,
        tax_amount=1800.0,
        total_amount=11900.0,   # deliberately wrong
        currency="INR",
        tax_split="single",
        gst_rate=18.0,
        items=[{"description": "Copper wire", "qty": 4, "rate": 2500.0, "amount": 10000.0}],
        arithmetic_consistent=False,
        injected_error=100.0,
    )
    path = settings.dataset_dir
    import pathlib

    pdf = pathlib.Path(path) / "broken.pdf"
    render(spec, pdf)
    document, _ = ingest_file(pdf, settings)
    result = Pipeline(settings).process_document(document)
    codes = [issue.code for issue in result.issues]
    assert "ARITHMETIC_MISMATCH" in codes
    assert result.status is DocStatus.NEEDS_REVIEW
