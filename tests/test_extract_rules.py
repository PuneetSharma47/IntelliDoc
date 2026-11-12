"""Extractor behaviour on hand-built pages and on a rendered invoice."""

from docintel.extract.base import ExtractionContext
from docintel.extract.rules import FlatRegexExtractor, RulesExtractor, build_rows
from docintel.layout import index_document
from docintel.normalize import parse_date
from docintel.schema import BBox, Document, FieldName, Page, Token


def make_doc(rows: list[list[tuple[str, float, float]]], width=600, height=800) -> Document:
    """rows: list of [(text, x0, y)] laid out on a single page."""
    tokens = []
    for row in rows:
        for text, x0, y in row:
            tokens.append(
                Token(
                    text=text,
                    bbox=BBox(x0=x0, y0=y, x1=x0 + 6 * len(text), y1=y + 11),
                    page=0,
                )
            )
    page = Page(number=0, width=width, height=height, tokens=tokens)
    return index_document(Document(doc_id="t", filename="t.pdf", pages=[page]))


def extract(doc, extractor=None):
    """Best candidate per field, mirroring what the pipeline does."""
    from docintel.extract.base import pick_best

    ex = extractor or RulesExtractor()
    grouped: dict = {}
    for candidate in ex.extract(ExtractionContext(document=doc)):
        grouped.setdefault(candidate.field, []).append(candidate)
    return {field: pick_best(cands) for field, cands in grouped.items()}


def pack(texts: list[str], x: float, y: float, char_width: float = 5.2, gap: float = 4.0):
    """Lay tokens out left to right with a normal word gap (one column)."""
    out = []
    cursor = x
    for text in texts:
        out.append((text, cursor, y))
        cursor += char_width * len(text) + gap
    return out


def cols(*groups, y: float):
    """Build one visual row from several column blocks."""
    row = []
    for x, texts in groups:
        row.extend(pack(texts, x, y))
    return row


INVOICE_ROWS = [
    cols((40, ["Nandini", "Traders", "Pvt", "Ltd"]), (470, ["TAX", "INVOICE"]), y=60),
    cols((40, ["Invoice", "No:", "INV-1048"]), (350, ["Invoice", "Date:", "05/03/2026"]), y=120),
    cols((40, ["Bill", "To"]), y=150),
    cols((40, ["Rakesh", "Automobiles"]), y=168),
    cols((40, ["Subtotal"]), (480, ["10,000.00"]), y=400),
    cols((40, ["CGST", "@", "9%"]), (480, ["900.00"]), y=420),
    cols((40, ["SGST", "@", "9%"]), (480, ["900.00"]), y=440),
    cols((40, ["Grand", "Total"]), (480, ["11,800.00"]), y=470),
]


def test_layout_extractor_reads_all_core_fields():
    candidates = extract(make_doc(INVOICE_ROWS))
    assert candidates[FieldName.INVOICE_NUMBER].value == "INV-1048"
    assert candidates[FieldName.ISSUE_DATE].value == parse_date("05/03/2026")
    assert candidates[FieldName.SUBTOTAL].value == 10000.0
    assert candidates[FieldName.TOTAL_AMOUNT].value == 11800.0
    assert candidates[FieldName.VENDOR_NAME].value == "Nandini Traders Pvt Ltd"
    assert candidates[FieldName.BUYER_NAME].value == "Rakesh Automobiles"


def test_split_tax_components_are_summed():
    candidates = extract(make_doc(INVOICE_ROWS))
    assert candidates[FieldName.TAX_AMOUNT].value == 1800.0
    assert "sum" in candidates[FieldName.TAX_AMOUNT].provenance.rule


def test_grand_total_wins_over_distractor_total_items():
    rows = INVOICE_ROWS + [cols((40, ["Total", "Items:", "4"]), y=300)]
    candidates = extract(make_doc(rows))
    assert candidates[FieldName.TOTAL_AMOUNT].value == 11800.0


def test_invoice_number_never_sums_into_tax():
    rows = INVOICE_ROWS + [cols((40, ["Invoice", "No:", "GST-8809"]), y=200)]
    candidates = extract(make_doc(rows))
    assert candidates[FieldName.TAX_AMOUNT].value == 1800.0


def test_value_below_label_is_used_when_aligned():
    rows = [
        cols((40, ["Subtotal"]), y=400),
        cols((240, ["10,000.00"]), y=416),
        cols((40, ["Grand", "Total"]), (480, ["11,800.00"]), y=440),
    ]
    candidates = extract(make_doc(rows))
    assert candidates[FieldName.SUBTOTAL].value == 10000.0


def test_flat_baseline_is_allowed_to_be_wrong():
    """The baseline has no geometry, so it grabs the first 'total' it sees."""
    rows = INVOICE_ROWS + [cols((40, ["Total", "Items:", "4"]), y=300)]
    doc = make_doc(rows)
    flat = extract(doc, FlatRegexExtractor())
    layout = extract(doc)
    assert layout[FieldName.TOTAL_AMOUNT].value == 11800.0
    assert flat[FieldName.TOTAL_AMOUNT].score < layout[FieldName.TOTAL_AMOUNT].score


def test_rows_are_column_aware():
    doc = make_doc(INVOICE_ROWS)
    rows = build_rows(doc)
    subtotal_row = next(r for r in rows if "Subtotal" in r.text)
    assert len(subtotal_row.chunks()) == 2
    assert [t.text for t in subtotal_row.chunks()[1]] == ["10,000.00"]
