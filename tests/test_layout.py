from docintel.layout import index_document, index_page, key_value_pairs, lines_of
from docintel.schema import BBox, Document, Page, Token


def tok(text, x0, y0, x1, y1):
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


def test_line_reconstruction_groups_same_baseline():
    page = Page(
        number=0,
        width=600,
        height=800,
        tokens=[
            tok("Invoice", 40, 100, 90, 112),
            tok("No:", 92, 100, 118, 112),
            tok("INV-1048", 122, 100, 190, 112),
            tok("Bill", 40, 130, 66, 142),
            tok("To", 70, 130, 88, 142),
        ],
    )
    index_page(page)
    lines = lines_of(page)
    assert len(lines) == 2
    assert lines[0].text == "Invoice No: INV-1048"
    assert lines[1].text == "Bill To"


def test_key_value_pairs_splits_on_wide_gutter():
    page = Page(
        number=0,
        width=600,
        height=800,
        tokens=[tok("Subtotal", 40, 500, 100, 512), tok("10,000.00", 500, 500, 560, 512)],
    )
    index_page(page)
    line = lines_of(page)[0]
    pairs = key_value_pairs(line)
    assert pairs
    label, value = pairs[0]
    assert [t.text for t in label] == ["Subtotal"]
    assert [t.text for t in value] == ["10,000.00"]


def test_subscripts_do_not_merge_into_row():
    page = Page(
        number=0,
        width=600,
        height=800,
        tokens=[
            tok("Total", 40, 500, 80, 512),
            tok("1", 400, 480, 404, 487),   # superscript reference marker
            tok("11,800.00", 480, 500, 560, 512),
        ],
    )
    index_page(page)
    lines = lines_of(page)
    texts = [line.text for line in lines]
    # the superscript forms its own line rather than being glued onto the amount
    assert "Total 11,800.00" in texts or "Total" in texts
    assert not any("1 11,800.00" in t for t in texts)


def test_index_document_assigns_ids():
    page = Page(number=0, width=600, height=800, tokens=[tok("a", 10, 10, 20, 20), tok("b", 30, 10, 40, 20)])
    doc = index_document(Document(doc_id="x", filename="x.pdf", pages=[page]))
    assert [t.token_id for t in doc.pages[0].tokens] == [0, 1]
    assert all(t.line_id == 0 for t in doc.pages[0].tokens)
