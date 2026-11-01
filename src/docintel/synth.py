"""Synthetic document generation for the labelled evaluation set.

Why synthetic data: invoices are confidential, so there is no shippable public
corpus that matches Indian GST layouts. The generator renders PDFs with a real
text layer, three visually different layouts, Indian number grouping, three date
formats and a configurable error injection rate -- plus a JSON label file per
document. It is a *test harness*, not a claim about real-world accuracy; see
docs/DATASET.md for the limitations and what a production rollout would need.

Every variation the extractor is expected to survive (label/value column gutters,
split CGST/SGST, distractor rows, missing due dates, arithmetic errors) is a knob
here, so the evaluation can isolate which variation breaks which extractor.
"""

from __future__ import annotations

import contextlib
import glob
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pymupdf

#: The built-in PDF fonts are Latin-1 only, which turns ``₹`` into a middle dot.
#: If a Unicode TTF is present, register it so the rupee sign renders for real and
#: the extractor is tested against the character it will actually see in the wild.
_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "C:/Windows/Fonts",
    str(Path.home() / ".fonts"),
]
_FONT_FILES = {"body": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "italic": "DejaVuSans-Oblique.ttf"}


def _discover_fonts() -> dict[str, str]:
    found: dict[str, str] = {}
    for directory in _FONT_DIRS:
        for alias, filename in _FONT_FILES.items():
            if alias in found:
                continue
            candidate = Path(directory) / filename
            if candidate.exists():
                found[alias] = str(candidate)
        if len(found) == len(_FONT_FILES):
            break
    for alias, filename in _FONT_FILES.items():
        if alias in found:
            continue
        hits = glob.glob(f"/usr/share/fonts/**/{filename}", recursive=True)
        if hits:
            found[alias] = hits[0]
    return found


FONTS = _discover_fonts()
_FALLBACK_ALIAS = {"helv": "helv", "hebo": "hebo", "heit": "heit"}
_ALIAS = {"helv": "body", "hebo": "bold", "heit": "italic"}


A4 = (595.0, 842.0)
RECEIPT = (320.0, 620.0)

VENDORS = [
    "Nandini Traders Pvt Ltd",
    "Shree Balaji Enterprises",
    "Vertex Industrial Solutions",
    "Kaveri Textiles & Co",
    "Aurora Logistics LLP",
    "Deccan Precision Industries",
    "Meghna Agro Distributors",
    "Sunrise Electricals",
]

BUYERS = [
    "Rakesh Automobiles",
    "Bluewave Retail Pvt Ltd",
    "Hindustan Paints Depot",
    "Greenfield Hotels",
    "Ashirwad Supermarket",
    "Nova Pharma Distributors",
]

PRODUCTS = [
    ("Copper wire 1.5 sq mm (90m)", 1450, 2450),
    ("MS angle 40x40x5 (6m)", 890, 1350),
    ("LED panel 18W", 320, 640),
    ("Cotton bedsheet queen", 780, 1290),
    ("Steel fastener pack", 145, 320),
    ("Hydraulic oil 5L", 1250, 1980),
    ("Corrugated box 12x10x8", 42, 78),
    ("PVC conduit 25mm", 96, 165),
]

GST_SLABS = [5.0, 12.0, 18.0, 28.0]

DATE_STYLES = ["dmy_slash", "dmy_dash", "dMy", "ymd"]
CURRENCY_STYLES = ["rupee_symbol", "rs_prefix", "inr_prefix", "bare"]


def indian_format(value: float) -> str:
    """Format 118000.0 as ``1,18,000.00`` (Indian digit grouping)."""
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    sign = "-" if value < 0 else ""
    return f"{sign}{whole}.{frac}"


def format_money(value: float, style: str) -> str:
    body = indian_format(value)
    if style == "rupee_symbol":
        return f"₹{body}"
    if style == "rs_prefix":
        return f"Rs. {body}"
    if style == "inr_prefix":
        return f"INR {body}"
    return body


def format_date(value: date, style: str) -> str:
    if style == "dmy_slash":
        return value.strftime("%d/%m/%Y")
    if style == "dmy_dash":
        return value.strftime("%d-%m-%Y")
    if style == "dMy":
        return value.strftime("%d %b %Y")
    return value.strftime("%Y-%m-%d")


def make_gstin(rng: random.Random) -> str:
    """Build a GSTIN whose check digit is actually valid."""
    from docintel.normalize import gstin_checksum

    state = f"{rng.randint(1, 37):02d}"
    pan = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    pan += f"{rng.randint(0, 9999):04d}"
    pan += rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    entity = rng.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    prefix = f"{state}{pan}{entity}Z"
    return prefix + gstin_checksum(prefix)


def make_invoice_number(rng: random.Random, doc_type: str) -> str:
    if doc_type == "receipt":
        return f"RCPT-{rng.randint(1000, 9999)}"
    prefix = rng.choice(["INV", "INV", "BILL", "GST"])
    year = rng.choice([2025, 2026])
    sep = rng.choice(["-", "/"])
    if rng.random() < 0.35:
        return f"{prefix}{sep}{year}{sep}{rng.randint(100, 999):03d}"
    return f"{prefix}-{rng.randint(1000, 9999)}"


@dataclass
class LineItem:
    description: str
    qty: int
    rate: float

    @property
    def amount(self) -> float:
        return round(self.qty * self.rate, 2)


@dataclass
class SynthDoc:
    doc_id: str
    template: str
    doc_type: str
    invoice_number: str
    issue_date: str
    due_date: str | None
    vendor_name: str
    vendor_gstin: str
    buyer_name: str
    subtotal: float
    tax_amount: float
    total_amount: float
    currency: str
    tax_split: str
    gst_rate: float
    items: list[dict] = field(default_factory=list)
    arithmetic_consistent: bool = True
    injected_error: float = 0.0
    pages: int = 1
    date_style: str = "dmy_slash"
    currency_style: str = "rupee_symbol"
    has_distractors: bool = False

    def labels(self) -> dict:
        data = asdict(self)
        data.pop("items", None)
        data["labels"] = {
            "invoice_number": self.invoice_number,
            "issue_date": self.issue_date,
            "due_date": self.due_date,
            "vendor_name": self.vendor_name,
            "vendor_gstin": self.vendor_gstin,
            "buyer_name": self.buyer_name,
            "currency": self.currency,
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "total_amount": self.total_amount,
            "document_type": self.doc_type,
        }
        return data


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def _resolved(font: str) -> str:
    alias = _ALIAS.get(font, font)
    return alias if alias in FONTS else _FALLBACK_ALIAS.get(font, "helv")


def _font_len(text: str, size: float, font: str = "helv") -> float:
    alias = _resolved(font)
    if alias in FONTS:
        return pymupdf.Font(fontfile=FONTS[alias]).text_length(text, size)
    return pymupdf.Font(font).text_length(text, size)


def new_page(doc, width: float, height: float):
    page = doc.new_page(width=width, height=height)
    for alias, path in FONTS.items():
        page.insert_font(fontname=alias, fontfile=path)
    return page


def _write(page, x: float, y: float, text: str, size: float = 9.5, font: str = "helv") -> None:
    page.insert_text((x, y), text, fontname=_resolved(font), fontsize=size, color=(0, 0, 0))


def _write_right(page, x_right: float, y: float, text: str, size: float = 9.5, font: str = "helv") -> None:
    _write(page, x_right - _font_len(text, size, font), y, text, size, font)


def _rule(page, x0: float, y: float, x1: float, width: float = 0.8) -> None:
    page.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1, y), color=(0.35, 0.35, 0.35), width=width)


# --------------------------------------------------------------------------- #
# Layout templates
# --------------------------------------------------------------------------- #


def _totals_rows(spec: SynthDoc) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("Subtotal", format_money(spec.subtotal, spec.currency_style))]
    if spec.tax_split == "cgst_sgst":
        half = round(spec.gst_rate / 2, 2)
        cgst = round(spec.tax_amount / 2, 2)
        rows.append((f"CGST @ {half:g}%", format_money(cgst, spec.currency_style)))
        rows.append((f"SGST @ {half:g}%", format_money(spec.tax_amount - cgst, spec.currency_style)))
    elif spec.tax_split == "igst":
        rows.append((f"IGST @ {spec.gst_rate:g}%", format_money(spec.tax_amount, spec.currency_style)))
    elif spec.tax_split == "single":
        rows.append((f"GST @ {spec.gst_rate:g}%", format_money(spec.tax_amount, spec.currency_style)))
    return rows


def _draw_classic(page, spec: SynthDoc) -> None:
    width = page.rect.width
    right = width - 40
    _write(page, 40, 62, spec.vendor_name, 15, "hebo")
    _write(page, 40, 78, f"{spec.vendor_gstin}  |  GSTIN", 8.5)
    _write(page, 40, 92, "12 Market Road, Sector 4, Pune 411001", 8.5)
    _write_right(page, right, 58, "TAX INVOICE" if spec.doc_type != "receipt" else "RECEIPT", 13, "hebo")
    _write_right(page, right, 76, f"Invoice No: {spec.invoice_number}", 9.5)
    _write_right(page, right, 90, f"Invoice Date: {format_date(date.fromisoformat(spec.issue_date), spec.date_style)}", 9.5)
    if spec.due_date:
        _write_right(page, right, 104, f"Due Date: {format_date(date.fromisoformat(spec.due_date), spec.date_style)}", 9.5)
    _rule(page, 40, 118, right)

    _write(page, 40, 140, "Bill To", 9.5, "hebo")
    _write(page, 40, 154, spec.buyer_name, 9.5)
    _write(page, 40, 168, "Plot 22, Industrial Area, Nagpur 440016", 8.5)

    y = 200
    _write(page, 40, y, "Description", 9, "hebo")
    _write_right(page, 300, y, "Qty", 9, "hebo")
    _write_right(page, 400, y, "Rate", 9, "hebo")
    _write_right(page, right, y, "Amount", 9, "hebo")
    _rule(page, 40, y + 4, right, 0.5)
    y += 20
    for item in spec.items:
        _write(page, 40, y, item["description"], 9)
        _write_right(page, 300, y, str(item["qty"]), 9)
        _write_right(page, 400, y, indian_format(item["rate"]), 9)
        _write_right(page, right, y, indian_format(item["amount"]), 9)
        y += 16

    if spec.has_distractors:
        _write(page, 40, y + 6, f"Total Items: {len(spec.items)}", 9)
        _write(page, 40, y + 20, f"PO Number: PO-{spec.invoice_number[-4:]}", 9)

    y += 34
    _rule(page, 340, y - 14, right, 0.5)
    for label, value in _totals_rows(spec):
        _write(page, 340, y, label, 9.5)
        _write_right(page, right, y, value, 9.5)
        y += 16
    _rule(page, 340, y - 8, right, 0.5)
    _write(page, 340, y + 8, "Grand Total", 10.5, "hebo")
    _write_right(page, right, y + 8, format_money(spec.total_amount, spec.currency_style), 10.5, "hebo")

    _write(page, 40, page.rect.height - 60, "Payment terms: 15 days from invoice date.", 8.5)
    _write(page, 40, page.rect.height - 48, "This is a computer generated document.", 8, "heit")


def _draw_compact(page, spec: SynthDoc) -> None:
    width = page.rect.width
    right = width - 36
    _write(page, 36, 54, spec.vendor_name, 13.5, "hebo")
    _write(page, 36, 68, "GSTIN: " + spec.vendor_gstin, 8.5)
    _write_right(page, right, 54, "INVOICE", 11, "hebo")
    _rule(page, 36, 76, right)

    _write(page, 36, 96, "BILLED TO", 8.5, "hebo")
    _write(page, 36, 110, spec.buyer_name, 9.5)
    _write(page, 36, 123, "Nagpur 440016, MH", 8.5)

    _write(page, 300, 96, "Invoice Number", 8.5)
    _write_right(page, right, 96, spec.invoice_number, 9.5)
    _write(page, 300, 112, "Date", 8.5)
    _write_right(page, right, 112, format_date(date.fromisoformat(spec.issue_date), spec.date_style), 9.5)
    if spec.due_date:
        _write(page, 300, 128, "Payment Due", 8.5)
        _write_right(page, right, 128, format_date(date.fromisoformat(spec.due_date), spec.date_style), 9.5)

    y = 158
    _write(page, 36, y, "Item", 8.5, "hebo")
    _write_right(page, right, y, "Amount", 8.5, "hebo")
    _rule(page, 36, y + 4, right, 0.5)
    y += 18
    for item in spec.items:
        _write(page, 36, y, f"{item['description']}  x{item['qty']}", 9)
        _write_right(page, right, y, indian_format(item["amount"]), 9)
        y += 15

    if spec.has_distractors:
        _write(page, 36, y + 6, "Freight: included", 8.5)

    y += 30
    for label, value in _totals_rows(spec):
        _write(page, 250, y, label, 9)
        _write_right(page, right, y, value, 9)
        y += 15
    _rule(page, 250, y - 6, right, 0.6)
    _write(page, 250, y + 10, "AMOUNT DUE", 10, "hebo")
    _write_right(page, right, y + 10, format_money(spec.total_amount, spec.currency_style), 10, "hebo")
    _write(page, 36, page.rect.height - 50, "Goods once sold will not be taken back.", 8, "heit")


def _draw_receipt(page, spec: SynthDoc) -> None:
    width = page.rect.width
    right = width - 16
    centre = width / 2
    _write(page, centre - _font_len(spec.vendor_name, 11, "hebo") / 2, 40, spec.vendor_name, 11, "hebo")
    _write(page, centre - _font_len("CASH MEMO", 7.5) / 2, 52, "CASH MEMO", 7.5)
    _rule(page, 16, 60, right, 0.5)
    _write(page, 16, 76, f"Receipt No: {spec.invoice_number}", 8)
    _write(page, 16, 88, f"Date: {format_date(date.fromisoformat(spec.issue_date), spec.date_style)}", 8)
    _write(page, 16, 100, f"Bill To: {spec.buyer_name}", 8)
    _rule(page, 16, 108, right, 0.5)

    y = 124
    for item in spec.items:
        _write(page, 16, y, item["description"][:34], 7.5)
        _write_right(page, right, y, indian_format(item["amount"]), 7.5)
        y += 13
    _rule(page, 16, y - 4, right, 0.4)

    y += 10
    for label, value in _totals_rows(spec):
        _write(page, 16, y, label, 8)
        _write_right(page, right, y, value, 8)
        y += 13
    _rule(page, 16, y - 4, right, 0.6)
    _write(page, 16, y + 12, "TOTAL", 9.5, "hebo")
    _write(page, 16, y + 28, f"GSTIN: {spec.vendor_gstin}", 7)
    _write_right(page, right, y + 12, format_money(spec.total_amount, spec.currency_style), 9.5, "hebo")
    _write(page, centre - _font_len("Thank you, visit again", 7.5) / 2, page.rect.height - 40, "Thank you, visit again", 7.5, "heit")


def _draw_second_page(doc, spec: SynthDoc) -> None:
    page = new_page(doc, A4[0], A4[1])
    width = page.rect.width
    _write(page, 40, 60, spec.vendor_name, 11, "hebo")
    _write(page, 40, 74, f"Invoice No: {spec.invoice_number}", 9)
    _rule(page, 40, 84, width - 40)
    _write(page, 40, 110, "Terms & Conditions", 10, "hebo")
    terms = [
        "1. Payment is due within 15 days of the invoice date.",
        "2. Interest at 18% p.a. applies on delayed payments.",
        "3. Goods remain the property of the seller until paid in full.",
        "4. Disputes are subject to Pune jurisdiction.",
    ]
    y = 128
    for term in terms:
        _write(page, 40, y, term, 9)
        y += 16
    _write(page, 40, a4_bottom(y + 20), "Bank: HDFC Bank | A/c 50100XXXXXXX | IFSC HDFC0000123", 9)
    _write_right(page, width - 40, a4_bottom(10), "Page 2 of 2", 8)


def a4_bottom(offset: float) -> float:
    return A4[1] - offset


def _draw_ledger(page, spec: SynthDoc) -> None:
    """Held-out layout: unfamiliar wording, value printed left of its label.

    Authored *after* the rule set was frozen and never used to tune it, so it
    measures generalisation to unseen phrasing rather than recall of the
    templates the rules were written against. Used by the "holdout" evaluation
    slice; see docs/EVALUATION.md.
    """
    width = page.rect.width
    right = width - 40
    _write(page, 40, 60, spec.vendor_name.upper(), 12, "hebo")
    _write(page, 40, 74, "SELLER:", 8.5, "hebo")
    _write(page, 40, 87, spec.vendor_name, 9)
    _write(page, 40, 99, "GSTIN/UIN: " + spec.vendor_gstin, 8.5)
    _write(page, 40, 118, "CONSIGNEE:", 8.5, "hebo")
    _write(page, 40, 131, spec.buyer_name, 9)
    _write_right(page, right, 74, f"Ref No: {spec.invoice_number}", 9)
    _write_right(page, right, 88, f"Dt. {format_date(date.fromisoformat(spec.issue_date), spec.date_style)}", 9)
    if spec.due_date:
        _write_right(page, right, 102, f"Pay by {format_date(date.fromisoformat(spec.due_date), spec.date_style)}", 9)
    _rule(page, 40, 146, right)

    y = 168
    _write(page, 40, y, "Particulars", 9, "hebo")
    _write_right(page, 330, y, "Qty", 9, "hebo")
    _write_right(page, right, y, "Value", 9, "hebo")
    _rule(page, 40, y + 4, right, 0.5)
    y += 20
    for item in spec.items:
        _write(page, 40, y, item["description"], 9)
        _write_right(page, 330, y, str(item["qty"]), 9)
        _write_right(page, right, y, indian_format(item["amount"]), 9)
        y += 16

    y += 30
    # values on the left of the caption, captions right-aligned: the inverse of
    # the usual layout, which is exactly what breaks position-based heuristics
    for label, value in _totals_rows(spec):
        _write_right(page, 430, y, value, 9.5)
        _write(page, 450, y, "Value of Supply" if label == "Subtotal" else label.replace("CGST", "Output Tax CGST").replace("SGST", "Output Tax SGST"), 9.5)
        y += 16
    _write_right(page, 430, y + 8, format_money(spec.total_amount, spec.currency_style), 10.5, "hebo")
    _write(page, 450, y + 8, "Net Receivable", 10.5, "hebo")
    _write(page, 40, page.rect.height - 60, "E. & O.E.", 8, "heit")


TEMPLATES = {
    "classic": _draw_classic,
    "compact": _draw_compact,
    "receipt": _draw_receipt,
    "ledger": _draw_ledger,
}

#: Layout families kept out of rule development, evaluated as a separate slice.
HOLDOUT_TEMPLATES = {"ledger"}


# --------------------------------------------------------------------------- #
# Spec + render
# --------------------------------------------------------------------------- #


def build_spec(rng: random.Random, idx: int, template: str) -> SynthDoc:
    doc_type = "receipt" if template == "receipt" else "invoice"
    n_items = rng.randint(1, 4)
    items = []
    seen = set()
    for _ in range(n_items):
        desc, lo, hi = rng.choice(PRODUCTS)
        if desc in seen:
            continue
        seen.add(desc)
        qty, rate = rng.randint(1, 6), float(rng.randint(lo, hi))
        items.append({"description": desc, "qty": qty, "rate": rate, "amount": round(qty * rate, 2)})
    if not items:
        desc, lo, hi = PRODUCTS[0]
        items = [{"description": desc, "qty": 2, "rate": float(lo), "amount": round(2 * float(lo), 2)}]

    subtotal = round(sum(it["qty"] * it["rate"] for it in items), 2)
    gst_rate = rng.choice(GST_SLABS)
    tax_amount = round(subtotal * gst_rate / 100.0, 2)
    total = round(subtotal + tax_amount, 2)

    injected = 0.0
    consistent = True
    if rng.random() < 0.12:
        injected = round(rng.choice([-1, 1]) * rng.uniform(5, 900), 2)
        total = round(total + injected, 2)
        consistent = False

    issue = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    has_due = rng.random() > 0.3 and template != "receipt"
    due = (issue + timedelta(days=rng.choice([7, 15, 30, 45]))).isoformat() if has_due else None
    if rng.random() < 0.02 and due:  # malformed due date: before issue date
        due = (issue - timedelta(days=5)).isoformat()

    split = rng.choices(["cgst_sgst", "igst", "single"], weights=[0.6, 0.25, 0.15])[0]
    if gst_rate == 0.0:
        split = "single"

    return SynthDoc(
        doc_id=f"{template}-{idx:04d}",
        template=template,
        doc_type=doc_type,
        invoice_number=make_invoice_number(rng, doc_type),
        issue_date=issue.isoformat(),
        due_date=due,
        vendor_name=rng.choice(VENDORS),
        vendor_gstin=make_gstin(rng),
        buyer_name=rng.choice(BUYERS),
        subtotal=subtotal,
        tax_amount=tax_amount,
        total_amount=total,
        currency="INR",
        tax_split=split,
        gst_rate=gst_rate,
        items=items,
        arithmetic_consistent=consistent,
        injected_error=injected,
        pages=2 if rng.random() < 0.15 else 1,
        date_style=rng.choice(DATE_STYLES),
        currency_style=rng.choice(CURRENCY_STYLES),
        has_distractors=rng.random() < 0.4,
    )


def render(spec: SynthDoc, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    size = RECEIPT if spec.template == "receipt" else A4
    doc = pymupdf.open()
    page = new_page(doc, size[0], size[1])
    TEMPLATES[spec.template](page, spec)
    if spec.pages == 2 and spec.template != "receipt":
        _draw_second_page(doc, spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):  # older PyMuPDF builds lack subsetting
        doc.subset_fonts()  # keeps generated PDFs small enough to commit
    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    return out_path


def generate_dataset(
    out_dir: str | Path = "data",
    n: int = 240,
    seed: int = 20260101,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
) -> dict:
    """Render ``n`` documents plus labels and a deterministic train/dev/test split."""
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "raw"
    label_dir = out_dir / "ground_truth"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    order = ["classic", "compact", "receipt", "ledger"]
    records = []
    split_counts = {"train": 0, "dev": 0, "test": 0}
    for idx in range(n):
        template = order[idx % len(order)]
        spec = build_spec(rng, idx, template)
        pdf_path = pdf_dir / f"{spec.doc_id}.pdf"
        render(spec, pdf_path)
        labels = spec.labels()
        labels["file"] = str(pdf_path)
        labels["holdout"] = spec.template in HOLDOUT_TEMPLATES
        labels["split"] = (
            "train" if idx < int(n * train_ratio)
            else "dev" if idx < int(n * (train_ratio + dev_ratio))
            else "test"
        )
        split_counts[labels["split"]] += 1
        (label_dir / f"{spec.doc_id}.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        records.append(labels)

    manifest = {
        "seed": seed,
        "n_docs": n,
        "splits": split_counts,
        "templates": order,
        "error_injection_rate": round(sum(1 for r in records if not r["arithmetic_consistent"]) / n, 4),
        "docs": records,
    }
    (label_dir / "labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
