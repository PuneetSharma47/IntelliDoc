"""Rule-based extractors.

Two variants live here on purpose:

``FlatRegexExtractor`` (``rules_flat``)
    Reads the document as one long string and applies regexes in sequence. This
    is the honest baseline: it is what a first attempt at this problem looks
    like, and it is the thing the layout-aware extractor must beat.

``RulesExtractor`` (``rules_layout``)
    Uses label anchors *plus geometry*: same-row value lookup, column gutters,
    header bands and the "money column" on the right of the page. Every candidate
    carries the box it came from, so a reviewer can verify it.

Both produce :class:`~docintel.extract.base.Candidate` objects; arbitration,
confidence and validation are separate stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..layout import Line
from ..normalize import (
    GSTIN_LOOSE_RE,
    NormalizationError,
    detect_currency,
    looks_like_date,
    looks_like_money,
    parse_date,
    parse_money,
)
from ..schema import BBox, Document, FieldName, Provenance, Token
from .base import Candidate, ExtractionContext

# --------------------------------------------------------------------------- #
# Anchor vocabularies
# --------------------------------------------------------------------------- #

ID_PATTERN = (
    r"\b(?:INV|INVOICE|BILL|REC|RCPT|TXN|CRN|DOC)[-/ ]?[A-Z0-9]{1,12}(?:[-/][A-Z0-9]{1,8})?\b"
)

LABELS: dict[str, list[str]] = {
    "invoice_number": [
        r"invoice\s*(?:no|num|number|#|ref)\.?",
        r"inv\s*(?:no|#)\.?",
        r"bill\s*(?:no|number)\.?",
        r"receipt\s*(?:no|number|#)\.?",
        r"voucher\s*(?:no|#)\.?",
        r"doc(?:ument)?\s*no\.?",
    ],
    "issue_date": [
        r"invoice\s*date",
        r"bill\s*date",
        r"date\s*of\s*(?:issue|invoice)",
        r"dated",
        r"dt\.?",
        r"date",
    ],
    "due_date": [r"due\s*date", r"payment\s*due(?:\s*date)?", r"due\s*on", r"due\s*by"],
    "subtotal": [
        r"sub\s*-?\s*total",
        r"taxable\s*(?:value|amount)",
        r"net\s*amount",
        r"amount\s*(?:before|excl\.?)\s*tax",
        r"basic\s*amount",
    ],
    "tax": [
        r"\bcgst\b",
        r"\bsgst\b",
        r"\bigst\b",
        r"\bgst\b",
        r"\bvat\b",
        r"total\s*tax",
        r"tax\s*amount",
        r"\btax\b",
    ],
    "total_amount": [
        r"grand\s*total",
        r"total\s*amount(?:\s*due)?",
        r"invoice\s*total",
        r"amount\s*(?:payable|due)",
        r"net\s*payable",
        r"bill\s*amount",
        r"\btotal\b",
    ],
    "vendor": [r"(?:from|seller|vendor|supplier|issued\s*by)\s*[:\-]"],
    "buyer": [
        r"bill(?:ed)?\s*to",
        r"buyer(?:\s*name)?",
        r"customer(?:\s*name)?",
        r"sold\s*to",
        r"ship(?:ped)?\s*to",
        r"consignee",
    ],
    "gstin": [r"gstin", r"gst\s*(?:in|no)\.?", r"gst\s*number", r"tax\s*id"],
}

COMPILED = {k: re.compile("|".join(f"(?:{p})" for p in v), re.I) for k, v in LABELS.items()}
#: Deliberately case-sensitive: with ``re.IGNORECASE`` the ``[A-Z0-9]`` body also
#: matches lowercase, so "Receivable" would parse as ``REC`` + body. Callers
#: upper-case the text before matching.
ID_RE = re.compile(ID_PATTERN)


def _looks_like_id(text: str) -> bool:
    """An identifier must contain a digit.

    The prefix alternation matches plain words otherwise: ``INV`` + ``OICE`` makes
    "INVOICE" look like an invoice number, and ``BILL`` + ``TO`` makes "Bill To"
    look like one too.
    """
    upper = text.upper()
    match = ID_RE.search(upper)
    if match is None:
        return False
    return any(ch.isdigit() for ch in match.group(0))
MONEY_TOKEN_RE = re.compile(r"(?:(?:₹|rs\.?|inr|\$|€|£)\s*)?\d[\d,]*(?:\.\d{1,2})?", re.I)
RATE_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?\s*%")

#: Rows that mention "total" but are not the invoice total.
TOTAL_EXCLUDE = re.compile(
    r"sub\s*-?\s*total|total\s*(?:items?|qty|quantity|tax|discount|weight)|tax\s*total|"
    r"total\s*(?:cgst|sgst|igst)|items?\s*:",
    re.I,
)

#: Rows in the header band that are clearly not the vendor name.
VENDOR_EXCLUDE = re.compile(
    r"invoice|\btax\b|receipt|cash\s*memo|credit\s*note|statement|bill\s*to|billed\s*to|"
    r"gstin|gst\s*no|due\s*date|po\s*(?:no|number)|original\s+for|duplicate|\bpage\b|"
    r"https?://|@|www\.|thank\s*you|\bmemo\b|^\W*$",
    re.I,
)

#: Words that make a header line look like a business name rather than a banner.
BUSINESS_WORD = re.compile(
    r"\b(electricals?|traders?|agencies|agency|textiles?|automobiles?|pharma|pharmaceuticals|"
    r"enterprises?|industries|solutions?|services?|distributors?|supermarket|hotels?|stores?|"
    r"mart|logistics|precision|agro|retail|depot|suppliers?|manufactur\w+|\w+\s+co)\b",
    re.I,
)

#: Any of these in a candidate value means it is a field label, not a party name.
LABEL_TEXT = re.compile(
    r"(?:invoice|bill|receipt|date|gstin|total|subtotal|amount|due|number|no\.)\b",
    re.I,
)

LEGAL_SUFFIX = re.compile(
    r"\b(pvt|private|ltd|limited|llp|inc|co\.?|corp|enterprises?|traders?|industries|"
    r"solutions?|services?|technologies|associates|agency|stores?|mart|distributors?|"
    r"& co|and sons)\b",
    re.I,
)


# --------------------------------------------------------------------------- #
# Row helper
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    """A reconstructed line plus a character-offset map back to its tokens.

    A single visual row routinely contains two unrelated columns: ``Subtotal`` on
    the left of the totals block and its amount on the right, or the vendor name
    and ``TAX INVOICE`` on the same baseline. :meth:`chunks` splits a row at wide
    horizontal gutters so a label is only ever paired with the values in its own
    column (plus the next column, for the standard label/value pair).
    """

    line: Line
    text: str
    spans: list[tuple[int, int, Token]]
    page_width: float
    page_height: float
    _chunks: list[list[Token]] | None = None

    # -- column splitting ---

    def chunks(self) -> list[list[Token]]:
        if self._chunks is None:
            self._chunks = split_chunks(list(self.line.tokens), self.line.bbox.height)
        return self._chunks

    def chunk_index(self, token: Token) -> int:
        for idx, chunk in enumerate(self.chunks()):
            if any(t is token for t in chunk):
                return idx
        return -1

    def _label_tokens(self, match: re.Match[str]) -> list[Token]:
        return [tok for (s, e, tok) in self.spans if s < match.end() and e > match.start()]

    def value_groups(self, label_tokens: list[Token]) -> list[tuple[list[Token], str]]:
        """Candidate value spans for a label, nearest first.

        Order: the remainder of the label's own column block, then the blocks to
        its right. Rates and units sit inside the first group (``CGST @ 14%``),
        which is why callers scan each group for a *typed* value instead of taking
        the first token blindly.
        """
        if not label_tokens:
            return []
        last = label_tokens[-1]
        chunks = self.chunks()
        ci = self.chunk_index(last)
        if ci < 0:
            return []
        groups: list[tuple[list[Token], str]] = []
        remainder = [t for t in chunks[ci] if t.bbox.x0 >= last.bbox.x1 - 0.5]
        if remainder:
            groups.append((remainder, "same_chunk"))
        for offset, chunk in enumerate(chunks[ci + 1: ci + 3], start=1):
            groups.append((chunk, "next_chunk" if offset == 1 else "far_chunk"))
        return groups

    def value_after(self, label_tokens: list[Token]) -> tuple[list[Token], str]:
        """Tokens that continue the label: same column first, then next column."""
        if not label_tokens:
            return [], ""
        last = label_tokens[-1]
        chunks = self.chunks()
        ci = self.chunk_index(last)
        if ci < 0:
            return [], ""
        within = [t for t in chunks[ci] if t.bbox.x0 >= last.bbox.x1 - 0.5]
        if within:
            return within, "same_chunk"
        if ci + 1 < len(chunks):
            return chunks[ci + 1], "next_chunk"
        return [], ""

    def _after(self, tok: Token) -> list[Token]:
        return [t for t in self.line.tokens if t.bbox.x0 >= tok.bbox.x1 - 0.5]

    def _before(self, tok: Token) -> list[Token]:
        return [t for t in self.line.tokens if t.bbox.x1 <= tok.bbox.x0 + 0.5]

    def label_chunk_x(self, label_tokens: list[Token]) -> tuple[float, float]:
        chunks = self.chunks()
        ci = self.chunk_index(label_tokens[-1])
        if ci < 0:
            return label_tokens[0].bbox.x0, label_tokens[-1].bbox.x1
        chunk = chunks[ci]
        return min(t.bbox.x0 for t in chunk), max(t.bbox.x1 for t in chunk)

    @property
    def y(self) -> float:
        return self.line.bbox.cy

    @property
    def rel_y(self) -> float:
        return self.y / max(1.0, self.page_height)

    def in_money_column(self, tok: Token) -> bool:
        return tok.bbox.x0 > self.page_width * 0.5


def split_chunks(tokens: list[Token], line_height: float, max_gap_ratio: float = 0.85) -> list[list[Token]]:
    """Split a row into column blocks at gaps wider than the line height."""
    if not tokens:
        return []
    threshold = max(6.0, line_height * max_gap_ratio)
    chunks: list[list[Token]] = [[tokens[0]]]
    for prev, tok in zip(tokens, tokens[1:], strict=False):
        if tok.bbox.x0 - prev.bbox.x1 > threshold:
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    return chunks


def _next_row_aligned(row: Row, rows: list[Row], idx: int, label_tokens: list[Token]) -> Row | None:
    """The row immediately below, if it is inside the same column block."""
    if idx + 1 >= len(rows):
        return None
    nxt = rows[idx + 1]
    if nxt.line.page != row.line.page:
        return None
    if abs(nxt.y - row.y) > row.page_height * 0.06:
        return None
    return nxt


def build_rows(doc: Document) -> list[Row]:
    rows: list[Row] = []
    for page in doc.pages:
        for line in document_lines_for(page):
            parts: list[str] = []
            spans: list[tuple[int, int, Token]] = []
            pos = 0
            for tok in line.tokens:
                spans.append((pos, pos + len(tok.text), tok))
                parts.append(tok.text)
                pos += len(tok.text) + 1
            rows.append(
                Row(
                    line=line,
                    text=" ".join(parts),
                    spans=spans,
                    page_width=page.width,
                    page_height=page.height,
                )
            )
    return rows


def document_lines_for(page):
    from ..layout import lines_of

    return lines_of(page)


def _ocr_factor(tokens: list[Token]) -> float:
    """Down-weight candidates whose tokens OCR was unsure about."""
    confs = [t.ocr_confidence for t in tokens if t.ocr_confidence is not None]
    if not confs:
        return 1.0
    mean = sum(confs) / len(confs)
    return 0.65 + 0.35 * max(0.0, min(1.0, mean))


def _prov(tokens: list[Token], rule: str, source: str) -> Provenance:
    box: BBox | None = None
    for tok in tokens:
        box = tok.bbox if box is None else box.union(tok.bbox)
    page = tokens[0].page if tokens else 0
    return Provenance(
        page=page,
        bbox=box,
        source=source,
        raw_text=" ".join(t.text for t in tokens),
        rule=rule,
    )


def _join_id(tokens: list[Token], limit: int = 3) -> str:
    """Join tokens that belong to one identifier (``INV`` + ``1048``)."""
    if not tokens:
        return ""
    out = [tokens[0].text]
    for nxt in tokens[1:limit]:
        prev = out[-1]
        if prev.endswith(("-", "/", ":")) or ":" in prev:
            out[-1] = prev.rstrip(":") + ("" if prev.endswith(("-", "/")) else "")
            out.append(nxt.text)
        elif nxt.text[:1].isdigit() and prev.isalpha() and len(prev) <= 4:
            out.append(nxt.text)
        else:
            break
    return "".join(out).strip(" :;-")


CURRENCY_PREFIX = re.compile(r"(?i)^(₹|rs\.?|inr|usd|eur|gbp|\$|€|£)")


def _is_amount_token(text: str) -> bool:
    """Reject identifiers that merely contain digits (``GST-8809``, ``PO-4471``).

    Without this guard a total can be summed with an invoice number: both are
    numeric-looking tokens and both sit next to a keyword.
    """
    if RATE_TOKEN_RE.search(text) and "%" in text:
        return False
    stripped = CURRENCY_PREFIX.sub("", text.strip())
    letters = sum(ch.isalpha() for ch in stripped)
    if letters >= 2:
        return False
    return looks_like_money(text)


def _first_money(tokens: list[Token]) -> Token | None:
    for tok in tokens:
        if _is_amount_token(tok.text):
            return tok
    return None


def _first_date(tokens: list[Token]) -> Token | None:
    for tok in tokens:
        if looks_like_date(tok.text):
            return tok
    return None


def _first_date_multi(tokens: list[Token], dayfirst: bool = True):
    """Find a date that may be split across tokens ("22", "Feb", "2026")."""
    for tok in tokens:
        if looks_like_date(tok.text):
            try:
                return parse_date(tok.text, dayfirst=dayfirst), tok.text, [tok]
            except NormalizationError:
                continue
    for n in (2, 3):
        for i in range(0, max(0, len(tokens) - n + 1)):
            window = tokens[i: i + n]
            if window[-1].bbox.x0 - window[0].bbox.x1 > 60:
                continue
            joined = " ".join(t.text for t in window).strip(" ,.")
            if not joined or not any(ch.isdigit() for ch in joined):
                continue
            try:
                return parse_date(joined, dayfirst=dayfirst), joined, window
            except NormalizationError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Layout-aware extractor
# --------------------------------------------------------------------------- #


class RulesExtractor:
    """Geometry-aware label/value extraction."""

    name = "rules_layout"

    def __init__(self, dayfirst: bool = True) -> None:
        self.dayfirst = dayfirst

    # -- helpers ------------------------------------------------------------ #

    def _currency_hint(self, rows: list[Row]) -> str:
        for row in rows:
            found = detect_currency(row.text)
            if found:
                return found
        return "INR"

    def _next_row_ok(self, row: Row, rows: list[Row], idx: int, value: Token, label_tokens: list[Token]) -> bool:
        """Guard for the "value on the line below" fallback.

        A value found on the next line only counts when it is in the same column
        block as the label. Without this, ``Subtotal`` happily picks up the house
        number from the address printed underneath it.
        """
        nxt = _next_row_aligned(row, rows, idx, label_tokens)
        if nxt is None:
            return False
        label_x0, label_x1 = row.label_chunk_x(label_tokens)
        return not (value.bbox.x0 < label_x1 - 12 and label_x0 > row.page_width * 0.45)

    # -- per field ---------------------------------------------------------- #

    def _ids(self, rows: list[Row]) -> list[Candidate]:
        out: list[Candidate] = []
        label_re = COMPILED["invoice_number"]
        for idx, row in enumerate(rows):
            match = label_re.search(row.text)
            if not match:
                continue
            label_tokens = row._label_tokens(match)
            if not label_tokens:
                continue
            value, score, rule, tail, how = "", 0.0, "", [], ""
            for group, group_kind in row.value_groups(label_tokens):
                candidate = _join_id(group)
                if candidate:
                    value, tail, how = candidate, group, group_kind
                    break
            if value and _looks_like_id(value):
                score, rule = 0.94, f"label+{how}"
            elif value and re.fullmatch(r"[A-Z0-9][A-Z0-9/\-]{2,19}", value.upper()) and any(ch.isdigit() for ch in value):
                score, rule = 0.90, f"label+{how}"
            elif value:
                score, rule = 0.66, f"label+{how}_loose"
            if not value and idx + 1 < len(rows):
                nxt = _next_row_aligned(row, rows, idx, label_tokens)
                if nxt is not None:
                    value = _join_id(nxt.line.tokens)
                    score, rule = 0.80, "label+next_row"
            value = value.strip(".,;:")
            if not value:
                continue
            used = tail or row.line.tokens
            out.append(
                Candidate(
                    field=FieldName.INVOICE_NUMBER,
                    value=value,
                    raw=value,
                    score=score * _ocr_factor(used),
                    provenance=_prov(used, rule, self.name),
                )
            )
        out.extend(self._id_pattern_fallback(rows))
        return out

    def _id_pattern_fallback(self, rows: list[Row]) -> list[Candidate]:
        out: list[Candidate] = []
        for row in rows:
            for match in ID_RE.finditer(row.text.upper()):
                text = row.text[match.start(): match.end()].strip()  # keep the printed case
                if len(text) < 5 or not any(ch.isdigit() for ch in text):
                    continue
                out.append(
                    Candidate(
                        field=FieldName.INVOICE_NUMBER,
                        value=text,
                        raw=text,
                        score=0.55 * _ocr_factor(row.line.tokens),
                        provenance=_prov(row.line.tokens, "id_pattern", self.name),
                    )
                )
                break
        return out

    def _dates(self, rows: list[Row], field: FieldName, key: str) -> list[Candidate]:
        out: list[Candidate] = []
        label_re = COMPILED[key]
        for idx, row in enumerate(rows):
            match = label_re.search(row.text)
            if not match:
                continue
            if key == "issue_date":
                due = COMPILED["due_date"].search(row.text)
                if due is not None and due.start() <= match.start():
                    continue  # a due-date row also contains the bare word "date"
            label_tokens = row._label_tokens(match)
            if not label_tokens:
                continue
            parsed = None
            score, rule = 0.0, ""
            for group, how in row.value_groups(label_tokens):
                parsed = _first_date_multi(group, self.dayfirst)
                if parsed is not None:
                    score, rule = (0.88 if how == "far_chunk" else 0.92), f"label+{how}"
                    break
            if parsed is None:
                parsed = _first_date_multi(row.line.tokens, self.dayfirst)
                score, rule = 0.80, "row_date"
            if parsed is None and idx + 1 < len(rows):
                nxt = _next_row_aligned(row, rows, idx, label_tokens)
                if nxt is not None:
                    parsed = _first_date_multi(nxt.line.tokens, self.dayfirst)
                    score, rule = 0.76, "label+next_row"
            if parsed is None:
                continue
            value, raw, tokens = parsed
            out.append(
                Candidate(
                    field=field,
                    value=value,
                    raw=raw,
                    score=score * _ocr_factor(tokens),
                    provenance=_prov(tokens, rule, self.name),
                )
            )
        return out

    def _money(self, rows: list[Row], field: FieldName, key: str, agg: str = "single") -> list[Candidate]:
        """Extract one money field.

        ``agg='sum'`` adds up all matched components (CGST + SGST + IGST) and
        keeps the weakest component's score: a correct sum built from one bad
        addend is still wrong.
        """
        label_re = COMPILED[key]
        grouped: list[tuple[Row, Token, float, str]] = []
        for idx, row in enumerate(rows):
            if key == "total_amount" and TOTAL_EXCLUDE.search(row.text):
                continue
            if key == "subtotal" and re.search(r"grand\s*total|amount\s*due", row.text, re.I):
                continue
            if key == "tax" and GSTIN_LOOSE_RE.search(row.text.upper().replace("-", "")):
                continue
            if key == "tax" and (_looks_like_id(row.text) or COMPILED["invoice_number"].search(row.text)):
                continue  # "Invoice No: GST-8809" is an identifier, not a tax amount
            match = label_re.search(row.text)
            if not match:
                continue
            label_tokens = row._label_tokens(match)
            if not label_tokens:
                continue
            tok, score, rule = None, 0.0, ""
            for group, how in row.value_groups(label_tokens):
                cand_tok = _first_money(group)
                if cand_tok is not None:
                    tok = cand_tok
                    score = 0.86 if how == "far_chunk" else 0.91
                    rule = f"label+{how}"
                    break
            if tok is None:
                tok, score, rule = _first_money(row._before(label_tokens[0])), 0.78, "value+left_of_label"
            if tok is None:
                money_anywhere = [t for t in row.line.tokens if looks_like_money(t.text)]
                if money_anywhere:
                    tok, score, rule = money_anywhere[-1], 0.70, "row_rightmost"
            if tok is None and idx + 1 < len(rows):
                nxt = _next_row_aligned(row, rows, idx, label_tokens)
                if nxt is not None:
                    cand = _first_money(nxt.line.tokens)
                    if cand is not None and self._next_row_ok(row, rows, idx, cand, label_tokens):
                        tok, score, rule = cand, 0.72, "label+next_row"
            if tok is None:
                continue
            if key == "total_amount" and re.search(r"grand\s*total|amount\s*(?:payable|due)|invoice\s*total", row.text, re.I):
                score = min(0.97, score + 0.04)
            if key == "total_amount" and row.rel_y > 0.5:
                score = min(0.97, score + 0.02)
            grouped.append((row, tok, score * _ocr_factor([tok]), rule))

        if not grouped:
            return []

        if agg == "sum" and len(grouped) > 1:
            amounts: list[float] = []
            for tok in (t for _r, t, _s, _rule in grouped):
                try:
                    amounts.append(parse_money(tok.text).amount)
                except NormalizationError:
                    continue
            if not amounts:
                return []
            total = round(sum(amounts), 2)
            tokens = [t for _r, t, _s, _rule in grouped]
            raw = " + ".join(t.text for t in tokens)
            return [
                Candidate(
                    field=field,
                    value=total,
                    raw=raw,
                    score=min(s for _r, _t, s, _rule in grouped) * (0.96 if len(amounts) > 1 else 1.0),
                    provenance=_prov(tokens, "sum_of_components", self.name),
                    note=f"{len(amounts)} components summed",
                )
            ]

        out: list[Candidate] = []
        currency_hint = self._currency_hint(rows)
        for _row, tok, score, rule in grouped:
            try:
                money = parse_money(tok.text, default_currency=currency_hint)
            except NormalizationError:
                continue
            out.append(
                Candidate(
                    field=field,
                    value=money.amount,
                    raw=tok.text,
                    score=score,
                    provenance=_prov([tok], rule or "money_anchor", self.name),
                )
            )
        return out

    def _currency(self, rows: list[Row]) -> list[Candidate]:
        hint = self._currency_hint(rows)
        out: list[Candidate] = [
            Candidate(
                field=FieldName.CURRENCY,
                value=hint,
                raw=hint,
                score=0.62,
                provenance=Provenance(page=0, source=self.name, rule="currency_default", raw_text=hint),
            )
        ]
        for row in rows:
            if COMPILED["total_amount"].search(row.text) and detect_currency(row.text):
                code = detect_currency(row.text)
                out.append(
                    Candidate(
                        field=FieldName.CURRENCY,
                        value=code,
                        raw=row.text,
                        score=0.88,
                        provenance=_prov(row.line.tokens, "symbol_near_total", self.name),
                    )
                )
        return out

    def _vendor(self, rows: list[Row]) -> list[Candidate]:
        out: list[Candidate] = []
        # (a) explicit "From: / Seller:" label
        for idx, row in enumerate(rows):
            match = COMPILED["vendor"].search(row.text)
            if not match:
                continue
            label_tokens = row._label_tokens(match)
            if not label_tokens:
                continue
            tail, _how = row.value_after(label_tokens)
            name = " ".join(t.text for t in tail).strip(" :-") if tail else ""
            if name and (LABEL_TEXT.search(name) or looks_like_money(name) or _looks_like_id(name)):
                name = ""  # neighbouring column holds a different field
            if name and not VENDOR_EXCLUDE.search(name):
                out.append(
                    Candidate(
                        field=FieldName.VENDOR_NAME,
                        value=name,
                        raw=name,
                        score=0.90,
                        provenance=_prov(tail, "from_label", self.name),
                    )
                )
            elif idx + 1 < len(rows):
                nxt = _next_row_aligned(row, rows, idx, label_tokens)
                if nxt is not None:
                    name = " ".join(t.text for t in nxt.line.tokens).strip(" :-")
                    if name and not VENDOR_EXCLUDE.search(name):
                        out.append(
                            Candidate(
                                field=FieldName.VENDOR_NAME,
                                value=name,
                                raw=name,
                                score=0.85,
                                provenance=_prov(nxt.line.tokens, "from_label_next_row", self.name),
                            )
                        )

        # (b) the most prominent non-boilerplate text block in the header band
        for idx, row in enumerate(rows):
            if row.rel_y > 0.34:
                break
            best_chunk: list[Token] | None = None
            best_score = 0.0
            for chunk in row.chunks():
                text = " ".join(t.text for t in chunk).strip()
                if not text or len(text) > 70 or len(text) < 3:
                    continue
                if VENDOR_EXCLUDE.search(text):
                    continue
                if not any(ch.isalpha() for ch in text):
                    continue
                if looks_like_money(text) or looks_like_date(text.replace(" ", "")):
                    continue
                mean_h = sum(t.bbox.height for t in chunk) / len(chunk)
                score = 0.52
                score += 0.18 if text.isupper() else 0.0
                score += 0.15 if LEGAL_SUFFIX.search(text) or BUSINESS_WORD.search(text) else 0.0
                score += 0.10 * min(1.0, mean_h / 14.0)
                score += max(0.0, 0.12 - 0.04 * idx)
                score *= _ocr_factor(chunk)
                if score > best_score:
                    best_score, best_chunk = score, chunk
            if best_chunk is not None:
                text = " ".join(t.text for t in best_chunk).strip(" :-")
                out.append(
                    Candidate(
                        field=FieldName.VENDOR_NAME,
                        value=text,
                        raw=text,
                        score=min(0.89, best_score),
                        provenance=_prov(best_chunk, "header_band", self.name),
                    )
                )
        return out

    def _buyer(self, rows: list[Row]) -> list[Candidate]:
        out: list[Candidate] = []
        for idx, row in enumerate(rows):
            match = COMPILED["buyer"].search(row.text)
            if not match:
                continue
            label_tokens = row._label_tokens(match)
            if not label_tokens:
                continue
            tail, how = row.value_after(label_tokens)
            name = " ".join(t.text for t in tail).strip(" :-")
            if name and (len(name) < 3 or LABEL_TEXT.search(name) or looks_like_money(name)):
                # the neighbouring column belongs to a different field, not the buyer
                name = ""
            if name:
                score, rule, used = 0.88, f"bill_to_{how}", tail
            else:
                nxt = _next_row_aligned(row, rows, idx, label_tokens)
                if nxt is None or COMPILED["gstin"].search(nxt.text) or not nxt.text.strip():
                    continue
                label_x0 = row.label_chunk_x(label_tokens)[0]
                chunks = nxt.chunks()
                aligned = min(chunks, key=lambda c: abs(c[0].bbox.x0 - label_x0))
                if aligned is chunks[-1] and len(chunks) > 1 and abs(aligned[0].bbox.x0 - label_x0) > 90:
                    aligned = chunks[0]
                name = " ".join(t.text for t in aligned).strip(" :-")
                if not name or VENDOR_EXCLUDE.search(name) or LABEL_TEXT.search(name) or looks_like_money(name):
                    continue
                score, rule, used = 0.84, "bill_to_next_row", aligned
            out.append(
                Candidate(
                    field=FieldName.BUYER_NAME,
                    value=name,
                    raw=name,
                    score=score * _ocr_factor(used),
                    provenance=_prov(used, rule, self.name),
                )
            )
        return out

    def _gstins(self, rows: list[Row]) -> list[Candidate]:
        """Vendor GSTIN: the first GSTIN printed on the document.

        Buyer GSTINs are out of scope for this schema, so only the earliest one
        is kept. On documents where the buyer block is printed above the seller
        block this rule picks the wrong party -- a documented failure case.
        """
        out: list[Candidate] = []
        for row in rows:
            for gstin in GSTIN_LOOSE_RE.findall(row.text.upper()):
                tokens = [t for t in row.line.tokens if gstin in t.text.upper().replace("-", "")] or row.line.tokens
                out.append(
                    Candidate(
                        field=FieldName.VENDOR_GSTIN,
                        value=gstin,
                        raw=gstin,
                        score=0.90 * _ocr_factor(tokens),
                        provenance=_prov(tokens, "gstin_first", self.name),
                    )
                )
                return out  # first GSTIN only
        return out

    def _doc_type_hint(self, doc: Document) -> str:
        text = doc.text.lower()
        if "credit note" in text:
            return "credit_note"
        if "tax invoice" in text or "invoice" in text:
            return "invoice"
        if "receipt" in text or "cash memo" in text:
            return "receipt"
        return "invoice"

    # -- entry point -------------------------------------------------------- #

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:
        rows = build_rows(ctx.document)
        candidates: list[Candidate] = []
        candidates += self._ids(rows)
        candidates += self._dates(rows, FieldName.ISSUE_DATE, "issue_date")
        candidates += self._dates(rows, FieldName.DUE_DATE, "due_date")
        candidates += self._money(rows, FieldName.SUBTOTAL, "subtotal")
        candidates += self._money(rows, FieldName.TAX_AMOUNT, "tax", agg="sum")
        candidates += self._money(rows, FieldName.TOTAL_AMOUNT, "total_amount")
        candidates += self._currency(rows)
        candidates += self._vendor(rows)
        candidates += self._buyer(rows)
        candidates += self._gstins(rows)
        return candidates


# --------------------------------------------------------------------------- #
# Flat baseline
# --------------------------------------------------------------------------- #


class FlatRegexExtractor:
    """No-geometry baseline: one text blob, regexes applied in order.

    Kept intentionally naive. It is the control condition in the evaluation, and
    the place where its naive choices fail (grabbing "Total Items" counts, the
    first date on the page instead of the invoice date) is documented in
    docs/FAILURE_CASES.md.
    """

    name = "rules_flat"

    def __init__(self, dayfirst: bool = True) -> None:
        self.dayfirst = dayfirst

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:
        doc = ctx.document
        text = doc.text
        flat = re.sub(r"[ \t]+", " ", text)
        out: list[Candidate] = []
        prov = Provenance(page=0, source=self.name, rule="flat_regex")

        id_match = re.search(r"(?:invoice|bill|inv|receipt)\s*(?:no|number|#)?\.?\s*[:\-]?\s*(" + ID_PATTERN + ")", flat, re.I)
        if id_match:
            out.append(Candidate(FieldName.INVOICE_NUMBER, id_match.group(1), id_match.group(1), 0.62, prov))
        else:
            loose = ID_RE.search(flat)
            if loose:
                out.append(Candidate(FieldName.INVOICE_NUMBER, loose.group(0), loose.group(0), 0.45, prov))

        date_hits = list(
            re.finditer(
                r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2}",
                flat,
            )
        )
        if date_hits:
            try:
                value = parse_date(date_hits[0].group(0), dayfirst=self.dayfirst)
                out.append(Candidate(FieldName.ISSUE_DATE, value, date_hits[0].group(0), 0.55, prov))
            except NormalizationError:
                pass
            if len(date_hits) > 1:
                try:
                    value = parse_date(date_hits[1].group(0), dayfirst=self.dayfirst)
                    out.append(Candidate(FieldName.DUE_DATE, value, date_hits[1].group(0), 0.35, prov))
                except NormalizationError:
                    pass

        def amount_after(keyword: str, exclude: str | None = None) -> tuple[float, str] | None:
            for m in re.finditer(keyword, flat, re.I):
                window = flat[m.end(): m.end() + 40]
                if exclude and re.search(exclude, flat[max(0, m.start() - 20): m.end() + 40], re.I):
                    continue
                hit = MONEY_TOKEN_RE.search(window)
                if not hit:
                    continue
                try:
                    return parse_money(hit.group(0)).amount, hit.group(0)
                except NormalizationError:
                    continue
            return None

        for keyword, field, score, exclude in [
            (r"grand\s*total|total\s*amount|amount\s*due|invoice\s*total", FieldName.TOTAL_AMOUNT, 0.6, None),
            (r"sub\s*-?\s*total|taxable\s*value", FieldName.SUBTOTAL, 0.5, None),
            (r"c[gs]st|i[gs]st|\bgst\b|\bvat\b|tax\s*amount", FieldName.TAX_AMOUNT, 0.45, None),
        ]:
            hit = amount_after(keyword, exclude)
            if hit:
                out.append(Candidate(field, hit[0], hit[1], score, prov))
                continue
            if field is FieldName.TOTAL_AMOUNT:
                hit = amount_after(r"\btotal\b", r"sub\s*-?\s*total")
                if hit:
                    out.append(Candidate(field, hit[0], hit[1], 0.35, prov))

        gstins = GSTIN_LOOSE_RE.findall(flat.upper())
        if gstins:
            out.append(Candidate(FieldName.VENDOR_GSTIN, gstins[0], gstins[0], 0.55, prov))

        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if first_line and len(first_line) <= 70:
            out.append(Candidate(FieldName.VENDOR_NAME, first_line, first_line, 0.45, prov))

        out.append(Candidate(FieldName.CURRENCY, detect_currency(flat) or "INR", "", 0.5, prov))
        return out
