"""Geometry helpers: line reconstruction, neighbour lookup and token features.

Two documents with identical word sequences can mean different things depending
on where the words sit. "Total" on the left of a row and "11,800" on the right
is a label/value pair; the same two tokens one column apart on different rows
are unrelated. Everything in this module exists to make that distinction cheap,
because the rule extractors and the learned token tagger both consume it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schema import BBox, Document, Page, Token

#: Two tokens belong to the same line when their vertical centres differ by less
#: than this fraction of the median glyph height. 0.55 was picked by sweeping
#: 0.3-0.9 on the synthetic dev split (see docs/EVALUATION.md).
LINE_TOLERANCE = 0.55


def _median_height(tokens: list[Token]) -> float:
    heights = sorted(t.bbox.height for t in tokens if t.bbox.height > 0)
    if not heights:
        return 10.0
    return heights[len(heights) // 2]


def index_page(page: Page, tolerance: float = LINE_TOLERANCE) -> Page:
    """Assign ``line_id`` and ``token_id`` to every token of a page.

    Tokens are bucketed by vertical centre; a bucket becomes a line only when it
    is horizontally consistent, which keeps superscripts and stray headers from
    merging into the row above.
    """
    tokens = [t for t in page.tokens if t.text.strip()]
    if not tokens:
        page.tokens = []
        return page

    tol = max(1.5, _median_height(tokens) * tolerance)
    ordered = sorted(tokens, key=lambda t: (t.bbox.cy, t.bbox.x0))

    lines: list[list[Token]] = []
    for tok in ordered:
        placed = False
        for line in lines:
            if abs(line[0].bbox.cy - tok.bbox.cy) <= tol:
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])

    lines.sort(key=lambda ln: min(t.bbox.y0 for t in ln))
    flat: list[Token] = []
    for line_id, line in enumerate(lines):
        line.sort(key=lambda t: t.bbox.x0)
        for tok in line:
            tok.line_id = line_id
            tok.token_id = len(flat)
            flat.append(tok)
    page.tokens = flat
    return page


def index_document(doc: Document) -> Document:
    for page in doc.pages:
        index_page(page)
    return doc


@dataclass(frozen=True)
class Line:
    """A reconstructed row of text, kept out of the pydantic model layer."""

    line_id: int
    page: int
    tokens: tuple[Token, ...]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> BBox:
        box = self.tokens[0].bbox
        for tok in self.tokens[1:]:
            box = box.union(tok.bbox)
        return box

    @property
    def y(self) -> float:
        return self.bbox.cy

    def tokens_right_of(self, token: Token) -> list[Token]:
        return [t for t in self.tokens if t.bbox.x0 >= token.bbox.x1 - 0.5]

    def tokens_left_of(self, token: Token) -> list[Token]:
        return [t for t in self.tokens if t.bbox.x1 <= token.bbox.x0 + 0.5]

    def __iter__(self):
        return iter(self.tokens)


def lines_of(page: Page) -> list[Line]:
    rows: dict[int, list[Token]] = defaultdict(list)
    for tok in page.tokens:
        rows[tok.line_id].append(tok)
    return [
        Line(line_id=lid, page=page.number, tokens=tuple(sorted(rows[lid], key=lambda t: t.bbox.x0)))
        for lid in sorted(rows)
    ]


def document_lines(doc: Document) -> list[Line]:
    out: list[Line] = []
    for page in doc.pages:
        out.extend(lines_of(page))
    return out


def key_value_pairs(line: Line, max_gap_ratio: float = 0.85) -> list[tuple[list[Token], list[Token]]]:
    """Split a line into (label, value) chunks separated by wide horizontal gaps.

    A gap wider than ``max_gap_ratio`` of the line height is treated as a column
    break. Invoices are extremely regular about this: ``Subtotal`` and ``10,000``
    sit in different columns with a big whitespace gutter between them, while
    ``Invoice No:`` and ``INV-1048`` are separated by a single space.
    """
    toks = list(line.tokens)
    if not toks:
        return []
    threshold = max(6.0, line.bbox.height * max_gap_ratio)
    chunks: list[list[Token]] = [[toks[0]]]
    for prev, tok in zip(toks, toks[1:], strict=False):
        gap = tok.bbox.x0 - prev.bbox.x1
        if gap > threshold:
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    if len(chunks) == 1:
        return []
    return list(zip(chunks, chunks[1:], strict=False))


def same_row_neighbours(page: Page, token: Token, window: int = 3) -> list[Token]:
    line = [t for t in page.tokens if t.line_id == token.line_id]
    line.sort(key=lambda t: t.bbox.x0)
    try:
        idx = line.index(token)
    except ValueError:
        return []
    return line[max(0, idx - window): idx] + line[idx + 1: idx + 1 + window]


def below_tokens(page: Page, token: Token, max_lines: int = 2, max_x_shift: float = 60.0) -> list[Token]:
    """Tokens on the next lines that start near ``token``'s left edge.

    Used for the stacked invoice-number layout, where the label sits on one line
    and the value directly underneath it.
    """
    out: list[Token] = []
    for line in lines_of(page):
        if line.y <= token.bbox.cy + 1.0:
            continue
        if line.y - token.bbox.cy > token.bbox.height * (max_lines + 1) * 2.2:
            break
        cand = [t for t in line.tokens if abs(t.bbox.x0 - token.bbox.x0) <= max_x_shift]
        if cand:
            out.extend(cand)
            break
    return out


def rightmost_column(tokens: list[Token], page_width: float) -> list[Token]:
    """Tokens in the right third of the page — where totals live on invoices."""
    return [t for t in tokens if t.bbox.x0 >= page_width * 0.55]


def normalize_x(token: Token, page_width: float) -> float:
    return token.bbox.x0 / max(1.0, page_width)


def normalize_y(token: Token, page_height: float) -> float:
    return token.bbox.cy / max(1.0, page_height)


def relative_height(token: Token, page: Page) -> float:
    return token.bbox.height / max(1.0, page.height)


def token_features(token: Token, page: Page, doc_lines: dict[int, Line]) -> list[float]:
    """Feature vector consumed by :mod:`docintel.extract.learned`.

    Deliberately geometry-heavy and text-light: the point of a layout-aware
    baseline is to exploit *where* tokens are, so character-level identity is
    reduced to a handful of coarse flags plus a hashed bag of the token text.
    """
    from .normalize import looks_like_date, looks_like_money

    line = doc_lines.get(id(token))
    left_gap = 0.0
    right_gap = 0.0
    rel_x = normalize_x(token, page.width)
    if line is not None:
        toks = list(line.tokens)
        i = toks.index(token)
        if i > 0:
            left_gap = min(120.0, token.bbox.x0 - toks[i - 1].bbox.x1) / 120.0
        if i + 1 < len(toks):
            right_gap = min(120.0, toks[i + 1].bbox.x0 - token.bbox.x1) / 120.0

    text = token.text
    flags = [
        1.0 if text.endswith(":") else 0.0,
        1.0 if text.isupper() else 0.0,
        1.0 if looks_like_money(text) else 0.0,
        1.0 if looks_like_date(text) else 0.0,
        1.0 if token.is_numeric else 0.0,
        1.0 if any(ch.isdigit() for ch in text) else 0.0,
        min(1.0, len(text) / 24.0),
        1.0 if rel_x > 0.6 else 0.0,          # in the money column
        1.0 if rel_x < 0.35 else 0.0,         # in the label column
        1.0 if line is not None and line.y < page.height * 0.28 else 0.0,  # header band
        1.0 if line is not None and line.y > page.height * 0.55 else 0.0,  # footer band
    ]
    return [
        rel_x,
        normalize_y(token, page.height),
        relative_height(token, page),
        left_gap,
        right_gap,
        *flags,
    ]


FEATURE_NAMES = [
    "rel_x",
    "rel_y",
    "rel_h",
    "left_gap",
    "right_gap",
    "ends_colon",
    "is_upper",
    "money_value",
    "date_value",
    "numeric",
    "has_digit",
    "text_len",
    "in_money_column",
    "in_label_column",
    "in_header_band",
    "in_footer_band",
]
