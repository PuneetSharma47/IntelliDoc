"""Page rendering with field highlights.

"Show me where that number came from" is the review workflow's most common
request. Every extracted field carries a page number and a box, so the server can
render the page with the boxes drawn on it and the reviewer can verify in one
glance instead of re-reading the document.
"""

from __future__ import annotations

from pathlib import Path

# Colour by confidence band, matching the review policy in docintel.review.
COLORS = {
    "high": (0.05, 0.55, 0.25),
    "medium": (0.85, 0.55, 0.05),
    "low": (0.75, 0.15, 0.15),
}


def confidence_color(confidence: float) -> tuple[float, float, float]:
    if confidence >= 0.85:
        return COLORS["high"]
    if confidence >= 0.6:
        return COLORS["medium"]
    return COLORS["low"]


def render_page(
    path: str | Path,
    page_no: int,
    boxes: list[tuple[float, float, float, float]] | None = None,
    dpi: int = 120,
    colors: list[tuple[float, float, float]] | None = None,
) -> bytes:
    """Render one page to PNG, optionally drawing highlight boxes.

    ``boxes`` are PDF-point rectangles; ``colors`` is parallel to ``boxes`` and
    lets the caller highlight by confidence.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf

    doc = pymupdf.open(path)
    try:
        if page_no < 0 or page_no >= doc.page_count:
            raise IndexError(f"page {page_no} out of range (0..{doc.page_count - 1})")
        page = doc[page_no]
        for idx, box in enumerate(boxes or []):
            color = (colors[idx] if colors and idx < len(colors) else COLORS["high"])
            page.draw_rect(pymupdf.Rect(*box), color=color, width=1.2, dashes=None)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_thumbnails(path: str | Path, dpi: int = 60, max_pages: int = 4) -> list[bytes]:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf

    doc = pymupdf.open(path)
    try:
        out = []
        for number in range(min(doc.page_count, max_pages)):
            out.append(doc[number].get_pixmap(dpi=dpi).tobytes("png"))
        return out
    finally:
        doc.close()
