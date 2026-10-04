"""Native PDF text-layer reader (PyMuPDF).

Digital invoices -- the overwhelming majority of B2B documents in practice -- do
not need OCR at all, and running OCR on them only adds errors. Calling this
engine first is both faster and more accurate; the OCR path exists for scans.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import BBox, Page, SourceKind, Token
from .base import OCREngineError


class PdfTextEngine:
    name = "pdftext"
    source = SourceKind.PDF_TEXT

    def __init__(self, min_chars_per_page: int = 40) -> None:
        self.min_chars_per_page = min_chars_per_page

    def available(self) -> bool:
        try:
            import pymupdf  # noqa: F401
        except ImportError:  # pragma: no cover
            try:
                import fitz  # noqa: F401
            except ImportError:
                return False
        return True

    @staticmethod
    def _open(path: Path):
        try:
            import pymupdf as fitz
        except ImportError:  # pragma: no cover
            import fitz
        doc = fitz.open(path)
        if doc.is_encrypted:
            doc.close()
            raise OCREngineError(f"encrypted PDF: {path}")
        return doc

    def has_text_layer(self, path: Path) -> bool:
        doc = self._open(path)
        try:
            for page in doc:
                if len(page.get_text("text").strip()) >= self.min_chars_per_page:
                    return True
            return False
        finally:
            doc.close()

    def read_pages(self, path: Path) -> list[Page]:
        doc = self._open(path)
        pages: list[Page] = []
        try:
            for number, page in enumerate(doc):
                words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
                tokens = [
                    Token(
                        text=w[4],
                        bbox=BBox(x0=float(w[0]), y0=float(w[1]), x1=float(w[2]), y1=float(w[3])),
                        page=number,
                    )
                    for w in words
                    if str(w[4]).strip()
                ]
                pages.append(
                    Page(number=number, width=float(page.rect.width), height=float(page.rect.height), tokens=tokens)
                )
        finally:
            doc.close()
        if not pages:
            raise OCREngineError(f"no pages in {path}")
        return pages
