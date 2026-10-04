"""Tesseract OCR engine (optional dependency).

Rasterises each PDF page with PyMuPDF at a configurable DPI, then calls
``pytesseract.image_to_data`` and rescales the boxes from pixels back to PDF
points so downstream geometry is identical for scanned and digital documents.

Install with ``pip install "intellidoc[ocr]"`` plus the ``tesseract``
system binary. When either is missing, :meth:`available` returns ``False`` and
:func:`docintel.ocr.registry.resolve_engine` falls back instead of crashing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..schema import BBox, Page, SourceKind, Token
from .base import OCREngineError


class TesseractEngine:
    name = "tesseract"
    source = SourceKind.OCR

    def __init__(self, dpi: int = 300, lang: str = "eng", psm: int = 6) -> None:
        self.dpi = dpi
        self.lang = lang
        self.psm = psm

    def available(self) -> bool:
        if shutil.which("tesseract") is None:
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def _render(self, path: Path):
        try:
            import pymupdf as fitz
        except ImportError:  # pragma: no cover
            import fitz
        import io

        from PIL import Image

        doc = fitz.open(path)
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=self.dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                yield page.rect.width, page.rect.height, img
        finally:
            doc.close()

    def read_pages(self, path: Path) -> list[Page]:
        if not self.available():
            raise OCREngineError(
                "tesseract engine unavailable: install the 'ocr' extra and the tesseract binary"
            )
        import pytesseract
        from pytesseract import Output

        scale = 72.0 / self.dpi  # pixels -> PDF points
        pages: list[Page] = []
        for number, (width, height, image) in enumerate(self._render(path)):
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=f"--psm {self.psm}",
                output_type=Output.DICT,
            )
            tokens: list[Token] = []
            for i, text in enumerate(data["text"]):
                text = (text or "").strip()
                if not text:
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    conf = -1.0
                if conf < 0:
                    conf = 0.0
                x, y = data["left"][i] * scale, data["top"][i] * scale
                w, h = data["width"][i] * scale, data["height"][i] * scale
                if w <= 0 or h <= 0:
                    continue
                tokens.append(
                    Token(
                        text=text,
                        bbox=BBox(x0=x, y0=y, x1=x + w, y1=y + h),
                        page=number,
                        ocr_confidence=conf / 100.0,
                    )
                )
            pages.append(Page(number=number, width=width, height=height, tokens=tokens))
        if not pages:
            raise OCREngineError(f"no pages rendered from {path}")
        return pages
