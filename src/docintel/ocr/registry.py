"""Engine selection policy."""

from __future__ import annotations

from pathlib import Path

from ..config import OCRConfig
from ..schema import Page, SourceKind
from .base import OCREngine, OCREngineError
from .pdf_text import PdfTextEngine
from .tesseract import TesseractEngine

_BUILDERS = {
    "pdftext": lambda cfg: PdfTextEngine(min_chars_per_page=cfg.min_text_chars_per_page),
    "tesseract": lambda cfg: TesseractEngine(dpi=cfg.dpi, lang=cfg.lang, psm=cfg.psm),
}


def resolve_engine(cfg: OCRConfig, path: Path) -> tuple[OCREngine, str]:
    """Pick an engine for ``path`` and return ``(engine, note)``.

    ``auto`` prefers the embedded text layer and only falls back to OCR when a
    page has too little text to be a real text layer. The note explains the
    decision and is surfaced in the API response, because "why did this document
    go through OCR" is a question reviewers ask.
    """
    if cfg.engine == "auto":
        text_engine = _BUILDERS["pdftext"](cfg)
        if path.suffix.lower() == ".pdf" and text_engine.available() and text_engine.has_text_layer(path):
            return text_engine, "native text layer detected"
        ocr_engine = _BUILDERS["tesseract"](cfg)
        if ocr_engine.available():
            return ocr_engine, "scanned document, OCR used"
        if text_engine.available():
            return text_engine, "OCR unavailable, fell back to text layer"
        raise OCREngineError("no usable OCR engine: install PyMuPDF or the ocr extra")

    builder = _BUILDERS.get(cfg.engine)
    if builder is None:
        raise OCREngineError(f"unknown OCR engine {cfg.engine!r}")
    engine = builder(cfg)
    if not engine.available():
        raise OCREngineError(f"engine {cfg.engine!r} not available in this environment")
    return engine, f"explicit engine {cfg.engine}"


def engine_by_name(name: str, cfg: OCRConfig) -> OCREngine:
    builder = _BUILDERS.get(name)
    if builder is None:
        raise OCREngineError(f"unknown OCR engine {name!r}")
    return builder(cfg)


def source_of(engine: OCREngine) -> SourceKind:
    return getattr(engine, "source", SourceKind.PDF_TEXT)


def read_pages(engine: OCREngine, path: Path) -> list[Page]:
    return engine.read_pages(path)
