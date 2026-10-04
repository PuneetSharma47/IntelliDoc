"""OCR package: pluggable text extraction from PDFs and images."""

from .base import OCREngine, OCREngineError
from .pdf_text import PdfTextEngine
from .registry import engine_by_name, resolve_engine
from .tesseract import TesseractEngine

__all__ = [
    "OCREngine",
    "OCREngineError",
    "PdfTextEngine",
    "TesseractEngine",
    "engine_by_name",
    "resolve_engine",
]
