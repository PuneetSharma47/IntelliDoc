"""OCR engines behind a small protocol.

The pipeline never talks to Tesseract or PyMuPDF directly; it asks for an engine
by name and receives :class:`~docintel.schema.Page` objects. That keeps the
scanned-document path testable on machines without OCR binaries and lets the
evaluation harness swap engines without touching the extractors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import Page, SourceKind


class OCREngine(Protocol):
    name: str
    source: SourceKind

    def available(self) -> bool:  # pragma: no cover - trivial
        ...

    def read_pages(self, path: Path) -> list[Page]:  # pragma: no cover - trivial
        ...


@runtime_checkable
class SupportsAvailability(Protocol):
    def available(self) -> bool:
        ...


class OCREngineError(RuntimeError):
    pass
