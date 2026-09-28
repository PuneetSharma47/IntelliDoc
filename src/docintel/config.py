"""Configuration objects, loaded from YAML with environment overrides.

Thresholds are configuration, not constants: the review-routing threshold has a
direct precision/recall trade-off against human labelling cost, so it has to be
tunable and recorded next to every evaluation run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OCRConfig(BaseModel):
    engine: str = "auto"              # auto | pdftext | tesseract
    dpi: int = 300
    lang: str = "eng"
    psm: int = 6
    min_text_chars_per_page: int = 40  # below this, treat the page as scanned


class ExtractionConfig(BaseModel):
    extractor: str = "rules_layout"    # rules_flat | rules_layout | learned | ensemble
    dayfirst: bool = True
    learned_model: str = "models/learned_tagger.joblib"
    layoutlm_model: str = "models/layoutlmv3-finetuned"
    agreement_bonus: float = 0.06      # confidence boost when two extractors agree


class ReviewConfig(BaseModel):
    auto_approve_threshold: float = 0.86
    review_threshold: float = 0.55
    field_auto_threshold: float = 0.80
    #: Per-field overrides. Money and identifiers must clear a higher bar than
    #: low-risk metadata, otherwise a defaulted currency pulls every document
    #: into the review queue and reviewers stop trusting the queue.
    field_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "total_amount": 0.85,
            "subtotal": 0.85,
            "tax_amount": 0.85,
            "invoice_number": 0.85,
            "vendor_gstin": 0.85,
            "issue_date": 0.85,
            "due_date": 0.80,
            "vendor_name": 0.75,
            "buyer_name": 0.75,
            "currency": 0.55,
            "document_type": 0.60,
        }
    )
    block_on_arithmetic: bool = True
    max_review_tasks: int = 25


class ValidationConfig(BaseModel):
    money_tolerance: float = 0.01
    tax_tolerance_ratio: float = 0.02         # 2% slack on expected tax amount
    max_future_days: int = 1
    max_age_days: int = 3650
    require_gstin_checksum: bool = False       # off by default: OCR noise breaks it
    duplicate_invoice_window_days: int = 90


class StoreConfig(BaseModel):
    db_path: str = "data/docintel.db"
    search_top_k: int = 10
    snippet_chars: int = 220


class Settings(BaseModel):
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    dataset_dir: str = "data"
    benchmarks_dir: str = "benchmarks/results"

    def fingerprint(self) -> str:
        """Stable hash of everything that can change pipeline output."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from YAML, then apply ``DOCINTEL_*`` environment overrides.

    Environment overrides use ``__`` as the nesting separator, e.g.
    ``DOCINTEL_REVIEW__AUTO_APPROVE_THRESHOLD=0.9``.
    """
    data: dict[str, Any] = {}
    if path is None:
        default = Path("configs/default.yaml")
        if default.exists():
            path = default
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    for key, raw in os.environ.items():
        if not key.startswith("DOCINTEL_"):
            continue
        path_parts = key[len("DOCINTEL_"):].lower().split("__")
        cursor = data
        for part in path_parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path_parts[-1]] = yaml.safe_load(raw)
    return Settings(**data)
