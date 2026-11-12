"""Shared fixtures: settings, a tiny in-memory dataset and a synthetic document."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from docintel.config import Settings, load_settings  # noqa: E402
from docintel.ingest import ingest_file  # noqa: E402
from docintel.synth import build_spec, render  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    loaded = load_settings(ROOT / "configs" / "default.yaml")
    loaded.store.db_path = str(tmp_path / "test.db")
    loaded.dataset_dir = str(tmp_path / "data")
    loaded.benchmarks_dir = str(tmp_path / "bench")
    return loaded


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """A deterministic classic invoice with known field values."""
    import random

    rng = random.Random(4242)
    spec = build_spec(rng, 1, "classic")
    spec.vendor_name = "Nandini Traders Pvt Ltd"
    spec.buyer_name = "Rakesh Automobiles"
    spec.invoice_number = "INV-1048"
    path = tmp_path_factory.mktemp("pdf") / "invoice.pdf"
    render(spec, path)
    return path, spec.labels()["labels"] | {"arithmetic_consistent": spec.arithmetic_consistent}


@pytest.fixture()
def fixture_doc(fixture_pdf, settings):
    path, labels = fixture_pdf
    document, _note = ingest_file(path, settings)
    return document, labels
