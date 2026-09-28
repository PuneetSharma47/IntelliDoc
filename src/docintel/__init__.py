"""IntelliDoc: layout-aware extraction with validation and human review.

The public surface is intentionally small:

    from docintel.config import load_settings
    from docintel.pipeline import Pipeline

    pipeline = Pipeline(load_settings())
    result = pipeline.process_path("invoice.pdf").result
    result.status, result.fields[FieldName.TOTAL_AMOUNT].value
"""

from __future__ import annotations

__version__ = "0.9.0"
__all__ = ["__version__"]
