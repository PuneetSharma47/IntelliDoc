"""Extractors: each returns candidates for the same field schema.

``rules_flat``   regexes over the page text, no geometry (baseline)
``rules_layout`` label anchors plus column geometry (default)
``learned``      linear tagger over layout + text features
``layoutlmv3``   fine-tuned LayoutLMv3 token classifier (optional)
``ensemble``     merges several extractors and rewards agreement
"""

from __future__ import annotations

from .base import Candidate, ExtractionContext, Extractor, pick_best

__all__ = ["Candidate", "ExtractionContext", "Extractor", "pick_best"]
