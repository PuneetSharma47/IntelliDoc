"""Pipeline orchestration: ingest -> classify -> extract -> score -> validate -> route.

One place that composes the stages, so the CLI, the API and the evaluation harness
all run *the same* code path. Anything that only exists in the API is untested
behaviour, and anything that only exists in the eval script produces numbers that
do not describe the deployed system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .classify import classify
from .confidence import score_all
from .config import Settings, load_settings
from .extract.base import ExtractionContext, Extractor
from .extract.ensemble import EnsembleExtractor, group_by_field
from .extract.layoutlm import LayoutLMv3Extractor, missing_reason
from .extract.learned import LearnedTagger
from .extract.rules import FlatRegexExtractor, RulesExtractor
from .ingest import ingest_file
from .review import route
from .schema import (
    DocClass,
    Document,
    ExtractionResult,
    FieldName,
    FieldValue,
    Issue,
    Severity,
)
from .validate import validate


@dataclass
class PipelineOutcome:
    result: ExtractionResult
    engine_note: str = ""
    warnings: list[str] | None = None


class Pipeline:
    """Configurable extraction pipeline shared by CLI, API and evaluation."""

    def __init__(self, settings: Settings | None = None, tagger: LearnedTagger | None = None) -> None:
        self.settings = settings or load_settings()
        self._tagger = tagger
        self._tagger_loaded = tagger is not None

    # -- extractor selection ------------------------------------------------ #

    def _tagger_instance(self) -> LearnedTagger | None:
        if self._tagger_loaded:
            return self._tagger
        self._tagger_loaded = True
        path = Path(self.settings.extraction.learned_model)
        if path.exists():
            try:
                self._tagger = LearnedTagger.load(path, dayfirst=self.settings.extraction.dayfirst)
            except Exception:  # noqa: BLE001 - a stale model must not break extraction
                self._tagger = None
        return self._tagger

    def build_extractor(self, name: str | None = None) -> tuple[Extractor | None, str]:
        """Return ``(extractor, note)``; note explains any fallback."""
        name = name or self.settings.extraction.extractor
        dayfirst = self.settings.extraction.dayfirst
        if name == "rules_flat":
            return FlatRegexExtractor(dayfirst=dayfirst), ""
        if name == "rules_layout":
            return RulesExtractor(dayfirst=dayfirst), ""
        if name == "learned":
            tagger = self._tagger_instance()
            if tagger is None:
                return RulesExtractor(dayfirst=dayfirst), (
                    f"learned model missing at {self.settings.extraction.learned_model}; used rules_layout"
                )
            return tagger, ""
        if name == "layoutlmv3":
            extractor = LayoutLMv3Extractor(self.settings.extraction.layoutlm_model)
            if not extractor.available():
                return RulesExtractor(dayfirst=dayfirst), (
                    f"LayoutLMv3 unavailable ({missing_reason(self.settings.extraction.layoutlm_model)}); "
                    "used rules_layout"
                )
            return extractor, ""
        if name == "ensemble":
            members: list[Extractor] = [RulesExtractor(dayfirst=dayfirst)]
            note = ""
            tagger = self._tagger_instance()
            if tagger is not None:
                members.append(tagger)
            else:
                note = "ensemble degraded to rules only (no learned model)"
            return EnsembleExtractor(members, self.settings.extraction.agreement_bonus), note
        raise ValueError(f"unknown extractor {name!r}")

    # -- processing --------------------------------------------------------- #

    def process_path(self, path: str | Path, extractor: str | None = None) -> PipelineOutcome:
        document, note = ingest_file(path, self.settings)
        warnings = [note] if "fell back" in note or "unavailable" in note else []
        result = self.process_document(document, extractor=extractor)
        return PipelineOutcome(result=result, engine_note=note, warnings=warnings)

    def process_document(self, document: Document, extractor: str | None = None) -> ExtractionResult:
        started = time.perf_counter()
        classification = classify(document)
        doc_type = classification.doc_type
        if doc_type is DocClass.UNKNOWN:
            doc_type = DocClass.INVOICE

        ex, note = self.build_extractor(extractor)
        assert ex is not None
        ctx = ExtractionContext(document=document, dayfirst=self.settings.extraction.dayfirst, doc_type=doc_type.value)

        agreed: set[FieldName] = set()
        disagreements: dict[FieldName, list] = {}
        if isinstance(ex, EnsembleExtractor):
            ensemble = ex.run(ctx)
            candidates = ensemble.candidates
            agreed = ensemble.agreed_fields
            disagreements = ensemble.disagreements
        else:
            candidates = ex.extract(ctx)

        grouped = group_by_field(candidates)
        fields = score_all(document, grouped, self.settings, agreements=agreed)
        fields.setdefault(
            FieldName.CURRENCY,
            FieldValue(field=FieldName.CURRENCY, value="INR", confidence=0.45, provenance={"rule": "default"}),
        )
        if doc_type is not DocClass.UNKNOWN:
            fields[FieldName.DOCUMENT_TYPE] = FieldValue(
                field=FieldName.DOCUMENT_TYPE,
                value=doc_type.value,
                confidence=classification.confidence,
                provenance={"rule": "classifier", "raw_text": classification.evidence},
            )

        issues = validate(fields, doc_type, document, self.settings)

        # disagreements surfaced by the ensemble are demoted to review tasks
        for field_name, values in disagreements.items():
            if field_name in fields:
                existing = fields[field_name].alternatives
                for value in values:
                    if value not in existing:
                        existing.append(value)

        result = ExtractionResult(
            document=document,
            doc_type=doc_type,
            doc_type_confidence=classification.confidence,
            fields=fields,
            issues=issues,
            extractor=ex.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        result = route(result, self.settings)
        if note:
            result.issues.append(Issue(code="EXTRACTOR_FALLBACK", message=note, severity=Severity.INFO))
        return result

    # -- convenience -------------------------------------------------------- #

    def explain(self, result: ExtractionResult) -> dict:
        """Compact, JSON-serialisable view used by the UI and the CLI."""
        return {
            "doc_id": result.document.doc_id,
            "filename": result.document.filename,
            "doc_type": result.doc_type.value,
            "doc_type_confidence": result.doc_type_confidence,
            "status": result.status.value,
            "extractor": result.extractor,
            "latency_ms": result.latency_ms,
            "fields": {
                name.value: {
                    "value": value.value,
                    "confidence": value.confidence,
                    "page": value.provenance.page,
                    "bbox": list(value.provenance.bbox.as_tuple()) if value.provenance.bbox else None,
                    "rule": value.provenance.rule,
                    "source": value.provenance.source,
                    "alternatives": value.alternatives,
                    "confidence_parts": value.confidence_parts,
                }
                for name, value in result.fields.items()
            },
            "issues": [issue.model_dump(mode="json") for issue in result.issues],
            "review_tasks": [task.model_dump(mode="json") for task in result.review_tasks],
        }
