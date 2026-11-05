"""Evaluation harness: field-level precision / recall / exact match.

Metrics are computed the way an accounts-payable team would count them:

* a field is **scored** only when the document actually contains it (a missing
  due date is not a false negative);
* a wrong value costs one false positive *and* one false negative -- inventing a
  number is as bad as missing one;
* ``exact match`` is TP over scored fields, the number to quote as accuracy;
* **straight-through accuracy** is the share of auto-approved documents that were
  fully correct, the number that decides whether the review threshold is usable;
* **routing precision / recall** ask whether the review queue catches the wrong
  fields without dragging in correct ones.

The harness runs the same :class:`~docintel.pipeline.Pipeline` used by the API,
so the numbers describe the deployed path rather than a research-only script.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .degrade import degrade_document
from .ingest import ingest_file
from .normalize import canonical_number
from .pipeline import Pipeline
from .schema import DocStatus, Document, FieldName

SCORED_FIELDS: tuple[FieldName, ...] = (
    FieldName.INVOICE_NUMBER,
    FieldName.ISSUE_DATE,
    FieldName.DUE_DATE,
    FieldName.VENDOR_NAME,
    FieldName.VENDOR_GSTIN,
    FieldName.BUYER_NAME,
    FieldName.CURRENCY,
    FieldName.SUBTOTAL,
    FieldName.TAX_AMOUNT,
    FieldName.TOTAL_AMOUNT,
    FieldName.DOCUMENT_TYPE,
)

TEXT_FIELDS = {FieldName.VENDOR_NAME, FieldName.BUYER_NAME}


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #


def _norm_text(value: Any) -> str:
    text = str(value).lower()
    for ch in ".,;:()-–—_/&":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _norm(value: Any, field_name: FieldName) -> Any:
    if value is None:
        return None
    if field_name in (FieldName.SUBTOTAL, FieldName.TAX_AMOUNT, FieldName.TOTAL_AMOUNT):
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return None
    if field_name in (FieldName.ISSUE_DATE, FieldName.DUE_DATE):
        from .normalize import NormalizationError, parse_date

        if isinstance(value, str):
            try:
                return parse_date(value).isoformat()
            except NormalizationError:
                pass
        return str(value)[:10]
    if field_name is FieldName.INVOICE_NUMBER:
        return canonical_number(str(value))
    if field_name is FieldName.VENDOR_GSTIN:
        return str(value).upper().replace(" ", "").replace("-", "")
    if field_name is FieldName.CURRENCY:
        return str(value).upper()
    if field_name in TEXT_FIELDS:
        return _norm_text(value)
    return str(value).strip().lower()


def values_match(pred: Any, gt: Any, field_name: FieldName) -> bool:
    p, g = _norm(pred, field_name), _norm(gt, field_name)
    if p is None or g is None:
        return False
    if isinstance(g, float):
        return abs(p - g) <= 0.011
    return p == g


def token_f1(pred: Any, gt: Any) -> float:
    """Soft overlap score for free-text fields such as party names."""
    if pred is None or gt is None:
        return 0.0
    p = _norm_text(pred).split()
    g = _norm_text(gt).split()
    if not p or not g:
        return 0.0
    common = 0
    remaining = list(g)
    for tok in p:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Report structures
# --------------------------------------------------------------------------- #


@dataclass
class FieldScore:
    field: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    scored: int = 0
    soft_f1_sum: float = 0.0
    soft_f1_n: int = 0
    flagged: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match(self) -> float:
        return self.tp / self.scored if self.scored else 0.0

    @property
    def soft_f1(self) -> float:
        return self.soft_f1_sum / self.soft_f1_n if self.soft_f1_n else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "exact_match": round(self.exact_match, 4),
                "soft_f1": round(self.soft_f1, 4),
            }
        )
        return data


@dataclass
class EvalReport:
    extractor: str
    split: str
    noise_cer: float
    n_docs: int
    fields: dict[str, FieldScore] = field(default_factory=dict)
    per_template: dict[str, dict[str, float]] = field(default_factory=dict)
    doc_exact_match: float = 0.0
    docs_exact: int = 0
    auto_approved: int = 0
    auto_approved_correct: int = 0
    needs_review: int = 0
    rejected: int = 0
    routing_field_precision: float = 0.0
    routing_field_recall: float = 0.0
    arithmetic_tp: int = 0
    arithmetic_fp: int = 0
    arithmetic_fn: int = 0
    arithmetic_tn: int = 0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_doc_confidence: float = 0.0
    errors: int = 0
    settings_fingerprint: str = ""
    created_at: str = ""

    # -- derived ----------------------------------------------------------- #

    @property
    def field_macro_f1(self) -> float:
        f1s = [f.f1 for f in self.fields.values() if f.scored]
        return sum(f1s) / len(f1s) if f1s else 0.0

    @property
    def field_macro_em(self) -> float:
        ems = [f.exact_match for f in self.fields.values() if f.scored]
        return sum(ems) / len(ems) if ems else 0.0

    @property
    def micro_precision(self) -> float:
        tp = sum(f.tp for f in self.fields.values())
        fp = sum(f.fp for f in self.fields.values())
        return tp / (tp + fp) if (tp + fp) else 0.0

    @property
    def micro_recall(self) -> float:
        tp = sum(f.tp for f in self.fields.values())
        fn = sum(f.fn for f in self.fields.values())
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def straight_through_accuracy(self) -> float:
        return self.auto_approved_correct / self.auto_approved if self.auto_approved else 0.0

    @property
    def auto_approval_rate(self) -> float:
        return self.auto_approved / self.n_docs if self.n_docs else 0.0

    @property
    def arithmetic_precision(self) -> float:
        denom = self.arithmetic_tp + self.arithmetic_fp
        return self.arithmetic_tp / denom if denom else 0.0

    @property
    def arithmetic_recall(self) -> float:
        denom = self.arithmetic_tp + self.arithmetic_fn
        return self.arithmetic_tp / denom if denom else 0.0

    def to_dict(self) -> dict:
        return {
            "extractor": self.extractor,
            "split": self.split,
            "noise_cer": self.noise_cer,
            "n_docs": self.n_docs,
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.field_macro_f1, 4),
            "macro_exact_match": round(self.field_macro_em, 4),
            "doc_exact_match": round(self.doc_exact_match, 4),
            "auto_approval_rate": round(self.auto_approval_rate, 4),
            "straight_through_accuracy": round(self.straight_through_accuracy, 4),
            "routing_field_precision": round(self.routing_field_precision, 4),
            "routing_field_recall": round(self.routing_field_recall, 4),
            "arithmetic_precision": round(self.arithmetic_precision, 4),
            "arithmetic_recall": round(self.arithmetic_recall, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "errors": self.errors,
            "settings_fingerprint": self.settings_fingerprint,
            "fields": {name: score.to_dict() for name, score in self.fields.items()},
            "per_template": self.per_template,
        }

    def to_markdown(self) -> str:
        lines = [
            f"### {self.extractor} ({self.split}, n={self.n_docs}, noise={self.noise_cer})",
            "",
            "| field | P | R | F1 | EM | scored |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, score in self.fields.items():
            if not score.scored:
                continue
            lines.append(
                f"| {name} | {score.precision:.3f} | {score.recall:.3f} | {score.f1:.3f} | "
                f"{score.exact_match:.3f} | {score.scored} |"
            )
        lines += [
            "",
            f"- micro F1 **{self.micro_f1:.3f}** (P {self.micro_precision:.3f} / R {self.micro_recall:.3f})",
            f"- document exact match **{self.doc_exact_match:.3f}**",
            f"- auto-approval rate {self.auto_approval_rate:.3f}, "
            f"straight-through accuracy {self.straight_through_accuracy:.3f}",
            f"- review routing: precision {self.routing_field_precision:.3f}, "
            f"recall {self.routing_field_recall:.3f}",
            f"- arithmetic check: precision {self.arithmetic_precision:.3f}, "
            f"recall {self.arithmetic_recall:.3f}",
            f"- latency mean {self.mean_latency_ms:.1f} ms / p95 {self.p95_latency_ms:.1f} ms",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


@dataclass
class Example:
    doc_id: str
    template: str
    split: str
    labels: dict
    document: Document


def load_dataset(dataset_dir: str | Path, settings: Settings, splits: Iterable[str] = ("test",)) -> list[Example]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "ground_truth" / "labels.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found - run `python scripts/generate_dataset.py` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(splits)
    examples: list[Example] = []
    for record in manifest["docs"]:
        if record["split"] not in wanted:
            continue
        document, _note = ingest_file(record["file"], settings)
        examples.append(
            Example(
                doc_id=record["doc_id"],
                template=record["template"],
                split=record["split"],
                labels=record["labels"]
                | {
                    "arithmetic_consistent": record["arithmetic_consistent"],
                    "holdout": bool(record.get("holdout", False)),
                    "template": record["template"],
                },
                document=document,
            )
        )
    return examples


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _empty_scores() -> dict[str, FieldScore]:
    return {name.value: FieldScore(field=name.value) for name in SCORED_FIELDS}


def evaluate_examples(
    examples: list[Example],
    settings: Settings,
    extractor: str,
    split: str = "test",
    noise_cer: float = 0.0,
    pipeline: Pipeline | None = None,
) -> EvalReport:
    pipe = pipeline or Pipeline(settings)
    report = EvalReport(
        extractor=extractor,
        split=split,
        noise_cer=noise_cer,
        n_docs=len(examples),
        settings_fingerprint=settings.fingerprint(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    report.fields = _empty_scores()

    template_stats: dict[str, list[bool]] = {}
    latencies: list[float] = []
    doc_confidences: list[float] = []
    routing_flagged = 0
    routing_wrong = 0
    routing_caught = 0

    for example in examples:
        doc = degrade_document(example.document, noise_cer) if noise_cer > 0 else example.document
        try:
            result = pipe.process_document(doc, extractor=extractor)
        except Exception as exc:  # noqa: BLE001 - an extractor crash is a data point, not a stop
            report.errors += 1
            print(f"  ! {example.doc_id}: {exc}")
            continue
        # latency for the extraction stage only once the pipeline is warm
        latencies.append(result.latency_ms)
        doc_confidences.append(
            statistics.mean([f.confidence for f in result.fields.values()]) if result.fields else 0.0
        )

        flagged_fields = {task.field for task in result.review_tasks if task.field is not None}
        doc_all_correct = True
        any_scored = False
        for name in SCORED_FIELDS:
            gt = example.labels.get(name.value)
            if gt is None or gt == "":
                continue
            any_scored = True
            score = report.fields[name.value]
            score.scored += 1
            predicted = result.fields.get(name)
            pred_value = predicted.value if predicted is not None else None
            if name in TEXT_FIELDS and pred_value is not None:
                score.soft_f1_sum += token_f1(pred_value, gt)
                score.soft_f1_n += 1
            correct = pred_value is not None and values_match(pred_value, gt, name)
            if name in flagged_fields:
                score.flagged += 1
                routing_flagged += 1
            if correct:
                score.tp += 1
            else:
                doc_all_correct = False
                routing_wrong += 1
                if name in flagged_fields:
                    routing_caught += 1
                if pred_value is None:
                    score.fn += 1
                else:
                    score.fp += 1
                    score.fn += 1

        if any_scored:
            report.docs_exact += 1 if doc_all_correct else 0
            template_stats.setdefault(example.template, []).append(doc_all_correct)
            if result.status is DocStatus.AUTO_APPROVED:
                report.auto_approved += 1
                report.auto_approved_correct += 1 if doc_all_correct else 0
            elif result.status is DocStatus.NEEDS_REVIEW:
                report.needs_review += 1
            else:
                report.rejected += 1

        # arithmetic consistency detection
        gt_consistent = bool(example.labels.get("arithmetic_consistent", True))
        flagged_arith = any(issue.code == "ARITHMETIC_MISMATCH" for issue in result.issues)
        if not gt_consistent and flagged_arith:
            report.arithmetic_tp += 1
        elif not gt_consistent and not flagged_arith:
            report.arithmetic_fn += 1
        elif gt_consistent and flagged_arith:
            report.arithmetic_fp += 1
        else:
            report.arithmetic_tn += 1

    report.doc_exact_match = report.docs_exact / report.n_docs if report.n_docs else 0.0
    report.routing_field_precision = routing_caught / routing_flagged if routing_flagged else 0.0
    # with nothing wrong to catch, recall is vacuously 1.0 (reported in the docs)
    report.routing_field_recall = routing_caught / routing_wrong if routing_wrong else 1.0
    report.per_template = {
        template: {
            "n": len(flags),
            "doc_exact_match": round(sum(flags) / len(flags), 4) if flags else 0.0,
        }
        for template, flags in template_stats.items()
    }
    if latencies:
        report.mean_latency_ms = statistics.mean(latencies)
        ordered = sorted(latencies)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        report.p95_latency_ms = ordered[idx]
    if doc_confidences:
        report.mean_doc_confidence = statistics.mean(doc_confidences)
    return report


# --------------------------------------------------------------------------- #
# Threshold sweep + comparison table
# --------------------------------------------------------------------------- #


def threshold_sweep(
    examples: list[Example],
    settings: Settings,
    extractor: str,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Coverage/precision trade-off of the field auto-approve threshold.

    Answers the operational question: if we only auto-accept fields above
    confidence *t*, how much of the document do we process untouched, and how
    often is a silently accepted field wrong?
    """
    thresholds = thresholds or [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    pipe = Pipeline(settings)
    predictions: list[tuple[Example, dict]] = []
    for example in examples:
        result = pipe.process_document(example.document, extractor=extractor)
        predictions.append((example, result.fields))

    rows = []
    for threshold in thresholds:
        accepted = wrong = 0
        docs_all_accepted = 0
        docs_accepted_correct = 0
        for example, fields in predictions:
            accepted_in_doc = 0
            correct_in_doc = 0
            for name in SCORED_FIELDS:
                gt = example.labels.get(name.value)
                if gt is None or gt == "":
                    continue
                fv = fields.get(name)
                if fv is None or fv.confidence < threshold:
                    continue
                accepted += 1
                accepted_in_doc += 1
                if values_match(fv.value, gt, name):
                    correct_in_doc += 1
                else:
                    wrong += 1
            if accepted_in_doc:
                docs_all_accepted += 1
                if correct_in_doc == accepted_in_doc:
                    docs_accepted_correct += 1
        coverage = accepted / max(1, sum(1 for ex in examples for n in SCORED_FIELDS if ex.labels.get(n.value) not in (None, "")))
        rows.append(
            {
                "threshold": threshold,
                "field_coverage": round(coverage, 4),
                "accepted_fields": accepted,
                "wrong_accepted": wrong,
                "accepted_precision": round(1 - wrong / accepted, 4) if accepted else 0.0,
                "docs_fully_accepted": docs_all_accepted,
                "docs_fully_accepted_correct": docs_accepted_correct,
                "straight_through_accuracy": round(docs_accepted_correct / docs_all_accepted, 4) if docs_all_accepted else 0.0,
            }
        )
    return rows


def comparison_table(reports: list[EvalReport]) -> str:
    header = (
        "| extractor | slice | n | micro F1 | macro F1 | doc EM | auto-approved | STP accuracy "
        "| review rate | arith P | arith R |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [
        f"| `{r.extractor}` | {r.split} | {r.n_docs} | {r.micro_f1:.3f} | {r.field_macro_f1:.3f} | "
        f"{r.doc_exact_match:.3f} | {r.auto_approval_rate:.3f} | {r.straight_through_accuracy:.3f} | "
        f"{r.needs_review / max(1, r.n_docs):.3f} | {r.arithmetic_precision:.3f} | {r.arithmetic_recall:.3f} |"
        for r in reports
    ]
    return "\n".join([header, *rows])


def save_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def save_field_csv(reports: list[EvalReport], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["extractor", "split", "noise_cer", "field", "precision", "recall", "f1", "exact_match", "soft_f1", "scored"])
        for report in reports:
            for name, score in report.fields.items():
                if not score.scored:
                    continue
                writer.writerow(
                    [report.extractor, report.split, report.noise_cer, name,
                     round(score.precision, 4), round(score.recall, 4), round(score.f1, 4),
                     round(score.exact_match, 4), round(score.soft_f1, 4), score.scored]
                )
    return path


# --------------------------------------------------------------------------- #
# Suite runner (used by the CLI and scripts/run_eval.py)
# --------------------------------------------------------------------------- #

DEFAULT_NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)


def run_suite(
    settings: Settings,
    splits: Iterable[str] = ("test",),
    extractors: Iterable[str] = ("rules_layout",),
    noise: Iterable[float] | None = None,
    noise_extractor: str | None = None,
    verbose: bool = True,
) -> int:
    """Run every requested extractor over every requested split and archive results.

    Slices reported per extractor: documents from the layout families the rules
    were written against ("seen") and the held-out layout family ("holdout").
    Reporting them separately is the point -- a single blended number hides the
    generalisation gap that decides whether the system is deployable.
    """
    settings = settings
    results_dir = Path(settings.benchmarks_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    examples: dict[str, list[Example]] = {}
    for split in splits:
        examples[split] = load_dataset(settings.dataset_dir, settings, splits=(split,))
    if verbose:
        for split, items in examples.items():
            print(f"loaded {len(items)} {split} documents")

    pipeline = Pipeline(settings)
    reports: list[EvalReport] = []
    for split, items in examples.items():
        seen = [e for e in items if not e.labels.get("holdout")]
        holdout = [e for e in items if e.labels.get("holdout")]
        for extractor in extractors:
            for slice_name, slice_items in (("seen", seen), ("holdout", holdout)):
                if not slice_items:
                    continue
                report = evaluate_examples(
                    slice_items,
                    settings,
                    extractor,
                    split=f"{split}/{slice_name}",
                    pipeline=pipeline,
                )
                reports.append(report)
                save_report(report, results_dir / f"{extractor}__{split}__{slice_name}.json")
                if verbose:
                    print()
                    print(report.to_markdown())

    noise_rows: list[dict] = []
    levels = list(noise) if noise is not None else list(DEFAULT_NOISE_LEVELS)
    extractor_list = list(extractors)
    if levels:
        primary = noise_extractor or ("rules_layout" if "rules_layout" in extractor_list else extractor_list[0])
        base = examples[next(iter(splits))]
        for cer in levels:
            report = evaluate_examples(base, settings, primary, split=f"noise@{cer}", noise_cer=cer, pipeline=pipeline)
            noise_rows.append(
                {
                    "cer": cer,
                    "micro_f1": round(report.micro_f1, 4),
                    "macro_f1": round(report.field_macro_f1, 4),
                    "doc_exact_match": round(report.doc_exact_match, 4),
                    "auto_approval_rate": round(report.auto_approval_rate, 4),
                    "straight_through_accuracy": round(report.straight_through_accuracy, 4),
                    "routing_field_recall": round(report.routing_field_recall, 4),
                    "field_exact_match": {k: round(v.exact_match, 4) for k, v in report.fields.items() if v.scored},
                }
            )
            if verbose:
                print(f"  noise CER {cer:.2f}: micro F1 {report.micro_f1:.3f}, "
                      f"doc EM {report.doc_exact_match:.3f}, auto-approve {report.auto_approval_rate:.3f}")

    sweep_rows: list[dict] = []
    if "dev" in examples:
        primary = noise_extractor or ("rules_layout" if "rules_layout" in extractor_list else extractor_list[0])
        sweep_rows = threshold_sweep(examples["dev"], settings, primary)
        if verbose:
            print("\nthreshold sweep (dev):")
            for row in sweep_rows:
                print(f"  t={row['threshold']:.2f} coverage={row['field_coverage']:.3f} "
                      f"precision={row['accepted_precision']:.3f} STP accuracy={row['straight_through_accuracy']:.3f}")

    save_field_csv(reports, results_dir / "field_metrics.csv")
    summary = {
        "created_at": created,
        "settings_fingerprint": settings.fingerprint(),
        "settings": settings.model_dump(mode="json"),
        "comparison_markdown": comparison_table(reports),
        "reports": [r.to_dict() for r in reports],
        "noise_ablation": noise_rows,
        "threshold_sweep": sweep_rows,
    }
    (results_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print()
        print(summary["comparison_markdown"])
        print(f"\nwrote {results_dir}/latest.json and field_metrics.csv")
    return 0
