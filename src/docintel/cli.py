"""Command line interface.

    docintel ingest invoice.pdf          # extract one document, print JSON
    docintel ingest data/raw/*.pdf --store --review
    docintel search "GSTIN 27AABCU"      # page-level search over the library
    docintel stats
    docintel evaluate --split test        # full evaluation, writes benchmarks/
    docintel generate-data --n 300
    docintel train-learned
    docintel serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import Pipeline


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    pipeline = Pipeline(settings)
    store = None
    if args.store:
        from .store import Store

        store = Store(settings.store.db_path)

    paths: list[Path] = []
    for pattern in args.paths:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pdf")))
        elif "*" in pattern:
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(p)
    if not paths:
        print("no input files", file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        try:
            outcome = pipeline.process_path(path, extractor=args.extractor)
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            print(f"! {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        result = outcome.result
        if store is not None:
            store.save_result(result, settings_fp=settings.fingerprint())
        if args.json:
            print(json.dumps(pipeline.explain(result), indent=2, default=str))
            continue
        print(f"\n{path.name}  [{result.status.value}]  {result.doc_type.value} "
              f"({result.doc_type_confidence:.2f})  {result.latency_ms:.1f} ms  via {result.extractor}")
        if outcome.engine_note:
            print(f"  engine: {outcome.engine_note}")
        for name, value in sorted(result.fields.items(), key=lambda kv: kv[0].value):
            print(f"  {name.value:16s} {_fmt(value.value):>16s}   conf {value.confidence:.2f}   "
                  f"p{value.provenance.page} {value.provenance.rule}")
        for issue in result.issues:
            print(f"  ! [{issue.severity.value}] {issue.code}: {issue.message}")
        if result.review_tasks and args.review:
            print("  review:")
            for task in result.review_tasks[: args.max_tasks]:
                label = task.field.value if task.field else "(document)"
                print(f"    - {label:16s} {task.reason.value:22s} priority {task.priority:.2f}  {task.detail}")
    return 1 if failures else 0


def cmd_search(args: argparse.Namespace) -> int:
    from .search import Searcher
    from .store import Store

    settings = load_settings(args.config)
    store = Store(settings.store.db_path)
    hits = Searcher(store, top_k=args.top_k).search(args.query)
    if not hits:
        print("no matches")
        return 0
    for hit in hits:
        print(f"{hit.filename}  p{hit.page + 1}  score {hit.score:.3f}")
        print(f"   {hit.snippet}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .store import Store

    settings = load_settings(args.config)
    store = Store(settings.store.db_path)
    print(json.dumps(store.stats(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_suite

    settings = load_settings(args.config)
    return run_suite(settings, splits=args.split, extractors=args.extractors, noise=args.noise, verbose=True)


def cmd_generate_data(args: argparse.Namespace) -> int:
    from .synth import generate_dataset

    settings = load_settings(args.config)
    manifest = generate_dataset(settings.dataset_dir, n=args.n, seed=args.seed)
    print(f"wrote {manifest['n_docs']} documents to {settings.dataset_dir}/raw")
    print(f"splits: {manifest['splits']}")
    print(f"arithmetic error rate: {manifest['error_injection_rate']:.3f}")
    return 0


def cmd_train_learned(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    from .evaluate import evaluate_examples, load_dataset
    from .extract.learned import LearnedTagger

    train_examples = load_dataset(settings.dataset_dir, settings, splits=("train",))
    dev_examples = load_dataset(settings.dataset_dir, settings, splits=("dev",))
    print(f"training on {len(train_examples)} documents")
    tagger = LearnedTagger().fit([(e.document, e.labels) for e in train_examples])
    path = tagger.save(settings.extraction.learned_model)
    print(f"saved model to {path}")
    report = evaluate_examples(dev_examples, settings, "learned", split="dev", pipeline=Pipeline(settings, tagger=tagger))
    print(report.to_markdown())
    return 0


def cmd_seed_demo(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    from .store import Store

    store = Store(settings.store.db_path)
    pipeline = Pipeline(settings)
    demo = Path(settings.dataset_dir) / "demo"
    if not demo.exists():
        print("no demo documents bundled", file=sys.stderr)
        return 2
    count = 0
    for path in sorted(demo.glob("*.pdf")):
        outcome = pipeline.process_path(path)
        store.save_result(outcome.result, settings_fp=settings.fingerprint())
        count += 1
    print(f"indexed {count} demo documents into {settings.store.db_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from api.main import create_app

    settings = load_settings(args.config)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docintel", description="Document extraction, validation and review")
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract fields from one or more files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--extractor", default=None, choices=["rules_flat", "rules_layout", "learned", "layoutlmv3", "ensemble"])
    p.add_argument("--store", action="store_true", help="persist results so they appear in the review console")
    p.add_argument("--review", action="store_true", help="print the review queue")
    p.add_argument("--max-tasks", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="search page text across the library")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="show library statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("evaluate", help="run the evaluation suite")
    p.add_argument("--split", nargs="+", default=["test"])
    p.add_argument("--extractors", nargs="+", default=["rules_layout"])
    p.add_argument("--noise", nargs="*", type=float, default=None, help="simulated character error rates")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("generate-data", help="regenerate the labelled synthetic corpus")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260101)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train-learned", help="train the linear layout tagger")
    p.set_defaults(func=cmd_train_learned)

    p = sub.add_parser("seed-demo", help="index the bundled demo documents")
    p.set_defaults(func=cmd_seed_demo)

    p = sub.add_parser("serve", help="run the review console")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
