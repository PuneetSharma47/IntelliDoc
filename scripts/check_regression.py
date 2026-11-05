#!/usr/bin/env python3
"""Fail CI if the committed benchmark floors are not met.

Reads benchmarks/results/latest.json (written by scripts/run_eval.py) and checks the
quality floors this project refuses to regress below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

FLOORS = {
    # (extractor, slice prefix) -> {metric: minimum}
    ("rules_layout", "seen"): {"micro_f1": 0.95, "doc_exact_match": 0.85, "arithmetic_recall": 0.90},
    ("rules_layout", "holdout"): {"micro_f1": 0.70},
    ("rules_flat", "seen"): {"micro_f1": 0.60},
}


def main(path: str = "benchmarks/results/latest.json") -> int:
    report_path = Path(path)
    if not report_path.exists():
        print(f"missing {report_path}: run scripts/run_eval.py first")
        return 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for row in payload["reports"]:
        slice_name = row["split"].split("/")[-1]
        for (extractor, wanted), metrics in FLOORS.items():
            if row["extractor"] != extractor or slice_name != wanted:
                continue
            checked += 1
            for metric, minimum in metrics.items():
                value = row.get(metric, 0.0)
                status = "ok " if value >= minimum else "FAIL"
                print(f"[{status}] {extractor:14s} {slice_name:8s} {metric:18s} {value:.3f} (floor {minimum:.2f})")
                if value < minimum:
                    failures.append(f"{extractor}/{slice_name}/{metric}: {value:.3f} < {minimum:.2f}")
    if not checked:
        print("no matching reports found; nothing checked")
        return 2
    if failures:
        print("\nregression floors missed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nall {checked} floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
