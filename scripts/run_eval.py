#!/usr/bin/env python3
"""Run the evaluation suite and archive the results under benchmarks/results/."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from docintel.config import load_settings
from docintel.evaluate import run_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", nargs="+", default=["dev", "test"])
    parser.add_argument("--extractors", nargs="+", default=["rules_flat", "rules_layout", "learned"])
    parser.add_argument("--noise", nargs="*", type=float, default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    return run_suite(
        settings,
        splits=args.split,
        extractors=args.extractors,
        noise=args.noise,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
