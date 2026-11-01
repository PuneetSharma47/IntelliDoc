#!/usr/bin/env python3
"""Generate the labelled synthetic corpus used by the evaluation harness."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from docintel.synth import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--n", type=int, default=300, help="number of documents")
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()

    manifest = generate_dataset(args.out, n=args.n, seed=args.seed)
    print(f"documents : {manifest['n_docs']} -> {args.out}/raw")
    print(f"labels    : {args.out}/ground_truth/labels.json")
    print(f"splits    : {manifest['splits']}")
    print(f"arithmetic errors injected: {manifest['error_injection_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
