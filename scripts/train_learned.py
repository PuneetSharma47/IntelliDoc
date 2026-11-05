#!/usr/bin/env python3
"""Train the linear layout tagger on the training split."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from docintel.config import load_settings
from docintel.evaluate import evaluate_examples, load_dataset
from docintel.extract.learned import LearnedTagger
from docintel.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    train = load_dataset(settings.dataset_dir, settings, splits=(args.train_split,))
    dev = load_dataset(settings.dataset_dir, settings, splits=(args.eval_split,))
    print(f"training documents: {len(train)}  |  dev documents: {len(dev)}")

    tagger = LearnedTagger().fit([(e.document, e.labels) for e in train])
    out = tagger.save(args.out or settings.extraction.learned_model)
    print(f"model written to {out}")
    report = evaluate_examples(dev, settings, "learned", split=args.eval_split, pipeline=Pipeline(settings, tagger=tagger))
    print()
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
