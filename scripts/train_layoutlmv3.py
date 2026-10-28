#!/usr/bin/env python3
"""Fine-tune LayoutLMv3 for token-level field tagging.

Two modes:

``--export-only``
    Writes the word-level BIO JSONL and the label list. Runs with only the core
    dependencies installed, so the training data is reproducible on any machine.

(default)
    Trains ``microsoft/layoutlmv3-base`` on that JSONL and saves a checkpoint to
    ``models/layoutlmv3-finetuned``, which :class:`docintel.extract.layoutlm.LayoutLMv3Extractor`
    picks up automatically: ``docintel ingest invoice.pdf --extractor layoutlmv3``.

Training tips for this task size (~200 documents, ~40k words):

* 15-20 epochs, lr 5e-5, batch size 2 with gradient accumulation 8 fits a 8 GB GPU.
* Keep ``--fp16`` off on CPU-only machines; the script falls back automatically.
* Evaluate with the same harness as the other extractors so the comparison table
  in docs/EVALUATION.md stays apples-to-apples:

      python scripts/train_layoutlmv3.py --export-only
      python scripts/train_layoutlmv3.py
      python scripts/run_eval.py --extractors rules_layout learned layoutlmv3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from docintel.config import load_settings
from docintel.evaluate import load_dataset
from docintel.extract.layoutlm import training_records, write_jsonl
from docintel.extract.learned import LABELS


def export(settings, split: str, out_dir: Path) -> Path:
    examples = load_dataset(settings.dataset_dir, settings, splits=(split,))
    records = training_records([(e.document, e.labels) for e in examples])
    path = write_jsonl(records, out_dir / f"{split}.jsonl")
    (out_dir / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    print(f"{split}: {len(records)} page records -> {path}")
    return path


def train(args: argparse.Namespace) -> int:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        print(f"training dependencies missing: {exc}")
        print('install them with: pip install "intellidoc[ml]"')
        return 2

    label2id = {label: idx for idx, label in enumerate(LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    def load_records(path: Path) -> Dataset:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return Dataset.from_list(
            [
                {
                    "tokens": row["tokens"],
                    "bboxes": row["bboxes"],
                    "labels": [label2id.get(tag, 0) for tag in row["labels"]],
                }
                for row in rows
            ]
        )

    out_dir = Path(args.out)
    train_ds = load_records(export(args.settings, "train", out_dir))
    dev_ds = load_records(export(args.settings, "dev", out_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    def encode(batch):
        enc = tokenizer(
            batch["tokens"],
            boxes=batch["bboxes"],
            truncation=True,
            max_length=512,
            padding="max_length",
            word_labels=batch["labels"],
        )
        return enc

    train_ds = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(encode, batched=True, remove_columns=dev_ds.column_names)

    collator = DataCollatorForTokenClassification(tokenizer)
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        fp16=use_cuda and args.fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )
    trainer.train()
    final_dir = out_dir / "layoutlmv3-finetuned"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"checkpoint saved to {final_dir}")
    print("evaluate with: python scripts/run_eval.py --extractors rules_layout learned layoutlmv3")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=float, default=18)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    args.settings = load_settings(args.config)
    out_dir = Path(args.out)
    if args.export_only:
        export(args.settings, "train", out_dir)
        export(args.settings, "dev", out_dir)
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
