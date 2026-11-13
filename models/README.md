# Model artefacts

Neither model is committed, both are reproducible locally.

| file / directory | produced by | notes |
| --- | --- | --- |
| `learned_tagger.joblib` | `python scripts/train_learned.py` | ~1 s/document to train; a multinomial logistic regression over layout + text-hash features |
| `layoutlmv3-finetuned/` | `python scripts/train_layoutlmv3.py` | ~500 MB checkpoint; see `../docs/LAYOUTLM.md` |
| `train.jsonl`, `dev.jsonl`, `labels.json` | `python scripts/train_layoutlmv3.py --export-only` | word-level BIO records for the transformer |

Without `learned_tagger.joblib` the `learned` and `ensemble` extractors fall back to
`rules_layout` and say so (`EXTRACTOR_FALLBACK` in the issue list).
