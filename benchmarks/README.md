# Benchmark results

`results/latest.json` is the full output of the most recent
`python scripts/run_eval.py` run: per-field precision/recall/F1/exact-match, document
exact match, straight-through accuracy, routing quality, the noise ablation, the
threshold sweep, and the exact settings fingerprint that produced all of it.

`results/field_metrics.csv` is the same data flattened to one row per
extractor × slice × field, for plotting or diffing between runs.

`results/<extractor>__<split>__<slice>.json` is one report per pair, so a single
regression can be traced without re-running everything.

Run-to-run diffs are the intended workflow:

```bash
python scripts/run_eval.py --split test --extractors rules_layout learned
git diff benchmarks/results/
```

The numbers currently committed are summarised in `../docs/EVALUATION.md`.
