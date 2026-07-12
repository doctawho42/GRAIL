#!/usr/bin/env bash
# Regenerate every theory-spine headline number in order, from committed
# checkpoints + the symlinked dataset (grail_metabolism/data/). Run from repo root.
# This is the one-command regen referenced by the TAME provenance table in
# docs/GRAIL_FRAMING.md (§Reproducibility & provenance).
set -euo pipefail

python scripts/run_benchmark.py                       # results/benchmark_report.json          (~20-30 min: rule-bank ceiling 0.735 over 1170 + SyGMa baseline)
python scripts/factorize_recall.py                    # results/recall_factorization.json       (~90 min: deployed pipeline + parallel full-bank ceiling on all 1170)
python scripts/ceiling_external_validity.py           # results/ceiling_external_validity.json  (~5 min: internal vs GLORYx-37 uncapped ceiling + covariate OLS)
python scripts/anchor_certification.py                # results/anchor_certification.json       (~13 min: SyGMa re-derive + paired bootstrap + McNemar on the common set)
python scripts/make_factorization_figure.py           # docs/benchmark/factorization_waterfall.{png,svg}  (<1 min: reads recall_factorization.json)

echo "headline numbers regenerated under results/ and docs/benchmark/"
