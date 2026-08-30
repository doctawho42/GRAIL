#!/usr/bin/env bash
# Validation pools at a range of rule budgets, so the deployed budget is a point on a measured
# curve rather than the value the released checkpoint happened to carry. The budget of 30 already
# exists as results/valpools_k30/all.json and is not rebuilt.
set -euo pipefail
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for K in 10 20 50 100; do
  OUT="results/valpools_k${K}/all.json"
  if [ -f "$OUT" ]; then echo "have $OUT"; continue; fi
  mkdir -p "$(dirname "$OUT")"
  echo "=== rule budget $K ==="
  $PY scripts/typed_edit/build_val_pools.py --top-k "$K" --out "$OUT"
done
echo "budget sweep done"
