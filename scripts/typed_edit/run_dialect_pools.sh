#!/usr/bin/env bash
# Rebuild the comparison-set pools with every substrate handed to the matcher as the declared
# standardiser draws it, so the substrate presentation can be swept like every other declared
# axis. The pools stay keyed by the corpus string, so the two dialects are paired substrate by
# substrate against one annotation.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
# The released pair, read from the pool builder's own defaults rather than repeated here. This
# script used to name full5000_priors' filter, which is the run the repository once shipped and
# no longer does, so the standardised pools were scored by one model and the pools they are
# compared against by another, and the difference read as an effect of the drawing.
GEN=$($PY -c "import re,pathlib;print(re.search(r'--gen-ckpt\", default=str\(ROOT / \"([^\"]+)', pathlib.Path('scripts/typed_edit/build_wide_pools.py').read_text()).group(1))")
FLT=$($PY -c "import re,pathlib;print(re.search(r'--filter-ckpt\", default=str\(ROOT / \"([^\"]+)', pathlib.Path('scripts/typed_edit/build_wide_pools.py').read_text()).group(1))")
echo "generator: $GEN"
echo "filter:    $FLT"
N=${N:-291}
SHARDS=${SHARDS:-4}
mkdir -p results/widepools_std results/widepools_k30_std

step=$(( (N + SHARDS - 1) / SHARDS ))
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  a=$(( i * step )); b=$(( a + step )); [ $b -gt $N ] && b=$N
  [ $a -ge $N ] && continue
  $PY scripts/typed_edit/build_wide_pools.py --start $a --end $b --present standardised \
      --gen-ckpt $GEN --filter-ckpt $FLT --top-k 7581 \
      --out results/widepools_std/w$i.json >/dev/null 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "exhaustive shards done"

$PY scripts/typed_edit/build_wide_pools.py --start 0 --end $N --present standardised \
    --gen-ckpt $GEN --filter-ckpt $FLT --top-k 30 \
    --out results/widepools_k30_std/all.json >/dev/null 2>&1
echo "interactive done"
