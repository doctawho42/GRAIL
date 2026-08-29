#!/usr/bin/env bash
# The two coarse shards covering [0,73) and [146,219) hold the largest substrates and have run
# far longer than the two that finished. A shard writes only when it completes, so four hours of
# work is unwritten in each and killing them loses it. This runs the same two ranges again in
# pieces small enough that progress is durable: whatever finishes is on disk, and the coarse
# shards remain the preferred source where they land.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/Users/nikitapolomosnov/anaconda3/envs/test_grail/bin/python}
GEN=artifacts/full5000_implicit/checkpoints/generator.pt
FLT=artifacts/full5000_priors/checkpoints/filter.pt
STEP=${STEP:-12}
CONCURRENCY=${CONCURRENCY:-3}

pieces=()
for a in $(seq 0 $STEP 72); do b=$((a+STEP)); [ $b -gt 73 ] && b=73; pieces+=("$a $b"); done
for a in $(seq 146 $STEP 218); do b=$((a+STEP)); [ $b -gt 219 ] && b=219; pieces+=("$a $b"); done

for piece in "${pieces[@]}"; do
  set -- $piece
  out="results/widepools_std_fine/p$1_$2.json"
  [ -f "$out" ] && continue
  $PY scripts/typed_edit/build_wide_pools.py --start "$1" --end "$2" --present standardised \
      --gen-ckpt $GEN --filter-ckpt $FLT --top-k 7581 --out "$out" >/dev/null 2>&1 &
  # `wait -n` is bash 4; macOS ships 3.2, where the fallback waits for ALL children and
  # serialises the whole run on its slowest piece. Poll for a free slot instead.
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$CONCURRENCY" ]; do sleep 10; done
done
wait
echo "fine pieces written: $(ls results/widepools_std_fine/ 2>/dev/null | wc -l | tr -d ' ')"
