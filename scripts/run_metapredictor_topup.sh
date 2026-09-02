#!/bin/bash
# Decode a handful of substrates at the deployed beam and merge them into the natural-drawing
# predictions.
#
# The re-run on the standardiser's drawing covered 79 of the 81 substrates the standardiser
# actually moves: two differ only in how a charge is written, a sulfoxide as S=O against
# S$^{+}$--O$^{-}$ and a phenol against its phenolate, and an earlier test of "moved" did not
# count them. Two of 291 is small and it is not nothing: inside a column captioned as one drawing
# they are the other one, and the check that reads that column refuses while they are.
#
#   bash scripts/run_metapredictor_topup.sh <ids.csv> <index_map.json> <out.json>
#
# ids.csv carries "id,drawn_smiles" and index_map.json maps the same ids to the STORED smiles,
# because the prediction file this merges into is keyed by the substrate as the corpus stores it.
set -euo pipefail
CONDA=${CONDA:-/opt/homebrew/bin/conda}
ENV=${ENV:-metapredictor}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=$ROOT/artifacts/tier2/metapredictor_src
IN_CSV=$1
IDX=$2
TARGET=$3
OUT=$REPO/prediction_topup
mkdir -p "$OUT"
cd "$REPO"

$CONDA run -n $ENV python prepare_input_file.py \
    -input_file "$IN_CSV" -output_file "$REPO/processed_topup.txt"
echo "  lines: $(wc -l < "$REPO/processed_topup.txt")"

# The deployed setting, not the wide one: this fills gaps in the deployed-beam re-run.
$CONDA run -n $ENV bash "$REPO/predict-cpu-wide.sh" \
    "$REPO/processed_topup.txt" "$OUT" 8 8 2 8

$CONDA run -n $ENV python "$ROOT/scripts/tier2_metapredictor_to_json.py" \
    --input-csv "$IN_CSV" \
    --metabolite-txt "$OUT/metabolite.txt" \
    --sub-index-map "$IDX" \
    --per-parent 16 \
    --out "$OUT/topup.json"

python - "$OUT/topup.json" "$TARGET" <<'MERGE'
import json, sys
from pathlib import Path

topup, target = Path(sys.argv[1]), Path(sys.argv[2])
new = json.loads(topup.read_text())
have = json.loads(target.read_text()) if target.exists() else {}
overlap = sorted(set(new) & set(have))
if overlap:
    print(f"refusing: {len(overlap)} of the substrates decoded here are already in the target, "
          f"and overwriting a prediction is not a top-up", file=sys.stderr)
    raise SystemExit(1)
have.update(new)
target.write_text(json.dumps(have, indent=2))
print(f"merged {len(new)} substrates; the file now holds {len(have)}")
MERGE
echo "METAPREDICTOR_TOPUP_DONE"
