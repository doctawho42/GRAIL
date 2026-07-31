#!/bin/bash
# MetaPredictor on the 150 shared substrates at a wide beam, for the coverage analogue of the
# decomposition. Identical checkpoints, pipeline and seed to the deployed run; only n_best and
# beam_size differ, so the two decodes are comparable and their candidate sets nest.
# Deployed: 8x8 / 2x8 = 16 candidates. Here: 16x16 / 4x16 = 64.
set -euo pipefail
CONDA=/opt/homebrew/bin/conda
ENV=metapredictor
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=$ROOT/artifacts/tier2/metapredictor_src
OUT=$REPO/prediction_wide150
mkdir -p "$OUT"
cd "$REPO"

echo "[1/3] tokenise the 150 parents"
$CONDA run -n $ENV python prepare_input_file.py \
    -input_file "$ROOT/artifacts/tier2/mp_input.csv" \
    -output_file "$REPO/processed_wide150.txt"
echo "  lines: $(wc -l < "$REPO/processed_wide150.txt")"

echo "[2/3] wide-beam two-stage translate (16/16 then 4/16)"
$CONDA run -n $ENV bash "$REPO/predict-cpu-wide.sh" "$REPO/processed_wide150.txt" "$OUT" 16 16 4 16

echo "[3/3] parse to JSON, 64 ranked lines per valid parent"
$CONDA run -n $ENV python "$ROOT/scripts/tier2_metapredictor_to_json.py" \
    --input-csv "$ROOT/artifacts/tier2/mp_input.csv" \
    --metabolite-txt "$OUT/metabolite.txt" \
    --sub-index-map "$ROOT/artifacts/tier2/sub_index_map.json" \
    --per-parent 64 \
    --out "$ROOT/artifacts/tier2/metapredictor_preds_wide64.json"

echo "METAPREDICTOR_WIDEBEAM_DONE"
