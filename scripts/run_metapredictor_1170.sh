#!/bin/bash
# MetaPredictor inference on the FULL 1170-substrate clean test (native conda, CPU).
# Same pipeline as run_metapredictor.sh but reads artifacts/tier2_1170/mp_input.csv and writes
# artifacts/tier2_1170/metapredictor_preds.json. 9-model CPU ensemble -> expect many hours.
set -euo pipefail

CONDA=/opt/homebrew/bin/conda
ENV=metapredictor
ROOT=/Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746
TIER2_IN=$ROOT/artifacts/tier2_1170
REPO=$ROOT/artifacts/tier2/metapredictor_src
SCRIPTS=$ROOT/scripts

cd "$REPO"
mkdir -p prediction_1170

echo "[1/3] prepare_input_file (canonicalise + tokenise 1170 parents)"
$CONDA run -n $ENV python prepare_input_file.py \
    -input_file "$TIER2_IN/mp_input.csv" \
    -output_file "$REPO/processed_data_1170.txt"
echo "  processed_data lines: $(wc -l < "$REPO/processed_data_1170.txt")"

echo "[2/3] two-stage CPU translate (4-model SoM ensemble -> 5-model metabolite ensemble)"
$CONDA run -n $ENV bash "$REPO/predict-cpu.sh" "$REPO/processed_data_1170.txt" "$REPO/prediction_1170"

echo "[3/3] parse raw ranked metabolite.txt -> JSON"
$CONDA run -n $ENV python "$SCRIPTS/tier2_metapredictor_to_json.py" \
    --input-csv "$TIER2_IN/mp_input.csv" \
    --metabolite-txt "$REPO/prediction_1170/metabolite.txt" \
    --sub-index-map "$TIER2_IN/sub_index_map.json" \
    --out "$TIER2_IN/metapredictor_preds.json"

echo "METAPREDICTOR_1170_RUN_DONE"
