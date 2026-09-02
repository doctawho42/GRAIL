#!/bin/bash
# MetaPredictor's emission knob, swept on the comparison set the way SyGMa's scenario is.
#
# The paper asks other work to declare the setting that decides how many candidates a system may
# emit, and sweeps SyGMa's because it is a knob the authors can turn. MetaPredictor's beam is the
# same kind of knob and the same kind of ours: it is a local checkout with its own environment.
# Leaving it at one setting while sweeping the other comparator's is the asymmetry this closes.
#
# Deployed is n_best 8 / beam 8 then 2 / 8, sixteen candidates. Wide is 16 / 16 then 4 / 16,
# sixty-four. Everything else is identical, seed included, so the two decodes are comparable and
# the wide one's candidate set contains the deployed one's.
set -euo pipefail
CONDA=${CONDA:-/opt/homebrew/bin/conda}
ENV=${ENV:-metapredictor}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=$ROOT/artifacts/tier2/metapredictor_src
OUT=$REPO/prediction_wide291
N1=${N1:-16}; B1=${B1:-16}; N2=${N2:-4}; B2=${B2:-16}
PER_PARENT=${PER_PARENT:-64}
mkdir -p "$OUT"
cd "$REPO"

echo "[1/3] tokenise the comparison set"
$CONDA run -n $ENV python prepare_input_file.py \
    -input_file "$ROOT/artifacts/tier2/mp_input_291.csv" \
    -output_file "$REPO/processed_291.txt"
echo "  lines: $(wc -l < "$REPO/processed_291.txt")"

echo "[2/3] two-stage translate at n_best/beam $N1/$B1 then $N2/$B2"
$CONDA run -n $ENV bash "$REPO/predict-cpu-wide.sh" \
    "$REPO/processed_291.txt" "$OUT" "$N1" "$B1" "$N2" "$B2"

echo "[3/3] parse to JSON, $PER_PARENT ranked lines per valid parent"
$CONDA run -n $ENV python "$ROOT/scripts/tier2_metapredictor_to_json.py" \
    --input-csv "$ROOT/artifacts/tier2/mp_input_291.csv" \
    --metabolite-txt "$OUT/metabolite.txt" \
    --sub-index-map "$ROOT/artifacts/tier2/sub_index_map_291.json" \
    --per-parent "$PER_PARENT" \
    --out "$ROOT/results/metapredictor_wide_beam_preds.json"

echo "METAPREDICTOR_BEAM_SWEEP_DONE"
