#!/usr/bin/env bash
# Regenerate everything downstream of a comparator's predictions, in dependency order.
#
# There was no single entry point and no written order. Twelve scripts read a comparator's
# prediction file and eight of them write a tracked artifact, so correcting one arm meant knowing,
# from nowhere, which to run and in what sequence; running them in the wrong order leaves a table
# built on one state of an artifact beside a table built on another, and nothing downstream can
# see that it happened. This file is that order, executable.
#
# RUN A SNAPSHOT OF THIS FILE, NOT THIS FILE: bash reads a script incrementally by byte offset and
# editing one mid-run makes the interpreter resume at a shifted position. Hence GRAIL_ROOT.
set -uo pipefail
ROOT="${GRAIL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
LOG="${REGEN_LOG:-$ROOT/results/.regen}"
mkdir -p "$LOG"

step () {                      # step <name> <command...>
  local name="$1"; shift
  printf '\n=== %-34s %s\n' "$name" "$(date '+%H:%M:%S')"
  if "$@" > "$LOG/$name.txt" 2>&1; then
    tail -2 "$LOG/$name.txt" | sed 's/^/    /'
  else
    echo "    FAILED (exit $?) -- see $LOG/$name.txt" >&2
    tail -6 "$LOG/$name.txt" | sed 's/^/    /' >&2
    FAILED="${FAILED:-} $name"
  fi
}

P=python

# 1. The table every other comparison reads. Nothing that consumes contrasts may run before it.
step deployment_table   $P scripts/typed_edit/deployment_table.py

# 2. Arms that read the predictions directly. Independent of each other, all downstream of 1
#    only in the sense that they must describe the same arm file.
step criterion_sweep    $P scripts/typed_edit/criterion_sweep.py
step precision_table    $P scripts/typed_edit/precision_table.py
step matched_length     $P scripts/typed_edit/matched_length.py
step parent_drop_effect $P scripts/typed_edit/parent_drop_effect.py
step error_by_chemistry $P scripts/typed_edit/error_by_chemistry.py
step drawing_equalised  $P scripts/typed_edit/drawing_equalised.py

# 3. The comparator's own section, scored from the corrected predictions rather than from the CSV
#    that produced the defective arm: that CSV is gone, and artifacts/tier2/bt_out.csv is a
#    150-substrate pilot, not it. --predictions exists for exactly this.
#    BOTH settings, with the labels paper2_numbers.py indexes by. This producer OVERWRITES its
#    --out with whatever it was given, so passing one setting silently deletes the other from the
#    report and the manuscript's drawing-sensitivity numbers vanish with it. The first draft of
#    this file passed one, and with a stray comma in the label at that.
#    The natural-drawing arm is NOT re-run: it is a different input drawing and no corrected run of
#    it exists, so it still carries the provenance of the arm that was replaced. The SI says so.
step biotransformer_arm $P scripts/typed_edit/biotransformer_arm.py \
     --predictions "allHuman one step=$ROOT/results/biotransformer_allhuman_one_step_preds.json" \
     --predictions "allHuman one step, natural drawing=$ROOT/results/biotransformer_allhuman_one_step_natural_drawing_preds.json"

# A gate that checks the thing that actually went wrong, not that the step exited zero.
step arm_carries_both_settings $P - <<'CHECK'
import json, sys
want = {"allHuman one step", "allHuman one step, natural drawing"}
got = set(json.load(open("results/biotransformer_arm.json"))["by_setting"])
if got != want:
    sys.exit(f"REFUSING: biotransformer_arm.json carries {sorted(got)}, expected {sorted(want)}")
print("both settings present:", sorted(got))
CHECK

# 4. The knob sweep and the whole-population axis.
step biotransformer_steps $P scripts/typed_edit/biotransformer_steps.py
step population_definition $P scripts/typed_edit/population_definition.py

# 5. The correction over the family, which reads the table from 1.
step multiplicity       $P scripts/typed_edit/multiplicity.py

# 6. The revision-era recomputations that quote the same cells.
step phase1_tmain       $P revision/phase1_tmain.py
step phase2_comparators $P revision/phase2_comparators.py

# 7. Numbers, then the macros that carry them, then what prints them.
step paper2_numbers     $P scripts/paper2_numbers.py
step paper2_macros      $P scripts/paper2_macros.py
step paper2_tables      $P scripts/paper2_tables.py
step paper2_si_tables   $P scripts/paper2_si_tables.py
step paper2_figures     $P scripts/paper2_figures.py

echo
if [ -n "${FAILED:-}" ]; then
  echo "STEPS THAT FAILED:${FAILED}" >&2
  exit 1
fi
echo "chain complete $(date '+%H:%M:%S'); build the documents next with scripts/build_paper2.sh"
