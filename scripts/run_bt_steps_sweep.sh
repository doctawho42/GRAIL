#!/usr/bin/env bash
# Sweep BioTransformer's --steps, the one declared knob of a comparator this paper never swept.
#
# The Supporting Information answers that gap with a PREDICTION rather than a measurement. It can
# only become a measurement against a steps=1 arm run by the same harness on the same day, and
# the reason is measured rather than assumed. Neither frozen BioTransformer artifact records the
# -s it ran under; they were written by different producers (the 291 file by
# typed_edit/biotransformer_arm.py joining an external CSV that is no longer in the repo, the
# 1170 file by bt_predict.py per parent); and over the 291 parents they share they disagree in
# BOTH directions -- 80 parents carry a metabolite the other lacks, 15 are empty in one and not
# the other, 1 the reverse. On top of that the runner that made them could not tell a crash or a
# timeout from a genuine empty. So comparing steps=2 against those numbers would price a harness
# defect, an unrecorded flag and a generation together, and report the sum as the generation.
#
# Hence four runs, in falling order of what they settle:
#   1. steps=1 on the 291-substrate comparison set  -- the baseline the sweep verdicts sit on
#   2. steps=2 on the same 291                      -- the contrast, against arm 1 and nothing else
#   3. steps=1 on the 1170-substrate evaluated set  -- baseline for the whole-population axis
#   4. steps=2 on the same 1170
# Each is --resume, so a kill costs the parent in flight and not the hours behind it, and each
# writes a sidecar naming its flags, its runtime, its per-parent failures and their reasons.
#
# The parents come from the KEYS of the frozen prediction files. That is not a convenience: the
# runner stores `preds[p]` for the same `p` it passes to -ismi, so the keys ARE the strings the
# earlier run fed BioTransformer, and taking them back guarantees the fresh arm sees the same
# input rather than a re-derivation of the population that might spell it differently.
set -uo pipefail
# RUN A SNAPSHOT OF THIS FILE, NOT THIS FILE. bash reads a script incrementally, by byte offset,
# so editing it while it runs makes the interpreter resume at a shifted position: correcting a
# comment here mid-run lengthened the header by four lines and bash re-executed an earlier `run`
# line, which is how arm 1 started a second time and the steps=2 arm was at risk of being skipped.
# Hence GRAIL_ROOT: copy this file somewhere stable, export GRAIL_ROOT to the checkout, and run
# the copy. Editing the original is then harmless.
ROOT="${GRAIL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

OUT="$ROOT/results/bt_steps"
mkdir -p "$OUT"

# The bundled JNI InChI artefact is cached for MAC-X86_64; on arm64 the jar exits before
# predicting anything under any other JDK. Named here so a failure to find it is a clear error
# rather than 1170 parents of silence.
JAVA_HOME_X86="${BIOTRANSFORMER_JAVA_HOME:-/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home}"
export BIOTRANSFORMER_JAVA="$JAVA_HOME_X86/bin/java"
if [ ! -x "$BIOTRANSFORMER_JAVA" ]; then
  echo "REFUSING: no x86_64 Java 8 at $BIOTRANSFORMER_JAVA" >&2
  echo "  set BIOTRANSFORMER_JAVA_HOME to a JDK 8 home, or the jar will exit before predicting." >&2
  exit 1
fi

WORKERS="${BT_WORKERS:-5}"          # 5 JVMs is about 10 GB on this 16 GB machine
TIMEOUT="${BT_TIMEOUT:-1800}"       # steps=2 on a large substrate is minutes, not seconds

run () {  # run <steps> <parents-file> <tag>
  local steps="$1" parents="$2" tag="$3"
  local out="$OUT/preds_s${steps}_${tag}.json"
  echo
  echo "=== steps=$steps on $tag  -> $(basename "$out")  ($(date '+%H:%M:%S'))"
  python "$ROOT/scripts/bt_predict.py" \
      --parents "$parents" --steps "$steps" --out "$out" \
      --workers "$WORKERS" --timeout "$TIMEOUT" --every 10 --resume
}

P291="$OUT/parents_n291.txt"
P1170="$OUT/parents_n1170.txt"
python - "$P291" "$P1170" <<'PY'
import json, sys
for src, dst in ((("results/biotransformer_allhuman_one_step_preds.json"), sys.argv[1]),
                 (("results/biotransformer_fulltest_preds.json"),          sys.argv[2])):
    keys = [k for k in json.load(open(src)) if isinstance(k, str) and k.strip()]
    open(dst, "w").write("\n".join(keys) + "\n")
    print(f"  {dst}: {len(keys)} parents from {src}")
PY

run 1 "$P291"  n291
run 2 "$P291"  n291
run 1 "$P1170" n1170
run 2 "$P1170" n1170

echo
echo "=== all four arms finished ($(date '+%H:%M:%S'))"
for f in "$OUT"/preds_s*.json; do
  case "$f" in *.run.json) continue;; esac
  python - "$f" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
s = json.load(open(p + ".run.json"))
print(f"  {p.split('/')[-1]:<24} {len(d):>5} parents  "
      f"{sum(len(v) for v in d.values()):>7} metabolites  "
      f"{s.get('n_failed', 0):>3} failed  {s.get('elapsed_s', 0)/3600:.2f} h")
PY
done
