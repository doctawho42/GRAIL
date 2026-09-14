#!/usr/bin/env python3
"""Phase 2: what would be submitted to MetaTox for the evaluated population.

Writes revision/metatox_submission_1170/ (substrate CSV, the builder's own files, a manifest).

MetaTox is a web service and this repository records no programmatic interface to it, so the
deliverable here is the submission itself rather than a column: the substrates that have no MetaTox
row, in the format the existing builder consumes, with the join key written down. Running it is the
author's action.

Why the set is not taken from a prediction CSV. scripts/make_metatox_input.py reads the substrate
column of a GRAIL prediction CSV, and every full-test CSV in this repository holds 1,169 of the
1,170: one CoA thioester is in the references and in none of them. A set drawn from a CSV would be
short by that molecule and the returned column's denominator would quietly disagree with the table
it joins, so the population file is the source and the builder is handed a CSV built from it.

    python revision/phase2_metatox_submission.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POPULATION_FILE = "results/test_references.json"
BASELINE = "results/metatox_smirks_preds.json"
OUTDIR = ROOT / "revision" / "metatox_submission_1170"
BUILDER = "scripts/make_metatox_input.py"


def _population() -> list:
    """The evaluated substrates, through the accessor the comparator runs themselves use."""
    from gloryx_via_service import population
    return population("evaluated1170")


def _held() -> set:
    """The substrates MetaTox has already answered, read through the declared accessor.

    The baseline is the SMIRKS run every published MetaTox number is computed from, not the
    248-substrate side analysis that sits beside it under a similar name.
    """
    blob = json.loads((ROOT / BASELINE).read_text())
    return set(blob["predictions"])


def to_submit() -> list:
    """The population minus what is held, in the population's own deterministic order."""
    held = _held()
    return [s for s in _population() if s not in held]


def manifest() -> dict:
    """What this submission is, by name rather than by inference."""
    subs = to_submit()
    held = _held()
    return {
        "what_this_is": ("the MetaTox submission set for the evaluated population: the substrates "
                         "with no MetaTox row"),
        "population": "evaluated1170",
        "source": POPULATION_FILE,
        "baseline": BASELINE,
        "held": len(held),
        "to_submit": len(subs),
        "join_key": ("the substrate string exactly as the corpus stores it, which is what the "
                     "scoring joins on; the builder submits the natural tautomer and asserts both "
                     "share a tautomer InChIKey"),
        "why_not_a_predictions_csv": ("the full-test prediction CSVs hold 1169 of the 1170; one CoA "
                                      "thioester is absent from all of them"),
        "next_step": ("run this batch in MetaTox, layer 1, and return the predictions keyed by the "
                      "id or the substrate string in substrate_map.csv"),
    }


def write_substrate_csv(path) -> Path:
    """A one-column CSV the existing builder can read, so the submission format cannot drift."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Newline "\n" rather than "": csv.writer terminates rows with CRLF by default, git stores the
    # file normalised to LF, and the working tree then differs from the commit the moment anyone
    # re-runs this. A producer whose own output makes the tree dirty is not rerunnable.
    with open(path, "w", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["substrate"])
        w.writeheader()
        for s in to_submit():
            w.writerow({"substrate": s})
    return path


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = write_substrate_csv(OUTDIR / "substrates_to_submit.csv")
    man = manifest()

    # One builder for every batch: the join key and the submission drawing are its guarantees, and
    # a second implementation here is how they would drift apart between batches.
    cmd = [sys.executable, BUILDER, "--predictions", str(csv_path), "--outdir", str(OUTDIR),
           "--split", "test-1170", "--purpose",
           "the Phase 2 MetaTox column on all 1,170 evaluated substrates"]
    print("running the repository's own submission builder:")
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    man["builder"] = {"command": cmd[1:], "returncode": proc.returncode,
                      "stdout_tail": proc.stdout.strip().splitlines()[-12:],
                      "stderr_tail": proc.stderr.strip().splitlines()[-8:]}
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        print(f"REFUSING to record a submission the builder could not produce "
              f"(exit {proc.returncode})")
        (OUTDIR / "manifest.json").write_text(json.dumps(man, indent=1))
        return 1

    try:
        from _provenance import stamp
        man = {"provenance": stamp(__file__), **man}
    except Exception as e:
        man = {"provenance": {"unavailable": f"{e.__class__.__name__}: {e}"}, **man}
    (OUTDIR / "manifest.json").write_text(json.dumps(man, indent=1))

    print(f"\n  population {man['to_submit'] + man['held']}  held {man['held']}  "
          f"to submit {man['to_submit']}")
    for f in sorted(OUTDIR.iterdir()):
        print(f"  {f.stat().st_size:>9} bytes  {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
