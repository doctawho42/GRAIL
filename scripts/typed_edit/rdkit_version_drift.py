#!/usr/bin/env python3
"""What changes if the standardiser runs under a different RDKit than the pinned one.

The stack pins rdkit==2022.09.5 and the reason is in requirements.txt: tautomer canonicalisation is
not stable across releases, and the key every recall figure in this work is scored under is a
tautomer-canonical InChIKey. That pin is also a constraint on where this work can run. A free GPU
platform offering only Python 3.12 offers no RDKit older than 2025, so a retraining there would
build its graphs with a standardiser this project has never used.

Whether that matters is a measurement rather than a judgement, and this makes it. Both versions run
the pipeline's own routine, replicated with rdkit alone so no other dependency has to be satisfied
twice: Cleanup, FragmentParent, Uncharger, TautomerEnumerator.Canonicalize, then a SMILES without
stereochemistry. The comparison is on training substrates, because that is what a retraining would
standardise, and it reports two things separately: whether the standardised structure is the same,
and whether the matching key is.

It needs a second interpreter with the other RDKit in it. The path to that interpreter is an
argument rather than something guessed, and the artifact records the two versions compared, so a
number measured against 2026.03.6 cannot later be read as though it were measured against whatever
is installed at the time.

    python scripts/typed_edit/rdkit_version_drift.py --other /path/to/venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

# The pipeline's standardisation, written out so it can run under an interpreter that has rdkit and
# nothing else of this project. It must stay identical to utils.preparation.standardize_mol; the
# self-check below holds it to that on this side rather than trusting the copy.
PROBE = '''
import json, sys
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog("rdApp.*")
U, T = rdMolStandardize.Uncharger(), rdMolStandardize.TautomerEnumerator()

def standardize(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        t = T.Canonicalize(U.uncharge(rdMolStandardize.FragmentParent(
            rdMolStandardize.Cleanup(m))))
    except Exception:
        return None
    return Chem.MolToSmiles(t, isomericSmiles=False) if t is not None else None

def key(s):
    v = standardize(s)
    if v is None:
        return None
    m = Chem.MolFromSmiles(v)
    try:
        return MolToInchiKey(m) if m is not None else None
    except Exception:
        return None

import rdkit
inp = json.loads(open(sys.argv[1]).read())
json.dump({"rdkit": rdkit.__version__,
           "std": {s: standardize(s) for s in inp},
           "key": {s: key(s) for s in inp}}, open(sys.argv[2], "w"))
'''


def _run(python: str, probe: Path, sample: Path, out: Path) -> dict:
    rc = subprocess.call([python, str(probe), str(sample), str(out)])
    if rc != 0:
        raise SystemExit(f"REFUSING: {python} could not run the probe (exit {rc})")
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True,
                    help="interpreter with the other RDKit, e.g. a venv's bin/python")
    ap.add_argument("--n", type=int, default=3000, help="substrates to compare")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from missing_types_in_train import annotated_pairs

    subs = sorted({a for a, _ in annotated_pairs("train", None)})
    rng = random.Random(args.seed)
    rng.shuffle(subs)
    sample = subs[:args.n]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        probe, inp = td / "probe.py", td / "in.json"
        probe.write_text(PROBE)
        inp.write_text(json.dumps(sample))
        here = _run(sys.executable, probe, inp, td / "a.json")
        there = _run(args.other, probe, inp, td / "b.json")

    # The replica has to agree with the pipeline's own routine on this side, or the comparison is
    # between the other RDKit and a paraphrase rather than between two RDKits.
    from grail_metabolism.utils.preparation import standardize_mol
    checked = 0
    for s in sample[:200]:
        try:
            mine = str(standardize_mol(s))
        except Exception:
            continue
        if here["std"][s] is not None and mine != here["std"][s]:
            print(f"REFUSING: the replica disagrees with utils.preparation.standardize_mol on "
                  f"{s}: {mine} against {here['std'][s]}", file=sys.stderr)
            return 1
        checked += 1

    rows = {}
    for field in ("std", "key"):
        both = [s for s in sample if here[field][s] is not None and there[field][s] is not None]
        differ = [s for s in both if here[field][s] != there[field][s]]
        rows[field] = {"comparable": len(both), "agree": len(both) - len(differ),
                       "differ": len(differ),
                       "share_differing": round(float(len(differ)) / len(both), 6) if both else None,
                       "unavailable_on_one_side": len(sample) - len(both)}

    examples = [{"input": s, "pinned": here["std"][s], "other": there["std"][s]}
                for s in sample if here["std"][s] and there["std"][s]
                and here["std"][s] != there["std"][s]][:8]

    report = {
        "provenance": stamp(__file__),
        "question": ("what a retraining under a different RDKit would standardise differently, "
                     "since the pinned version is not installable on the platform that has the GPU"),
        "pinned_version": here["rdkit"],
        "other_version": there["rdkit"],
        "population": {"substrates_compared": len(sample), "drawn_from": "the training split",
                       "seed": args.seed},
        "replica_checked_against_the_pipeline_on": checked,
        "standardised_structure": rows["std"],
        "matching_key": rows["key"],
        "examples": examples,
        "reading": ("A share near zero means a retraining elsewhere is comparable with what this "
                    "paper reports and the difference can be declared as a bound rather than "
                    "argued about; a share that is not near zero means the platform is unusable "
                    "for this and no amount of declaring fixes it."),
    }
    out = ROOT / "results" / "rdkit_version_drift.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"\n  {here['rdkit']} against {there['rdkit']}, {len(sample)} training substrates")
    for field, label in (("std", "standardised structure"), ("key", "matching key")):
        r = rows[field]
        print(f"    {label:24s} {r['differ']} of {r['comparable']} differ "
              f"({(r['share_differing'] or 0) * 100:.3f}%)")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
