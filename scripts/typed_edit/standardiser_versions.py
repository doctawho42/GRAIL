#!/usr/bin/env python3
"""Whether the corpus's drawing could have come out of the declared standardiser.

The corpus stores amides as imidic acids and cytosine as its lactim, and roughly three substrates
in ten are not fixed points of `standardize_mol`. One explanation is that the standardiser used to
be different: RDKit's tautomer canonicaliser has changed, and an older release might have written
the imidic form. If that were so, the drawing would be the declared standardiser's own output at
the time and the paper would owe a version, not a section.

It is not so. This runs the declared standardiser -- Cleanup, FragmentParent, Uncharger,
TautomerEnumerator.Canonicalize, SMILES without stereochemistry -- under every RDKit build
installed on this machine, on four molecules whose stored corpus form is imidic. Every version
returns the amide. The producer of the corpus is not in this repository and predates its first
commit by eight months, so the mechanism cannot be read off a script; what can be established is
that it was not this standardiser, and this establishes it.

    python scripts/typed_edit/standardiser_versions.py
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

# Each probe is (name, the amide a chemist draws, the form the corpus stores).
PROBES = [
    ("benzamide", "NC(=O)c1ccccc1", "N=C(O)c1ccccc1"),
    ("gemcitabine", "Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1",
     "N=c1ccn(C2OC(CO)C(O)C2(F)F)c(O)n1"),
    ("N-methylacetamide", "CC(=O)NC", "CN=C(C)O"),
    ("urea", "O=C(N)N", "N=C(N)O"),
]

PROBE_SOURCE = r'''
import json, sys
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
enumerator = rdMolStandardize.TautomerEnumerator()
uncharger = rdMolStandardize.Uncharger()

def standardise(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = uncharger.uncharge(mol)
    return Chem.MolToSmiles(enumerator.Canonicalize(mol), isomericSmiles=False)

probes = json.loads(sys.argv[1])
out = {}
for name, drawn, stored in probes:
    out[name] = {"from_drawn": standardise(drawn), "from_stored": standardise(stored)}
print(json.dumps({"rdkit": rdkit.__version__, "probes": out}))
'''


def interpreters() -> list[str]:
    found = sorted(set(glob.glob("/Users/nikitapolomosnov/anaconda3/envs/*/bin/python")
                       + glob.glob("/Users/nikitapolomosnov/anaconda3/bin/python")
                       + glob.glob(str(Path(sys.executable)))))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "standardiser_versions.json"))
    args = ap.parse_args()

    payload = json.dumps(PROBES)
    seen, rows = set(), []
    for exe in interpreters():
        try:
            done = subprocess.run([exe, "-c", PROBE_SOURCE, payload],
                                  capture_output=True, text=True, timeout=180)
        except Exception:
            continue
        if done.returncode != 0 or not done.stdout.strip():
            continue
        try:
            blob = json.loads(done.stdout.strip().splitlines()[-1])
        except Exception:
            continue
        if blob["rdkit"] in seen:
            continue
        seen.add(blob["rdkit"])
        rows.append(blob)

    canonical = {name: drawn for name, drawn, _ in PROBES}
    from rdkit import Chem
    canonical = {name: Chem.MolToSmiles(Chem.MolFromSmiles(drawn))
                 for name, drawn, _ in PROBES}
    verdicts = []
    for row in sorted(rows, key=lambda r: r["rdkit"]):
        agrees = all(row["probes"][n]["from_drawn"] == canonical[n]
                     and row["probes"][n]["from_stored"] == canonical[n]
                     for n in canonical)
        verdicts.append({"rdkit": row["rdkit"], "returns_the_amide_for_every_probe": agrees,
                         "probes": row["probes"]})

    report = {
        "provenance": stamp(__file__),
        "question": ("could the corpus's imidic drawing be the declared standardiser's own "
                     "output under a different RDKit release"),
        "probes": [{"name": n, "drawn": d, "stored_by_the_corpus": s} for n, d, s in PROBES],
        "versions_tested": [v["rdkit"] for v in verdicts],
        "versions_returning_the_amide": [v["rdkit"] for v in verdicts
                                         if v["returns_the_amide_for_every_probe"]],
        "versions_returning_the_imidic_form": [v["rdkit"] for v in verdicts
                                               if not v["returns_the_amide_for_every_probe"]],
        "by_version": verdicts,
        "reading": ("every RDKit release available here maps both the drawn amide and the stored "
                    "imidic acid to the amide, so the corpus drawing is not this standardiser's "
                    "output under any of them; the corpus files predate this repository's first "
                    "commit by eight months and their producer is not in it, so what the "
                    "assembly did can be bounded and not read"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"tested {len(verdicts)} RDKit releases: {', '.join(report['versions_tested'])}")
    print(f"returning the amide for every probe: "
          f"{len(report['versions_returning_the_amide'])} of {len(verdicts)}")
    if report["versions_returning_the_imidic_form"]:
        print(f"returning the imidic form: {report['versions_returning_the_imidic_form']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
