#!/usr/bin/env python3
"""What the annotated references actually are, chemically, on the split the paper evaluates on.

The paper reports recall against these references at length and never describes them. That is the
wrong way round: a recall figure is a statement about an annotation, and a reader cannot weigh it
without knowing what the annotation contains. Two defects were already visible in passing --- two
substrates carry their own key among their references, and the annotation is positive-unlabelled
so absence is not evidence --- and both were found while measuring something else.

This is the census that should have been there. Nothing here is a model output: it is the corpus
read back.

  what they are        how each reference stands to its parent in heavy atoms, so cleavages,
                       substitutions and conjugations are counted rather than assumed
  what repeats         references shared by more than one substrate, and references that collapse
                       onto each other inside one substrate's set under the paper's own key
  what is degenerate   references equal to their own substrate, which the parent-drop convention
                       discards, and references that do not parse or do not standardise
  what is unusual      the element inventory, and the references carrying an element the rule
                       bank never introduces or removes

    python scripts/typed_edit/reference_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

# Heavy-atom deltas below this are read as a cleavage and above it as a conjugation; between,
# as a substitution. The bounds are the size of a hydroxylation on either side, which is the
# smallest edit the corpus contains, so nothing is classified by a threshold tuned to the data.
CLEAVAGE, CONJUGATION = -1, +1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "reference_audit.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.metrics import _tautomer_inchikey as key_of

    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    rows = json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]
    substrates = [r["sub"] for r in rows]
    truth = {s: truth[s] for s in substrates if s in truth}

    stats = Counter()
    delta_hist, element_hist = Counter(), Counter()
    by_key = defaultdict(set)
    self_reference, unparsed, unkeyable = [], [], []
    collapsed_within = 0
    heavy_deltas = []

    for substrate, references in truth.items():
        parent = Chem.MolFromSmiles(substrate)
        if parent is None:
            stats["substrate_unparsed"] += 1
            continue
        parent_key = key_of(substrate)
        parent_heavy = parent.GetNumHeavyAtoms()
        seen_keys = set()
        for reference in references:
            stats["references"] += 1
            mol = Chem.MolFromSmiles(reference)
            if mol is None:
                stats["unparsed"] += 1
                if len(unparsed) < 10:
                    unparsed.append(reference[:90])
                continue
            key = key_of(reference)
            if not key:
                stats["unkeyable"] += 1
                if len(unkeyable) < 10:
                    unkeyable.append(reference[:90])
                continue
            if key == parent_key:
                stats["equal_to_their_own_substrate"] += 1
                if len(self_reference) < 10:
                    self_reference.append({"substrate": substrate[:80], "reference": reference[:80]})
            if key in seen_keys:
                collapsed_within += 1
            seen_keys.add(key)
            by_key[key].add(substrate)

            delta = mol.GetNumHeavyAtoms() - parent_heavy
            heavy_deltas.append(delta)
            delta_hist[max(-20, min(20, delta))] += 1
            if delta <= CLEAVAGE:
                stats["smaller_than_the_parent"] += 1
            elif delta >= CONJUGATION:
                stats["larger_than_the_parent"] += 1
            else:
                stats["same_heavy_atom_count"] += 1
            for atom in mol.GetAtoms():
                element_hist[atom.GetSymbol()] += 1

    shared = {k: v for k, v in by_key.items() if len(v) > 1}
    heavy_deltas.sort()
    n = len(heavy_deltas)

    def pct(p):
        return heavy_deltas[min(n - 1, int(p * n))] if n else None

    report = {
        "provenance": stamp(__file__),
        "population": {"substrates": len(truth), "references": stats["references"],
                       "source": "results/test_references.json on the evaluated test set"},
        "counts": dict(stats),
        "distinct_reference_keys": len(by_key),
        "keys_shared_by_more_than_one_substrate": len(shared),
        "references_collapsing_onto_another_within_one_substrate": collapsed_within,
        "heavy_atom_delta": {
            "min": heavy_deltas[0] if n else None, "max": heavy_deltas[-1] if n else None,
            "p05": pct(0.05), "median": pct(0.5), "p95": pct(0.95),
            "histogram_clipped_at_20": {str(k): v for k, v in sorted(delta_hist.items())},
        },
        "elements": dict(element_hist.most_common()),
        "examples": {"equal_to_their_own_substrate": self_reference,
                     "unparsed": unparsed, "unkeyable": unkeyable},
        "classification": (f"a reference is read as a cleavage at a heavy-atom delta of "
                           f"{CLEAVAGE} or below and as a conjugation at {CONJUGATION} or above, "
                           "which are the bounds of a single hydroxylation and are not tuned"),
        "reading": ("the annotation is positive-unlabelled: an absent pair is not a negative, "
                    "which is why precision is reported and not used to order systems"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    c = report["counts"]
    print(f"{c['references']} references over {len(truth)} substrates, "
          f"{len(by_key)} distinct under the paper's key")
    print(f"  smaller than the parent  {c.get('smaller_than_the_parent', 0)}")
    print(f"  same heavy-atom count    {c.get('same_heavy_atom_count', 0)}")
    print(f"  larger than the parent   {c.get('larger_than_the_parent', 0)}")
    print(f"  equal to their substrate {c.get('equal_to_their_own_substrate', 0)}")
    print(f"  unparsed / unkeyable     {c.get('unparsed', 0)} / {c.get('unkeyable', 0)}")
    print(f"  keys shared by >1 substrate {report['keys_shared_by_more_than_one_substrate']}")
    print(f"  collapsing within one set   {collapsed_within}")
    d = report["heavy_atom_delta"]
    print(f"  heavy-atom delta: min {d['min']}, p05 {d['p05']}, median {d['median']}, "
          f"p95 {d['p95']}, max {d['max']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
