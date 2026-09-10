#!/usr/bin/env python3
"""What is in the 'other' class, which is the second largest and which no method recovers well.

The per-class table leaves a class called 'other': the references whose formula delta matches none
of the named biotransformations. It is 142 of the comparison set's 665 references, a fifth of them,
and no method reaches 0.24 on it. A class that large and that unrecovered is worth a name, or worth
showing it cannot have one.

This clusters the 'other' references by their formula delta -- the same delta the classifier reads,
imported rather than restated -- and groups the deltas into chemical families by conservative
rules. A formula delta does not fix a mechanism (an added oxygen is a hydroxylation or an N-oxide
or an epoxide-then-hydrolysis), so a family is named for what the delta is and the ambiguity is
kept, exactly as the per-class table does for its named classes.

The finding the clustering is built to test: whether 'other' is one unnamed transformation or a
long tail of rare ones. The answer decides what a reader should take from it -- a gap in the
vocabulary, or a floor on what any template bank mined from few examples can reach.

    python scripts/other_class_clustering.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B          # noqa: E402
from error_by_chemistry import classify     # noqa: E402


def family(delta: dict) -> str:
    """A conservative chemical family for a formula delta, or 'unclassified tail'.

    The rules name what the delta is, not the mechanism it implies. They are deliberately loose:
    a family groups deltas a chemist would read as the same kind of change, and everything that
    does not fit a family stays in the tail rather than being forced into one.
    """
    C, H, N, O = (delta.get(e, 0) for e in "CHNO")
    S, P = delta.get("S", 0), delta.get("P", 0)
    hal = sum(delta.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    if P > 0:
        return "phosphate conjugation / nucleotide adduct (P added)"
    if S > 0 and C >= 5 and N >= 1:
        return "thiol conjugation (glutathione / cysteine / mercapturate)"
    if C >= 4 and O >= 2 and H >= 4:
        return "large O-bearing conjugate (glucuronide-like / acyl)"
    if set(delta) <= {"O"} and O > 0:
        return f"poly-oxygenation (+{O} O, no other change)"
    if set(delta) <= {"O"} and O < 0:
        return "deoxygenation (O removed, no other change)"
    if H == -2 and O == -1 and not (N or S or P or C or hal):
        return "dehydration (loss of H2O)"
    if H <= -2 and set(delta) <= {"H"}:
        return f"desaturation ({H} H, no other change)"
    if hal < 0 and O > 0:
        return "oxidative dehalogenation (halogen out, O in)"
    if hal < 0:
        return "dehalogenation (halogen removed)"
    if C < 0 and O > 0:
        return "oxidative cleavage (C lost, O gained)"
    return "unclassified tail"


def delta_str(d: dict) -> str:
    return " ".join(f"{el}{'+' if n > 0 else ''}{n}" for el, n in sorted(d.items())) or "(none)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "other_class_clustering.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    pools, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs_raw.update(blob["references"])
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(set(pools) & set(truth) & set(refs_raw))

    deltas = defaultdict(list)          # delta_str -> rows
    fams = defaultdict(list)            # family -> rows
    total = 0
    for s in subs:
        sm = Chem.MolFromSmiles(s)
        if sm is None:
            continue
        pool_keys = {c["key"] for c in pools[s]}
        for met in truth.get(s, []):
            mm = Chem.MolFromSmiles(met)
            if mm is None:
                continue
            name, delta = classify(sm, mm)
            if name != "other":
                continue
            total += 1
            row = {"substrate": s, "metabolite": met, "delta": delta_str(delta),
                   "in_pool": B._key(met) in pool_keys}
            deltas[delta_str(delta)].append(row)
            fams[family(delta)].append(row)

    def summarise(d):
        out = []
        for k, rows in sorted(d.items(), key=lambda kv: -len(kv[1])):
            inp = sum(r["in_pool"] for r in rows)
            out.append({"name": k, "references": len(rows), "in_pool": inp,
                        "pool_coverage": round(inp / len(rows), 3),
                        "example": rows[0]["metabolite"]})
        return out

    fam_rows = summarise(fams)
    delta_rows = summarise(deltas)
    singletons = sum(1 for v in deltas.values() if len(v) == 1)
    tail_refs = sum(len(v) for k, v in fams.items() if k == "unclassified tail")

    rep = {"config": {**B._code_version(), "population": len(subs),
                      "pools": "results/widepools_implicit/w*.json"},
           "other_references": total,
           "distinct_formula_deltas": len(deltas),
           "singleton_deltas": singletons,
           "reading": ("a long tail: most 'other' references are a formula delta seen once, so the "
                       "class is not one unnamed transformation but many rare ones, and a bank "
                       "mined from few examples has few templates for any of them"
                       if singletons > total / 3 else
                       "a few named families cover most of 'other'"),
           "families": fam_rows,
           "largest_deltas": delta_rows[:20]}

    print(f"  'other': {total} references, {len(deltas)} distinct formula deltas, "
          f"{singletons} seen once")
    print(f"\n  {'family':<52}{'refs':>5}{'in_pool':>9}{'cov':>6}")
    for r in fam_rows:
        print(f"  {r['name']:<52}{r['references']:>5}{r['in_pool']:>9}{r['pool_coverage']:>6.2f}")
    print(f"\n  {rep['reading']}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
