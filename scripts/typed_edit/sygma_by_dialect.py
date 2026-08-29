#!/usr/bin/env python3
"""SyGMa on the comparison set, in the dialect the corpus stores and in the one a chemist draws.

The substrate presentation is an axis, and the arms of this comparison did not all meet it the
same way. GRAIL, SyGMa and MetaPredictor were each handed the substrate exactly as the corpus
stores it. MetaTox was not: scripts/make_metatox_input.py re-tautomerises the submission form for
79 of the 291 because, in its own words, the stored form is an unnatural imidic acid and sending
it to an external tool would be unfair to that tool.

That asymmetry has a direction. SyGMa's published rules are written in the natural amide
notation -- not one of its 175 templates requires an imidic reactant -- while the mined half of
GRAIL's bank requires one in 684 templates against 628 for the amide. A comparison run entirely
in the corpus dialect is run in the dialect one bank was mined in and the other was not.

This measures the size of that. SyGMa is re-run on the same substrates presented both ways,
scored against the same annotation under the same tautomer-aware key, with paired intervals on
the difference.

    python scripts/typed_edit/sygma_by_dialect.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0
_SC = None


def _enumerate(shown: str) -> list[str]:
    """SyGMa's ranked prediction structures for one presentation of one substrate.

    Only the enumeration happens here. Keying is done by the caller through the project's cached
    tautomer table: canonicalising a tautomer is a search rather than a lookup and dominates
    everything else, and the table already holds what SyGMa produces on these substrates.
    """
    global _SC
    import sygma
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    if _SC is None:
        _SC = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    mol = Chem.MolFromSmiles(shown)
    if mol is None:
        return []
    try:
        tree = _SC.run(mol)
        tree.calc_scores()
        return [e[0] for e in tree.to_smiles()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "sygma_by_dialect.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.utils.preparation import standardize_mol

    four = json.loads((ROOT / "results" / "four_method_291.json").read_text())
    subs = sorted(four["per_substrate"]) if isinstance(four.get("per_substrate"), dict) else None
    if subs is None:
        from vs_metatox import population
        subs, _, _ = population()
    refs_blob = json.loads((ROOT / "results" / "test_references.json").read_text())
    from bank_without_selection import _key  # the cached tautomer table
    refs = {s: {k for k in (_key(p) for p in refs_blob.get(s, [])) if k} for s in subs}
    subs = [s for s in subs if refs[s]]
    print(f"population: {len(subs)} substrates")

    presented = {}
    for s in subs:
        try:
            presented[s] = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(s)))
        except Exception:
            presented[s] = s
    moved = sum(1 for s in subs if presented[s] != Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    print(f"substrates whose drawing changes: {moved} of {len(subs)}")

    parent = {s: _key(s) for s in subs}
    arms = {}
    for name in ("stored", "standardised"):
        per, t0 = {}, time.perf_counter()
        for i, substrate in enumerate(subs, 1):
            shown = substrate if name == "stored" else presented[substrate]
            out, seen = [], set()
            for smiles in _enumerate(shown):
                k = _key(smiles)
                # the parent is dropped on the same rule every other arm uses, and by the corpus
                # key in both presentations, so the convention cannot itself become a dialect effect
                if not k or k == parent[substrate] or k in seen:
                    continue
                seen.add(k)
                out.append(k)
            per[substrate] = out
            if i % 50 == 0 or i == len(subs):
                print(f"  {name}: {i}/{len(subs)} ({time.perf_counter() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
        arms[name] = per
        print(f"  {name}: mean list {np.mean([len(v) for v in per.values()]):.1f}")

    U = np.array([len(refs[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    out = {}
    for k in KS:
        hits = {name: np.array([len(set(per[s][:k]) & refs[s]) for s in subs], dtype=float)
                for name, per in arms.items()}
        d = hits["standardised"] - hits["stored"]
        bt = d[idx].sum(axis=1) / den
        lo, hi = np.percentile(bt, [2.5, 97.5])
        out[str(k)] = {
            "stored": round(float(hits["stored"].sum() / U.sum()), 4),
            "standardised": round(float(hits["standardised"].sum() / U.sum()), 4),
            "difference": round(float(d.sum() / U.sum()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "separates": bool(lo > 0 or hi < 0),
        }

    rep = {"provenance": stamp(__file__), "n": len(subs),
           "substrates_whose_drawing_changes": moved,
           "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
           "by_budget": out,
           "reading": ("the difference is standardised minus stored, so a positive value is "
                       "recall SyGMa was denied by being handed the corpus's drawing")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{'k':>4}{'stored':>10}{'standardised':>14}{'difference':>12}   interval")
    for k in KS:
        r = out[str(k)]
        mark = " *" if r["separates"] else "  "
        print(f"{k:>4}{r['stored']:>10.4f}{r['standardised']:>14.4f}{r['difference']:>+12.4f}"
              f"{mark} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
