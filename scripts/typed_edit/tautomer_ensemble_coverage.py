"""Free first stage of queue point 4: does applying the bank to a tautomer ensemble raise coverage?

The gemcitabine case is the problem in miniature: on the form a chemist draws the interactive arm
finds 0 of 4, on the corpus form 1 of 4. The bank is applied to one drawing. Applying it to a small
tautomer ensemble of the substrate and unioning the pools can only raise coverage, monotonically,
because the base form is included. This measures how much, with no model: coverage of the deployed
(corpus-form) pool against coverage of the union over the substrate's tautomers.

Coverage is the tautomer-aware key match the whole paper scores under. The base pool is taken from
the frozen widepools rather than recomputed; only the extra tautomers are applied, so the union is
base ∪ (bank applied to each enumerated tautomer). Whole bank, a substrate sample, because applying
7,581 templates per tautomer is the cost.

    python scripts/typed_edit/tautomer_ensemble_coverage.py --substrates 40 --max-tautomers 5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402
RDLogger.DisableLog("rdApp.*")
import bank_without_selection as B  # noqa: E402
from engine_knobs import apply_with  # the deployed application loop  # noqa: E402
from grail_metabolism.metrics import _tautomer_inchikey  # noqa: E402
from _provenance import stamp  # noqa: E402

TE = rdMolStandardize.TautomerEnumerator()


def keyset(smiles_list):
    out = set()
    for s in smiles_list:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=0, help="0 = all that qualify")
    ap.add_argument("--max-tautomers", type=int, default=4)
    ap.add_argument("--heavy-cap", type=int, default=40, help="skip substrates larger than this")
    ap.add_argument("--incomplete-only", action="store_true",
                    help="only substrates whose base pool misses a reference -- where the ensemble "
                         "can help and the gemcitabine question is even asked")
    ap.add_argument("--out", default=str(ROOT / "results" / "tautomer_ensemble_coverage.json"))
    args = ap.parse_args()

    rules = [ln.split()[0] for ln in
             (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    wide, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        b = json.loads(Path(f).read_text())
        wide.update(b["pools"])
        refs_raw.update(b["references"])
    subs = sorted(set(wide) & set(refs_raw))
    if args.incomplete_only:
        subs = [s for s in subs
                if refs_raw[s] and (set(refs_raw[s]) - {c["key"] for c in wide[s]})]
    if args.substrates:
        subs = subs[: args.substrates]

    tot_refs = base_hit = ens_hit = 0
    gained_refs = 0
    per = []
    t0 = time.time()
    for i, s in enumerate(subs, 1):
        if i % 5 == 0:
            print(f"  {i}/{len(subs)} ({time.time()-t0:.0f}s) base={base_hit} ens={ens_hit}",
                  flush=True)
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumHeavyAtoms() > args.heavy_cap:
            continue
        refs = set(refs_raw[s])
        if not refs:
            continue
        base_keys = {c["key"] for c in wide[s]}          # deployed corpus-form pool
        ens_keys = set(base_keys)                          # union starts from the base -> monotone
        try:
            tauts = list(TE.Enumerate(m))[: args.max_tautomers]
        except Exception:
            tauts = []
        for tm in tauts:
            try:
                prods = apply_with(tm, rules, False, "canonical", False)
            except Exception:
                continue
            ens_keys |= keyset(prods)
        b_hit = len(refs & base_keys)
        e_hit = len(refs & ens_keys)
        tot_refs += len(refs)
        base_hit += b_hit
        ens_hit += e_hit
        if e_hit > b_hit:
            gained_refs += e_hit - b_hit
            per.append({"substrate": s, "refs": len(refs), "base": b_hit, "ensemble": e_hit,
                        "tautomers": len(tauts)})

    rep = {"provenance": stamp(__file__),
           "substrates": len(subs), "max_tautomers": args.max_tautomers,
           "population": "incomplete base coverage" if args.incomplete_only else "sample",
           "references": tot_refs,
           "base_coverage": round(base_hit / tot_refs, 4) if tot_refs else None,
           "ensemble_coverage": round(ens_hit / tot_refs, 4) if tot_refs else None,
           "references_gained": gained_refs,
           "coverage_gain": round((ens_hit - base_hit) / tot_refs, 4) if tot_refs else None,
           "substrates_helped": len(per),
           "examples": per[:10],
           "reading": (
               "coverage can only rise, so the number is how much the tautomer ensemble is worth "
               "as a ceiling: a gain means the bank reaches references on a tautomer it misses on "
               "the corpus drawing (the gemcitabine case), and zero means the corpus form already "
               "sees what the bank can reach. This is a coverage ceiling, not recall -- the ranker "
               "still has to bring the newly reachable references into the budget.")}
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n  base coverage {rep['base_coverage']}  ->  ensemble {rep['ensemble_coverage']}  "
          f"(+{rep['coverage_gain']})")
    print(f"  references gained: {gained_refs} over {len(per)} substrates")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
