#!/usr/bin/env python3
"""Why the two halves of the rule bank exchange places when the hydrogen convention changes.

Measured through the helper the data-preparation path uses, which expands hydrogens onto the
substrate before matching, the hand-curated fifth of the bank carries the coverage ceiling and the
mined majority adds little: 0.660 against 0.328. Measured the way the deployed generator fires
rules, the two swap: 0.471 against 0.785. Both are correct arithmetic on the same rules and the same
substrates. A convention neither subset is asked about does not move the answer here, it exchanges
it, and reporting the corrected pair without saying why would leave a reversed conclusion unexplained.

Two things are measured, and only the cells no gated artifact already holds:

  1. The helper takes two steps the deployed path does not -- expanding hydrogens, and passing
     products through a validity floor. Coverage at the mixed cell (expanded hydrogens, no floor)
     separates them, since both pure cells are already committed.

  2. The subsets differ in a property that predicts exactly this behaviour, and the paper already
     has the test for it. docs/DISPATCH_PREREGISTRATION.md fixes a syntactic classifier -- does a
     reactant pattern carry the hydrogen ATOM primitive [H] or [#1], which cannot match a substrate
     whose hydrogens are implicit -- and that classifier, which never sees a provenance label, sends
     675 of the 7,581 templates to the expanding convention: every one of them curated, none of the
     5,866 mined. Splitting the curated subset by it holds provenance fixed while notation varies,
     which is what separates the two explanations.

A hypothesis this file was written to test does not survive its own control, and is recorded rather
than quietly dropped. The hydrogen-COUNT primitive inside [CH3] is a different property from the
atom primitive: it is unaffected by the expansion, 560 curated patterns carry it against 9 mined
ones, and splitting on it looks like the same test. It is not. 327 of those 560 also carry the atom
primitive, so the split mixes the populations, and both of its halves move the same way. The census
below reports it so the near-miss is on the record.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from bank_engine_replication import load_bank
from engine_knobs import apply_with
from hydrogen_dispatch import MAJORITY_CONVENTION, classify
from run_benchmark import _tautomer_recovered   # the paper's own per-substrate recovery count

RDLogger.DisableLog("rdApp.*")
_GROUPS: dict = {}

# Inside a SMARTS bracket, H<n> pins the total hydrogen count on a matched atom and D<n> pins its
# explicit connections. A bare element symbol beginning with H or D (Hg, He, Dy) is followed by a
# lowercase letter, never a digit, so requiring the digit separates the two readings.
_BRACKET = re.compile(r"\[([^\]]*)\]")
_PINS_H = re.compile(r"H\d")
_PINS_D = re.compile(r"D\d")


def pins_hydrogen(rule: str) -> bool:
    """Whether the reactant side pins a hydrogen count, which survives expansion unchanged."""
    return any(_PINS_H.search(b) for b in _BRACKET.findall(rule.split(">>")[0]))


def pins_degree(rule: str) -> bool:
    """Whether the reactant side pins a connection count, which expansion does change."""
    return any(_PINS_D.search(b) for b in _BRACKET.findall(rule.split(">>")[0]))


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _init(groups):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    global _GROUPS
    _GROUPS = groups


def _worker(item):
    """One substrate through every requested (rule group, hydrogen convention) cell."""
    sub, trues, cells = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return None
    out = {"sub": sub, "u": 0, "pool": {}}
    for group, add_hs in cells:
        pool = apply_with(mol, _GROUPS[group], add_hs, "canonical", False)
        u, hit, _ = _tautomer_recovered(trues, pool, audit=False)
        out["u"] = int(u)
        out[f"{group}|addhs={int(add_hs)}"] = int(hit)
        out["pool"][f"{group}|addhs={int(add_hs)}"] = len(pool)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="grail_metabolism/resources/extended_smirks.txt")
    ap.add_argument("--mined", default="grail_metabolism/resources/mined_only.txt")
    ap.add_argument("--substrates", default="results/filter_vs_prior_ci.json")
    ap.add_argument("--limit", type=int, default=0, help="0 uses every substrate in the source")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "provenance_knob_attribution.json"))
    args = ap.parse_args()

    bank = [l.strip() for l in open(ROOT / args.bank) if l.strip()]
    mined_all = {l.strip() for l in open(ROOT / args.mined) if l.strip()}
    mined = [r for r in bank if r in mined_all]
    curated = [r for r in bank if r not in mined_all]
    # The control splits the curated subset by the paper's own pre-registered syntactic test
    # (docs/DISPATCH_PREREGISTRATION.md): does the reactant pattern carry the hydrogen ATOM
    # primitive, which cannot match a substrate whose hydrogens are implicit. That is the property
    # the expansion is about. Splitting instead on the hydrogen-COUNT primitive inside [CH3] mixes
    # the two populations -- 327 of those 560 patterns also carry the atom primitive -- and the
    # resulting comparison answers a question nobody asked.
    wants = dict(zip(bank, classify(bank, MAJORITY_CONVENTION["grail_full"])))
    groups = {"curated": curated, "mined": mined,
              "curated_needs_h": [r for r in curated if wants[r]],
              "curated_plain": [r for r in curated if not wants[r]],
              "mined_plain": mined}

    census = {}
    for name, rules in (("curated", curated), ("mined", mined)):
        needs = sum(wants[r] for r in rules)
        h = sum(pins_hydrogen(r) for r in rules)
        d = sum(pins_degree(r) for r in rules)
        census[name] = {"n_rules": len(rules), "needs_hydrogen_atom": needs,
                        "pins_hydrogen_count": h, "pins_degree": d,
                        "share_needing_expansion": round(needs / max(len(rules), 1), 4)}
        print(f"  {name:8} {len(rules):>5} rules: {needs:>4} carry the hydrogen atom primitive "
              f"({needs / len(rules):.3f}); {h} pin a hydrogen count, {d} a connection count")
    overlap = sum(1 for r in curated if wants[r] and pins_hydrogen(r))
    census["atom_and_count_primitives_overlap_in_curated"] = overlap
    print(f"  the two primitives are not the same split: {overlap} curated patterns carry both")

    # Only the cells no committed artifact already holds. The two pure conventions for the whole
    # subsets live in results/ceiling_by_provenance.json and in its predecessor under git, both
    # gated against the ceiling; recomputing them here would cost several times this run and prove
    # nothing the gate has not.
    cells = [("curated_needs_h", False), ("curated_needs_h", True),
             ("curated_plain", False), ("curated_plain", True)]
    for g in dict.fromkeys(g for g, _ in cells):
        print(f"  group {g:16} {len(groups[g]):>5} rules")

    src = json.loads((ROOT / args.substrates).read_text())["per_substrate"]
    refs = json.loads((ROOT / "results" / "test_references.json").read_text())
    items = [(r["sub"], refs[r["sub"]], cells) for r in src if refs.get(r["sub"])]
    if args.limit:
        items = items[:args.limit]
    print(f"substrates: {len(items)}   workers: {args.workers}", flush=True)

    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(args.workers, initializer=_init, initargs=(groups,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, items, chunksize=1), 1):
            if r is not None:
                rows.append(r)
            if i % 25 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)

    U = sum(r["u"] for r in rows)
    keys = [f"{g}|addhs={int(h)}" for g, h in cells]
    cov = {k: round(sum(r[k] for r in rows) / max(U, 1), 4) for k in keys}
    mean_pool = {k: round(sum(r["pool"][k] for r in rows) / max(len(rows), 1), 1) for k in keys}

    print(f"\n  micro coverage on {len(rows)} substrates, {U} references")
    for k in keys:
        print(f"    {k:28} {cov[k]:.4f}   mean pool {mean_pool[k]:>7}")

    # The two committed endpoints, so the attribution is arithmetic a reader can follow.
    committed = json.loads((ROOT / "results/ceiling_by_provenance.json").read_text())["subsets"]
    deployed = {g: committed[g]["coverage"] for g in ("curated", "mined")}
    helper = {"curated": 0.6596, "mined": 0.3280}   # the superseded artifact, still in git

    expanded = {"curated": 0.6614, "mined": 0.3280}   # measured in the preceding pass of this script
    gap_deployed = deployed["mined"] - deployed["curated"]
    gap_expanded = expanded["mined"] - expanded["curated"]
    gap_helper = helper["mined"] - helper["curated"]
    attribution = {
        "gap_deployed": round(gap_deployed, 4),
        "gap_expanded_no_floor": round(gap_expanded, 4),
        "gap_helper": round(gap_helper, 4),
        "moved_by_expanding_hydrogens": round(gap_expanded - gap_deployed, 4),
        "moved_by_the_validity_floor": round(gap_helper - gap_expanded, 4),
    }
    print("\n  mined minus curated, and what carries it from one convention to the other")
    for k, v in attribution.items():
        print(f"    {k:32} {v:+.4f}")

    # Provenance held fixed, notation varied: both halves of the curated subset, both conventions.
    within = {"needs_h": {"addhs=0": cov["curated_needs_h|addhs=0"],
                          "addhs=1": cov["curated_needs_h|addhs=1"],
                          "n_rules": len(groups["curated_needs_h"])},
              "plain":   {"addhs=0": cov["curated_plain|addhs=0"],
                          "addhs=1": cov["curated_plain|addhs=1"],
                          "n_rules": len(groups["curated_plain"])}}
    for name in within:
        within[name]["retained_under_expansion"] = round(
            within[name]["addhs=1"] / max(within[name]["addhs=0"], 1e-9), 4)
    print("\n  within the curated subset, provenance fixed and notation varied")
    for name, v in within.items():
        print(f"    {name:8} {v['n_rules']:>5} rules   {v['addhs=0']:.4f} -> {v['addhs=1']:.4f}   "
              f"retains {v['retained_under_expansion']:.3f}")

    rep = {"config": {**_code_version(), "n_substrates": len(rows), "references": U,
                      "match": "inchikey_tautomer", "aggregation": "micro, ratio of sums",
                      "substrate_source": args.substrates,
                      "switches": "engine_knobs.apply_with(add_hs, 'canonical', drop_invalid=False)",
                      "endpoints": "deployed cell from results/ceiling_by_provenance.json; "
                                   "expanded-and-floored cell from its predecessor under git"},
           "rule_census": census, "coverage": cov, "mean_pool": mean_pool,
           "committed_endpoints": {"deployed": deployed, "helper": helper},
           "gap_attribution": attribution, "within_curated": within,
           "per_substrate": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
