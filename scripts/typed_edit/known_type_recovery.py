#!/usr/bin/env python3
"""Do the 98 known-type misses fall inside the 3% the primitive relaxation adds?

Appendix E.1 splits the 475 references the bank misses in one step: 337 need a reaction type
the bank does not hold, 40 do not type, and 98 need a type it DOES hold whose rule did not
fire on that substrate. Those 98 are the fifth of the ceiling gap the architecture note calls
reachable by loosening rules that already exist, and hypotheses H1 and H5 are written on them.

The relaxation ladder says loosening the convention-dependent primitives -- the hydrogen
count and the connectivity constraints of Table 7 -- adds 3% more admissible sites. That is
a statement about sites, not about references. This asks the question H1 actually needs: of
the 98, how many does that 3% recover?

Phase A reproduces the E.1 decomposition and dumps the pairs, using the same application
convention (`apply_with(..., add_hs=False, norm="canonical", drop_invalid=False)`), the same
tautomer-aware key, and the same MCS typing route as scripts/coverage_gap_types.py, so the
count it reaches is checkable against the committed 98 rather than asserted.

Phase B applies the relaxed bank to the substrates carrying those misses. Only the reactant
side of each SMIRKS is rewritten -- the product template still says what to build, and the
atom maps are preserved -- and the rewrite is skipped inside a recursive `$(...)` group,
where a relaxation under negation would tighten instead.

Recovery is reported against its cost: a bank that emits more matches more for no good
reason, so the extra candidates per substrate are counted beside the references recovered.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _provenance import stamp  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402

from engine_knobs import apply_with  # noqa: E402
from grail_metabolism.metrics import _tautomer_inchikey  # noqa: E402
from grail_metabolism.model.reaction_types import canonical_type  # noqa: E402
from grail_metabolism.utils.preparation import load_default_rules  # noqa: E402
from relaxation_ladder import _strip  # noqa: E402
from scripts.coverage_gap_types import pair_to_type  # noqa: E402
from scripts.run_benchmark import load_test_map  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def relax(smirks: str, drop_h: bool = True, drop_deg: bool = True) -> str:
    """Rewrite the reactant side only. The product template builds, so it is left alone."""
    parts = smirks.split(">")
    if len(parts) < 2:
        return smirks
    parts[0] = _strip(parts[0], drop_h, drop_deg)
    return ">".join(parts)


def _keys(products) -> set:
    out = set()
    for p in products:
        try:
            out.add(_tautomer_inchikey(p))
        except Exception:
            continue
    return out


def phase_a(rules, bank_types, items, log_every=25) -> dict:
    """Reproduce the E.1 split and keep the pairs, not just the counts."""
    cov, gap, known_pairs = Counter(), Counter(), []
    t0 = time.time()
    for i, (sub, true_prods) in enumerate(items, 1):
        if i % log_every == 0 or i == len(items):
            print(f"  A {i}/{len(items)} ({time.time() - t0:.0f}s) covered={cov['covered']} "
                  f"uncovered={cov['uncovered']} known={gap['known_type']} "
                  f"novel={gap['novel_type']} untypeable={gap['untypeable']}",
                  file=sys.stderr, flush=True)
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        covered_keys = _keys(apply_with(sub_mol, rules, False, "canonical", False))
        for met in true_prods:
            try:
                mk = _tautomer_inchikey(met)
            except Exception:
                continue
            if mk in covered_keys:
                cov["covered"] += 1
                continue
            cov["uncovered"] += 1
            met_mol = Chem.MolFromSmiles(met)
            t = pair_to_type(sub_mol, met_mol) if met_mol is not None else None
            if t is None:
                gap["untypeable"] += 1
            elif t in bank_types:
                gap["known_type"] += 1
                known_pairs.append({"substrate": sub, "metabolite": met, "key": mk})
            else:
                gap["novel_type"] += 1
    return {"cov": dict(cov), "gap": dict(gap), "known_pairs": known_pairs,
            "n_substrates": len(items)}


# The rungs of the ladder, as applicable rules rather than as queries. `as_written` is the
# control: it has to recover none of the misses, since a miss is by definition a reference
# the deployed bank does not reach, and a control that recovers some means the phase-A
# convention and the phase-B convention have drifted apart.
ARMS = [("as_written", False, False), ("no_H", True, False), ("no_H_no_deg", True, True)]


def phase_b(rules, known_pairs) -> dict:
    """Apply each rung of the relaxed bank to the substrates carrying a known-type miss."""
    banks, broke = {}, {}
    for name, dh, dd in ARMS:
        wide = [relax(r, dh, dd) for r in rules]
        keep, bad = [], 0
        for original, rewritten in zip(rules, wide):
            usable = rewritten == original
            if not usable:
                # ReactionFromSmarts RAISES on a malformed reactant template rather than
                # returning None, so a rewrite that does not parse has to be caught, not
                # tested for falsiness
                try:
                    usable = Chem.rdChemReactions.ReactionFromSmarts(rewritten) is not None
                except Exception:
                    usable = False
            if usable:
                keep.append(rewritten)
            else:
                bad += 1
                keep.append(original)          # a rewrite that does not parse falls back
        banks[name], broke[name] = keep, bad

    by_sub = {}
    for row in known_pairs:
        by_sub.setdefault(row["substrate"], []).append(row)

    per_arm = {name: {"recovered": [], "still_missed": [], "products": 0}
               for name, _, _ in ARMS}
    monotonicity = []
    t0 = time.time()
    for i, (sub, rows) in enumerate(by_sub.items(), 1):
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            continue
        got = {}
        for name, _, _ in ARMS:
            products = apply_with(mol, banks[name], False, "canonical", False)
            keys = _keys(products)
            per_arm[name]["products"] += len(products)
            got[name] = {row["key"] for row in rows if row["key"] in keys}
            for row in rows:
                (per_arm[name]["recovered"] if row["key"] in keys
                 else per_arm[name]["still_missed"]).append(row)
        # A looser arm cannot recover FEWER references than a tighter one on the same
        # substrate. Counting matches does not guarantee this: a relaxed query can trade one
        # site for another, and the trade is invisible in a total. Compared as sets of
        # reference keys, which is what recovery means.
        for a, b in zip([n for n, _, _ in ARMS], [n for n, _, _ in ARMS][1:]):
            if not got[a] <= got[b]:
                monotonicity.append({"substrate": sub, "tighter": a, "looser": b,
                                     "lost": sorted(got[a] - got[b])})
        print(f"  B {i}/{len(by_sub)} ({time.time() - t0:.0f}s) " +
              "  ".join(f"{n}={len(per_arm[n]['recovered'])}" for n, _, _ in ARMS),
              file=sys.stderr, flush=True)

    base_products = per_arm["as_written"]["products"]
    out = {"n_known_pairs": len(known_pairs), "n_substrates": len(by_sub), "arms": {}}
    for name, _, _ in ARMS:
        a = per_arm[name]
        gained = len(a["recovered"]) - len(per_arm["as_written"]["recovered"])
        out["arms"][name] = {
            "recovered": len(a["recovered"]), "still_missed": len(a["still_missed"]),
            "share_recovered": round(len(a["recovered"]) / max(len(known_pairs), 1), 4),
            "products": a["products"],
            "product_inflation": round(a["products"] / max(base_products, 1), 3),
            "extra_products_per_reference_recovered":
                round((a["products"] - base_products) / gained, 1) if gained > 0 else None,
            "rewrites_that_did_not_parse": broke[name],
            "recovered_pairs": a["recovered"],
        }
    ctrl = out["arms"]["as_written"]["recovered"]
    out["control_recovers"] = ctrl
    out["control_ok"] = ctrl == 0
    out["monotonicity_violations"] = monotonicity
    out["monotonicity_ok"] = not monotonicity
    if ctrl:
        print(f"  CONTROL FAIL: the unrelaxed bank recovers {ctrl} of the misses, so phase A "
              f"and phase B are not applying rules the same way", file=sys.stderr)
    for v in monotonicity[:5]:
        print(f"  MONOTONICITY FAIL: {v['looser']} lost {len(v['lost'])} reference(s) that "
              f"{v['tighter']} recovered on {v['substrate'][:40]}", file=sys.stderr)
    return out


def merge(pattern: str, out_path: str) -> int:
    """Sum the phase-A shards and check the total against the committed decomposition.

    The E.1 split is already in results/coverage_gap_types.json. Recomputing it here is
    worth nothing unless the recomputation is checked against it: a merge that lands on a
    different 98 has changed the application convention, the matcher or the typing route,
    and the pairs it dumps are then not the pairs the appendix is about.
    """
    import glob

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no shard matched {pattern}", file=sys.stderr)
        return 1
    cov, gap, pairs, subs, slices = Counter(), Counter(), [], 0, []
    for p in paths:
        blob = json.loads(Path(p).read_text())["phase_a"]
        cov.update(blob["cov"])
        gap.update(blob["gap"])
        pairs.extend(blob["known_pairs"])
        subs += blob["n_substrates"]
        slices.append(blob["slice"])
        print(f"  + {Path(p).name}: {blob['n_substrates']} substrates, "
              f"known {blob['gap'].get('known_type', 0)}", file=sys.stderr)

    committed = json.loads((ROOT / "results" / "coverage_gap_types.json").read_text())
    checks = {
        "n_substrates": (subs, committed["n_substrates"]),
        "covered_pairs": (cov["covered"], committed["covered_pairs"]),
        "uncovered_pairs": (cov["uncovered"], committed["uncovered_pairs"]),
        "known_type": (gap["known_type"], committed["gap"]["known_type"]),
        "novel_type": (gap["novel_type"], committed["gap"]["novel_type"]),
        "untypeable": (gap["untypeable"], committed["gap"]["untypeable"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    for k, (got, want) in checks.items():
        print(f"  {k:<16} recomputed {got:>6}   committed {want:>6}"
              f"{'   MISMATCH' if got != want else ''}", file=sys.stderr)

    out = {"phase_a": {"cov": dict(cov), "gap": dict(gap), "known_pairs": pairs,
                       "n_substrates": subs, "slices": sorted(slices)},
           "reproduces_committed_decomposition": not bad,
           "mismatches": {k: {"recomputed": v[0], "committed": v[1]} for k, v in bad.items()}}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--known-in", default="", help="phase B: a merged phase-A dump")
    ap.add_argument("--merge", default="", help="glob of phase-A shards to sum")
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_known_type_recovery.json"))
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out)

    rules = load_default_rules()
    bank_types = {t for t in (canonical_type(r) for r in rules) if t is not None}
    print(f"bank: {len(rules)} rules, {len(bank_types)} radius-0 types",
          file=sys.stderr, flush=True)

    out = {"provenance": stamp(__file__), "bank": {"n_rules": len(rules), "n_types": len(bank_types)}}
    if args.phase in ("a", "ab"):
        items = list(load_test_map(None, 42).items())
        sl = items[args.start:(args.end or None)]
        print(f"phase A on [{args.start}:{args.end or len(items)}] of {len(items)}",
              file=sys.stderr, flush=True)
        out["phase_a"] = phase_a(rules, bank_types, sl)
        out["phase_a"]["slice"] = [args.start, args.end or len(items)]
    if args.phase in ("b", "ab"):
        known = (json.loads(Path(args.known_in).read_text())["phase_a"]["known_pairs"]
                 if args.known_in else out["phase_a"]["known_pairs"])
        out["phase_b"] = phase_b(rules, known)

    Path(args.out).write_text(json.dumps(out, indent=1))
    short = json.loads(json.dumps(out))
    if "phase_a" in short:
        short["phase_a"]["known_pairs"] = len(short["phase_a"]["known_pairs"])
    if "phase_b" in short:
        # the per-arm detail moved inside `arms` when phase B gained its rungs; summarising
        # it at the top level looked for keys that are no longer there
        for arm in short["phase_b"].get("arms", {}).values():
            arm["recovered_pairs"] = len(arm.get("recovered_pairs", []))
    print(json.dumps(short, indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
