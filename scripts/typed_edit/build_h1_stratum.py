#!/usr/bin/env python3
"""The membership file H1 is registered on, built before the freeze rather than after the run.

H1 predicts that typing the label space moves recall where it moves supervision: onto the
substrates whose reference transformation needs a label that is sparse when rules are the
labels and dense when types are. Rules as labels leave 35.6% of the training pairs in labels
with fewer than five examples; types as labels leave 22.1%. This writes down exactly which
substrates the 13.5 points that move belong to, so the stratum cannot be drawn around the
answer once the answer is known.

For each reference transformation in the clean test split:

    rule-level label   the SMIRKS the mining route derives for that (substrate, product) pair,
                       looked up in the mined catalog. A transformation whose exact rule the
                       bank does not hold has no label at all, which is a support of zero: that
                       is what a rule-indexed selector sees.
    type-level label   the step-0 signature of that same SMIRKS at radius 0 with hydrogen
                       dropped from the identity of an edit, pooled over every bank rule
                       sharing it. This is the vocabulary the typed model would learn over, and
                       it is the one the 22.1% figure was computed from, so the stratum and the
                       claim are keyed alike.

A substrate enters the stratum when at least one of its references has rule-level support below
five and type-level support at or above five. It cannot enter on a reference whose type the
bank lacks: those need new chemistry, not a coarser label.

Shardable, and the merge checks that the shards tile the split.

    python scripts/typed_edit/build_h1_stratum.py --start 0 --end 195 --out .../s0.json
    python scripts/typed_edit/build_h1_stratum.py --merge 'results/h1shards/s*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdChemReactions, rdFMCS  # noqa: E402

from grail_metabolism.utils.preparation import load_default_rules  # noqa: E402
from scripts.mine_rules import (  # noqa: E402
    MCS_TIMEOUT_SECONDS, build_smirks, expand_center, find_reaction_center,
)
from scripts.run_benchmark import load_test_map  # noqa: E402
from type_curve import signature  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"
CURVE = ROOT / "results" / "typed_edit_type_curve.json"
STRATA = ROOT / "strata"
DENSE = 5          # the support at which a label stops being sparse, as in the type curve


def pair_to_smirks(sub_mol, prod_mol):
    """The mining route's SMIRKS for one (substrate, product) pair, minus the self-test gate.

    This is the same derivation `coverage_gap_types.pair_to_type` uses; it returns the SMIRKS
    rather than only its type, because the rule-level label is that string.
    """
    try:
        mcs = rdFMCS.FindMCS(
            [sub_mol, prod_mol], timeout=MCS_TIMEOUT_SECONDS, matchValences=False,
            ringMatchesRingOnly=True, completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
    except Exception:
        return None
    if mcs.canceled or mcs.numAtoms == 0:
        return None
    if mcs.numAtoms < 0.4 * min(sub_mol.GetNumAtoms(), prod_mol.GetNumAtoms()):
        return None
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return None
    sub_matches = sub_mol.GetSubstructMatches(mcs_mol, maxMatches=10)
    prod_matches = prod_mol.GetSubstructMatches(mcs_mol, maxMatches=10)
    if not sub_matches or not prod_matches:
        return None
    best, best_size = None, float("inf")
    for sm in sub_matches[:5]:
        for pm in prod_matches[:5]:
            cs, cp = find_reaction_center(sub_mol, prod_mol, sm, pm)
            size = len(cs) + len(cp)
            if 0 < size < best_size:
                best_size, best = size, (sm, pm, cs, cp)
    if best is None:
        return None
    sm, pm, cs, cp = best
    return build_smirks(sub_mol, prod_mol, sm, pm,
                        expand_center(sub_mol, cs, 1), expand_center(prod_mol, cp, 1))


def _sig(smirks):
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        rxn.Initialize()
    except Exception:
        return None
    return signature(rxn, 0, "no_explicit_H")


def type_support(counts):
    """type signature -> pooled training pairs, over the bank rules the catalog knows."""
    pooled = Counter()
    n_typed = 0
    for smirks in load_default_rules():
        if smirks not in counts:
            continue
        sig = _sig(smirks)
        if sig is None:
            continue
        pooled[sig] += counts[smirks]
        n_typed += 1
    return pooled, n_typed


def classify(items, counts, pooled, log_every=25):
    rows, stats = [], Counter()
    t0 = time.time()
    for i, (sub, prods) in enumerate(items, 1):
        if i % log_every == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time() - t0:.0f}s) in-stratum pairs "
                  f"{stats['in_stratum']}  substrates {stats['stratum_substrates']}",
                  file=sys.stderr, flush=True)
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        hit = False
        for met in prods:
            met_mol = Chem.MolFromSmiles(met)
            if met_mol is None:
                stats["unparseable_product"] += 1
                continue
            smirks = pair_to_smirks(sub_mol, met_mol)
            if smirks is None:
                stats["untypeable"] += 1
                rows.append({"substrate": sub, "metabolite": met, "smirks": None,
                             "rule_support": None, "type_support": None, "in_stratum": False})
                continue
            rule_sup = counts.get(smirks, 0)
            stats["rule_in_catalog"] += int(smirks in counts)
            sig = _sig(smirks)
            type_sup = pooled.get(sig, 0) if sig else 0
            inside = rule_sup < DENSE <= type_sup
            stats["in_stratum"] += int(inside)
            stats["sparse_at_rule"] += int(rule_sup < DENSE)
            stats["dense_at_type"] += int(type_sup >= DENSE)
            stats["pairs"] += 1
            hit = hit or inside
            rows.append({"substrate": sub, "metabolite": met, "smirks": smirks,
                         "rule_support": rule_sup, "type_support": type_sup,
                         "in_stratum": inside})
        stats["substrates"] += 1
        stats["stratum_substrates"] += int(hit)
    return rows, dict(stats)


def merge(pattern, out):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no shard matched", file=sys.stderr)
        return 1
    rows, stats, slices = [], Counter(), []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows += d["rows"]
        stats.update(d["stats"])
        slices.append(tuple(d["slice"]))
        print(f"  + {Path(p).name}: {d['slice']} pairs {len(d['rows'])}", file=sys.stderr)

    covered = set()
    for a, b in slices:
        covered |= set(range(a, b))
    n_all = stats["substrates"]
    gaps = sorted(set(range(max(max(s) for s in slices))) - covered)
    if gaps:
        print(f"FAIL: the shards do not tile the split; {len(gaps)} substrates are in none",
              file=sys.stderr)
        return 1

    in_sub = sorted({r["substrate"] for r in rows if r["in_stratum"]})
    all_sub = sorted({r["substrate"] for r in rows})
    STRATA.mkdir(exist_ok=True)
    (STRATA / "sparse_at_rule_dense_at_type.txt").write_text("\n".join(in_sub) + "\n")
    (STRATA / "sparse_at_rule_dense_at_type_complement.txt").write_text(
        "\n".join(s for s in all_sub if s not in set(in_sub)) + "\n")

    report = {
        "provenance": stamp(__file__),
        "definition": {
            "rule_label": "the mining route's SMIRKS for the pair, looked up in the mined "
                          "catalog; absent means a support of zero",
            "type_label": "step-0 signature at radius 0, no_explicit_H, pooled over bank rules",
            "dense_at": DENSE,
            "in_stratum": "rule support < 5 and type support >= 5",
        },
        "slices": [list(s) for s in sorted(slices)],
        "stats": dict(stats),
        "n_pairs": len(rows),
        "n_substrates": len(all_sub),
        "n_stratum_substrates": len(in_sub),
        "share_substrates": round(len(in_sub) / max(len(all_sub), 1), 4),
        "share_pairs": round(stats["in_stratum"] / max(stats["pairs"], 1), 4),
        "rows": rows,
    }
    Path(out).write_text(json.dumps(report, indent=1))
    short = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(short, indent=1))
    print(f"wrote {out} and {STRATA}/sparse_at_rule_dense_at_type.txt "
          f"({len(in_sub)} of {n_all} substrates)", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "h1_stratum.json"))
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out)

    counts = {k: int(v.get("count", 0)) for k, v in json.loads(CATALOG.read_text()).items()}
    pooled, n_typed = type_support(counts)

    # the vocabulary has to be the one the 22.1% was computed from, or the stratum and the
    # claim are keyed to different partitions
    curve = json.loads(CURVE.read_text())["by_variant"]["no_explicit_H"]
    dense_types = sum(1 for v in pooled.values() if v >= DENSE)
    pairs_in_dense = sum(v for v in pooled.values() if v >= DENSE) / max(sum(pooled.values()), 1)
    assert dense_types == curve["types_with_ge5_pairs"], (
        f"dense types {dense_types} != the curve's {curve['types_with_ge5_pairs']}")
    assert abs(pairs_in_dense - curve["train_pairs_in_dense_types"]) < 5e-4, (
        f"pairs in dense types {pairs_in_dense:.4f} != the curve's "
        f"{curve['train_pairs_in_dense_types']}")
    print(f"vocabulary: {len(pooled)} types with support, {dense_types} dense, "
          f"{pairs_in_dense:.3f} of pairs in them ({n_typed} bank rules typed)",
          file=sys.stderr, flush=True)

    items = list(load_test_map(None, 42).items())
    sl = items[args.start:(args.end or None)]
    print(f"substrates [{args.start}:{args.end or len(items)}] of {len(items)}",
          file=sys.stderr, flush=True)
    rows, stats = classify(sl, counts, pooled)
    Path(args.out).write_text(json.dumps(
        {"slice": [args.start, args.end or len(items)], "stats": stats, "rows": rows}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
