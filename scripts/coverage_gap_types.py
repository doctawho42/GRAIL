#!/usr/bin/env python3
"""Decompose the rule-bank coverage gap into 'known-type' vs 'novel-type' transformations.

The bank covers ~0.735 of test transformations (§6); the remaining ~26.5% is the coverage gap that
binds recall (§10). This script asks *why* each uncovered transformation is missed, to decide which
D1 lever can close it:

  known-type  : the transformation's radius-0 reaction TYPE (element-typed changed-bond multiset,
                `reaction_types.canonical_type`) IS already present in the bank -> the bank has a
                template of this chemistry, just not one specific enough / general enough to match
                this substrate. Fixable by TEMPLATE GENERALIZATION (in-repo, no new data).
  novel-type  : the type is ABSENT from the bank -> genuinely new chemistry. Needs NEW REACTION
                SOURCES (external corpora, out of current scope).
  untypeable  : MCS/reaction-center derivation failed (large/awkward pair) -> uncounted either way.

For each test substrate we apply the full bank (the §6 ceiling computation) to find which true
metabolites are NOT recovered (tautomer-InChIKey), then type each uncovered pair via the mining
MCS route (mine_rules) MINUS the self-test gate. If known-type dominates the gap, D1's path-a
(generalization) is viable; if novel-type dominates, D1 needs path-b (new data).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import rdFMCS

from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.reaction_types import canonical_type
from grail_metabolism.utils.preparation import apply_rules_to_molecule, load_default_rules
from scripts.mine_rules import MCS_TIMEOUT_SECONDS, build_smirks, expand_center, find_reaction_center
from scripts.run_benchmark import load_test_map


def pair_to_type(sub_mol, prod_mol):
    """Radius-0 reaction type of a (substrate, product) pair via the mining MCS route, no self-test."""
    try:
        mcs = rdFMCS.FindMCS(
            [sub_mol, prod_mol], timeout=MCS_TIMEOUT_SECONDS, matchValences=False,
            ringMatchesRingOnly=True, completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareAny, atomCompare=rdFMCS.AtomCompare.CompareElements,
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
    smirks = build_smirks(sub_mol, prod_mol, sm, pm, expand_center(sub_mol, cs, 1), expand_center(prod_mol, cp, 1))
    if smirks is None:
        return None
    return canonical_type(smirks)


def _build_report(cov, gap, n_substrates, n_rules, n_bank_types):
    n_unc = cov["uncovered"]
    typed = gap["known_type"] + gap["novel_type"]
    denom = cov["covered"] + n_unc
    return {
        "n_substrates": n_substrates,
        "n_rules": n_rules,
        "n_bank_types": n_bank_types,
        "covered_pairs": cov["covered"],
        "uncovered_pairs": n_unc,
        "coverage": round(cov["covered"] / denom, 4) if denom else 0.0,
        "gap": dict(gap),
        "gap_known_type_frac_of_uncovered": round(gap["known_type"] / n_unc, 4) if n_unc else 0.0,
        "gap_novel_type_frac_of_uncovered": round(gap["novel_type"] / n_unc, 4) if n_unc else 0.0,
        "gap_known_type_frac_of_typed": round(gap["known_type"] / typed, 4) if typed else 0.0,
    }


def _print_report(report):
    n_unc = report["uncovered_pairs"]
    gap = report["gap"]
    print("\n=== coverage-gap type decomposition ===", flush=True)
    print(f"  coverage {report['coverage']}  ({report['covered_pairs']} covered / {n_unc} uncovered)", flush=True)
    print(f"  of {n_unc} uncovered: known-type {gap.get('known_type',0)} ({report['gap_known_type_frac_of_uncovered']}), "
          f"novel-type {gap.get('novel_type',0)} ({report['gap_novel_type_frac_of_uncovered']}), "
          f"untypeable {gap.get('untypeable',0)}", flush=True)
    print(f"  of TYPED uncovered: known-type frac {report['gap_known_type_frac_of_typed']} "
          f"-> {'generalization viable (path a)' if report['gap_known_type_frac_of_typed'] >= 0.5 else 'skews to new sources (path b)'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = full test set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0, help="shard slice start into the test map")
    ap.add_argument("--end", type=int, default=0, help="shard slice end (0 = to the end)")
    ap.add_argument("--counts-out", default="", help="dump raw {cov,gap} counters for this shard slice and skip the report")
    ap.add_argument("--merge", default="", help="glob of shard counter files to sum into the headline")
    ap.add_argument("--out", default=str(ROOT / "results" / "coverage_gap_types.json"))
    args = ap.parse_args()

    if args.merge:
        paths = sorted({p for p in glob.glob(args.merge)})
        if not paths:
            print("no shard files matched", flush=True)
            return 1
        cov, gap = Counter(), Counter()
        n_rules = n_bank_types = n_subs = 0
        for p in paths:
            blob = json.loads(Path(p).read_text())
            cov.update(blob["cov"])
            gap.update(blob["gap"])
            n_rules = blob.get("n_rules", n_rules)
            n_bank_types = blob.get("n_bank_types", n_bank_types)
            n_subs += blob.get("n_substrates", 0)
            print(f"  + {Path(p).name}: {blob.get('n_substrates')} subs", flush=True)
        report = _build_report(cov, gap, n_subs, n_rules, n_bank_types)
        Path(args.out).write_text(json.dumps(report, indent=2))
        _print_report(report)
        print(f"Wrote {args.out}", flush=True)
        return 0

    rules = load_default_rules()
    bank_types = {t for t in (canonical_type(r) for r in rules) if t is not None}
    print(f"bank: {len(rules)} rules, {len(bank_types)} distinct radius-0 types", flush=True)

    test_map = load_test_map(args.sample or None, args.seed)
    all_items = list(test_map.items())
    items = all_items[args.start : (args.end or None)] if (args.start or args.end) else all_items
    print(f"test substrates: {len(items)} [{args.start}:{args.end or len(all_items)}]", flush=True)

    cov = Counter()          # covered vs uncovered true pairs
    gap = Counter()          # among uncovered: known_type / novel_type / untypeable
    t0 = time.time()
    for i, (sub, true_prods) in enumerate(items, 1):
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)  covered={cov['covered']} "
                  f"uncovered={cov['uncovered']}  known={gap['known_type']} novel={gap['novel_type']} untypeable={gap['untypeable']}", flush=True)
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        products = apply_rules_to_molecule(sub_mol, rules, normalization_mode="canonical")
        covered_keys = set()
        for p in products:
            try:
                covered_keys.add(_tautomer_inchikey(p))
            except Exception:
                continue
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
            if met_mol is None:
                gap["untypeable"] += 1
                continue
            t = pair_to_type(sub_mol, met_mol)
            if t is None:
                gap["untypeable"] += 1
            elif t in bank_types:
                gap["known_type"] += 1
            else:
                gap["novel_type"] += 1

    if args.counts_out:
        Path(args.counts_out).write_text(json.dumps(
            {"n_substrates": len(items), "n_rules": len(rules), "n_bank_types": len(bank_types),
             "cov": dict(cov), "gap": dict(gap)}))
        print(f"\nShard done: {len(items)} subs -> {args.counts_out}", flush=True)
        return 0

    report = _build_report(cov, gap, len(items), len(rules), len(bank_types))
    Path(args.out).write_text(json.dumps(report, indent=2))
    _print_report(report)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
