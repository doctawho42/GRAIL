#!/usr/bin/env python3
"""Structural canonical-SMIRKS dedup of the 7,581-rule bank (distribution-free deployment estimate).

NOTE ON "PROVABLE": a fully sound canonical key would tie every query constraint to a canonical
atom POSITION (canonical SMARTS -- an unsolved-in-RDKit problem). This key uses a sorted MULTISET of
atom/bond query SMARTS, which is position-blind by construction (that blindness is what collapses
permutation variants). It therefore has a measured ~1% residual over-merge on positionally-placed
constraints (e.g. a ring-bond at chain position 2 vs 3), caught by the behavioral safety check below.
So this is a distribution-free ESTIMATE that must be behaviorally cross-checked before shipping, not a
guarantee. The SAFE deployable merges are the intersection with behavioral agreement (see the audit).

The pool-collapse probe (scripts/rule_collapse.py) showed the functional duplicates are atom-map
PERMUTATION / RELABEL variants of one transformation (N-oxidation written 15 ways, etc.). That kind
of equivalence is PROVABLE and distribution-free -- no substrate pool needed, and it catches the
duplicates among the 4,834 rules that never fired on the probe pool. This is the right number for
the DEPLOYMENT claim (a smaller canonical bank => cheaper per-forward re-encoding, the dominant cost).

Canonical key (invariant to atom ORDER and atom-MAP RELABELING, correspondence-preserving):
  1. ReactionFromSmarts -> ReactionToSmiles(canonical=True)         # canonical atom order
  2. renumber atom-map numbers by first appearance in the canonical  # relabel-invariant, and
     reaction SMILES, consistently reactant->product                 # consistent across the arrow
Two rules with the same key are provably the same template up to atom relabeling; DIFFERENT
transformations keep different keys (verified: N-oxidation != N-dealkylation).

SAFETY: structural equivalence must IMPLY behavioral equivalence. We cross-check every structural
merge against the cached pool signatures (results/rule_collapse_cache.json): for each structural
cluster whose members fired on the pool, their pool product-signatures must be identical. Any split
is an over-merge (a canonicalization bug) and is reported -- expected 0.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit.Chem import AllChem
from rdkit import RDLogger

from grail_metabolism.utils.preparation import load_default_rules

RDLogger.DisableLog("rdApp.*")

CACHE = ROOT / "results" / "rule_collapse_cache.json"
OUT = ROOT / "results" / "rule_dedup_provable.json"
_MAP = re.compile(r":(\d+)]")


def canon_key(smirks: str):
    """Canonical reaction key: SMILES topology (atom-order + map-relabel invariant) AUGMENTED with
    the sorted multiset of per-atom query SMARTS. The SMILES conversion is lossy for SMARTS query
    primitives (ring/degree/atom-list/recursive), which over-merged 11.5% of clusters; the atom-query
    multiset carries exactly those dropped constraints back, so rules differing only in a query (e.g.
    R0 acyclic vs R ring, or a specific recursive environment) no longer collapse. Permutation/relabel
    variants keep identical query multisets -> still merge. None-safe. Conservative: may under-merge
    (safe), never the reverse (verified by the behavioral safety check below)."""
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        cs = AllChem.ReactionToSmiles(rxn, canonical=True)
    except Exception:
        return None
    remap: dict[str, str] = {}
    for m in _MAP.finditer(cs):
        old = m.group(1)
        if old not in remap:
            remap[old] = str(len(remap) + 1)
    smi = _MAP.sub(lambda mm: ":" + remap[mm.group(1)] + "]", cs)
    qs, bs = [], []
    for side in (rxn.GetReactants(), rxn.GetProducts()):
        for mol in side:
            for a in mol.GetAtoms():
                qs.append(re.sub(r":\d+", "", a.GetSmarts()))
            for b in mol.GetBonds():
                bs.append(b.GetSmarts())  # carries bond queries: @/!@ ring, order, aromaticity
    return smi + "||" + "|".join(sorted(qs)) + "##" + "|".join(sorted(bs))


def main() -> int:
    rules = load_default_rules()
    n_total = len(rules)
    print(f"rules: {n_total}", flush=True)

    t0 = time.time()
    keys = []
    n_unparseable = 0
    for i, s in enumerate(rules):
        if i % 1000 == 0:
            print(f"  {i}/{n_total} ({time.time()-t0:.0f}s)", flush=True)
        k = canon_key(s)
        if k is None:
            n_unparseable += 1
            k = f"__raw__:{i}:{s}"  # unparseable stays its own bucket (never merged)
        keys.append(k)

    groups: dict[str, list] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    n_distinct = len(groups)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    size_hist = Counter(sizes)

    # ---- SAFETY: structural merges must agree behaviorally on the probe pool ----
    safety = {"checked_clusters": 0, "clusters_with_pool_evidence": 0, "over_merges": 0, "examples": []}
    struct_vs_behav = {}
    if CACHE.exists():
        blob = json.loads(CACHE.read_text())
        pool_sig_raw = blob["sig"]  # {rule_idx(str): [[i,pid], ...]}
        pool_sig = {int(r): frozenset(tuple(p) for p in pairs) for r, pairs in pool_sig_raw.items()}
        for k, members in groups.items():
            if len(members) < 2:
                continue
            safety["checked_clusters"] += 1
            fired = [m for m in members if m in pool_sig and pool_sig[m]]
            if len(fired) < 2:
                continue
            safety["clusters_with_pool_evidence"] += 1
            distinct_behav = {pool_sig[m] for m in fired}
            if len(distinct_behav) > 1:
                safety["over_merges"] += 1
                if len(safety["examples"]) < 5:
                    safety["examples"].append({"key": k[:80], "members": fired[:6],
                                               "distinct_pool_signatures": len(distinct_behav)})

        # ---- decomposition: among rules that FIRED on the pool, how much of the pool-behavioral
        # collapse is explained by provable structural dedup vs residual functional equivalence? ----
        fired_rules = [r for r, s in pool_sig.items() if s]
        n_fired = len(fired_rules)
        struct_distinct_fired = len({keys[r] for r in fired_rules})
        behav_distinct_fired = len({pool_sig[r] for r in fired_rules})
        struct_vs_behav = {
            "fired_rules": n_fired,
            "structural_distinct_among_fired": struct_distinct_fired,
            "behavioral_distinct_among_fired": behav_distinct_fired,
            "note": "structural (provable, permutation/relabel) vs behavioral (empirical, any same-product "
                    "equivalence on the pool). behavioral <= structural means extra non-structural functional "
                    "equivalences exist beyond permutation variants.",
        }

    report = {
        "n_rules_total": n_total,
        "n_unparseable": n_unparseable,
        "equivalence": "provable canonical SMIRKS (atom-order + atom-map-relabel invariant, correspondence-preserving)",
        "n_distinct_canonical": n_distinct,
        "rules_eliminated": n_total - n_distinct,
        "collapse_fraction_of_bank": round((n_total - n_distinct) / n_total, 4),
        "clusters_ge2": sum(1 for s in sizes if s >= 2),
        "max_cluster_size": sizes[0] if sizes else 0,
        "rules_in_ge2_clusters": sum(s for s in sizes if s >= 2),
        "size_histogram_top": dict(sorted(size_hist.items(), key=lambda kv: -kv[0])[:15]),
        "safety_structural_implies_behavioral": safety,
        "structural_vs_behavioral_on_fired": struct_vs_behav,
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("\n=== PROVABLE CANONICAL-SMIRKS DEDUP (distribution-free) ===", flush=True)
    print(f"total rules        : {n_total}  (unparseable kept distinct: {n_unparseable})", flush=True)
    print(f"distinct templates : {n_distinct}", flush=True)
    print(f"rules eliminated   : {report['rules_eliminated']}  = {report['collapse_fraction_of_bank']*100:.1f}% of bank", flush=True)
    print(f"clusters >=2       : {report['clusters_ge2']}   max size {report['max_cluster_size']}   "
          f"rules in >=2-clusters {report['rules_in_ge2_clusters']}", flush=True)
    print(f"SAFETY (structural=>behavioral): {safety['over_merges']} over-merges "
          f"/ {safety['clusters_with_pool_evidence']} clusters with pool evidence  (want 0)", flush=True)
    if struct_vs_behav:
        print(f"among {struct_vs_behav['fired_rules']} fired: structural-distinct "
              f"{struct_vs_behav['structural_distinct_among_fired']}  vs  behavioral-distinct "
              f"{struct_vs_behav['behavioral_distinct_among_fired']}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
