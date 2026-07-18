#!/usr/bin/env python3
"""Rule-bank collapse: how much of the 7,581-rule bank is functionally redundant?

The premise under test (measure BEFORE any 'mining at the center' thesis): P2 says the learned
selector loses to a frequency prior because the label is 7,581-dim and near-zero per substrate --
extreme multi-label. But if the bank is a union of 4 source banks plus mining, a chunk of those
7,581 are plausibly FUNCTIONAL DUPLICATES (e.g. SyGMa hydroxylation encoded as several SMIRKS
strings). Collapsing duplicates is a *reachability-preserving coarsening* of the label space --
it lowers the effective dimensionality that degenerates selection, with ZERO coverage loss by
construction (unlike type-gating, which grouped DIFFERENT-product rules and cost reachability).

We do NOT assume; we measure the collapse:

  signature(rule) = frozenset{ (substrate_i, tautomer_inchikey(product)) } over a probe pool.
  Two rules are EMPIRICALLY EQUIVALENT on the pool iff identical signatures (same substrates,
  same products). Distinct signatures among FIRED rules = effective rule count (a LOWER bound on
  redundancy -- conservative, never overstates the collapse).

Both outcomes are informative and pre-registered:
  - small collapse (7581 -> ~7000): P2 is real sparsity; merging is NOT the lever.
  - large collapse (7581 -> ~2500): half the 'extreme multi-label' is redundant-encoding artifact;
    merging IS a selection lever.

Three traps (the reviewer's, applied to the reviewer's own idea) are instrumented, not deferred:
  T1  equivalence is substrate-dependent. We split the pool A/B and report the fraction of
      pool-A equivalence PAIRS that persist on pool-B (merge stability). Low => pool artifact.
  T2  redundancy secretly encodes a prior: a product reachable by k duplicates gets k x mass in
      noisy-or (multiplicity ~= frequency), and the frequency prior is exactly what beats the
      selector. We report cluster-size (multiplicity) distribution -- the signal that must be
      re-injected as an explicit per-rule frequency weight if we merge.
  T3  (generalization -> candidate explosion) is not tested here; it belongs to the coverage lever.

Also stratifies the collapse by signature THICKNESS (fires on >=1,2,3,5,10 substrates): collapse
among thick-signature rules is real redundancy; collapse concentrated in thin signatures is a
thin-evidence artifact.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.utils.preparation import apply_rules_to_molecule, load_default_rules

RDLogger.DisableLog("rdApp.*")

SDF = ROOT / "grail_metabolism" / "data" / "test.sdf"
CACHE = ROOT / "results" / "rule_collapse_cache.json"
OUT = ROOT / "results" / "rule_collapse.json"


def build_pool(n: int, seed: int, min_heavy: int = 10, max_heavy: int = 60):
    """Seeded-random drug-like probe molecules from the test SDF (dedup by canonical SMILES)."""
    sup = Chem.SDMolSupplier(str(SDF))
    uniq = {}
    for m in sup:
        if m is None:
            continue
        hv = m.GetNumHeavyAtoms()
        if hv < min_heavy or hv > max_heavy:
            continue
        s = Chem.MolToSmiles(m)
        if s not in uniq:
            uniq[s] = True
    smis = sorted(uniq)
    rng = random.Random(seed)
    rng.shuffle(smis)
    return smis[:n]


def compute_signatures(pool, rules):
    """rule_idx -> set of (substrate_i, pid); pid ids the CANONICAL PRODUCT SMILES.

    apply_rules_to_molecule already returns canonical-SMILES product keys, so functional-duplicate
    rules (same transformation, same product atoms) land on the same key with no extra cost. This
    is the conservative product key: tautomer-merging two DIFFERENT tautomers of one product could
    only *raise* the collapse, never lower it -- so this is a lower bound on redundancy.
    """
    pk_to_id: dict[str, int] = {}
    sig: dict[int, set] = defaultdict(set)
    t0 = time.time()
    for i, smi in enumerate(pool):
        if i % 20 == 0:
            print(f"  probe {i}/{len(pool)} ({time.time()-t0:.0f}s, {len(pk_to_id)} distinct products)", flush=True)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        prods = apply_rules_to_molecule(mol, rules, "canonical")  # {canonical_product_smiles: {rule_idx}}
        for prod_smi, rule_idxs in prods.items():
            pid = pk_to_id.get(prod_smi)
            if pid is None:
                pid = len(pk_to_id)
                pk_to_id[prod_smi] = pid
            for r in rule_idxs:
                sig[r].add((i, pid))
    return sig, pk_to_id


def cluster(sig_items):
    """sig_items: iterable of (rule_idx, frozenset). -> dict signature_key -> [rule_idx...]."""
    groups: dict = defaultdict(list)
    for r, s in sig_items:
        groups[s].append(r)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700, help="probe pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true", help="recompute stats from cache")
    args = ap.parse_args()

    rules = load_default_rules()
    n_total = len(rules)
    print(f"rules: {n_total}", flush=True)

    if args.analyze_only and CACHE.exists():
        blob = json.loads(CACHE.read_text())
        pool = blob["pool"]
        sig = {int(r): set(tuple(p) for p in pairs) for r, pairs in blob["sig"].items()}
        print(f"loaded cache: pool={len(pool)} fired_rules={len(sig)}", flush=True)
    else:
        pool = build_pool(args.n, args.seed)
        print(f"probe pool: {len(pool)} substrates (seed={args.seed})", flush=True)
        sig, _pk = compute_signatures(pool, rules)
        CACHE.write_text(json.dumps({
            "pool": pool,
            "sig": {str(r): sorted(list(s)) for r, s in sig.items()},
        }))
        print(f"cached -> {CACHE}", flush=True)

    npool = len(pool)
    # frozen signatures for fired rules (nonempty)
    fired = {r: frozenset(s) for r, s in sig.items() if s}
    n_fired = len(fired)
    n_unfired = n_total - n_fired

    # ---- primary collapse (all fired) ----
    groups = cluster(fired.items())
    n_distinct = len(groups)
    # effective bank: distinct fired signatures + unfired (each unassessed rule stays its own)
    effective_bank = n_distinct + n_unfired

    # ---- thickness stratification: fires on >=t distinct substrates ----
    def n_subs(fs):
        return len({i for (i, _pid) in fs})
    strata = {}
    for t in (1, 2, 3, 5, 10):
        sub = {r: fs for r, fs in fired.items() if n_subs(fs) >= t}
        g = cluster(sub.items())
        strata[f">={t}"] = {"rules": len(sub), "distinct": len(g),
                            "collapse_ratio": round(len(g) / len(sub), 4) if sub else None}

    # ---- T2: multiplicity (cluster size) distribution = the implicit frequency prior ----
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    size_hist = Counter(sizes)
    multiplicity = {
        "clusters_total": n_distinct,
        "singletons": size_hist.get(1, 0),
        "clusters_ge2": sum(1 for s in sizes if s >= 2),
        "max_cluster_size": sizes[0] if sizes else 0,
        "rules_in_ge2_clusters": sum(s for s in sizes if s >= 2),
        "size_histogram_top": dict(sorted(size_hist.items(), key=lambda kv: -kv[0])[:15]),
    }

    # ---- T1: A/B held-out merge stability ----
    # even-index substrates -> A, odd -> B (interleave avoids ordering bias)
    A = set(range(0, npool, 2))
    B = set(range(1, npool, 2))
    def restrict(fs, half):
        return frozenset((i, p) for (i, p) in fs if i in half)
    # rules firing in BOTH halves
    both = {r: fs for r, fs in fired.items() if restrict(fs, A) and restrict(fs, B)}
    sigA = {r: restrict(fs, A) for r, fs in both.items()}
    sigB = {r: restrict(fs, B) for r, fs in both.items()}
    classesA = cluster(sigA.items())
    # of all A-equivalent unordered pairs, fraction also B-equivalent
    pairs_A = pairs_AB = 0
    for members in classesA.values():
        if len(members) < 2:
            continue
        # cap huge classes to keep pair-count tractable while unbiased (sample within class)
        m = members if len(members) <= 200 else random.Random(0).sample(members, 200)
        for a, b in combinations(m, 2):
            pairs_A += 1
            if sigB[a] == sigB[b]:
                pairs_AB += 1
    t1 = {
        "rules_firing_both_halves": len(both),
        "A_equivalent_pairs": pairs_A,
        "A_pairs_preserved_on_B": pairs_AB,
        "merge_stability": round(pairs_AB / pairs_A, 4) if pairs_A else None,
    }

    # ---- example clusters (largest), with SMIRKS, to see if they're genuine duplicates ----
    examples = []
    for members in sorted(groups.values(), key=len, reverse=True)[:8]:
        if len(members) < 2:
            continue
        examples.append({
            "size": len(members),
            "n_substrates": n_subs(fired[members[0]]),
            "rule_indices": members[:12],
            "smirks_sample": [rules[m] for m in members[:4]],
        })

    report = {
        "n_rules_total": n_total,
        "pool_size": npool,
        "seed": args.seed,
        "product_key": "canonical_smiles (conservative; tautomer-merge can only raise collapse)",
        "equivalence": "identical (substrate,product) signature on probe pool (conservative lower bound)",
        "n_fired": n_fired,
        "n_unfired_on_pool": n_unfired,
        "n_distinct_signatures_fired": n_distinct,
        "effective_bank_conservative": effective_bank,
        "collapse_among_fired": round(n_distinct / n_fired, 4) if n_fired else None,
        "thickness_strata": strata,
        "multiplicity_T2": multiplicity,
        "merge_stability_T1": t1,
        "example_clusters": examples,
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("\n=== RULE-BANK COLLAPSE (probe pool n=%d, canonical-product-SMILES key) ===" % npool, flush=True)
    print(f"total rules            : {n_total}", flush=True)
    print(f"fired on pool          : {n_fired}   (unfired/unassessed: {n_unfired})", flush=True)
    print(f"distinct signatures    : {n_distinct}   among fired", flush=True)
    print(f"collapse among fired   : {n_distinct}/{n_fired} = {report['collapse_among_fired']}", flush=True)
    print(f"effective bank (cons.) : {effective_bank}   (= distinct fired + unfired)", flush=True)
    print("thickness strata (fires on >=t substrates): distinct/rules", flush=True)
    for k, v in strata.items():
        print(f"  {k:>4}: {v['distinct']}/{v['rules']} = {v['collapse_ratio']}", flush=True)
    print(f"T2 multiplicity: {multiplicity['clusters_ge2']} clusters>=2, "
          f"max size {multiplicity['max_cluster_size']}, "
          f"{multiplicity['rules_in_ge2_clusters']} rules in >=2-clusters", flush=True)
    print(f"T1 merge stability (A-pairs preserved on B): {t1['merge_stability']} "
          f"({t1['A_pairs_preserved_on_B']}/{t1['A_equivalent_pairs']})", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
