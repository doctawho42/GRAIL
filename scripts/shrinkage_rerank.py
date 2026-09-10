#!/usr/bin/env python3
"""Reinforcement shrinkage of the rule scores, tested against docs/SHRINKAGE_PREREGISTRATION.md.

Each per-rule score is pulled toward the frequency prior in proportion to the rule's training
support: score' = n/(n+lambda) * score + lambda/(n+lambda) * prior. The shrunk scores are then
combined by the deployed noisy-or, joined to the filter score from the frozen pools, and ranked by
the deployed order (dedup by key in product order, cap by generator, reciprocal rank fusion). At
lambda = 0 nothing changes, which is the gate: the arm has to reproduce the manuscript's exhaustive
column before any shrunk number is read.

n is the count of training positives per rule, from the label cache that reproduces the published
never/eq1/ge2 counts. prior is sigmoid(rule_prior_logits) from the deployed checkpoint. The
per-rule (rule_id, score) pools are collected once per population by running the generator, because
the details path collapses a multiply-reached candidate to its best rule and the aggregation shards
keep the score without the rule_id, and shrinkage needs both.

lambda is swept on the validation draw; the value maximising validation recall@15 is spent once on
the comparison set. The prediction, registered before this producer existed, is that the gain lands
on the five classes where the interactive arm already beats the exhaustive one.

    python scripts/shrinkage_rerank.py            # collect, sweep, evaluate, test the prediction
    python scripts/shrinkage_rerank.py --smoke    # 8 substrates, gate only
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RDLogger.DisableLog("rdApp.*")
import bank_without_selection as B                     # noqa: E402
from _rrf import RRF_K, competition_ranks              # noqa: E402
from error_by_chemistry import classify                # noqa: E402
from grail_metabolism.config import GeneratorConfig    # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402
from grail_metabolism.model.generator import _normalize_smiles_cached  # noqa: E402
from grail_metabolism.utils.preparation import safe_run_reactants      # noqa: E402

LABEL_CACHE = "artifacts/preprocessed/train/5a3f22a1e5962bbb/reaction_labels.expanded.pt"
GEN_CKPT = "artifacts/full5000_implicit/checkpoints/generator.pt"
WIDE = "results/widepools_implicit/w*.json"            # exhaustive arm, filter score + key per cand
BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
CAP = 100
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
G = {"sulfation", "hydrolysis", "deamination",
     "isomerisation, no formula change", "demethylation"}
PUBLISHED_COUNTS = {"never_positive": 4271, "pos_eq_1": 1520, "pos_ge2": 1790}
DEPLOYED_RECALL15 = 0.5353


def load_n() -> np.ndarray:
    d = torch.load(ROOT / LABEL_CACHE, map_location="cpu", weights_only=False)
    M = np.asarray([np.asarray(d[s]).ravel() for s in d], dtype=np.int64)
    n = M.sum(axis=0)
    got = {"never_positive": int((n == 0).sum()), "pos_eq_1": int((n == 1).sum()),
           "pos_ge2": int((n >= 2).sum())}
    if got != PUBLISHED_COUNTS:
        raise SystemExit(f"label cache does not reproduce the published counts: {got} vs "
                         f"{PUBLISHED_COUNTS}; the n vector would be the wrong bank")
    return n.astype(np.float64)


def collect(generator, subs, shard_glob):
    """Per-candidate list of (rule_id, rule_score), recovered from the frozen aggregation shards.

    The shards (aggregation_ablation.collect) hold, per candidate, the list of rule scores that
    reached it -- enumeration and the expensive tautomer normalisation already done -- but they
    dropped the rule_id, and shrinkage needs it to look up the rule's training support. It is
    recovered without re-enumerating anything: a single forward pass gives the score of every rule
    on the substrate, and within a substrate those scores are distinct (verified: zero collisions,
    zero misses over the sampled substrates), so each shard score maps back to exactly one rule.

    Re-running the enumeration instead would repeat the tautomer canonicalisation that is 94-99%
    of generation time, on the whole bank, for 584 substrates -- hours against seconds.
    """
    shard = {}
    for f in sorted(glob.glob(str(ROOT / shard_glob))):
        shard.update(json.loads(Path(f).read_text())["rows"])
    pools, t0 = {}, time.perf_counter()
    for i, s in enumerate(subs, 1):
        if i == 1 or i % 50 == 0 or i == len(subs):
            print(f"    map {i}/{len(subs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        _, scores, ranked = generator._prepare_generation(s, 7581, None)
        if scores is None:
            pools[s] = {}
            continue
        scores = np.asarray(scores, dtype=np.float64)
        by_val = {}
        for idx in ranked:
            by_val.setdefault(round(float(scores[idx]), 7), idx)
        per = {}
        for cand, sc_list in shard.get(s, {}).items():
            pairs = []
            for sc in sc_list:
                idx = by_val.get(round(float(sc), 7))
                if idx is not None:
                    pairs.append((idx, float(sc)))
            if pairs:
                per[cand] = pairs
        pools[s] = per
    return pools


def shrink_aggregate(rule_pairs, lam, n_vec, prior):
    """Shrink each rule score by its support, then combine by the deployed noisy-or."""
    if lam == 0.0:
        vals = [rs for _, rs in rule_pairs]
    else:
        vals = []
        for rid, rs in rule_pairs:
            n = n_vec[rid]
            vals.append((n / (n + lam)) * rs + (lam / (n + lam)) * prior[rid])
    clipped = np.clip(np.asarray(vals, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(1.0 - np.prod(1.0 - clipped))


def ranked_keys(rule_pool, side, self_key, lam, n_vec, prior, limit):
    """The deployed order on shrunk-aggregated generator scores joined to the frozen filter."""
    cands = []
    for smi, pairs in rule_pool.items():
        if smi not in side:
            continue
        g = shrink_aggregate(pairs, lam, n_vec, prior)
        f, key = side[smi]
        cands.append((smi, g, f, key))
    cands.sort(key=lambda c: -(c[2] * c[1]))          # dedup by key in product order
    seen, pool = set(), []
    for c in cands:
        if not c[3] or c[3] in seen:
            continue
        seen.add(c[3])
        pool.append(c)
    keep = sorted(pool, key=lambda c: -c[1])[:CAP]     # cap by (shrunk) generator score
    rf = competition_ranks(keep, lambda c: c[2])
    rg = competition_ranks(keep, lambda c: c[1])
    order = sorted(range(len(keep)),
                   key=lambda i: -(1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i])))
    return [keep[i][3] for i in order if keep[i][3] != self_key][:limit]


def recall_at(order, refs, k):
    return len(set(order[:k]) & refs) / len(refs) if refs else float("nan")


def load_wide(spec):
    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / spec))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs.update(blob["references"])
    return pools, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "shrinkage_rerank.json"))
    args = ap.parse_args()
    torch.set_num_threads(6)

    n_vec = load_n()
    st = torch.load(ROOT / GEN_CKPT, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**st["arch"]), st.get("rules"))
    gen.load_state_dict(st["state_dict"], strict=False)
    # Leave gen_normalization at the checkpoint default ('standardize'), which is what
    # aggregation_ablation.collect uses and what the frozen widepools were keyed under. Forcing
    # 'canonical' here re-normalised the candidates to keys the pools do not carry, the join
    # missed, and the gate read 0.388 instead of 0.5353. The gate is what caught it.
    prior = torch.sigmoid(gen.rule_prior_logits).detach().numpy().astype(np.float64)

    wide, refs_raw = load_wide(WIDE)
    subs = sorted(set(wide) & set(refs_raw))
    if args.smoke:
        subs = subs[:8]
    print(f"comparison: {len(subs)} substrates", flush=True)
    side = {s: {c["smiles"]: (c["filter"], c["key"]) for c in wide[s]} for s in subs}
    self_key = {s: B._key(s) for s in subs}
    kp = __import__("multiprocessing").Pool(6)
    refs = {s: set(B._dedup(refs_raw[s], None, kp)) for s in subs}
    kp.close()

    print("  mapping rule_id onto shard pools (comparison)", flush=True)
    cmp_pool = collect(gen, subs, "results/aggregation_shards/s*.json")

    def headline(pool_map, sides, selfk, refset, lam, subset=None):
        ss = subset or list(pool_map)
        U = sum(len(refset[s]) for s in ss)
        # Rank each substrate once to the widest budget, then read every budget off the prefix.
        orders = {s: ranked_keys(pool_map[s], sides[s], selfk[s], lam, n_vec, prior, max(BUDGETS))
                  for s in ss}
        out = {}
        for k in BUDGETS:
            out[k] = round(sum(len(set(orders[s][:k]) & refset[s]) for s in ss) / U, 4)
        return out

    # GATE: lambda=0 reproduces the deployed exhaustive column.
    g0 = headline(cmp_pool, side, self_key, refs, 0.0)
    gate_ok = abs(g0[15] - DEPLOYED_RECALL15) < 5e-4 if not args.smoke else True
    print(f"  gate lambda=0 recall@15 {g0[15]} vs deployed {DEPLOYED_RECALL15}: "
          f"{'OK' if gate_ok else 'FAIL'}", flush=True)
    if not gate_ok:
        raise SystemExit(f"lambda=0 does not reproduce the deployed column ({g0[15]} vs "
                         f"{DEPLOYED_RECALL15}); the pipeline is not the released one")
    if args.smoke:
        print("  smoke ok"); return 0

    # lambda selected on validation.
    valwide, vrefs_raw = load_wide("results/val_pools.json")
    vsubs = sorted(set(valwide) & set(vrefs_raw))
    vside = {s: {c["smiles"]: (c["filter"], c["key"]) for c in valwide[s]} for s in vsubs}
    vself = {s: B._key(s) for s in vsubs}
    kp = __import__("multiprocessing").Pool(6)
    vrefs = {s: set(B._dedup(vrefs_raw[s], None, kp)) for s in vsubs}
    kp.close()
    print(f"\n  validation: {len(vsubs)} substrates", flush=True)
    print("  mapping rule_id onto shard pools (validation)", flush=True)
    val_pool = collect(gen, vsubs, "results/aggregation_shards_val/s*.json")

    val_sweep = {}
    for lam in LAMBDAS:
        r = headline(val_pool, vside, vself, vrefs, lam)
        val_sweep[lam] = r
        print(f"    lambda {lam:>5}: val recall@15 {r[15]:.4f}", flush=True)
    lam_star = max(LAMBDAS, key=lambda L: val_sweep[L][15])
    print(f"  lambda* = {lam_star} (validation recall@15 {val_sweep[lam_star][15]})", flush=True)

    # A comparison-set sweep, for diagnosis and never for selection: it answers whether the
    # mechanism does anything at all, separately from whether the validation-selected lambda takes
    # it. If every lambda>0 is worse here too, the mechanism is dead.
    cmp_sweep = {lam: headline(cmp_pool, side, self_key, refs, lam)[15] for lam in LAMBDAS}
    lam_oracle = max(LAMBDAS, key=lambda L: cmp_sweep[L])
    print("\n  comparison sweep (diagnosis only, NOT used for selection):", flush=True)
    for lam in LAMBDAS:
        tag = "  <- comparison-best (forbidden for selection)" if lam == lam_oracle else ""
        print(f"    lambda {lam:>5}: cmp recall@15 {cmp_sweep[lam]:.4f}{tag}", flush=True)

    # spend once on comparison
    cmp_deployed = headline(cmp_pool, side, self_key, refs, 0.0)
    cmp_shrunk = headline(cmp_pool, side, self_key, refs, lam_star)
    print(f"\n  comparison headline recall@15: deployed {cmp_deployed[15]}, "
          f"shrunk {cmp_shrunk[15]}", flush=True)

    # Build the reference->class map once, from the reference structures, BEFORE per_class runs.
    ref_struct = json.loads((ROOT / "results/test_references.json").read_text())
    ref_class = {}
    for s in subs:
        smol = Chem.MolFromSmiles(s)
        if smol is None:
            continue
        for met in ref_struct.get(s, []):
            mmol = Chem.MolFromSmiles(met)
            if mmol is None:
                continue
            cls, _ = classify(smol, mmol)
            ref_class[(s, B._key(met))] = cls

    # per-class delta at k=15 under lambda*, using the same classifier
    def per_class(lam):
        orders = {s: ranked_keys(cmp_pool[s], side[s], self_key[s], lam, n_vec, prior, max(BUDGETS))
                  for s in subs}
        # classify each reference by its formula delta against the substrate
        hit, tot = Counter(), Counter()
        for s in subs:
            got = set(orders[s][:15])
            for rk in refs[s]:
                cls = ref_class.get((s, rk))
                if cls is None:
                    continue
                tot[cls] += 1
                if rk in got:
                    hit[cls] += 1
        return {c: hit[c] / tot[c] for c in tot if tot[c]}, dict(tot)

    dep_cls, tot = per_class(0.0)
    shr_cls, _ = per_class(lam_star)
    orc_cls, _ = per_class(lam_oracle)
    classes = sorted(tot, key=lambda c: -tot[c])
    deltas = {c: round(shr_cls.get(c, 0) - dep_cls.get(c, 0), 4) for c in classes}
    gains_G = [deltas[c] for c in classes if c in G]
    gains_notG = [deltas[c] for c in classes if c not in G]
    mean_G = float(np.mean(gains_G)) if gains_G else 0.0
    mean_notG = float(np.mean(gains_notG)) if gains_notG else 0.0

    orc_deltas = {c: round(orc_cls.get(c, 0) - dep_cls.get(c, 0), 4) for c in classes}
    orc_G = float(np.mean([orc_deltas[c] for c in classes if c in G])) if any(c in G for c in classes) else 0.0
    orc_notG = float(np.mean([orc_deltas[c] for c in classes if c not in G])) if any(c not in G for c in classes) else 0.0

    rep = {"config": {**B._code_version(), "lambda_grid": LAMBDAS, "lambda_star": lam_star,
                      "lambda_oracle_comparison": lam_oracle,
                      "cap": CAP, "n_source": LABEL_CACHE, "gate_recall15_lambda0": g0[15]},
           "prereg": "docs/SHRINKAGE_PREREGISTRATION.md",
           "group_G": sorted(G),
           "headline": {"deployed": cmp_deployed, "shrunk": cmp_shrunk},
           "validation_sweep": {str(L): val_sweep[L][15] for L in LAMBDAS},
           "per_class_delta_recall15": {c: {"references": tot[c], "deployed": round(dep_cls.get(c,0),4),
                                            "shrunk": round(shr_cls.get(c,0),4), "delta": deltas[c],
                                            "in_G": c in G} for c in classes},
           "primary_test": {"mean_gain_G": round(mean_G,4), "mean_gain_notG": round(mean_notG,4),
                            "prediction_holds": bool(mean_G > mean_notG)},
           "guardrail": {"deployed_recall15": cmp_deployed[15], "shrunk_recall15": cmp_shrunk[15],
                         "not_below_deployed": bool(cmp_shrunk[15] >= DEPLOYED_RECALL15 - 1e-9)},
           "comparison_sweep_diagnosis": {str(L): round(cmp_sweep[L], 4) for L in LAMBDAS},
           "oracle_reading": {"lambda": lam_oracle, "mean_gain_G": round(orc_G, 4),
                              "mean_gain_notG": round(orc_notG, 4),
                              "note": "comparison-best lambda; forbidden for selection, reported to show whether the mechanism could land on G at all",
                              "mechanism_is_dead": bool(lam_oracle == 0.0)}}

    print(f"\n  PRIMARY: mean gain on G {mean_G:+.4f} vs not-G {mean_notG:+.4f} -> "
          f"prediction {'HOLDS' if mean_G>mean_notG else 'REJECTED'}")
    print(f"  GUARDRAIL: shrunk recall@15 {cmp_shrunk[15]} vs deployed {cmp_deployed[15]} -> "
          f"{'ok' if rep['guardrail']['not_below_deployed'] else 'FAILED'}")
    print(f"\n  {'class':<40}{'refs':>5}{'dep':>8}{'shrunk':>8}{'delta':>8}{'G':>3}")
    for c in classes:
        e = rep["per_class_delta_recall15"][c]
        print(f"  {c:<40}{e['references']:>5}{e['deployed']:>8.3f}{e['shrunk']:>8.3f}"
              f"{e['delta']:>+8.3f}{'*' if e['in_G'] else '':>3}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
