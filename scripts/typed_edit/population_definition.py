#!/usr/bin/env python3
"""What actually defines the comparison population, and what the rest of the test set says.

The manuscript states that a substrate enters the comparison only if every method emitted
something for it, and names the cost: that rule would remove the cases where a comparator produced
nothing and a bank this size should win most clearly. The rule is not what the code does. The
population is the intersection of the substrates each method has an entry for, and the binding
constraint is one submission list: the comparator that is a web service was queried for 291
substrates and returned all 291. Nothing was dropped for emitting nothing, and the population
still contains substrates where a comparator's list is empty.

That correction changes what has to be measured. Three things are settled here.

First, the emission rule the manuscript describes, applied to the data: how many substrates each
comparator returns nothing for, inside the comparison set and over the whole evaluated test set.

Second, whether the 291 differ from the 879 they were drawn from. How the draw was made is not
recorded anywhere in this repository, which is the same defect the corpus assembly carries, so the
draw is not defended by its description but tested: the two halves are compared on the count of
annotated references, the size of the substrate, and the number of candidates the deployed system
emits, each by a permutation test.

Third, the comparison itself on the whole test set, for the three arms that have it. The deployed
system and two of the comparators ran over all 1,170 evaluated substrates, so their contrast can
be read on the population nobody selected, with a substrate a comparator answered nothing for
scored as zero rather than removed. The exhaustive arm exists only on the 291 and is absent here,
which is stated rather than worked around.

    python scripts/typed_edit/population_definition.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

KS = (5, 10, 15, 30, 50)
N_BOOT, SEED = 10000, 0
N_PERM = 10000
CAP = 100

TRUTH = ROOT / "results/test_references.json"
METATOX = ROOT / "results/metatox_smirks_preds.json"
# The deployed interactive arm, from the pools the comparison itself is read from. An earlier
# version of this script took it from results/scored_predictions.json, which is a narrower dump
# ranked by the product of the two component scores; that arm reaches half what the deployed one
# does and would have made every contrast here meaningless.
DEPLOYED_COMPARISON = ROOT / "results/widepools_k30/all.json"
DEPLOYED_FULLTEST = ROOT / "results/widepools_k30_fulltest"
# The exhaustive arm on the whole evaluated test set. It existed only on the comparison set, so
# the one ordering result this work still claims had never been read off the population nobody
# selected; these pools are that population.
EXHAUSTIVE_COMPARISON = ROOT / "results/widepools_implicit"
EXHAUSTIVE_FULLTEST = ROOT / "results/widepools_fulltest"
# The comparators that ran over the whole evaluated test set. BioTransformer joined them when it
# was run there: it is a jar this repository holds, and what had kept it on the comparison set
# was the cost of the run rather than an impossibility. MetaTox is the one that cannot follow,
# because a second submission to a web service is not ours to make.
WHOLE_TEST = {"sygma": ROOT / "results/sygma_fulltest_predictions.json",
              "metapredictor": ROOT / "artifacts/tier2_1170/metapredictor_preds.json",
              "biotransformer": ROOT / "results/biotransformer_fulltest_preds.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "population_definition.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey

    # Every file this run reads, collected where it is opened. A stamp says which code wrote the
    # artifact; it cannot say which pools the code was pointed at, and a pool that has since been
    # rebuilt, renamed or planted would leave the numbers here reading as current. Recorded so the
    # artifact can be checked against the files that produced it.
    read_paths = [TRUTH, METATOX, DEPLOYED_COMPARISON]
    truth = json.loads(TRUTH.read_text())
    metatox = json.loads(METATOX.read_text())["predictions"]
    deployed_rows = dict(json.loads(DEPLOYED_COMPARISON.read_text())["pools"])
    # The whole-test pools were built later and cover the comparison set too. Two builds of the
    # same configuration should agree candidate for candidate and score for score; where they do
    # not, one of them was scored by a different model and neither population can be trusted.
    disagree = 0
    for f in sorted(glob.glob(str(DEPLOYED_FULLTEST / "w*.json"))):
        read_paths.append(Path(f))
        for sub, pool in json.loads(Path(f).read_text())["pools"].items():
            if sub in deployed_rows:
                a = [(c["smiles"], round(c["generator"], 9), round(c["filter"], 9))
                     for c in deployed_rows[sub]]
                b = [(c["smiles"], round(c["generator"], 9), round(c["filter"], 9))
                     for c in pool]
                if a != b:
                    disagree += 1
            deployed_rows[sub] = pool
    if disagree:
        print(f"FAIL: {disagree} substrates are scored differently by the two builds of the "
              f"deployed arm; they are not the same configuration", file=sys.stderr)
        return 1
    # The exhaustive arm, assembled the same way and gated the same way: where the two builds
    # cover a substrate in common they must agree candidate for candidate, or they are not one
    # configuration and neither population can be read.
    exhaustive_rows, exh_disagree = {}, 0
    for pattern in (EXHAUSTIVE_COMPARISON / "w*.json", EXHAUSTIVE_FULLTEST / "w*.json"):
        for f in sorted(glob.glob(str(pattern))):
            read_paths.append(Path(f))
            for sub, pool in json.loads(Path(f).read_text())["pools"].items():
                if sub in exhaustive_rows:
                    a = [(c["smiles"], round(c["generator"], 9), round(c["filter"], 9))
                         for c in exhaustive_rows[sub]]
                    b = [(c["smiles"], round(c["generator"], 9), round(c["filter"], 9))
                         for c in pool]
                    if a != b:
                        exh_disagree += 1
                exhaustive_rows[sub] = pool
    if exh_disagree:
        print(f"FAIL: {exh_disagree} substrates are scored differently by the two builds of the "
              f"exhaustive arm; they are not the same configuration", file=sys.stderr)
        return 1

    others = {}
    for name, path in WHOLE_TEST.items():
        if path.exists():
            read_paths.append(path)
            others[name] = json.loads(path.read_text())
    # A comparator whose file is present but covers a fraction of the population would be read as
    # answering nothing on the rest, which is a recall of its own making. It is dropped with a
    # line rather than scored, and the artifact records which arms reached this population.
    for name in list(others):
        covered = len(set(others[name]) & set(truth))
        if covered < 0.99 * len(truth):
            print(f"  {name}: covers {covered} of {len(truth)} substrates, not enough to be "
                  f"read on this population; dropped", file=sys.stderr)
            others.pop(name)

    every = sorted(truth)
    inside = sorted(set(every) & set(metatox))
    outside = sorted(set(every) - set(metatox))

    # --- what the emission rule the manuscript describes would actually have done ---
    emission = {"comparison_set": {}, "whole_test_set": {}}
    for name, preds in others.items():
        emission["comparison_set"][name] = int(sum(1 for s in inside if not preds.get(s)))
        empty = [s for s in every if not preds.get(s)]
        emission["whole_test_set"][name] = {
            "substrates_with_no_prediction": len(empty),
            "references_they_carry": int(sum(len(truth[s]) for s in empty))}
    emission["comparison_set"]["metatox"] = int(sum(1 for s in inside if not metatox.get(s)))
    emission["comparison_set"]["grail_deployed"] = int(
        sum(1 for s in inside if not deployed_rows.get(s)))

    # --- are the 291 exchangeable with the 879 they were drawn from ---
    def heavy(s):
        mol = Chem.MolFromSmiles(s)
        return float(mol.GetNumHeavyAtoms()) if mol is not None else float("nan")

    features = {"annotated references": lambda s: float(len(truth[s])),
                "heavy atoms": heavy,
                "candidates the deployed system emits":
                    lambda s: float(len(deployed_rows.get(s, [])))}
    rng = np.random.default_rng(SEED)
    exchangeability = {}
    for label, fn in features.items():
        a = np.array([fn(s) for s in inside], dtype=float)
        b = np.array([fn(s) for s in outside], dtype=float)
        keep_a, keep_b = ~np.isnan(a), ~np.isnan(b)
        a, b = a[keep_a], b[keep_b]
        pooled = np.concatenate([a, b])
        observed = float(a.mean() - b.mean())
        n_a = len(a)
        ge = 0
        for _ in range(N_PERM):
            perm = rng.permutation(pooled)
            if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= abs(observed):
                ge += 1
        exchangeability[label] = {
            "mean_in_the_comparison_set": round(float(a.mean()), 3),
            "mean_outside_it": round(float(b.mean()), 3),
            "difference": round(observed, 3),
            "permutation_p": round((ge + 1) / (N_PERM + 1), 4),
            "n_in": int(n_a), "n_out": int(len(b))}

    # --- the same contrast on the population nobody selected ---
    def order_from(rows, subset):
        out = {}
        for s in subset:
            cands = rows.get(s) or []
            keep = sorted(cands, key=lambda c: -c["generator"])[:CAP]
            parent = tautkey(s)
            seen, ranked = set(), []
            for c in rrf_order(keep):
                # the pools carry the match key already; recomputing it costs an hour here
                k = c.get("key") or tautkey(c["smiles"])
                if not k or k == parent or k in seen:
                    continue
                seen.add(k)
                ranked.append(k)
            out[s] = ranked
        return out

    def deployed_order(subset):
        return order_from(deployed_rows, subset)

    def exhaustive_order(subset):
        return order_from(exhaustive_rows, subset)

    def comparator_order(preds, subset):
        return {s: [k for k in _dedup(preds.get(s, []), CAP + 5) if k and k != tautkey(s)]
                for s in subset}

    populations = {"the comparison set": inside}
    covered = [s for s in every if s in deployed_rows]
    if len(covered) == len(every):
        populations["the whole evaluated test set"] = every
    else:
        print(f"the deployed arm covers {len(covered)} of {len(every)} evaluated substrates; "
              f"the whole-test-set row is omitted rather than computed on a subset",
              file=sys.stderr)
    contrasts = {}
    for pop_label, subset in populations.items():
        real = {s: set(truth_keys) for s, truth_keys in
                ((s, [k for k in (tautkey(p) for p in truth[s]) if k]) for s in subset)}
        U = np.array([len(real[s]) for s in subset], dtype=float)
        ours = deployed_order(subset)
        boot = np.random.default_rng(SEED)
        idx = boot.integers(0, len(subset), (N_BOOT, len(subset)))
        denom = np.maximum(U[idx].sum(axis=1), 1)

        def hits(order, k):
            return np.array([len(set(order[s][:k]) & real[s]) for s in subset], dtype=float)

        # Both arms, where both cover the population. The exhaustive one carries the ordering
        # result this work claims, so reading it off the population nobody selected is the whole
        # point of this section.
        arms = {"deployed": ours}
        exh_covered = [s for s in subset if s in exhaustive_rows]
        if len(exh_covered) == len(subset):
            arms["exhaustive"] = exhaustive_order(subset)
        else:
            print(f"the exhaustive arm covers {len(exh_covered)} of {len(subset)} substrates of "
                  f"{pop_label}; its row is omitted rather than computed on a subset",
                  file=sys.stderr)

        row = {"n_substrates": len(subset), "n_references": int(U.sum()),
               "arms_present": sorted(arms),
               "grail_deployed_recall": {str(k): round(float(hits(ours, k).sum() / U.sum()), 4)
                                         for k in KS}}
        if "exhaustive" in arms:
            row["grail_exhaustive_recall"] = {
                str(k): round(float(hits(arms["exhaustive"], k).sum() / U.sum()), 4) for k in KS}
        for name, preds in others.items():
            theirs = comparator_order(preds, subset)
            cell = {"recall": {str(k): round(float(hits(theirs, k).sum() / U.sum()), 4)
                               for k in KS},
                    "substrates_with_no_prediction":
                        int(sum(1 for s in subset if not theirs[s]))}
            for arm_name, arm in arms.items():
                key = ("deployed_minus_comparator" if arm_name == "deployed"
                       else "exhaustive_minus_comparator")
                for k in KS:
                    d = hits(arm, k) - hits(theirs, k)
                    bt = d[idx].sum(axis=1) / denom
                    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                    cell.setdefault(key, {})[str(k)] = {
                        "difference": round(float(d.sum() / U.sum()), 4),
                        "ci95": [round(lo, 4), round(hi, 4)],
                        "excludes_zero": bool(lo > 0 or hi < 0)}
            row[name] = cell
        contrasts[pop_label] = row

    report = {
        "provenance": stamp(__file__),
        # The pools and prediction files this run was actually pointed at, so an artifact written
        # against a pool that has since been rebuilt or removed can be told from a current one.
        "inputs": record_inputs(read_paths),
        "what_defines_the_comparison_set": (
            "the intersection of the substrates each method has an entry for; the binding "
            "constraint is the 291-substrate submission list sent to the web-service comparator, "
            "which returned all 291. No substrate was removed for emitting nothing."),
        "how_the_291_were_drawn": (
            "not recorded: no script in this repository selects them and the submission "
            "directory documents the file format rather than the draw"),
        "emission": emission,
        "exchangeability": exchangeability,
        "contrasts": contrasts,
        "absent_arm": ("the exhaustive arm was built on the comparison set only, so the "
                       "whole-test-set row carries the deployed interactive arm and not it"),
        "deployed_arm_covers_the_whole_test_set": len(covered) == len(every),
        "permutation": {"n": N_PERM, "seed": SEED},
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "reading": (
            "The emission rule the manuscript describes is not the rule the code applies, and "
            "applying it would have removed almost nothing: over the whole test set the two "
            "comparators that ran on it answer nothing for a handful of substrates. The "
            "population's real defect is that the draw is unrecorded, which is tested here "
            "rather than described."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"comparison set {len(inside)} of {len(every)} evaluated substrates")
    print("\nsubstrates a method answers nothing for:")
    for name, n in emission["comparison_set"].items():
        print(f"  {name:16s} {n} inside the comparison set")
    for name, cell in emission["whole_test_set"].items():
        print(f"  {name:16s} {cell['substrates_with_no_prediction']} of {len(every)} over the "
              f"whole test set, carrying {cell['references_they_carry']} references")
    print("\nthe 291 against the 879 they were drawn from:")
    for label, cell in exchangeability.items():
        print(f"  {label:38s} {cell['mean_in_the_comparison_set']:8.3f} vs "
              f"{cell['mean_outside_it']:8.3f}  p={cell['permutation_p']:.4f}")
    print("\nthe deployed arm minus each comparator at 15 and 30:")
    for pop_label, row in contrasts.items():
        print(f"  {pop_label} ({row['n_substrates']} substrates, {row['n_references']} refs)")
        for name in others:
            for k in (15, 30):
                c = row[name]["deployed_minus_comparator"][str(k)]
                print(f"    {name:16s} k={k:<3d} {c['difference']:+.4f} "
                      f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                      f"{'  separates' if c['excludes_zero'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
