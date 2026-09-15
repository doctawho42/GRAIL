#!/usr/bin/env python3
"""Phase 1: one table of recall per (system, matching criterion, budget, population).

Writes revision/T_main.csv and revision/T_main_provenance.json. Re-tabulation only: no prediction
is generated here, and every cell that existing prediction files can support is emitted, including
the cells where this system loses.

Three things this module does NOT invent, and one it does.

Not invented. The ranking, the parent-drop convention, the pool cap, the budgets, the bootstrap
size and seed are taken from scripts/typed_edit/deployment_table.py, which produced the published
comparison-set table: cap 100, budgets (1, 3, 5, 8, 10, 15, 20, 30, 50), micro recall as the ratio
of sums, a prediction whose key equals the substrate's dropped BEFORE the budget for every arm
alike, 10,000 bootstrap draws at seed 0. A comparator's list is deduplicated in rank order to
`max(budget) + 5` keys, has the parent dropped, and is then truncated to `max(budget)`. The
reproduction gate in revision/tests asserts that the comparison-set cells under the tautomer
criterion equal the published floats.

Invented here, and labelled as such in the output. The repository stores no interval for a single
arm's recall: results/deployment_table.json holds bare floats for recall_micro and attaches
intervals only to contrasts, and scripts/_contrast.py implements the paired case alone. The
interval columns here are therefore a new quantity, a substrate-level bootstrap of the same micro
ratio at the same draw count and seed. They are not a re-tabulation of anything published.

Two joins that are easy to get silently wrong, handled explicitly.

References. On the comparison set the pools carry their own `references`, already as tautomer
keys, and the published table is computed from those; the gate only passes if they are used. Under
any other criterion those keys are the wrong alphabet, so references are re-keyed from the SMILES
in results/test_references.json instead, and the provenance file records which of the two supplied
each row.

Keying. scripts/bank_without_selection.py:_key falls back to returning the input SMILES when
canonicalisation fails, and such a "key" matches no reference, so the failure biases recall down
with nothing in the table to show it. Keying here reports the count and the inputs, and the
provenance file carries them.

    python revision/phase1_tmain.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAP = 100
KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0
CRITERIA = ("exact", "canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer")

# Where each arm's predictions come from, per population. Established by Phase 0 and recomputed
# there: revision/00_inventory_facts.json carries the counted coverage of every file named here.
ARMS = {
    "comparison291": {
        "whole bank": ("pool", "results/widepools_implicit/w*.json"),
        "trained budget": ("pool", "results/widepools_k30/all.json"),
        "metatox": ("list", "results/metatox_smirks_preds.json", "predictions"),
        "sygma": ("list", "results/sygma_fulltest_predictions.json", None),
        "metapredictor": ("list", "artifacts/tier2_1170/metapredictor_preds.json", None),
        "biotransformer": ("list", "results/biotransformer_allhuman_one_step_preds.json", None),
        "gloryx": ("list", "results/gloryx_service_preds.json", "predictions"),
    },
    "evaluated1170": {
        "whole bank": ("pool", "results/widepools_fulltest/w*.json"),
        "trained budget": ("pool", "results/widepools_k30_fulltest/w*.json"),
        "sygma": ("list", "results/sygma_fulltest_predictions.json", None),
        "metapredictor": ("list", "artifacts/tier2_1170/metapredictor_preds.json", None),
        "biotransformer": ("list", "results/biotransformer_fulltest_preds.json", None),
        # Produced by Phase 2. This is the MERGE of both GLORYx runs, not the service run's own
        # output: that one holds the 879 substrates the published comparison-set run lacked, so
        # naming it here would read an empty list for the other 291 and score them as misses.
        # revision/phase2_gloryx_merge.py refuses a union that does not cover the population, and
        # coverage_gaps() checks the arm as declared.
        "gloryx": ("list", "results/gloryx_service_preds_evaluated1170.json", "predictions"),
        # metatox still has no file on this population: it is a web service with no programmatic
        # interface recorded here, so Phase 2 wrote its submission files and stopped. The cells
        # stay absent rather than being filled from a narrower population.
    },
}


# --------------------------------------------------------------------------- keying

def _per_item_keyer(criterion):
    from grail_metabolism import metrics as M
    if criterion == "exact":
        return lambda s: s
    return {"canonical": M._canonical_key, "inchikey": M._inchikey,
            "inchikey_tautomer": M._tautomer_inchikey,
            "inchi_no_stereo": M._inchikey_skeleton, "tanimoto1": M._morgan_key}[criterion]


_TABLES: dict = {}


def _table(criterion):
    if criterion in _TABLES:
        return _TABLES[criterion]
    path = ROOT / "results" / "key_tables" / f"{criterion}.json"
    _TABLES[criterion] = json.loads(path.read_text()) if path.exists() else {}
    return _TABLES[criterion]


def match_keys(smiles_iter, criterion):
    """Keys for a list of SMILES under one criterion, plus a report of every fallback taken.

    The precomputed map is consulted first because canonicalisation is the cost of this script;
    anything it lacks is computed live, and anything that cannot be computed falls back to the raw
    string exactly as the repository's own keyer does, but counted rather than silent.
    """
    table, keyer = _table(criterion), _per_item_keyer(criterion)
    keys, report = [], {"from_table": 0, "computed_live": 0, "fallbacks": 0, "fallback_inputs": []}
    # A fallback cannot be detected by catching an exception, because this repository's keying
    # never raises: _tautomer_inchikey falls back to _inchikey per input, and _inchikey returns the
    # raw SMILES when RDKit cannot parse it. So an unkeyable structure arrives as a "key" that is
    # the SMILES itself, matches no reference, and biases recall down invisibly. It is detected
    # here by that signature. Two criteria are excluded because for them a key equal to the input
    # is correct rather than a failure: `exact` is the raw string by definition, and `canonical`
    # returns the input unchanged whenever the input is already canonical.
    detectable = criterion not in ("exact", "canonical")
    for s in smiles_iter:
        hit = table.get(s)
        if hit is not None:
            keys.append(hit)
            report["from_table"] += 1
            continue
        try:
            k = keyer(s)
            report["computed_live"] += 1
        except Exception:
            k = s
        keys.append(k)
        if detectable and k == s:
            report["fallbacks"] += 1
            if s not in report["fallback_inputs"]:
                report["fallback_inputs"].append(s)
    return keys, report


# --------------------------------------------------------------------------- list shaping

def drop_parent(keys, parent_key):
    """Drop the substrate's own key, before any budget is applied."""
    return [k for k in keys if k and k != parent_key]


def dedup_in_rank_order(keys, cap=None):
    out, seen = [], set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if cap and len(out) >= cap:
            break
    return out


# --------------------------------------------------------------------------- estimators

def micro_recall(hits, universe):
    """Ratio of sums: a substrate is weighted by how many references it carries."""
    return float(sum(hits.values())) / max(float(sum(universe.values())), 1.0)


def bootstrap_ci(hits, universe, n_boot=N_BOOT, seed=SEED):
    """Percentile interval for the micro ratio, resampling substrates.

    Computed here because the repository stores no interval for a single arm's recall.
    """
    subs = sorted(hits)
    h = np.array([hits[s] for s in subs], dtype=float)
    u = np.array([universe[s] for s in subs], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(subs), (n_boot, len(subs)))
    draws = h[idx].sum(axis=1) / np.maximum(u[idx].sum(axis=1), 1.0)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# --------------------------------------------------------------------------- loading

def _load_pools(pattern):
    pools, refs, read = {}, {}, []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs.update(blob.get("references") or {})
        read.append(str(Path(f).relative_to(ROOT)))
    return pools, refs, read


def _ordered_candidates(pool):
    """The release order over a pool, as the published table builds it, before any keying.

    The order is criterion-independent: it is a function of the two component scores. Only the
    keys the ordered candidates are then reduced to depend on the criterion, so the dicts are
    returned and keyed by the caller.
    """
    from _rrf import rrf_order
    keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
    return list(rrf_order(keep))


def _arm_keys(candidates, criterion):
    """Keys for an ordered candidate list under one criterion.

    Under the tautomer criterion the pool already stores the key the published table was computed
    from, so it is read rather than recomputed. That is not only cheaper -- re-canonicalising the
    whole-bank pools costs about forty-five minutes of RDKit for the comparison set alone -- it is
    also the only faithful choice: a different RDKit than the one that wrote the pools would
    produce different keys, and the gate would then fail for a reason that has nothing to do with
    the arithmetic under test. Every other criterion has no stored key and is computed from the
    candidate SMILES.
    """
    if criterion == "inchikey_tautomer":
        return [c["key"] for c in candidates], {"from_pool_key": len(candidates), "fallbacks": 0,
                                                "fallback_inputs": []}
    return match_keys([c["smiles"] for c in candidates], criterion)


def arm_key_lists(spec, subs, ordered_for_arm, parent, criterion):
    """One arm's per-substrate key lists, ready for scoring, or None when its file is absent.

    Factored out of `build_rows` so the family-wise recomputation can reuse the assembly that the
    reproduction gate certifies, instead of re-expressing it. Re-expressing it is how two analyses
    of one quantity drift apart, and this repository has paid for that before.

    A pool arm is ordered by the release ranking and keyed per criterion; a list arm is keyed,
    deduplicated in rank order to `max(budget) + 5`, has the parent dropped, and is truncated to
    `max(budget)`. Both drop the parent before any budget is applied, which is the published
    convention and has to hold for every arm alike or the arms are not measured on one axis.
    """
    if spec[0] == "pool":
        keys_by_sub = {}
        rep = {"fallbacks": 0, "fallback_inputs": [], "from_pool_key": 0}
        for s in subs:
            ks, r = _arm_keys(ordered_for_arm[s], criterion)
            rep["fallbacks"] += r["fallbacks"]
            rep["from_pool_key"] += r.get("from_pool_key", 0)
            keys_by_sub[s] = dedup_in_rank_order(drop_parent(ks, parent[s]))
        return keys_by_sub, rep, spec[1]

    path = ROOT / spec[1]
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    preds = blob[spec[2]] if spec[2] else blob
    keys_by_sub = {}
    rep = {"fallbacks": 0, "fallback_inputs": []}
    for s in subs:
        ks, r = match_keys(preds.get(s, []), criterion)
        rep["fallbacks"] += r["fallbacks"]
        cut = dedup_in_rank_order(ks, cap=max(KS) + 5)
        keys_by_sub[s] = drop_parent(cut, parent[s])[:max(KS)]
    return keys_by_sub, rep, spec[1]


def coverage_gaps(arms=None, members=None, populations=None):
    """Declared list arms whose file exists but does not cover its population.

    The invariant this table rests on. `arm_key_lists` reads a list arm with `preds.get(s, [])`,
    so a substrate the file lacks arrives as an empty list and is scored as a miss rather than as
    absent -- the comparator is understated and nothing says so. Every arm declared here covers
    its population, and the GLORYx column over the evaluated set came within one edit of breaking
    it: the service run's output holds only the substrates the published run lacked, so naming
    that file directly would have understated GLORYx on the other 291.

    An arm whose file is missing entirely is deliberately not a gap. `arm_key_lists` returns None
    and the arm yields no cells at all, which is the right treatment of a comparator that was
    never run; demanding coverage from it would demand a merge for something never attempted. The
    dangerous case is the file that exists and is short, because it looks like data.

    Pool arms are not checked this way: they are assembled from shards and indexed directly, so a
    missing substrate raises rather than becoming an empty list.

    `members` lets a caller supply each population's substrates; by default they are resolved
    through this module's own accessor, because three comparator files here carry 291 keys and one
    of them is a different 291, so a count is not a population.
    """
    arms = ARMS if arms is None else arms
    populations = tuple(arms) if populations is None else populations
    out = []
    for population in populations:
        if population not in arms:
            continue
        if members is not None and population in members:
            subs = set(members[population])
        else:
            subs = set(_population(population)[0])
        for arm, spec in arms[population].items():
            if spec[0] != "list":
                continue
            path = ROOT / spec[1]
            if not path.exists():
                continue
            blob = json.loads(path.read_text())
            preds = blob[spec[2]] if spec[2] else blob
            if not isinstance(preds, dict):
                out.append({"population": population, "arm": arm, "file": spec[1],
                            "n_missing": len(subs), "n_covered": 0,
                            "first_missing": sorted(subs)[0] if subs else None,
                            "why": "the declared accessor did not yield a substrate map"})
                continue
            missing = subs - set(preds)
            if missing:
                out.append({"population": population, "arm": arm, "file": spec[1],
                            "n_missing": len(missing), "n_covered": len(subs) - len(missing),
                            "first_missing": sorted(missing)[0]})
    return out


def _population(name):
    """Substrates, per-substrate reference SMILES, and the pool sources for one population."""
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    if name == "evaluated1170":
        big, _, read_b = _load_pools(ARMS[name]["whole bank"][1])
        small, _, read_s = _load_pools(ARMS[name]["trained budget"][1])
        subs = sorted(set(truth) & set(big) & set(small))
        return subs, {s: truth[s] for s in subs}, (big, small), read_b + read_s, None
    big, refs_b, read_b = _load_pools(ARMS[name]["whole bank"][1])
    small, refs_s, read_s = _load_pools(ARMS[name]["trained budget"][1])
    pool_refs = {**refs_b, **refs_s}
    subs = sorted(s for s in set(big) & set(small) if pool_refs.get(s))
    return subs, {s: truth[s] for s in subs}, (big, small), read_b + read_s, pool_refs


# --------------------------------------------------------------------------- the table

def build_rows(populations=("comparison291", "evaluated1170"), criteria=CRITERIA,
               n_boot=N_BOOT, seed=SEED, provenance=None):
    from bank_without_selection import _key as tautkey

    rows = []
    for population in populations:
        subs, ref_smiles, (big, small), pool_files, pool_refs = _population(population)
        parent_tauto = {s: tautkey(s) for s in subs}
        ordered = {"whole bank": {s: _ordered_candidates(big[s]) for s in subs},
                   "trained budget": {s: _ordered_candidates(small[s]) for s in subs}}
        for criterion in criteria:
            # references: the pools' own keys only where they are the right alphabet
            use_pool_refs = (criterion == "inchikey_tautomer" and pool_refs is not None)
            if use_pool_refs:
                real = {s: set(pool_refs[s]) for s in subs}
                ref_source, ref_report = "pool references (tautomer keys)", None
            else:
                flat, index = [], {}
                for s in subs:
                    index[s] = (len(flat), len(flat) + len(ref_smiles[s]))
                    flat.extend(ref_smiles[s])
                keys, ref_report = match_keys(flat, criterion)
                real = {s: set(keys[a:b]) for s, (a, b) in index.items()}
                ref_source = "results/test_references.json, re-keyed"
            universe = {s: len(real[s]) for s in subs}
            parent = parent_tauto if criterion == "inchikey_tautomer" else {
                s: k for s, k in zip(subs, match_keys(subs, criterion)[0])}

            for arm, spec in ARMS[population].items():
                built = arm_key_lists(spec, subs, ordered.get(arm), parent, criterion)
                if built is None:
                    continue
                keys_by_sub, rep, source = built
                for k in KS:
                    hits = {s: len(set(keys_by_sub[s][:k]) & real[s]) for s in subs}
                    lo, hi = bootstrap_ci(hits, universe, n_boot=n_boot, seed=seed)
                    rows.append({
                        "system": arm, "criterion": criterion, "k": k, "population": population,
                        "recall": round(micro_recall(hits, universe), 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "n_substrates": len(subs),
                        "n_references": int(sum(universe.values())),
                        "mean_emitted_at_k": round(
                            float(np.mean([min(len(keys_by_sub[s]), k) for s in subs])), 3),
                        "mean_emitted_untruncated": round(
                            float(np.mean([len(keys_by_sub[s]) for s in subs])), 3),
                        "predictions_from": source,
                        "references_from": ref_source,
                        "interval": "computed here; the repository stores none per arm",
                    })
                if provenance is not None:
                    provenance.setdefault("keying", []).append(
                        {"population": population, "criterion": criterion, "system": arm,
                         "fallbacks": rep["fallbacks"]})
            if provenance is not None and ref_report is not None:
                provenance.setdefault("reference_keying", []).append(
                    {"population": population, "criterion": criterion,
                     "fallbacks": ref_report["fallbacks"],
                     "fallback_inputs": ref_report["fallback_inputs"]})
        if provenance is not None:
            provenance.setdefault("pool_files", {})[population] = pool_files
    return rows


def main() -> int:
    from _provenance import stamp
    # Stamped like every other artifact here. The deposit recorded only the producer's filename,
    # which cannot say whether the file was written by this version of it; the digest can, and a
    # deposit that outlives an edit to its producer is the stale-artifact defect this repository
    # has already paid for once.
    provenance = {"provenance": stamp(__file__),
                  "generated_by": "revision/phase1_tmain.py", "cap": CAP, "budgets": list(KS),
                  "n_boot": N_BOOT, "seed": SEED,
                  "mechanics_copied_from": "scripts/typed_edit/deployment_table.py",
                  "arms": {p: {a: s[1] for a, s in d.items()} for p, d in ARMS.items()},
                  "cells_absent_by_construction": {
                      "evaluated1170": ["metatox"],
                      "why": ("MetaTox is a web service with no programmatic interface recorded "
                              "here, so Phase 2 wrote its submission files and stopped; GLORYx "
                              "was produced and is no longer absent")}}
    rows = build_rows(provenance=provenance)
    out = ROOT / "revision" / "T_main.csv"
    cols = ["system", "criterion", "k", "population", "recall", "ci_lo", "ci_hi",
            "n_substrates", "n_references", "mean_emitted_at_k", "mean_emitted_untruncated",
            "predictions_from", "references_from", "interval"]
    # LF, so a byte comparison against the committed table is meaningful instead of failing for a
    # line ending and looking like a change in the numbers. Both settings are needed and the first
    # attempt here set only one: `newline=` governs whether Python translates newlines on write,
    # while csv.writer emits its own row terminator, and that defaults to CRLF. Setting only the
    # open() newline left the output byte for byte unchanged -- measured, after it was claimed
    # fixed once already.
    with open(out, "w", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (ROOT / "revision" / "T_main_provenance.json").write_text(json.dumps(provenance, indent=1))
    by_pop: dict = {}
    for r in rows:
        by_pop.setdefault(r["population"], set()).add(r["system"])
    print(f"wrote {out.relative_to(ROOT)}: {len(rows)} rows")
    for pop, systems in sorted(by_pop.items()):
        print(f"  {pop}: {len(systems)} systems x {len(CRITERIA)} criteria x {len(KS)} budgets")
        print(f"    {sorted(systems)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
