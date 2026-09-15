#!/usr/bin/env python3
"""Phase 2: GLORYxR's two modes beside the service-derived GLORYx column, where that is legitimate.

Writes revision/T_gloryxr.csv and revision/T_gloryxr_provenance.json.

Why this table exists rather than a gloryxr arm in phase1_tmain.ARMS. The local column emits no
stereochemistry at all: 0 of 14,755 predictions carry a stereo InChIKey block against 725 of 14,784
in the service column. Beside the service arm it therefore disagrees by 25-26% under `exact`,
12-14% under `canonical` and 9-10% under `inchikey`, always against the local arm, and by about
1.2% under `inchi_no_stereo` and `inchikey_tautomer`. ARMS cannot restrict an arm to a subset of
criteria, so declaring one there would emit four knowingly invalid rows per mode. Here the
restriction is enforced: a stereo-sensitive criterion is refused rather than computed, because such
a number measures an alphabet difference and not either system.

The arms are assembled through phase1_tmain.arm_key_lists, the same helper the reproduction gate
certifies. Re-expressing dedup, parent-drop and truncation is how two readings of one column drift
apart, and this repository has paid for that already.

    python revision/phase2_gloryxr_table.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase1_tmain as T  # noqa: E402

# Reused, not re-expressed. The test pins this identity so a second implementation cannot creep in.
arm_key_lists = T.arm_key_lists

OUT = ROOT / "revision" / "T_gloryxr.csv"
PROV = ROOT / "revision" / "T_gloryxr_provenance.json"
POPULATION = "evaluated1170"

# The only two criteria under which the local and service columns are written in the same alphabet.
CRITERIA = ("inchi_no_stereo", "inchikey_tautomer")
# Refused by name, each with the disagreement that disqualifies it.
STEREO_SENSITIVE = {"exact": "25-26%", "canonical": "12-14%", "inchikey": "9-10%"}

ARMS = {
    "gloryxr default": ("list", "results/gloryxr_local_preds_default.json", "predictions"),
    "gloryxr strict": ("list", "results/gloryxr_local_preds_strict.json", "predictions"),
    "gloryx": ("list", "results/gloryx_service_preds_evaluated1170.json", "predictions"),
    "whole bank": ("pool", "results/widepools_fulltest/w*.json"),
    "trained budget": ("pool", "results/widepools_k30_fulltest/w*.json"),
}
ROUTE = {
    "gloryx": ("the authors' web service at module version 0.0.26, phase_1_and_2; its scores are "
               "rounded to three decimals and it emits some products a single rule application "
               "cannot reach"),
    "gloryxr default": ("run locally from the public source with the author-supplied model dumps, "
                        "one generation, scores at six decimals"),
    "gloryxr strict": ("the same run with strict SOM annotation, which restricts the site to the "
                       "most relevant atoms and changes ranking but not coverage"),
}


def coverage_disagreement(a: dict, b: dict) -> list:
    """Substrates on which two mode columns hold different product SETS.

    The two modes produce identical sets on all 1,170 substrates and differ only in order, so every
    coverage figure is equal between them by construction. If that stops holding, either a column
    was regenerated against different inputs or the modes no longer mean what they mean, and
    publishing two coverage numbers that cannot both be right is worse than refusing.
    """
    shared = set(a) & set(b)
    return sorted(s for s in shared if set(a[s]) != set(b[s]))


_POPULATION_CACHE: dict = {}


def _population_once():
    """The population, resolved once per process.

    Without this the test suite took 43:40: `rows()` is called for each admitted criterion and by
    three separate tests, and every call re-loaded the pool shards and re-derived the reference
    sets. The work is identical each time, so a gate paying for it repeatedly is paying for
    nothing.
    """
    if POPULATION not in _POPULATION_CACHE:
        _POPULATION_CACHE[POPULATION] = T._population(POPULATION)
    return _POPULATION_CACHE[POPULATION]


def rows(criterion: str = "inchikey_tautomer") -> list:
    """Recall at each budget for every arm, under one admitted criterion.

    No intervals, and no parameter offering them: these columns were obtained by different routes,
    so an interval around each would invite the paired reading the routes do not support. The
    reason is recorded in `provenance()["intervals"]` rather than as a dead argument here.
    """
    if criterion in STEREO_SENSITIVE:
        raise ValueError(
            f"REFUSING criterion {criterion!r}: the local GLORYxR column carries no "
            f"stereochemistry at all (0 of 14,755 predictions have a stereo InChIKey block, "
            f"against 725 of 14,784 in the service column), so this criterion disagrees by "
            f"{STEREO_SENSITIVE[criterion]} against the local arm and would measure the alphabet "
            f"rather than either system. Admitted criteria: {', '.join(CRITERIA)}")
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; admitted: {', '.join(CRITERIA)}")

    subs, ref_smiles, (big, small), _, pool_refs = _population_once()
    from bank_without_selection import _key as tautkey

    if criterion == "inchikey_tautomer" and pool_refs is not None:
        real = {s: set(pool_refs[s]) for s in subs}
    else:
        flat, index = [], {}
        for s in subs:
            index[s] = (len(flat), len(flat) + len(ref_smiles[s]))
            flat.extend(ref_smiles[s])
        keys, _r = T.match_keys(flat, criterion)
        real = {s: set(keys[a:b]) for s, (a, b) in index.items()}
    parent = ({s: tautkey(s) for s in subs} if criterion == "inchikey_tautomer"
              else {s: k for s, k in zip(subs, T.match_keys(subs, criterion)[0])})
    ordered = {"whole bank": {s: T._ordered_candidates(big[s]) for s in subs},
               "trained budget": {s: T._ordered_candidates(small[s]) for s in subs}}

    universe = {s: len(real[s]) for s in subs}
    out = []
    for system, spec in ARMS.items():
        built = arm_key_lists(spec, subs, ordered.get(system), parent, criterion)
        if built is None:
            continue
        keys_by_sub, _rep, source = built
        for k in T.KS:
            hits = {s: len(set(keys_by_sub[s][:k]) & real[s]) for s in subs}
            out.append({
                "system": system,
                "criterion": criterion,
                "k": k,
                "population": POPULATION,
                "recall": round(T.micro_recall(hits, universe), 4),
                "n_substrates": len(subs),
                "n_references": int(sum(universe.values())),
                "mean_emitted_at_k": round(
                    float(sum(min(len(keys_by_sub[s]), k) for s in subs) / len(subs)), 3),
                "predictions_from": source,
                "route": ROUTE.get(system, ""),
            })
    return out


def provenance() -> dict:
    """What this table is, and the six measured limitations that travel with it."""
    return {
        "what_this_is": ("GLORYxR's two SOM modes beside the service-derived GLORYx column and "
                         "GRAIL's two arms, on the evaluated population, under the two criteria "
                         "where the columns share an alphabet"),
        "population": POPULATION,
        "criteria_admitted": list(CRITERIA),
        "criteria_refused": {k: f"{v} disagreement, always against the local arm"
                             for k, v in STEREO_SENSITIVE.items()},
        "why_not_an_arm_in_t_main": (
            "phase1_tmain.ARMS cannot restrict an arm to a subset of criteria, so a gloryxr arm "
            "there would emit four knowingly invalid rows per mode under exact, canonical and "
            "inchikey. The restriction is a measurement, not a preference."),
        "limitations": [
            {"id": "stereo",
             "detail": ("0 of 14,755 local predictions carry a stereo InChIKey block against 725 "
                        "of 14,784 service ones, and 0 of 11,072 local SMILES contain @, / or \\ "
                        "against 1,982 of 35,175 service ones. This is why only two criteria are "
                        "admitted.")},
            {"id": "generation",
             "detail": ("the local column is strictly one generation, measured rather than read "
                        "off the source (results/gloryxr_mechanics.json): one pass of the bank "
                        "over the parent, with RunReactants enumerating every match of a "
                        "template, so a single firing of a single rule returns several products. "
                        "Over the 1,170 substrates, 9,230 of 20,923 rule firings returned more "
                        "than one product and the worst single firing returned 46. What does not "
                        "happen is recursion: react_one has one call site and no product is fed "
                        "back as a substrate. An earlier version of this record said each rule is "
                        "applied once, which was false, and made a claim about the service's "
                        "cascade that nothing here measured.")},
            {"id": "score_resolution",
             "detail": ("the service rounds scores to three decimals and the local column keeps "
                        "six, so on 298 shared substrates longer than 15 the k=15 cut is decided "
                        "by a lexicographic tie-break in 128 of 298 service lists against 106 "
                        "local ones -- a systematic difference in how the budget bites")},
            {"id": "heavy_atom_floor",
             "detail": ("GLORYxR discards products below 3 heavy atoms (reactions.py), while "
                        "GRAIL's own floor is 2 -- an undocumented asymmetry in coverage ceiling")},
            {"id": "tautomer_drawing",
             "detail": ("GLORYxR applies no tautomer canonicalisation to its output while GRAIL "
                        "does, and the corpus stores 155 of the 1,170 substrates as imidic acids. "
                        "Measured over all of them -- 122 pairs, 33 skipped where canonicalisation "
                        "leaves the string unchanged or the molecule will not parse -- by predicting "
                        "each substrate in both the corpus and the natural-tautomer drawing: "
                        "Jaccard of the product sets is a median 0.028 on raw SMILES (mean 0.030, "
                        "max 0.091) and a median 0.725 on the tautomer key (mean 0.704, min 0.125, "
                        "max 0.933). So the key absorbs most of the difference but not all of it, "
                        "and NOT ONE of the 122 reaches full agreement: 0 of 122. In absolute "
                        "terms a median 3 products appear only for the corpus drawing (mean 5.84, "
                        "max 21) and a median 7 only for the natural one (mean 11.52, max 49), "
                        "against a median 38 products in total. The drawing therefore changes "
                        "which rules fire, not merely how the result is written, on every affected "
                        "substrate.")},
            {"id": "duplicate_model",
             "detail": ("two of the eight per-subset dumps in multi_models/ are byte-identical "
                        "(sha256 2c186eb6...), so the 224 phase-1 rules across those two subsets "
                        "are scored by one and the same trained forest, loaded twice under two "
                        "keys -- 86% of the rule table by rule count. Nine dumps arrived in all, "
                        "those eight plus single_model.joblib, which no reported number uses")},
        ],
        "rule_bank_mechanics": (
            "the numbers behind the generation limitation, the 0.2 factor applied to rules of "
            "uncommon priority, and how many rules carry that priority are measured and recorded "
            "in results/gloryxr_mechanics.json rather than asserted here"),
        "modes_share_coverage": ("identical product sets on 1170/1170 substrates, mean list 31.95 "
                                 "in both; order agrees on 32 (2.7%) and top-15 differs on 838 "
                                 "(71.6%) at mean Jaccard 0.795. Any coverage figure is equal "
                                 "between them by construction."),
        "what_the_mode_costs_and_buys": (
            "strict SOM annotation is better at small budgets and worse at large ones, and the "
                "crossing sits near k=15. Under inchikey_tautomer the strict-minus-default "
                "difference runs +0.0269, +0.0316, +0.0246, +0.0161, +0.0166 at k=1,3,5,8,10, "
                "exactly 0.0000 at k=15, then -0.0058, -0.0096, -0.0016 at k=20,30,50; under "
                "inchi_no_stereo the same shape, differing on 9 of 9 budgets. That is what the "
                "tool's own tutorial predicts: restricting the site removes scores inflated by "
                "promiscuous atom matching, which cleans the head of the list and costs depth. "
                "Reading the pair at k=15 alone would suggest the mode does not affect retrieval "
                "at all, because k=15 is the one budget where the difference vanishes."),
        "against_grail_the_sign_changes_with_budget": (
            "the service-derived gloryx column leads GRAIL's whole bank on 5 of 9 budgets under "
                "inchikey_tautomer and 4 of 9 under inchi_no_stereo, and the sign is not random: "
                "gloryx is ahead through k=5..20 (by as much as +0.0292 at k=10) and behind at "
                "k=30 and k=50 (by 0.0327 and 0.0867). So neither system dominates; the ordering "
                "depends on the budget, which is the same budget-dependent reversal this work "
                "reports for its other comparators, and a single-budget statement in either "
                "direction would misdescribe it."),
        "environment": ("both columns computed under the project's own uv.lock (scikit-learn "
                        "1.9.0, cdpkit 1.2.3) and again under a fresh resolve (1.9.1, 1.3.0); the "
                        "substrate maps are byte-identical in both modes, so the version drift "
                        "moves no number on this population"),
        "intervals": ("none. These columns were obtained by different routes -- a web service and "
                      "a local run of a successor -- and an interval around each would invite the "
                      "paired reading the routes do not support."),
    }


def main() -> int:
    d = json.loads((ROOT / ARMS["gloryxr default"][1]).read_text())["predictions"]
    s = json.loads((ROOT / ARMS["gloryxr strict"][1]).read_text())["predictions"]
    bad = coverage_disagreement(d, s)
    if bad:
        print(f"REFUSING: the two modes disagree on the product set for {len(bad)} substrate(s), "
              f"e.g. {bad[0][:60]}. They must agree by construction; publishing two coverage "
              f"figures that cannot both be right is worse than refusing.")
        return 1

    all_rows = []
    for criterion in CRITERIA:
        all_rows.extend(rows(criterion=criterion))

    cols = ["system", "criterion", "k", "population", "recall", "n_substrates", "n_references",
            "mean_emitted_at_k", "predictions_from", "route"]
    with open(OUT, "w", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)

    prov = provenance()
    try:
        from _provenance import stamp
        prov = {"provenance": stamp(__file__), **prov}
    except Exception as e:
        prov = {"provenance": {"unavailable": f"{e.__class__.__name__}: {e}"}, **prov}
    PROV.write_text(json.dumps(prov, indent=1))

    print(f"wrote {OUT.relative_to(ROOT)}: {len(all_rows)} rows")
    for criterion in CRITERIA:
        here = [r for r in all_rows if r["criterion"] == criterion]
        print(f"  {criterion}: {len({r['system'] for r in here})} systems x {len(T.KS)} budgets")
        for system in sorted({r["system"] for r in here}):
            at15 = next((r["recall"] for r in here
                         if r["system"] == system and r["k"] == 15), None)
            print(f"    {system:18} recall@15 {at15}")
    print(f"wrote {PROV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
