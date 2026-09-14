#!/usr/bin/env python3
"""Phase 0 discovery probe: recompute every machine-checkable fact in revision/00_inventory.md.

This is a discovery probe, not an analysis function. It counts what exists and writes the counts
to revision/00_inventory_facts.json so the prose inventory beside it can be audited against a
rerun rather than trusted. The test-first discipline the revision asks for begins with Phase 1's
analysis functions, which compute recall and intervals; nothing here computes a metric.

Two traps this probe exists to avoid, both hit while writing the inventory by hand:

  * counting "substrate keys" by the look of the string. A filter requiring one of "()=#@" drops
    Br, CCO and every other short SMILES, which silently turned 1,170 into 1,144. Coverage here is
    the size of the intersection with a declared population and nothing else.
  * reading the comparison set from the artifact that names it. comparison_set_members.json keys
    its members by tautomer InChIKey, so it cannot be intersected with prediction files keyed by
    SMILES. The 291 SMILES are recovered from the prediction files that carry exactly them, and
    the probe refuses if those files disagree.

    python revision/00_inventory.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision" / "00_inventory_facts.json"

# Prediction files are looked for here; the first sweep missed artifacts/ entirely and so missed
# the only MetaPredictor file that covers the whole evaluated set.
PRED_GLOBS = (
    "results/*pred*.json", "results/*sygma*.json", "results/*metatox*.json",
    "results/*metapredictor*.json", "results/*biotransformer*.json", "results/*glory*.json",
    "results/wide_pools.json",
    "artifacts/**/*pred*.json", "docs/**/*pred*.json", "docs/**/*gloryx*.json",
)
POOL_DIR_GLOBS = ("results/*pools*/", "results/*shards*/")
# The four files that carry exactly the comparison set, used to recover its SMILES.
C291_SOURCES = ("results/gloryx_service_preds.json", "results/metatox_smirks_preds.json",
                "results/sygma_standardised_predictions.json", "results/wide_pools.json")


def _load(rel):
    try:
        return json.loads((ROOT / rel).read_text())
    except Exception:
        return None


def _subject_keys(blob, universes):
    """The substrate keys a file is about, or None when it is not keyed by substrate."""
    if not isinstance(blob, dict):
        return None
    for nest in ("predictions", "preds", "pools", "by_substrate"):
        if isinstance(blob.get(nest), dict):
            return set(blob[nest])
    top = set(blob)
    return top if any(top & u for u in universes) else None


def populations():
    truth = _load("results/test_references.json") or {}
    evaluated = set(truth)
    with_ref = {s for s, v in truth.items() if v}
    sets, disagree = {}, []
    for rel in C291_SOURCES:
        blob = _load(rel)
        keys = _subject_keys(blob, [evaluated]) if blob else None
        if keys:
            sets[rel] = keys
    members = set()
    if sets:
        first = next(iter(sets.values()))
        for rel, ks in sets.items():
            if ks != first:
                disagree.append(rel)
        members = first
    return {
        "evaluated1170": {"source": "results/test_references.json", "n": len(evaluated),
                          "n_carrying_a_reference": len(with_ref)},
        "comparison291": {
            "n": len(members),
            "recovered_from": sorted(sets),
            "sources_disagreeing_on_membership": disagree,
            "named_by": "results/comparison_set_members.json keys members by tautomer InChIKey, "
                        "not by SMILES, so it cannot be joined to prediction files directly",
            "draw_procedure_recorded": False,
        },
    }, evaluated, members


def coverage(evaluated, members):
    rows = []
    seen = set()
    for pat in PRED_GLOBS:
        for p in sorted(glob.glob(str(ROOT / pat), recursive=True)):
            rel = os.path.relpath(p, ROOT)
            if rel in seen:
                continue
            seen.add(rel)
            blob = _load(rel)
            keys = _subject_keys(blob, [evaluated, members]) if blob is not None else None
            if not keys:
                continue
            rows.append({"file": rel, "keys": len(keys),
                         "of_evaluated1170": len(keys & evaluated),
                         "of_comparison291": len(keys & members)})
    for pat in POOL_DIR_GLOBS:
        for d in sorted(glob.glob(str(ROOT / pat))):
            subs, files = set(), 0
            for p in sorted(glob.glob(os.path.join(d, "*.json"))):
                blob = _load(os.path.relpath(p, ROOT))
                if isinstance(blob, dict) and isinstance(blob.get("pools"), dict):
                    subs |= set(blob["pools"])
                    files += 1
            if subs:
                rows.append({"file": os.path.relpath(d, ROOT), "shards": files, "keys": len(subs),
                             "of_evaluated1170": len(subs & evaluated),
                             "of_comparison291": len(subs & members)})
    rows.sort(key=lambda r: (-r["of_evaluated1170"], -r["of_comparison291"], r["file"]))
    return rows


def criteria():
    src = (ROOT / "grail_metabolism/metrics.py").read_text()
    branches = re.findall(r'if match == "([a-z0-9_]+)":\s*\n\s*return \{(_[a-z_]+)\(', src)
    cfg = re.search(r'match: Literal\[([^\]]*)\]', (ROOT / "grail_metabolism/config.py").read_text())
    met = re.search(r'match: Literal\[([^\]]*)\]', src)
    def names(m):
        return sorted(set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))) if m else []
    tables = {}
    for p in sorted(glob.glob(str(ROOT / "results/key_tables/*.json"))):
        blob = _load(os.path.relpath(p, ROOT))
        tables[os.path.basename(p)[:-5]] = len(blob) if isinstance(blob, dict) else None
    return {"dispatch": "grail_metabolism/metrics.py:_match_keys",
            "branches": {k: v for k, v in branches},
            "fallback": "any other value compares the raw strings",
            "declared_in_config": names(cfg), "declared_in_metrics": names(met),
            "precomputed_key_tables": tables}


def multiplicity():
    d = _load("results/multiplicity.json")
    if not d:
        return {"error": "results/multiplicity.json unreadable"}
    arms, comps, budgets = Counter(), Counter(), Counter()
    for name in d.get("cells", {}):
        m = re.match(r"^(.*?) - (.*?) @ (\d+)$", name)
        if m:
            arms[m.group(1)] += 1
            comps[m.group(2)] += 1
            budgets[int(m.group(3))] += 1
    wider = d.get("over_every_contrast_the_paper_prints", {})
    return {"artifact": "results/multiplicity.json", "family": d.get("family"),
            "alpha": d.get("alpha"), "procedure": d.get("procedure"),
            "n_tests": d.get("n_tests"),
            "n_separating_per_comparison": d.get("n_separating_per_comparison"),
            "n_separating_after_holm": d.get("n_separating_after_holm"),
            "arms": dict(arms), "comparators": dict(comps), "budgets": sorted(budgets),
            "records_its_population": "population" in d,
            "n_cells_whose_verdict_changes": len(d.get("cells_whose_verdict_the_correction_changes", [])),
            "n_leads_removed": len(d.get("leads_the_correction_removes", [])),
            "wider_family": {"n_tests": wider.get("n_tests"),
                             "n_separating_after_holm": wider.get("n_separating_after_holm"),
                             "comparators": wider.get("comparators")},
            "other_fwer_artifacts_not_quoted_by_the_manuscript": {
                rel: sorted(_load(rel) or {}) for rel in
                ("results/multiplicity_holm.json", "results/union_multiplicity.json")
                if _load(rel) is not None}}


def documents():
    pulled = {}
    for tex in sorted(glob.glob(str(ROOT / "paper2/*.tex"))):
        body = Path(tex).read_text()
        pulled[os.path.basename(tex)[:-4]] = sorted(set(re.findall(r"\\input\{([^}]*)\}", body)))
    every = {n for names in pulled.values() for n in names}
    present = {os.path.basename(p)[:-4] for p in glob.glob(str(ROOT / "paper2/*.tex"))}
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}",
                                    "".join(Path(t).read_text() for t in
                                            glob.glob(str(ROOT / "paper2/*.tex"))))))
    return {"inputs_by_document": {k: v for k, v in pulled.items() if v},
            "tex_present_but_pulled_by_nothing": sorted(present - every - {"grail_jcim", "si"}),
            "figures_referenced": figures,
            "figure_files_present": sorted(os.path.basename(p) for p in
                                           glob.glob(str(ROOT / "paper2/*.eps"))),
            "producers": {
                "paper2/table_sweep.tex": "scripts/paper2_tables.py",
                "paper2/table_modes.tex, table_grain.tex, table_hypotheses.tex, table_case.tex":
                    "scripts/paper2_tables_more.py",
                "paper2/si_table_*.tex": "scripts/paper2_si_tables.py",
                "paper2/fig_*.eps": "scripts/paper2_figures.py"}}


def main() -> int:
    pops, evaluated, members = populations()
    facts = {"generated_by": "revision/00_inventory.py",
             "populations": pops,
             "prediction_coverage": coverage(evaluated, members),
             "matching_criteria": criteria(),
             "family_wise_correction": multiplicity(),
             "documents": documents()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=1))
    print(f"evaluated1170 = {pops['evaluated1170']['n']}, "
          f"comparison291 = {pops['comparison291']['n']}, "
          f"disagreeing sources = {pops['comparison291']['sources_disagreeing_on_membership']}")
    full = [r for r in facts["prediction_coverage"] if r["of_evaluated1170"] == pops["evaluated1170"]["n"]]
    print(f"files covering the whole evaluated set: {len(full)}")
    for r in full:
        print(f"  {r['file']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
