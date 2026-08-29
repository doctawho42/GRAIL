#!/usr/bin/env python3
"""One substrate, the ranked answer the server returns, and the rule behind every line of it.

The paper's main non-numerical claim is that a prediction is rule-attributable: a user can see
which transformation produced a candidate and where it fired, and reject it on chemical grounds
rather than on a score. That claim was asserted twice and shown nowhere, and the released pools
could not show it -- they carry the structure, the two component scores and the matching key,
because `build_val_pools` calls the generator with `compute_sites=False` and keeps `d[0]` and
`d[1]` of a four-tuple. The rule identity and the firing atoms are computed by the pipeline and
discarded on the way to the artifact.

This runs the deployed interactive configuration on one substrate and keeps all four fields: the
rule budget the checkpoint records, the H9 cap, RRF over the filter and generator orderings, and
the tautomer-aware key deciding which candidates are annotated. The default substrate is
gemcitabine, whose four annotated metabolites are one deamination and three sequential
phosphorylations, so the example holds transformations a reader can check by eye.

    python scripts/typed_edit/case_study.py
    python scripts/typed_edit/case_study.py --substrate "CCO" --show 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

# Gemcitabine exactly as the corpus stores it. That string is not the standardiser's output and
# not how a chemist draws the drug: every corpus structure is the InChI round-trip of its own
# record, and the round-trip places the mobile hydrogen on oxygen, so cytosine arrives as the
# 4-imino-2-hydroxy lactim. Stereochemistry is absent for a separate reason -- InChI's standard
# layer is written without it here, and the pipeline strips it anyway.
GEMCITABINE = "N=c1ccn(C2OC(CO)C(O)C2(F)F)c(O)n1"
# The same molecule as a chemist draws it, which is also the fixed point of the declared
# standardiser: 4-amino-2-oxo.
GEMCITABINE_DRAWN = "Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default=GEMCITABINE,
                    help="the corpus key: the string the annotation is filed under")
    ap.add_argument("--present", choices=("stored", "standardised"), default="stored",
                    help="how the substrate is handed to the matcher -- as the corpus stores it, "
                         "or as the declared standardiser draws it. The references are looked up "
                         "under the corpus key either way, so the two runs are scored against the "
                         "same annotation.")
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/filter.pt"))
    ap.add_argument("--top-k", type=int, default=30,
                    help="the interactive mode's rule budget, which the checkpoint records")
    ap.add_argument("--cap", type=int, default=100, help="the H9 pool cap")
    ap.add_argument("--show", type=int, default=15,
                    help="rows written to the table; the artifact keeps the whole ranked pool")
    ap.add_argument("--out", default=str(ROOT / "results/case_study.json"))
    args = ap.parse_args()

    import torch  # noqa: F401
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    from bank_without_selection import _key, _load
    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.workflows.factory import build_filter, build_generator

    corpus_key = args.substrate
    if Chem.MolFromSmiles(corpus_key) is None:
        raise SystemExit(f"unparseable substrate: {corpus_key}")
    if args.present == "standardised":
        from grail_metabolism.utils.preparation import standardize_mol
        s = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(corpus_key)))
    else:
        s = corpus_key
    sub_mol = Chem.MolFromSmiles(s)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    # the references, read from the frozen pools rather than recomputed, so the example is scored
    # against the same annotation the comparison table is scored against
    import glob
    refs = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        refs.update(json.loads(Path(f).read_text())["references"])
    reference_keys = set(refs.get(corpus_key, []))
    if not reference_keys:
        raise SystemExit(f"no references filed under the corpus key {corpus_key}")

    t0 = time.perf_counter()
    det = generator.generate_scored_with_details(s, top_k=args.top_k, threshold=None,
                                                 compute_sites=True)
    gen_seconds = time.perf_counter() - t0
    det.sort(key=lambda d: (-d[1], d[0]))
    det = det[:args.cap]

    cands = [d[0] for d in det]
    fs = filt.score_batch(s, cands) if cands else []

    # generator.rules maps the SMIRKS text to its graph; rule_names is the list the rule index
    # addresses, and it is the ordering rule_reactions was compiled in
    rules = generator.rule_names
    # curated or mined is decided by membership in the mined file, not by an index threshold:
    # the bank is a deduplicated union and nothing guarantees the curated rules come first
    mined = {ln.strip() for ln in
             (ROOT / "grail_metabolism/resources/mined_only.txt").read_text().splitlines()
             if ln.strip()}
    rows, seen = [], set()
    for (smiles, gscore, rule_id, sites), fscore in zip(det, fs):
        k = _key(smiles)
        if not k or k in seen:
            continue
        seen.add(k)
        rows.append({
            "smiles": smiles, "key": k,
            "generator": round(float(gscore), 4), "filter": round(float(fscore), 4),
            "rule_id": int(rule_id),
            "rule": rules[rule_id] if 0 <= rule_id < len(rules) else None,
            "rule_source": "mined" if (0 <= rule_id < len(rules)
                                       and rules[rule_id] in mined) else "curated",
            "firing_atoms": list(sites),
            "is_reference": k in reference_keys,
        })

    ranked = rrf_order(rows)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    found = [r["rank"] for r in ranked if r["is_reference"]]
    missing = sorted(reference_keys - {r["key"] for r in ranked})

    rep = {
        "provenance": stamp(__file__),
        "substrate": s,
        "corpus_substrate": corpus_key,
        "presentation": args.present,
        "substrate_heavy_atoms": sub_mol.GetNumHeavyAtoms(),
        "configuration": {
            "mode": "interactive", "rule_budget": args.top_k, "pool_cap": args.cap,
            "substrate_presentation": args.present,
            "ranking": "reciprocal rank fusion of the filter and generator orderings, k=60",
            "match": "inchikey_tautomer",
            "generator_checkpoint": str(Path(args.gen_ckpt).relative_to(ROOT)),
            "filter_checkpoint": str(Path(args.filter_ckpt).relative_to(ROOT)),
        },
        "generator_seconds": round(gen_seconds, 2),
        "n_candidates": len(ranked),
        "n_references": len(reference_keys),
        "reference_ranks": found,
        "references_not_in_pool": missing,
        "recall_at": {k: sum(r <= k for r in found) / max(len(reference_keys), 1)
                      for k in (1, 5, 10, 15, 30, 50)},
        "note": ("rule_id indexes the deployed bank and rule is its SMIRKS; firing_atoms are "
                 "substrate atom indices, from the generator's per-product localisation, which "
                 "the pool artifacts omit because they are built with compute_sites=False"),
        "candidates": ranked,
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"substrate: {s}  ({rep['substrate_heavy_atoms']} heavy atoms, "
          f"presented {args.present}; filed under {corpus_key})")
    print(f"{len(ranked)} candidates from a budget of {args.top_k} rules in {gen_seconds:.1f}s; "
          f"{len(reference_keys)} annotated, found at ranks {found}"
          + (f"; {len(missing)} not in the pool" if missing else ""))
    print(f"\n{'rank':>4} {'ref':>4} {'rule':>6} {'gen':>7} {'filt':>7}  sites  structure")
    for r in ranked[:args.show]:
        mark = "  *" if r["is_reference"] else "   "
        sites = ",".join(str(a) for a in r["firing_atoms"][:4]) or "-"
        print(f"{r['rank']:>4} {mark:>4} {r['rule_id']:>6} {r['generator']:>7.4f} "
              f"{r['filter']:>7.4f}  {sites:<6} {r['smiles']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
