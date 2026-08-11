#!/usr/bin/env python3
"""Are the rules labelled in the convention the pipeline fires them in?

This paper's claim is that a rule bank's reach is a property of bank and engine together, and that
a figure quoted without its application procedure is not comparable. The claim applies to us. The
generator is supervised by a label matrix that says, for each substrate and rule, whether applying
that rule yields an annotated metabolite; the deployed pipeline then fires the selected rules to
enumerate candidates. If the two use different hydrogen conventions the selector is taught about
rules other than the ones it will fire, and the selection factor of the decomposition measures a
mismatch as well as a policy.

Reading the code settles which convention each side uses. This settles how much it costs:

    positives that exist under one convention and not the other, per substrate and per rule
    the share of the label matrix that changes
    how the two halves of the bank move, since they are written in different conventions

Nothing here retrains anything. It applies the bank twice to the same substrates and compares the
two label matrices, which is exactly the quantity the supervision depends on.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger

from _contract import contract
from grail_metabolism.utils.preparation import (_clean_product_smiles, _iter_reaction_products,
                                                _normalize_smiles_cached, load_default_rules)

RDLogger.DisableLog("rdApp.*")

_CTX: dict = {}


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _init(rules):
    # `_iter_reaction_products` takes the rule as a SMIRKS string and compiles both a match pattern
    # and a reaction from it. Handing it a pre-compiled reaction makes its pattern compile fail and
    # every rule returns nothing, silently: the run reports zero positives under both conventions,
    # which reads like a measurement and is a broken one.
    _CTX["rules"] = list(rules)


def _productive(substrate, refs, contract_first=False) -> set[int]:
    """Indices of rules that yield an annotated metabolite from this presentation of the substrate.

    `contract_first` puts back the hydrogens an expansion drew before the product is read. Without
    it an expanded arm measures an unfinished loop as well as a convention, which is the split
    Section 4 of the paper reports; here it separates how much of the label disagreement is the
    convention from how much is the missing call.
    """
    out = set()
    for i, rule in enumerate(_CTX["rules"]):
        for product in _iter_reaction_products(substrate, rule):
            if contract_first:
                try:
                    product = contract(product)
                except Exception:
                    continue
            try:
                smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            for fragment in _clean_product_smiles(smiles):
                # Normalising a fragment can raise on exactly the products this audit is about:
                # an expanded substrate yields rings the toolkit will not kekulise. A fragment that
                # cannot be normalised cannot equal a normalised reference either, so it is not a
                # match in either arm, and swallowing it here favours neither convention.
                try:
                    key = _normalize_smiles_cached(fragment, "standardize")
                except Exception:
                    continue
                if key in refs:
                    out.add(i)
                    break
            if i in out:
                break
    return out


def _worker(item):
    sub, refs = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not refs:
        return sub, None, None, None
    keys = set()
    for r in refs:
        try:
            k = _normalize_smiles_cached(r, "standardize")
        except Exception:
            continue
        if k:
            keys.add(k)
    if not keys:
        return sub, None, None, None
    implicit = _productive(Chem.Mol(mol), keys)              # the convention the pipeline fires in
    expanded = _productive(Chem.AddHs(Chem.Mol(mol)), keys)  # the convention the labels are built in
    completed = _productive(Chem.AddHs(Chem.Mol(mol)), keys, contract_first=True)
    return sub, sorted(implicit), sorted(expanded), sorted(completed)


def load_training_pairs(limit: int, seed: int) -> list[tuple[str, list[str]]]:
    """Substrate -> annotated metabolites, from the clean training triples."""
    import random
    from grail_metabolism.utils.preparation import MolFrame  # noqa: F401  (import parity with training)

    sdf = ROOT / "grail_metabolism/data/train.sdf"
    triples = ROOT / "grail_metabolism/data/train_triples_clean.txt"
    if not sdf.exists() or not triples.exists():
        raise SystemExit("training split not present; this audit needs train.sdf and its triples")
    supplier = Chem.SDMolSupplier(str(sdf))
    smiles = []
    for m in supplier:
        smiles.append(Chem.MolToSmiles(m) if m is not None else None)
    pairs: dict[str, list[str]] = {}
    for line in triples.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[2] != "1":
            continue
        a, b = int(parts[0]), int(parts[1])
        if a >= len(smiles) or b >= len(smiles) or smiles[a] is None or smiles[b] is None:
            continue
        pairs.setdefault(smiles[a], []).append(smiles[b])
    items = sorted(pairs.items())
    if limit and len(items) > limit:
        random.Random(seed).shuffle(items)
        items = sorted(items[:limit])
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150,
                    help="substrates to audit; the full training set is the same measurement, slower")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default=str(ROOT / "results" / "label_convention_audit_3arm.json"),
                    help="the two-arm artifact this script used to write backs certified text and "
                         "is not overwritten: this version emits a third arm and writes beside it")
    args = ap.parse_args()

    rules = load_default_rules()
    items = load_training_pairs(args.limit, args.seed)
    print(f"{len(items)} training substrates with annotated metabolites, {len(rules)} rules, "
          f"applied under both conventions", flush=True)

    rows = []
    with multiprocessing.get_context("spawn").Pool(args.workers, _init, (rules,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_worker, items, 1), 1):
            if r[1] is not None:
                rows.append(r)
            if n % 10 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    n_impl = sum(len(r[1]) for r in rows)
    n_exp = sum(len(r[2]) for r in rows)
    n_comp = sum(len(r[3]) for r in rows)
    both = sum(len(set(r[1]) & set(r[2])) for r in rows)
    both_comp = sum(len(set(r[1]) & set(r[3])) for r in rows)
    union_comp = sum(len(set(r[1]) | set(r[3])) for r in rows)
    subs_disagree_comp = sum(1 for r in rows if set(r[1]) != set(r[3]))
    only_impl = n_impl - both
    only_exp = n_exp - both
    union = both + only_impl + only_exp
    subs_disagree = sum(1 for r in rows if set(r[1]) != set(r[2]))

    # The frequency prior is the per-rule share of positives in this matrix, so a matrix built in
    # the other convention is a different prior over different rules. The learned-against-prior
    # comparison has both arms reading it, which is why this is computed here rather than assumed.
    from collections import Counter
    imp_c, exp_c = Counter(), Counter()
    for _, i_, e_, _c in rows:
        imp_c.update(i_)
        exp_c.update(e_)
    allr = sorted(set(imp_c) | set(exp_c))
    def _rank(c):
        order = sorted(allr, key=lambda r: (-c.get(r, 0), r))
        return {r: i for i, r in enumerate(order)}
    ra, rb = _rank(imp_c), _rank(exp_c)
    n_u = len(allr)
    mu = (n_u - 1) / 2 if n_u else 0.0
    num = sum((ra[r] - mu) * (rb[r] - mu) for r in allr)
    den = (sum((ra[r] - mu) ** 2 for r in allr) * sum((rb[r] - mu) ** 2 for r in allr)) ** 0.5
    prior = {"rules_positive_where_it_fires": len(imp_c),
             "rules_positive_where_it_is_labelled": len(exp_c),
             "rules_that_fire_but_are_never_labelled": len(set(imp_c) - set(exp_c)),
             "rules_labelled_that_never_fire": len(set(exp_c) - set(imp_c)),
             "spearman_between_the_two_priors": round(num / den, 4) if den else None,
             "top_k_overlap": {str(k): len({r for r, _ in imp_c.most_common(k)} &
                                           {r for r, _ in exp_c.most_common(k)})
                               for k in (10, 30, 100)}}
    print(f"\n  frequency prior: {prior['rules_that_fire_but_are_never_labelled']} rules fire "
          f"productively and are never labelled positive; Spearman "
          f"{prior['spearman_between_the_two_priors']}")

    # The expanded arm as the labels are built today never contracts the product, so it carries the
    # unfinished loop as well as the convention. Completing it separates the two: what remains
    # between the completed arm and the fired one is the convention alone.
    completed_block = {
        "positives_completed": n_comp,
        "agreeing_with_the_fired_convention": both_comp,
        "jaccard": round(both_comp / max(union_comp, 1), 4),
        "substrates_whose_label_row_changes": subs_disagree_comp,
        "note": "the arm the labels would have if the expansion were completed; the gap from this "
                "to the fired convention is the convention alone, the gap from the uncompleted "
                "arm additionally carries the missing contraction"}
    rep = {"frequency_prior": prior, "completed_expansion": completed_block,
           "config": {**_code_version(), "n_substrates_requested": args.limit, "seed": args.seed,
                      "n_substrates_scored": len(rows), "n_rules": len(rules),
                      "split": "train, clean triples",
                      "note": "a rule counts as positive for a substrate when applying it under "
                              "that convention yields an annotated metabolite; this is what "
                              "MolFrame.label_reactions computes, and it computes it expanded"},
           "positives": {"implicit_only": only_impl, "expanded_only": only_exp,
                         "both": both, "union": union,
                         "implicit_total": n_impl, "expanded_total": n_exp},
           "agreement": {"jaccard": round(both / max(union, 1), 4),
                         "share_of_union_lost_by_expanding": round(only_impl / max(union, 1), 4),
                         "share_of_union_added_by_expanding": round(only_exp / max(union, 1), 4),
                         "substrates_whose_label_row_changes": subs_disagree,
                         "substrates_scored": len(rows)},
           "per_substrate": [{"substrate": s, "implicit": i, "expanded": e, "completed": c}
                             for s, i, e, c in rows]}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n  positives under the fired convention: {n_impl}")
    print(f"  positives under the labelled convention: {n_exp}")
    print(f"  agreeing: {both}   fired-only: {only_impl}   labelled-only: {only_exp}")
    print(f"  Jaccard between the two label matrices: {rep['agreement']['jaccard']}")
    print(f"  substrates whose label row changes: {subs_disagree} of {len(rows)}")
    print(f"\n  completing the expansion: {n_comp} positives, Jaccard against the fired convention "
          f"{completed_block['jaccard']}, rows still differing {subs_disagree_comp} of {len(rows)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
