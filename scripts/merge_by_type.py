#!/usr/bin/env python3
"""The free first stage of the type-merge (queue point 7): what merging by type would pool.

The bank is 7,581 templates over 4,417 canonical types, and noisy-or treats same-type templates as
independent witnesses. Merging a type's templates into one would make that hold by construction and
pool their training support. This counts what the pooling would be -- no model, no generalised
environment yet -- so the training is only spent if the pooling is real: how many types carry more
than one template, and how the per-unit training support moves from per-template to per-type.

Support per rule is the training positive count from the label cache that reproduces the published
never/eq1/ge2 counts, the same source the shrinkage experiment used.

    python scripts/merge_by_type.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B  # noqa: E402
from grail_metabolism.model.reaction_types import canonical_type  # noqa: E402

BANK = ROOT / "grail_metabolism/resources/extended_smirks.txt"
LABEL_CACHE = "artifacts/preprocessed/train/5a3f22a1e5962bbb/reaction_labels.expanded.pt"
PUBLISHED = {"never_positive": 4271, "pos_eq_1": 1520, "pos_ge2": 1790}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "merge_by_type.json"))
    args = ap.parse_args()

    rules = [ln.split()[0] for ln in BANK.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    d = torch.load(ROOT / LABEL_CACHE, map_location="cpu", weights_only=False)
    M = np.asarray([np.asarray(d[s]).ravel() for s in d], dtype=np.int64)
    support = M.sum(axis=0)
    got = {"never_positive": int((support == 0).sum()), "pos_eq_1": int((support == 1).sum()),
           "pos_ge2": int((support >= 2).sum())}
    if got != PUBLISHED or len(support) != len(rules):
        raise SystemExit(f"label cache does not line up with the bank: {got} vs {PUBLISHED}, "
                         f"{len(support)} supports vs {len(rules)} rules")

    # group rule indices by canonical type
    by_type = defaultdict(list)
    untypeable = 0
    for i, r in enumerate(rules):
        t = canonical_type(r)
        if t is None:
            untypeable += 1
            continue
        by_type[json.dumps(t, sort_keys=True)].append(i)

    sizes = [len(v) for v in by_type.values()]
    multi = [v for v in by_type.values() if len(v) > 1]
    per_rule = support.tolist()
    per_type = [int(support[v].sum()) for v in by_type.values()]

    def share_le(vals, k):
        return round(sum(1 for x in vals if x <= k) / len(vals), 4)

    rep = {"config": {**B._code_version(), "bank": str(BANK.relative_to(ROOT)),
                      "support_source": LABEL_CACHE},
           "rules": len(rules), "untypeable_rules": untypeable,
           "distinct_types": len(by_type),
           "types_with_more_than_one_template": len(multi),
           "rules_in_multi_template_types": sum(len(v) for v in multi),
           "largest_type_templates": max(sizes),
           "support_per_template": {
               "median": st.median(per_rule), "mean": round(st.mean(per_rule), 2),
               "share_le_1": share_le(per_rule, 1), "share_eq_0": round(float((support == 0).mean()), 4)},
           "support_per_type_after_merge": {
               "median": st.median(per_type), "mean": round(st.mean(per_type), 2),
               "share_le_1": share_le(per_type, 1)},
           "support_argument_holds": bool(share_le(per_type, 1) < share_le(per_rule, 1)
                                            and st.median(per_type) > st.median(per_rule)),
           "reading": (
               "the support argument does NOT hold: the median per-unit support is "
               f"{st.median(per_rule)} per template and {st.median(per_type)} per type, and the "
               f"share resting on <=1 pair RISES from {share_le(per_rule,1)} to {share_le(per_type,1)} "
               "rather than falling, because most types carry a single template and their share of "
               "the type space is higher than of the rule space. Merging pools support only in the "
               f"{len(multi)} types with more than one template; the bank-wide support does not "
               "improve. What the merge still buys is independence -- those "
               f"{len(multi)} types stop being counted as multiple witnesses by noisy-or -- so if "
               "point 7 is run it is for the independence, not the support, and the coverage of the "
               "merged bank (needing the generalised environment) is the deciding next stage."
               if not (share_le(per_type, 1) < share_le(per_rule, 1))
               else "merging pools support and lowers the low-support share")}

    print(f"  {len(rules)} rules -> {len(by_type)} types ({untypeable} untypeable)")
    print(f"  {len(multi)} types carry >1 template, holding {sum(len(v) for v in multi)} rules; "
          f"largest type has {max(sizes)}")
    print(f"  support per template: median {st.median(per_rule)}, <=1 pair {share_le(per_rule,1)}")
    print(f"  support per type after merge: median {st.median(per_type)}, <=1 pair {share_le(per_type,1)}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
