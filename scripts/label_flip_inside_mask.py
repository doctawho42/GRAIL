#!/usr/bin/env python3
"""How many training labels actually change, once the mask that gates the gradient is applied?

The label matrix is built by expanding the substrate with explicit hydrogens; every deployed firing
path passes the molecule as parsed. Correcting that changes the matrix, and the raw size of the
change is not the size of the effect a retrain could show: the generator masks its logits by whether
each rule's reactant pattern matches the substrate as parsed, and multiplies its loss by the same
mask. A label that flips on a rule the mask already zeroes carried no gradient before and carries
none after. Reporting the raw count would overstate the headroom, and a null result would then be
misattributed to the model rather than to there having been little to change.

The two label rows are not recomputed here. results/label_convention_audit.json already holds them
per substrate, at two full passes over the bank each; the mask is a SMARTS match and costs seconds.
Reading one and computing the other is the same measurement for a hundredth of the compute, and it
is also the only version that is guaranteed to intersect the rows the audit actually reported.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger

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
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = list(rules)
    # The same patterns the generator masks with, compiled the same way. Taking the template out of
    # a compiled reaction instead -- ReactionFromSmarts(...).GetReactantTemplate(0) -- returns a
    # query molecule RDKit has not sanitised, and matching against it segfaults the interpreter:
    # a worker pool respawns silently around that, so a run appears to hang rather than to crash.
    patterns = []
    for rule in _CTX["rules"]:
        try:
            patterns.append(Chem.MolFromSmarts(rule.split(">>")[0]))
        except Exception:
            patterns.append(None)
    _CTX["patterns"] = patterns


def _worker(item):
    substrate, implicit, expanded = item
    mol = Chem.MolFromSmiles(substrate)
    if mol is None:
        return None
    mask = set()
    for i, pattern in enumerate(_CTX["patterns"]):
        if pattern is None:
            continue
        try:
            if mol.GetSubstructMatches(pattern, uniquify=True, maxMatches=4):
                mask.add(i)
        except Exception:
            continue
    implicit, expanded = set(implicit), set(expanded)
    return {"substrate": substrate, "mask": len(mask),
            "implicit": len(implicit), "expanded": len(expanded),
            "implicit_in_mask": len(implicit & mask), "expanded_in_mask": len(expanded & mask),
            "gained_in_mask": len((implicit - expanded) & mask),
            "lost_in_mask": len((expanded - implicit) & mask),
            "gained_outside": len((implicit - expanded) - mask),
            "lost_outside": len((expanded - implicit) - mask),
            "row_changes": bool(implicit != expanded),
            "row_changes_in_mask": bool((implicit & mask) != (expanded & mask))}


def load_pairs(audit_path: Path) -> list:
    """Substrate with its two label rows, straight from the audit that computed them."""
    audit = json.loads(audit_path.read_text())
    return [(row["substrate"], row["implicit"], row["expanded"])
            for row in audit["per_substrate"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(ROOT / "results" / "label_convention_audit.json"),
                    help="the artifact holding both label rows per substrate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default=str(ROOT / "results" / "label_flip_inside_mask.json"))
    args = ap.parse_args()

    rules = load_default_rules()
    items = load_pairs(Path(args.audit))
    print(f"{len(items)} training substrates from {Path(args.audit).name}, {len(rules)} rules; "
          f"label rows read, mask computed", flush=True)

    rows = []
    with multiprocessing.get_context("spawn").Pool(args.workers, _init, (rules,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_worker, items, 1), 1):
            if r is not None:
                rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)

    def total(key):
        return sum(r[key] for r in rows)

    n = max(len(rows), 1)
    rep = {"config": {**_code_version(), "labels_from": str(Path(args.audit).name),
                      "n_substrates_scored": len(rows), "n_rules": len(rules),
                      "split": "train, clean triples",
                      "mask": "reactant template matches the substrate as parsed, which is what "
                              "Generator._rule_applicability computes and the loss multiplies by"},
           "positives": {"implicit": total("implicit"), "expanded": total("expanded"),
                         "implicit_in_mask": total("implicit_in_mask"),
                         "expanded_in_mask": total("expanded_in_mask")},
           "flips_inside_the_mask": {"gained": total("gained_in_mask"),
                                     "lost": total("lost_in_mask"),
                                     "net": total("gained_in_mask") - total("lost_in_mask")},
           "flips_outside_the_mask": {"gained": total("gained_outside"),
                                      "lost": total("lost_outside")},
           "rows": {"changed": sum(r["row_changes"] for r in rows),
                    "changed_inside_the_mask": sum(r["row_changes_in_mask"] for r in rows),
                    "scored": len(rows),
                    "share_changed_inside_the_mask":
                        round(sum(r["row_changes_in_mask"] for r in rows) / n, 4)},
           "gradient_carrying_multiplier":
               round(total("implicit_in_mask") / max(total("expanded_in_mask"), 1), 3),
           "per_substrate": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    f = rep["flips_inside_the_mask"]
    print(f"\n  positives that carry gradient: {rep['positives']['expanded_in_mask']} today, "
          f"{rep['positives']['implicit_in_mask']} corrected "
          f"({rep['gradient_carrying_multiplier']}x)")
    print(f"  inside the mask: {f['gained']} gained, {f['lost']} lost, net {f['net']:+}")
    print(f"  outside it, where no gradient flows: "
          f"{rep['flips_outside_the_mask']['gained']} gained, "
          f"{rep['flips_outside_the_mask']['lost']} lost")
    print(f"  rows that change at all: {rep['rows']['changed']} of {len(rows)}; "
          f"inside the mask: {rep['rows']['changed_inside_the_mask']} "
          f"({rep['rows']['share_changed_inside_the_mask']})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
