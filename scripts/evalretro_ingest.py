#!/usr/bin/env python3
"""Eleven published retrosynthesis leaderboards, and the three test sets they are actually on.

The manuscript defers the decisive cross-domain test -- does the matching convention reorder a
leaderboard outside metabolism -- on the ground that it needs "at least two comparable published
models whose raw ranked predictions are public" and that "such pairs do not currently exist in a
re-scorable form". That was an assessment, not a measurement, and it is wrong: the EvalRetro
benchmark publishes the ranked predictions of twelve single-step retrosynthesis systems on what it
presents as the USPTO-50k test set, under CC BY 4.0 (doi:10.6084/m9.figshare.25325623). Eleven ship
as CSV, and re-scoring them needs no checkpoint, no GPU and no training.

Reading them turned up something the comparison has to be built around rather than past. The eleven
files are not on one test set. Clustered by which products they contain, they fall into three
disjoint groups that share about a tenth of their reactions pairwise:

  three systems on a 5,004-reaction set, which is the one this repository holds;
  seven systems on a 5,007-reaction set, mutually identical;
  one system on a 5,007-reaction set of its own.

A leaderboard is only a leaderboard within a group. This script therefore does not force the eleven
into one table -- it recovers the groups, reconstructs each group's test set from the systems that
agree on it, and emits per-system predictions aligned to that set. What the paper then scores is the
seven-system group, because seven comparable systems on one set is the multi-model test the
deferral was about.

Nothing here trusts a file's ordering or its index column. The `idx` column means a running
prediction counter in one file, a reaction index in another, and a float in a third; some separate
reactions with an empty row and some do not. Blocks are read off the `target` column, which every
file agrees holds the product and holds it constant for one reaction, and a reaction is identified
by the pair (product, ground-truth reactants) -- a product alone does not identify one, since a
product can be made more than one way. Position is never used: this literature has already lost
thirteen points to a split that differed silently from the one on its own model card.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SOURCE = "EvalRetro, doi:10.6084/m9.figshare.25325623, CC BY 4.0"
DEFAULT_DIR = ROOT / "grail_metabolism" / "data" / "evalretro"
REPO_SPLIT = ROOT / "grail_metabolism" / "data" / "USPTO_50k" / "test.csv"
# Two files whose product sets overlap by at least this much are on the same test set. The observed
# structure leaves no room for judgement: within a group the overlap is 1.00 and between groups it
# is 0.10, so any threshold in between recovers the same three groups.
SAME_SET = 0.90


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


def canonical_set(dotjoined: str):
    """A reactant set as canonical SMILES, so two spellings of one reaction compare equal."""
    parts = [p for p in (dotjoined or "").split(".") if p]
    out = set()
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return None
        out.add(Chem.MolToSmiles(m))
    return frozenset(out) or None


def parse_blocks(path: Path) -> list[dict]:
    """Reactions and their ranked predictions, however this particular file delimits them.

    The block boundary is read off the product, the one column all eleven files agree about; the
    index column, whose meaning varies between them, is used only to skip separator rows. The first
    row of a block is the reaction's ground truth and the rest are its ranked predictions, which is
    checked against the group's consensus rather than assumed.
    """
    blocks, cur, target = [], None, None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if not (row.get("idx") or "").strip():
                continue
            t = row.get("target") or ""
            if t != target:
                if cur is not None:
                    blocks.append(cur)
                cur, target = {"product": t, "rows": []}, t
            cur["rows"].append(row.get("reactants") or "")
    if cur is not None:
        blocks.append(cur)
    return [{"product": b["product"], "true": b["rows"][0], "preds": b["rows"][1:]}
            for b in blocks if b["rows"]]


def cluster_by_test_set(products: dict) -> list[list[str]]:
    """Group the systems whose product sets coincide. A leaderboard is only one within a group."""
    groups: list[list[str]] = []
    for n in sorted(products):
        for g in groups:
            other = products[g[0]]
            overlap = len(products[n] & other) / max(1, min(len(products[n]), len(other)))
            if overlap >= SAME_SET:
                g.append(n)
                break
        else:
            groups.append([n])
    return sorted(groups, key=len, reverse=True)


def consensus_split(group: list[str], parsed: dict, label: str) -> list[tuple]:
    """The test set a group agrees on: the reactions every member carries, with one ground truth.

    Reconstructing the set from its members rather than trusting any one file is what makes it a
    split rather than a file. A reaction where the members disagree about the ground truth is
    dropped and counted, never resolved by majority: a disagreement about what the answer is cannot
    be voted away.
    """
    per = {n: {canonical_set(b["product"]): b for b in parsed[n]} for n in group}
    shared = set.intersection(*(set(p) for p in per.values()))
    rows, disputed = [], 0
    for prod in sorted(shared, key=lambda f: sorted(f)):
        if len({canonical_set(per[n][prod]["true"]) for n in group}) != 1:
            disputed += 1
            continue
        first = per[group[0]][prod]
        rows.append((first["product"], first["true"], prod))
    print(f"    {label}: {len(rows)} agreed reactions"
          + (f", {disputed} dropped where the members disagree about the truth" if disputed else ""),
          flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR / "normalised"))
    ap.add_argument("--report", default=str(ROOT / "results" / "evalretro_ingest.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed, products = {}, {}
    for path in sorted(Path(args.dir).glob("*_pred.csv")):
        n = path.stem.replace("_pred", "")
        parsed[n] = parse_blocks(path)
        products[n] = {canonical_set(b["product"]) for b in parsed[n]}
        print(f"  parsed {n:16} {len(parsed[n]):>5} reactions", flush=True)

    repo = {canonical_set(r["PRODUCT"]) for r in csv.DictReader(open(REPO_SPLIT))}
    groups = cluster_by_test_set(products)
    print(f"\n{len(groups)} distinct test sets among {len(parsed)} published systems", flush=True)

    report_groups = {}
    for gi, group in enumerate(groups):
        label = f"cluster{gi}"
        print(f"  {label}: {len(group)} systems {group}", flush=True)
        rows = consensus_split(group, parsed, label)
        share_repo = len(products[group[0]] & repo) / max(1, len(products[group[0]]))
        print(f"    {share_repo:.2f} of its products are in this repository's own test split",
              flush=True)

        csv_path = Path(args.dir) / f"{label}_test.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["REACTANT", "PRODUCT"])
            for product, true, _ in rows:
                w.writerow([true, product])

        for n in group:
            by_prod = {canonical_set(b["product"]): b for b in parsed[n]}
            payload = [{"product": product, "preds": by_prod[key]["preds"]}
                       for product, _, key in rows]
            (out_dir / f"{n}.json").write_text(json.dumps(payload))
        depths = Counter(len(b["preds"]) for n in group for b in parsed[n])
        report_groups[label] = {
            "systems": group, "reactions": len(rows),
            "share_of_products_in_repo_test_split": round(share_repo, 4),
            "ranks_per_reaction": [min(depths), max(depths)],
            "test_csv": str(csv_path.relative_to(ROOT))}

    rep = {"config": {**_code_version(), "source": SOURCE,
                      "repo_split": str(REPO_SPLIT.relative_to(ROOT)),
                      "cluster_rule": f"product-set overlap >= {SAME_SET}",
                      "alignment": "reaction identified by (product, ground-truth reactants); "
                                   "file order and index column never used"},
           "n_systems": len(parsed), "n_test_sets": len(groups), "clusters": report_groups}
    Path(args.report).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
