#!/usr/bin/env python3
"""What each arm actually hands back: an ordering, a named transformation, an enzyme, a site.

The manuscript claimed two things about the incumbent it sets out to replace: that this work
names the rule and the site behind every prediction, offered under a heading of what the
incumbent does not do, and that the incumbent "ranks part of its output and leaves the remainder
in file order". Neither was measured. The second is contradicted by the very file the comparison
was run on, in which every returned structure carries the service's own score.

Attribution and ordering are properties of an output format, and this repository holds the output
of all four comparators. So they are read off those files rather than argued about. Two things
are separated deliberately:

  measured here    what the file this repository holds actually carries. That is the only thing a
                   comparison run from these files can claim, and it is what the paper prints.
  from the method  what the method's own publication says it reports. It is recorded beside the
                   measurement, marked as a citation and never merged into it, because a channel
                   a submission route did not retain is not a channel the method lacks.

    python scripts/typed_edit/what_each_arm_returns.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

# Column names that carry an attribution rather than a structure or a bulk property. Matching is
# case-insensitive and exact, so a "Reaction ID" counts as naming the transformation and a
# "Precursor SMILES" does not accidentally count as anything.
TRANSFORMATION_COLUMNS = {"reaction", "reaction id", "biotransformation", "transformation"}
ENZYME_COLUMNS = {"enzyme(s)", "enzyme", "enzymes", "biosystem"}
SITE_COLUMNS = {"site", "site of metabolism", "som", "atoms"}


def metatox() -> dict:
    """MetaTox: the frozen return, which carries the service's own score for every structure."""
    blob = json.loads((ROOT / "results/metatox_smirks_preds.json").read_text())
    preds, scored = blob["predictions"], blob["predictions_with_scores"]
    returned = sum(len(v) for v in preds.values())
    with_score = sum(1 for s in scored for row in scored[s]
                     if len(row) > 1 and row[1] is not None)
    return {
        "file": "results/metatox_smirks_preds.json",
        "structures_returned": returned,
        "structures_carrying_a_score": with_score,
        "score_channel": blob.get("ranking"),
        "names_the_transformation": False,
        "names_an_enzyme": False,
        "names_a_site": False,
        "from_the_method": ("its publication reports a biotransformation type and a site of "
                            "metabolism; the file returned to this submission carries neither"),
    }


def sygma() -> dict:
    """SyGMa: the frozen list, ordered by the engine's own score before it was written out."""
    blob = json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text())
    returned = sum(len(v) for v in blob.values())
    return {
        "file": "results/sygma_fulltest_predictions.json",
        "structures_returned": returned,
        # The score was applied and then discarded: `tree.calc_scores()` runs before
        # `to_smiles()`, so the order is the ranking and the number behind it was not kept.
        "structures_carrying_a_score": 0,
        "score_channel": "the engine's own score, applied as an order and not retained",
        "ordered": True,
        "names_the_transformation": False,
        "names_an_enzyme": False,
        "names_a_site": False,
        "from_the_method": ("its rule set names each transformation, and the engine exposes the "
                            "applied rule per node; this run kept the ordered structures only"),
    }


def metapredictor() -> dict:
    blob = json.loads((ROOT / "artifacts/tier2_1170/metapredictor_preds.json").read_text())
    returned = sum(len(v) for v in blob.values())
    return {
        "file": "artifacts/tier2_1170/metapredictor_preds.json",
        "structures_returned": returned,
        "structures_carrying_a_score": 0,
        "score_channel": None,
        "ordered": False,
        "names_the_transformation": False,
        "names_an_enzyme": False,
        "names_a_site": False,
        "from_the_method": "a sequence-to-sequence translation; it emits structures and no rule",
    }


def biotransformer() -> dict:
    """BioTransformer: the CSV it wrote, whose columns are the attribution channel."""
    path = ROOT / "artifacts/tier2/bt_out.csv"
    cols, rows = [], 0
    if path.exists():
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            cols = [c.strip().lower() for c in next(reader, [])]
            rows = sum(1 for _ in reader)
    score_cols = [c for c in cols if "score" in c or c in ("probability", "confidence")]
    return {
        "file": "artifacts/tier2/bt_out.csv",
        "structures_returned": rows,
        "structures_carrying_a_score": 0 if not score_cols else rows,
        "score_channel": score_cols[0] if score_cols else None,
        "ordered": False,
        "names_the_transformation": bool(set(cols) & TRANSFORMATION_COLUMNS),
        "names_an_enzyme": bool(set(cols) & ENZYME_COLUMNS),
        "names_a_site": bool(set(cols) & SITE_COLUMNS),
        "columns": cols,
        "from_the_method": "its output names the reaction and the enzyme, and no atom indices",
    }


def ours() -> dict:
    """This work's own arm, scores from the pools and attribution from the arm's own output.

    The frozen pools keep a structure and its two scores and nothing else, because that is all
    scoring a comparison needs. The rule and the site are carried by the system's output rather
    than by the pools, so the attribution is read where it exists: the site instrument recovers
    the template and the atoms it fired on for every match it scores, and its scored count equal
    to its matched count is the evidence that neither channel is ever missing.
    """
    pools = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        pools.update(json.loads(Path(f).read_text())["pools"])
    cands = [c for pool in pools.values() for c in pool]
    site = json.loads((ROOT / "results/site_agreement.json").read_text())["counts"]
    attributed = site["scored"] == site["matched"] and site["matched"] > 0
    return {
        "file": "results/widepools_implicit/w*.json",
        "structures_returned": len(cands),
        "structures_carrying_a_score": sum(
            1 for c in cands if c.get("generator") is not None and c.get("filter") is not None),
        "score_channel": "both component scores, on every candidate",
        "ordered": True,
        "names_the_transformation": attributed,
        "names_an_enzyme": False,
        "names_a_site": attributed,
        "attribution_read_from": "results/site_agreement.json",
        "matches_the_site_instrument_could_attribute": f"{site['scored']} of {site['matched']}",
        "from_the_method": "the rule index and the matched atoms are carried with the candidate",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "what_each_arm_returns.json"))
    args = ap.parse_args()

    arms = {"GRAIL": ours(), "MetaTox": metatox(), "SyGMa": sygma(),
            "MetaPredictor": metapredictor(), "BioTransformer": biotransformer()}
    for row in arms.values():
        n = row["structures_returned"]
        row["share_carrying_a_score"] = (round(row["structures_carrying_a_score"] / n, 4)
                                         if n else None)
        row.setdefault("ordered", row["structures_carrying_a_score"] == n and n > 0)

    ranked = sorted(a for a, r in arms.items() if a != "GRAIL" and r["ordered"])
    unranked = sorted(a for a, r in arms.items() if a != "GRAIL" and not r["ordered"])
    attributing = sorted(a for a, r in arms.items()
                         if a != "GRAIL" and (r["names_the_transformation"] or r["names_a_site"]))

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([ROOT / r["file"] for r in arms.values()
                                 if "*" not in r["file"]]),
        "question": ("which arms return an ordering over their whole output, and which return an "
                     "attribution with it, read from the files this repository holds"),
        "by_arm": arms,
        "comparators_that_order_their_whole_output": ranked,
        "comparators_that_do_not": unranked,
        "comparators_whose_held_output_names_a_transformation_or_a_site": attributing,
        "reading": (
            "Ordering is not what separates this work from the incumbent: MetaTox scores every "
            "structure it returns and SyGMa orders its list, while MetaPredictor returns an "
            "unranked list and BioTransformer's file order is the only order it gives. "
            "Attribution is not a differentiator either, since BioTransformer's own output names "
            "the reaction and the enzyme. What is unusual is not that a rule and a site are "
            "reported but that the site is checked against the atoms that changed."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{'arm':16s} {'returned':>9s} {'scored':>8s} {'ordered':>8s}  attribution")
    for name, row in arms.items():
        attrib = ", ".join(k.split("_")[-1] for k in
                           ("names_the_transformation", "names_an_enzyme", "names_a_site")
                           if row[k]) or "none in the held output"
        print(f"{name:16s} {row['structures_returned']:9d} "
              f"{row['structures_carrying_a_score']:8d} {str(row['ordered']):>8s}  {attrib}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
