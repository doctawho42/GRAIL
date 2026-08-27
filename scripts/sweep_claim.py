"""The sentence the MetaTox comparison may be stated in, derived rather than written.

Section 0.2b fixes that the comparison is the sweep and not a cell. Written by hand, that
sentence was wrong within an hour: it said the point estimate leads at k=15 and k=20, which holds
for the whole bank and not for the trained budget, where the gap is -0.0346 and -0.0647 and the
second of those separates against us.

So the sentence is generated from `results/deployment_table.json` and a test holds the
preregistration to containing exactly what this produces. A claim that cannot drift from its
artifact is worth more than one that is merely correct today.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "results/deployment_table.json"
ARMS = {"whole bank": "whole bank", "trained budget": "trained budget"}


def _fmt(ks):
    ks = [str(k) for k in ks]
    if not ks:
        return "no budget"
    if len(ks) == 1:
        return f"k = {ks[0]}"
    return "k = " + ", ".join(ks[:-1]) + " and " + ks[-1]


def claim(table_path: Path = TABLE) -> str:
    d = json.loads(Path(table_path).read_text())
    budgets = [int(k) for k in d["recall_micro"]]
    lines = []
    for arm in ARMS:
        leads, sep, trails_sep = [], [], []
        for k in budgets:
            c = d["contrasts"][str(k)][f"{arm} - metatox"]
            if c["gap"] > 0:
                leads.append(k)
                if c["excludes_zero"]:
                    sep.append(k)
            elif c["excludes_zero"]:
                trails_sep.append(k)
        ties = [k for k in budgets if k not in sep and k not in trails_sep]
        line = (f"The {arm} leads MetaTox at {_fmt(leads)}, with the paired-bootstrap interval "
                f"excluding zero at {_fmt(sep)}.")
        if trails_sep:
            line += f" It trails with the interval excluding zero at {_fmt(trails_sep)}."
        if ties:
            line += f" At {_fmt(ties)} the interval does not separate."
        lines.append(line)
    n = d["population"]["n"]
    mean = d["mean_output_length"]
    lines.append(f"Mean list length: whole bank {mean['whole bank']}, trained budget "
                 f"{mean['trained budget']}, MetaTox {mean['metatox']}, on {n} substrates.")
    return "\n".join("> " + x for x in lines)


if __name__ == "__main__":
    print(claim())
