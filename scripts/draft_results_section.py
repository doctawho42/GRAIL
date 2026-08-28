"""The comparison section of the draft, generated from the artifact rather than typed.

The hand-written version of this section reported one comparator and claimed a lead at the tight
budgets. The artifact that defines the population carries three, and two of them lead both GRAIL
arms exactly there. The section is generated so that it cannot say otherwise again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

LABEL = {"whole bank": "GRAIL exhaustive", "trained budget": "GRAIL interactive",
         "metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor"}
OURS = ("whole bank", "trained budget")


def build():
    d = json.loads((ROOT / "results/deployment_table.json").read_text())
    rec, con = d["recall_micro"], d["contrasts"]
    out_len = d["mean_output_length"]
    ks = sorted(rec, key=int)
    arms = [a for a in LABEL if a in rec[ks[0]]]
    others = [a for a in arms if a not in OURS]

    L = []
    L.append("On the %d shared test substrates carrying %d annotated metabolites, micro recall "
             "by budget. A prediction equal to the substrate is dropped before the budget for "
             "every method alike, the convention `results/four_method_291.json` uses."
             % (d["population"]["n"], int(d["population"]["n_references"])))
    L.append("")
    L.append("| $k$ | " + " | ".join(LABEL[a] for a in arms) + " |")
    L.append("|---:|" + "---:|" * len(arms))
    for k in ks:
        best = max(arms, key=lambda a: rec[k][a])
        cells = []
        for a in arms:
            v = f"{rec[k][a]:.4f}"
            cells.append(f"**{v}**" if a == best else v)
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Mean list length: " + ", ".join(f"{LABEL[a]} {out_len[a]}" for a in arms) + ".")
    L.append("")

    # who leads where, and whether the interval separates
    lose, win, tie = [], [], []
    for k in ks:
        best_other = max(others, key=lambda a: rec[k][a])
        best_ours = max(OURS, key=lambda a: rec[k][a])
        c = con[k][f"{best_ours} - {best_other}"]
        row = (k, best_ours, best_other, c)
        if c["gap"] > 0 and c["excludes_zero"]:
            win.append(row)
        elif c["gap"] < 0 and c["excludes_zero"]:
            lose.append(row)
        else:
            tie.append(row)

    def fmt(rows):
        return ", ".join(f"$k={k}$ ({LABEL[o]} {rec[k][o]:.4f} against "
                         f"{LABEL[m]} {rec[k][m]:.4f})" for k, m, o, _ in rows)

    L.append("Read against the strongest comparator at each budget, the picture divides in three.")
    L.append("")
    if lose:
        L.append(f"**GRAIL trails, interval excluding zero:** {fmt(lose)}.")
        L.append("")
    if tie:
        L.append(f"**Neither separates:** {fmt(tie)}.")
        L.append("")
    if win:
        L.append(f"**GRAIL leads, interval excluding zero:** {fmt(win)}.")
        L.append("")
    L.append("The advantage is at depth and not at the head of the list. SyGMa leads at the "
             "tightest budgets and MetaPredictor in the middle; both saturate, MetaPredictor at "
             f"{rec['15']['metapredictor']:.4f} from $k=15$ on a mean list of "
             f"{out_len['metapredictor']}, SyGMa at {rec['50']['sygma']:.4f} on {out_len['sygma']}. "
             "GRAIL's exhaustive mode keeps rising because it keeps having candidates.")
    L.append("")
    L.append("Against MetaTox alone, the incumbent web service and the system this work set out "
             "to replace, the exhaustive mode leads at every budget: " +
             ", ".join(f"$k={k}$ {con[k]['whole bank - metatox']['gap']:+.4f}"
                       f"{'*' if con[k]['whole bank - metatox']['excludes_zero'] else ''}"
                       for k in ks) + ", where an asterisk marks an interval excluding zero.")
    L.append("")
    short = d["substrates_whose_list_is_shorter_than_the_budget"]
    L.append("Where a method runs out of candidates the budget stops measuring ranking, and the "
             f"counts are reported for every arm: at $k=15$, {short['15']['trained budget']} of "
             f"{d['population']['n']} interactive lists are shorter than the budget, "
             f"{short['15']['metapredictor']} of MetaPredictor's and "
             f"{short['15']['metatox']} of MetaTox's; at $k=50$ the counts are "
             f"{short['50']['trained budget']}, {short['50']['metapredictor']} and "
             f"{short['50']['metatox']}.")
    return "\n".join(L)


if __name__ == "__main__":
    out = ROOT / "paper2/draft_results.md"
    out.write_text(build() + "\n")
    print(build())
