"""The manuscript's comparison table, as LaTeX, generated from the artifact.

The hand-written table reported one comparator and claimed a lead the artifact does not support.
Generating it means the columns cannot fall behind the file that defines the population.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = {"whole bank": "GRAIL exh.", "trained budget": "GRAIL int.",
         "metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPred."}


def table():
    d = json.loads((ROOT / "results/deployment_table.json").read_text())
    rec, out = d["recall_micro"], d["mean_output_length"]
    # What each method emits, not what survives truncation at the widest budget: the row had been
    # printing the second under the first's name, understating SyGMa by nearly half.
    emit = d.get("mean_emitted_untruncated", {})
    ks = sorted(rec, key=int)
    arms = [a for a in LABEL if a in rec[ks[0]]]
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         "\\begin{tabular}{r" + "r" * len(arms) + "}", "\\toprule",
         "$k$ & " + " & ".join(LABEL[a] for a in arms) + " \\\\", "\\midrule"]
    # Nothing is bolded. Marking the largest point estimate at every budget asserts a leader at
    # the four budgets where the paper's own text says no arm separates, which is the discipline
    # of Section 2.8 broken by typography. The levels are here; the verdicts are in Table S3.
    for k in ks:
        L.append(f"{k} & " + " & ".join(f"{rec[k][a]:.4f}" for a in arms) + " \\\\")
    L += ["\\midrule",
          "emitted & " + " & ".join(f"{emit.get(a, out[a])}" for a in arms) + " \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{Micro recall at each output budget on the "
          f"{d['population']['n']} substrates every method predicts on, carrying "
          f"{int(d['population']['n_references'])} annotated metabolites. The last row gives the "
          "mean number of candidates each method emits, before any budget is applied: it is a "
          "property of the method and not of this table, and for two of the comparators it is "
          "larger than the widest budget shown. A prediction equal to the substrate is "
          "dropped before the budget for every method alike.}",
          "\\label{tab:sweep}", "\\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__":
    (ROOT / "paper2/table_sweep.tex").write_text(table() + "\n")
    print(table())
