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
    ks = sorted(rec, key=int)
    arms = [a for a in LABEL if a in rec[ks[0]]]
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         "\\begin{tabular}{r" + "r" * len(arms) + "}", "\\toprule",
         "$k$ & " + " & ".join(LABEL[a] for a in arms) + " \\\\", "\\midrule"]
    for k in ks:
        best = max(arms, key=lambda a: rec[k][a])
        cells = [(f"\\textbf{{{rec[k][a]:.4f}}}" if a == best else f"{rec[k][a]:.4f}")
                 for a in arms]
        L.append(f"{k} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule",
          "list & " + " & ".join(f"{out[a]}" for a in arms) + " \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{Micro recall at each output budget on the "
          f"{d['population']['n']} substrates every method predicts on, carrying "
          f"{int(d['population']['n_references'])} annotated metabolites. Bold marks the leader at "
          "that budget. The last row is the mean number of candidates each method emits. A "
          "prediction equal to the substrate is dropped before the budget for every method alike. "
          "Paired-bootstrap intervals on every difference are in the released artifact.}",
          "\\label{tab:sweep}", "\\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__":
    (ROOT / "paper2/table_sweep.tex").write_text(table() + "\n")
    print(table())
