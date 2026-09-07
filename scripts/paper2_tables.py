"""The manuscript's comparison table, as LaTeX, generated from the artifact.

The hand-written table reported one comparator and claimed a lead the artifact does not support.
Generating it means the columns cannot fall behind the file that defines the population.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _acs_table import acs_table

ROOT = Path(__file__).resolve().parent.parent
LABEL = {"whole bank": "GRAIL exh.", "trained budget": "GRAIL int.",
         "metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPred.",
         "biotransformer": "BioTrans.", "gloryx": "GLORYx"}


def blend_column():
    """The aggregation a selection on validation chooses, as a column beside the deployed one.

    The released rule is noisy-or and validation prefers the blend through a budget of twenty,
    which includes the budget this paper reads its headline at. Reporting that in the SI and
    printing only the deployed column here would leave a reader of the table unable to see what
    the configuration the paper's own discipline points to actually does.

    The column is admitted only if the re-derivation reproduces the deployed column exactly at
    every budget. The two are computed from different objects -- this one from per-template
    scores kept rather than collapsed -- so an unnoticed drift would otherwise put two
    incomparable numbers side by side.
    """
    path = ROOT / "results/aggregation_ablation.json"
    if not path.exists():
        return None
    a = json.loads(path.read_text())
    by = a.get("by_rule", {})
    if a.get("deployed") != "noisy_or" or "hybrid" not in by:
        return None
    return by["noisy_or"]["recall"], by["hybrid"]["recall"], a["join"]


def _emitted_note(emitted, out, arms, ks) -> str:
    """Which arms emit more than the widest budget, read from the row rather than remembered."""
    widest = max(int(k) for k in ks)
    over = [LABEL[a] for a in arms
            if float(emitted.get(a, out.get(a, 0))) > widest and a in LABEL]
    if not over:
        return f"No arm's mean emission exceeds the widest budget shown, {widest}."
    if len(over) == 1:
        return (f"One arm emits more than the widest budget shown, {widest}: {over[0]}, "
                f"whose column at that budget is therefore still a truncation.")
    return ("These arms emit more than the widest budget shown, "
            f"{widest}: " + ", ".join(over) + ".")


def table():
    d = json.loads((ROOT / "results/deployment_table.json").read_text())
    rec, out = d["recall_micro"], d["mean_output_length"]
    emit2 = d.get("mean_emitted_untruncated_2dp", {})
    # What each method emits, not what survives truncation at the widest budget: the row had been
    # printing the second under the first's name, understating SyGMa by nearly half.
    emit = d.get("mean_emitted_untruncated", {})
    ks = sorted(rec, key=int)
    arms = [a for a in LABEL if a in rec[ks[0]]]
    # A comparator the artifact carries and this map does not would be a column the sweep computes
    # and the paper does not print, while the abstract counts it: that happened, and a referee
    # found the fifth method missing from the table where a reader looks for all of them.
    missing = [a for a in rec[ks[0]] if a not in LABEL]
    if missing:
        raise SystemExit(f"deployment_table.json carries arms this table has no label for: "
                         f"{', '.join(missing)}; the abstract counts them")
    blend = blend_column()
    if blend and any(abs(blend[0][k] - rec[k]["whole bank"]) > 5e-5 for k in ks if k in blend[0]):
        blend = None          # the re-derivation has drifted; the column would not be comparable
    # The eighth column pushed the table past the two-column measure by 38pt, so the header is
    # the short name the caption then defines rather than a phrase.
    head = [LABEL[a] for a in arms] + (["blend"] if blend else [])
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
         "\\setlength{\\tabcolsep}{4.5pt}",
         "\\begin{tabular}{r" + "r" * len(head) + "}", "\\toprule",
         "$k$ & " + " & ".join(head) + " \\\\", "\\midrule"]
    # Nothing is bolded. Marking the largest point estimate at every budget asserts a leader at
    # the four budgets where the paper's own text says no arm separates, which is the discipline
    # of Section 2.8 broken by typography. The levels are here; the verdicts are in Table S3.
    for k in ks:
        cells = [f"{rec[k][a]:.4f}" for a in arms]
        if blend:
            cells.append(f"{blend[1][k]:.4f}")
        L.append(f"{k} & " + " & ".join(cells) + " \\\\")
    # The last row is a different quantity from the nine above it -- a list length, not a recall
    # -- and sat under the same rule structure reading as a tenth budget. It says what it is.
    emitted = [f"\\emph{{{float(emit2.get(a, emit.get(a, out[a]))):.2f}}}" for a in arms]
    if blend:
        emitted.append("\\emph{---}")
    L += ["\\midrule",
          "\\emph{mean emitted} & " + " & ".join(emitted) + " \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{Micro recall at each output budget on the "
          f"{d['population']['n']} substrates of the comparison set, carrying "
          f"{int(d['population']['n_references'])} annotated metabolites. The last row gives the "
          "mean number of candidates each method emits, before any budget is applied: it is a "
          "property of the method and not of this table. "
          # Counted rather than asserted. The caption said two comparators emit more than the
          # widest budget; at a widest budget of 50 only one does, and the sentence had been
          # written when the widest was narrower.
          + _emitted_note(emit2 or emit, out, arms, ks) +
          " A prediction equal to the substrate is "
          "dropped before the budget for every method alike."
          # What the blend column is, and nothing about why it is admissible: that argument is
          # the Supporting Information's and the main text carries the pointer.
          + ("" if not blend else
             " The last column, \\emph{blend}, re-ranks the exhaustive arm's own pool under the "
             "aggregation a selection on validation chooses. It is not the released "
             "configuration.") + "}",
          "\\label{tab:sweep}", "\\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__":
    (ROOT / "paper2/table_sweep.tex").write_text(acs_table(table()) + "\n")
    print(table())
