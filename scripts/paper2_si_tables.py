"""The supporting information's tables, generated from the artifacts that measured them.

The main text's tables are generated for the same reason these are: a table typed by hand falls
behind the artifact the first time a run changes. The supporting information is where that risk
is highest, because it is the part nobody re-reads.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper2"


def art(name):
    return json.loads((ROOT / "results" / name).read_text())


def thousands(x):
    return f"{x:,}".replace(",", "{,}")


def si_splits():
    lk = art("leakage_fix_report.json")["clean_split_stats"]
    ovl = art("external_overlap_audit.json")
    rows = [f"{n} & {thousands(lk[k]['remaining_substrates'])} & "
            f"{thousands(lk[k]['remaining_positive_pairs'])} & "
            f"{thousands(lk[k]['remaining_triples'])} \\\\"
            for k, n in (("train", "train"), ("val", "validation"), ("test", "test"))]
    return ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{lrrr}\n\\toprule\n"
            "split & substrates & annotated pairs & triples \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The three substrate-disjoint splits. Triples count every "
            "(substrate, product, label) row including the rule-applicable products that carry "
            "no annotation, which the objective treats as unlabelled rather than negative. "
            "A read-only audit verifies zero substrate overlap and zero annotated-pair overlap "
            "across all three pairs of splits; the comparison set contains "
            f"{ovl['MetaTox 291-substrate comparison set']['in_train_or_val']} substrates seen in "
            "train or validation.}\n\\label{tab:si-splits}\n\\end{table}\n")


def si_criteria():
    rows = [
        ("canonical", "RDKit canonical SMILES, equality of strings", "strictest"),
        ("inchikey", "the full InChIKey, including the stereochemistry block", "strict"),
        ("inchi\\_no\\_stereo",
         "the first block of the InChIKey. This is the skeleton hash, so it drops the "
         "protonation and isotope layers along with stereochemistry",
         "medium"),
        ("tanimoto1", "Tanimoto similarity of one on 1024-bit Morgan fingerprints", "loosest"),
        ("inchikey\\_tautomer",
         "canonical tautomer on both sides, then a canonical SMILES without stereochemistry",
         "the default"),
    ]
    body = "\n".join(f"\\texttt{{{a}}} & {b} & {c} \\\\" for a, b, c in rows)
    return ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{lp{7.2cm}l}\n\\toprule\n"
            "criterion & what counts as the same structure & strictness \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The five declared matching criteria. The references carry no "
            "stereochemistry and do not distinguish tautomers, and standard InChI normalises "
            "only a subset of tautomers, which is why the tautomer-aware key is the default "
            "rather than the full InChIKey.}\n\\label{tab:si-criteria}\n\\end{table}\n")


def si_criterion_sweep():
    d = art("criterion_sweep.json")
    crits = d["criteria"]
    ks = sorted((int(k) for k in d["by_criterion"][crits[0]]["verdict_by_budget"]), key=int)
    short = {"canonical": "canonical", "inchikey": "InChIKey",
             "inchi_no_stereo": "InChIKey, no stereo", "tanimoto1": "Tanimoto $=1$",
             "inchikey_tautomer": "tautomer (default)"}
    mark = {"leads": "$+$", "trails": "$-$", "neither": "$\\cdot$"}
    # Which comparator each verdict is read against. It is not the same one at every budget, nor
    # the same one under every criterion -- a sign at k = 20 stands against SyGMa under one
    # criterion and MetaTox under another -- and a grid that prints only the sign cannot be read
    # without it.
    tag = {"MetaTox": "M", "SyGMa": "S", "MetaPredictor": "P", "BioTransformer": "B",
           "metatox": "M", "sygma": "S", "metapredictor": "P", "biotransformer": "B"}
    rows, arms_by_k = [], {}
    for c in crits:
        v = d["by_criterion"][c]["verdict_by_budget"]
        m = d["by_criterion"][c].get("margin_by_budget", {})
        cells = []
        for k in ks:
            cell = mark[v[str(k)]]
            row = m.get(str(k)) or {}
            who = row.get("theirs")
            if who:
                cell += "\\textsuperscript{" + tag.get(who, who[:1]) + "}"
                arms_by_k.setdefault(str(k), set()).add(row.get("ours", ""))
            cells.append(cell)
        rows.append(f"{short.get(c, c)} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"${k}$" for k in ks)
    arm_row = ""
    if arms_by_k:
        def arm_label(k):
            names = {a for a in arms_by_k.get(str(k), set()) if a}
            if len(names) != 1:
                return "--"
            only = names.pop()
            return "exh." if "exhaustive" in only else "int."
        arm_row = ("\\midrule\nGRAIL arm & "
                   + " & ".join(arm_label(k) for k in ks) + " \\\\\n")
    moved = d["n_budgets_moving"]
    worst = max(moved, key=lambda c: moved[c])
    return ("\\begin{table*}[t]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'c' * len(ks)}}}\n\\toprule\n"
            f"criterion & \\multicolumn{{{len(ks)}}}{{c}}{{output budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n" + arm_row + "\\bottomrule\n\\end{tabular}\n"
            "\\caption{The verdict of the comparison under each declared matching criterion. "
            "$+$ marks a budget where GRAIL's better arm leads the strongest comparator with the "
            "paired interval excluding zero, $-$ one where it trails on the same terms, and "
            "$\\cdot$ one where the interval covers zero. Every cell is read from the interval "
            "and never from the point estimate. The superscript names the comparator the cell is "
            "read against, M for MetaTox, S for SyGMa, P for MetaPredictor and B for "
            "BioTransformer, and the last row names the GRAIL arm, \\emph{int.} or \\emph{exh.}, "
            "with a dash where the better arm is not the same one under all five criteria at "
            "that budget: neither the arm nor the comparator is constant across "
            "the grid, so a sign on its own does not say what was compared with what. The levels "
            "every cell is read from are Table~\\ref{SI-tab:si-criterion-levels}. Against "
            "the default criterion the verdict moves "
            f"at {moved[worst]} of {len(ks)} budgets under \\texttt{{{worst}}}.}}\n"
            "\\label{tab:criterion}\n\\end{table*}\n")


def si_criterion_levels():
    """The recall the verdict grid is read from, arm by arm, criterion by criterion.

    Table~\\ref{MS-tab:criterion} prints a sign per cell and the manuscript says all five
    criteria are reported for every comparison. They were not: only the verdicts were, so a
    reader could see that a sign moves and not how far the level moves with it, nor whether the
    arms reorder. This is the evidence behind that grid.
    """
    d = art("criterion_sweep.json")
    crits = d["criteria"]
    ks = sorted((int(k) for k in d["by_criterion"][crits[0]]["recall_micro"]
                 [next(iter(d["by_criterion"][crits[0]]["recall_micro"]))]), key=int)
    short = {"canonical": "canonical", "inchikey": "InChIKey",
             "inchi_no_stereo": "InChIKey, no stereo", "tanimoto1": "Tanimoto $=1$",
             "inchikey_tautomer": "tautomer (default)"}
    label = {"GRAIL exhaustive": "GRAIL exh.", "GRAIL interactive": "GRAIL int.",
             "MetaTox": "MetaTox", "SyGMa": "SyGMa", "MetaPredictor": "MetaPred.",
             "BioTransformer": "BioTrans."}
    arms = [a for a in label if a in d["by_criterion"][crits[0]]["recall_micro"]]
    rows = []
    for c in crits:
        rec = d["by_criterion"][c]["recall_micro"]
        rows.append("\\midrule\n\\multicolumn{%d}{l}{\\emph{%s}} \\\\"
                    % (len(ks) + 1, short.get(c, c)))
        for a in arms:
            cells = " & ".join(f"{rec[a][str(k)]:.4f}" for k in ks)
            rows.append(f"\\quad {label[a]} & {cells} \\\\")
    head = " & ".join(f"${k}$" for k in ks)
    missing = [a for a in label if a not in arms]
    note = ("" if not missing else
            " " + " and ".join(label[a] for a in missing) + " is absent from this grid.")
    # Five criterion blocks of six arms plus rules and a caption do not fit a page at \small: the
    # last rows ran under the folio, which printed over a data cell -- a referee could not read
    # MetaPredictor at a budget of eight. The type is smaller and the rows tighter so the table
    # ends above the footer.
    return ("\\begin{table*}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\renewcommand{\\arraystretch}{0.92}\n"
            f"\\begin{{tabular}}{{l{'r' * len(ks)}}}\n\\toprule\n"
            f"arm & \\multicolumn{{{len(ks)}}}{{c}}{{output budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall for every arm at every budget under each of the five declared "
            "matching criteria, on the "
            f"{d['population']['n']} substrates of the comparison set. "
            "Table~\\ref{MS-tab:criterion} reads its verdicts from these numbers: a cell there is "
            "the better arm of this work against the strongest comparator at that budget, with "
            "the paired interval, so a sign can move either because a level moved or because the "
            "arms reordered, and both are visible here. Ranking is settled before the key is "
            "taken, so a criterion changes what counts as a hit and never what is ranked, and "
            "the parent-drop convention is re-derived under each criterion. A difference taken "
            "between two columns here can disagree with the margin behind the verdict by one in "
            "the last digit: the margin is computed from the hits and rounded once, and these "
            "levels are rounded before anything is subtracted." + note + "}\n"
            "\\label{tab:si-criterion-levels}\n\\end{table*}\n")


def si_fusion_k():
    """The fusion constant's sweep, printed rather than described.

    The text said the constant was swept over six values spanning two orders of magnitude and
    that everything from 60 upward is indistinguishable from the deployed value. Neither the six
    values nor the levels appeared anywhere, so the claim could not be checked against anything.
    """
    d = art("fusion_knobs.json")
    ks = sorted((int(k) for k in next(iter(d["by_constant"].values()))["recall"]), key=int)
    consts = [str(c) for c in d["constants_swept"]]
    dep = str(d["deployed_constant"])
    rows = []
    for c in consts:
        row = d["by_constant"][c]
        cells = [f"{row['recall'][str(k)]:.4f}" for k in ks]
        cell = row.get("against_the_deployed_constant_at_15")
        if cell is None:
            gap = "\\emph{deployed}"
        else:
            star = "$^{*}$" if cell["excludes_zero"] else "\\phantom{$^{*}$}"
            gap = (f"${cell['gap']:+.4f}${star} {{\\scriptsize [{cell['ci95'][0]:+.3f},"
                   f"{cell['ci95'][1]:+.3f}]}}")
        mark = "$\\;\\leftarrow$" if c == dep else ""
        rows.append(f"${c}${mark} & " + " & ".join(cells) + f" & {gap} \\\\")
    head = " & ".join(f"$k={k}$" for k in ks)

    # The second panel. A null stated at one budget was read as a null at every budget, so the
    # contrast is printed at each of them for the constants the null is about, which are the ones
    # at or above the value where the sweep goes flat.
    flat = d["null_bound"]["constants"]
    panel = []
    for c in [str(x) for x in flat if str(x) != dep]:
        cells = []
        for k in ks:
            cell = d["by_constant"][c]["against_the_deployed_constant"][str(k)]
            star = "$^{*}$" if cell["excludes_zero"] else "\\phantom{$^{*}$}"
            cells.append(f"${cell['gap']:+.4f}${star} {{\\tiny [{cell['ci95'][0]:+.3f},"
                         f"{cell['ci95'][1]:+.3f}]}}")
        panel.append(f"${c}$ & " + " & ".join(cells) + " \\\\")
    bound = ", ".join(
        f"{d['null_bound']['by_budget'][str(k)]['widest_interval_endpoint']:.4f} at $k={k}$"
        for k in ks)

    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'r' * len(ks)}l}}\n\\toprule\n"
            f"$K$ & {head} & against the deployed $K$ at $k=15$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
            "\\vspace{4pt}\n\\scriptsize\n"
            f"\\begin{{tabular}}{{l{'l' * len(ks)}}}\n\\toprule\n"
            f"$K$ & {head} \\\\\n\\midrule\n"
            + "\n".join(panel) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\normalsize\n"
            "\\caption{Micro recall under each value of the reciprocal-rank-fusion constant $K$, "
            f"on the {d['population']['n_substrates']} substrates of the comparison set carrying "
            f"{d['population']['n_references']} references, with the paired difference against the "
            f"deployed $K={dep}$ at a budget of 15 and its 95\\% interval; $^{{*}}$ marks an "
            "interval excluding zero. The lower panel gives the same paired difference at every "
            f"budget for the constants at or above {min(flat)}, which are the ones the flatness "
            "claim is about; the widest endpoint any of those intervals reaches is "
            f"{bound}, and one cell separates. Recomputing the fusion from the stored component "
            "scores costs no model run, which is why this knob could be swept and the aggregation "
            "rule had to be re-derived. The deployed value is the one the method was published "
            "with and was not chosen here, so sweeping it on this population adopts nothing: the "
            "objection Section~\\ref{sec:si-aggregation} raises against selecting a parameter here "
            "applies to a parameter that would be changed, and this one is not.}\n"
            "\\label{tab:si-fusion-k}\n\\end{table}\n")


def si_applicability():
    """The substrates these numbers were measured on, described rather than assumed."""
    d = art("applicability_domain.json")
    pops = list(d["populations"])
    names = [n for n in d["populations"][pops[0]]["descriptors"]]
    rows = []
    for name in names:
        cells = []
        for pop in pops:
            q = d["populations"][pop]["descriptors"][name]["quantiles"]
            # A range written with a dash reads badly when an endpoint is negative, and
            # calculated logP has one, so the endpoints are separated by a comma in math mode.
            def num(x):
                return ("$-" + f"{abs(x):g}$") if x < 0 else f"${x:g}$"
            cells.append(f"{q['0.5']:g} ({num(q['0.05'])}, {num(q['0.95'])})")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for label, key in (("substrates", "n_substrates"),
                       ("Bemis--Murcko scaffolds", "bemis_murcko_scaffolds"),
                       ("scaffolds on one substrate", "scaffolds_carried_by_one_substrate"),
                       ("acyclic substrates", "acyclic_substrates")):
        cells = [str(d["populations"][pop][key]) for pop in pops]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(pop for pop in pops)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'l' * len(pops)}}}\n\\toprule\n"
            f" & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The substrates behind every number in this work, as median and the fifth "
            "to ninety-fifth percentile. The corpus is assembled from four drug-centric sources "
            "and the distribution is what that produces; the manuscript motivates the problem "
            "with environmental chemicals as well, and this table is what a reader should hold "
            "that motivation against. It describes the population and does not establish an "
            "applicability domain, which would need performance measured against these axes "
            "rather than the axes alone.}\n"
            "\\label{tab:si-applicability}\n\\end{table}\n")


def si_population_list():
    """The comparison set itself, so the Supporting Information names its own population.

    Every comparator number in this work is measured on 291 substrates and until now they were
    identifiable only by going to the repository. A referee should not have to.
    """
    d = art("comparison_set_members.json")
    keys = sorted(d["references_per_member"])
    cols = 3
    rows, i = [], 0
    while i < len(keys):
        chunk = keys[i:i + cols]
        cells = [f"\\texttt{{{k}}}" for k in chunk] + [""] * (cols - len(chunk))
        rows.append(" & ".join(cells) + " \\\\")
        i += cols
    # Three InChIKeys in a typewriter face overrun the one-column measure at the body size, by
    # 66pt; at footnote size they fit with room to spare and stay legible.
    return ("{\\footnotesize\n\\begin{longtable}{@{}lll@{}}\n"
            "\\caption{The \\numPopulationN{} substrates of the comparison set, as tautomer-aware "
            "InChIKeys, which is the key every comparison in this work is scored under. The list "
            "is here so that the Supporting Information names its own population rather than "
            "pointing at a repository for it; the structures themselves are in the deposit.}\\\\\n"
            "\\label{tab:si-population-list}\\\\\n\\toprule\n\\endfirsthead\n"
            "\\toprule\n\\endhead\n\\bottomrule\n\\endfoot\n"
            + "\n".join(rows) + "\n\\end{longtable}\n}\n")


def si_drawing_equalised():
    """The comparison with every arm that can be re-run on the drawing a user would submit."""
    d = art("drawing_equalised.json")
    # This instrument takes hours on the full population and has a --substrates probe, so a probe
    # artifact can sit where the real one belongs and look like a table. It is refused rather
    # than printed: the population it was computed on has to be the one the comparison uses.
    expected = art("deployment_table.json")["population"]["n"]
    if d["population"]["n_substrates"] != expected:
        raise FileNotFoundError(
            f"results/drawing_equalised.json holds {d['population']['n_substrates']} substrates, "
            f"not the comparison set's {expected}; re-run it without --substrates")
    # The population is not the only way this artifact can be wrong. The first full run read a
    # re-run file that covered 79 of the 291 substrates and scored the rest as empty, which put a
    # comparator's recall at a quarter of its real value and looked like an enormous drawing
    # effect. The instrument now records what its re-runs cover and whether the two runs agree on
    # the substrates the drawing does not move; a table is refused unless both are on record.
    missed = d.get("substrates_the_drawing_moves_that_a_re_run_does_not_cover")
    if missed is None or any(missed.values()):
        raise FileNotFoundError(
            "results/drawing_equalised.json does not show every re-run covering the substrates "
            f"the drawing moves: {missed}")
    if "unmoved_substrates_where_the_two_runs_disagree" not in d:
        raise FileNotFoundError(
            "results/drawing_equalised.json predates the coverage record; re-run it")
    bad = {k: v for k, v in d["unmoved_substrates_where_the_two_runs_disagree"].items() if v}
    if bad:
        raise FileNotFoundError(
            f"results/drawing_equalised.json reports two runs disagreeing on substrates the "
            f"drawing does not move: {bad}")
    rec, stored = d["recall_equalised"], d["recall_as_stored"]
    arms = [a for a in ("GRAIL exhaustive", "GRAIL interactive", "MetaTox", "SyGMa",
                        "MetaPredictor", "BioTransformer") if a in rec]
    label = {"GRAIL exhaustive": "GRAIL exh.", "GRAIL interactive": "GRAIL int.",
             "MetaTox": "MetaTox$^{\\dagger}$", "SyGMa": "SyGMa",
             "MetaPredictor": "MetaPred.", "BioTransformer": "BioTrans."}
    ks = sorted((int(k) for k in rec[arms[0]]), key=int)
    mark = {"leads": "$+$", "trails": "$-$", "neither": "$\\cdot$"}
    rows = []
    for a in arms:
        rows.append(f"{label[a]} & " + " & ".join(f"{rec[a][str(k)]:.4f}" for k in ks) + " \\\\")
    rows.append("\\midrule")
    # Three verdict rows, not two. The middle one is the honest reading of this table: MetaTox is
    # the strongest comparator at the wide budgets and it is the one arm that did not move, so a
    # cell read against it is not a cell of an equalised comparison. Leaving it out of the grid
    # says what the other five arms give on one drawing.
    for name, grid in (("verdict, as stored", d["verdicts_as_stored"]),
                       ("verdict, equalised", d["verdicts_equalised"]),
                       ("verdict, equalised, MetaTox set aside",
                        d.get("verdicts_equalised_without_metatox") or {})):
        if not grid:
            continue
        rows.append(f"\\emph{{{name}}} & "
                    + " & ".join(mark[grid[str(k)]["verdict"]] for k in ks) + " \\\\")
    head = " & ".join(f"${k}$" for k in ks)
    moved = d["budgets_whose_verdict_moves"]
    moved_text = ("no budget's verdict moves between the two" if not moved else
                  "the verdict moves at $k=" + "$, $k=".join(moved) + "$")
    pop = d["population"]
    return ("\\begin{table*}[t]\n\\centering\\scriptsize\n"
            f"\\begin{{tabular}}{{l{'r' * len(ks)}}}\n\\toprule\n"
            f"arm & \\multicolumn{{{len(ks)}}}{{c}}{{output budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall with every arm that can be re-run presented with the substrate "
            "as the declared standardiser draws it, which is the drawing a user submits, on the "
            f"{pop['n_substrates']} substrates of the comparison set; the standardiser changes "
            f"{pop['substrates_the_standardiser_moves']} of them. "
            "$^{\\dagger}$MetaTox is a web service with no re-run available to us, so its column "
            "is the same as in Table~\\ref{MS-tab:sweep} and is the one arm still on its own "
            "input: it received the natural tautomer for part of the submission to begin with, "
            "which is the asymmetry this table removes from the other five. It is also the "
            "strongest comparator at the wide budgets, so a verdict read against it is not a "
            "verdict of an equalised comparison, and the last row is the grid with it set aside. "
            "The verdict rows "
            "read the better arm of this work against the strongest comparator at each budget, "
            "$+$ where the paired interval excludes zero in this work's favour, $-$ where it "
            f"excludes zero against, $\\cdot$ where it covers zero; {moved_text}. References are "
            "looked up under the corpus string in both, so the two grids are scored against one "
            "annotation.}\n"
            "\\label{tab:si-equalised}\n\\end{table*}\n")


def si_oracle():
    d = art("oracle_by_grouping.json")
    h, b = d["headroom_over_fusion"], d["contrasts_between_arms"]
    g = d["groups_per_substrate"]
    name = {"formula": "molecular formula", "random_matched": "random, matched to type",
            "type": "transformation type", "both": "formula and type"}
    order = ["formula", "random_matched", "type", "both"]
    rows = [f"{name[a]} & {d['recall_micro'][str(d['k'])][a]:.4f} & "
            f"${h[a]['gap']:+.4f}$ $[{h[a]['ci95'][0]:+.4f}, {h[a]['ci95'][1]:+.4f}]$ & "
            f"{g[a]} \\\\" for a in order]
    return ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{lrlr}\n\\toprule\n"
            "partition & recall@15 & over the deployed ranking & groups \\\\\n\\midrule\n"
            f"none (deployed) & {d['recall_micro'][str(d['k'])]['fusion']:.4f} & --- & --- \\\\\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{An oracle that orders candidate groups by whether they contain a "
            "reference, under four partitions of one candidate pool. A finer partition wins at an "
            "oracle for nothing, so the random partition is a control: its group-size multiset is "
            "exactly the transformation type's on each substrate. Read against that control "
            "rather than against the deployed ranking, molecular formula does not separate "
            f"(${b['formula-random_matched']['gap']:+.4f}$, "
            f"$[{b['formula-random_matched']['ci95'][0]:+.4f}, "
            f"{b['formula-random_matched']['ci95'][1]:+.4f}]$) while transformation type does "
            f"(${b['type-random_matched']['gap']:+.4f}$, "
            f"$[{b['type-random_matched']['ci95'][0]:+.4f}, "
            f"{b['type-random_matched']['ci95'][1]:+.4f}]$).}}\n"
            "\\label{tab:si-oracle}\n\\end{table}\n")


def si_ranking():
    d = art("ranking_ablation.json")
    arms = d["arms"]
    ks = ["1", "5", "10", "15", "30", "50"]
    # The two stages are the filter and the generator everywhere else in both documents, and
    # naming them twice made a reader match a table to a section by inference.
    label = {"fusion": "fusion (deployed)", "filter": "filter alone",
             "generator": "generator alone", "product": "their product",
             "random": "seeded permutation"}
    blocks = []
    for pop in d["by_population"]:
        r = d["by_population"][pop]
        rows = "\n".join(
            f"{label[a]} & " + " & ".join(f"{r['recall_micro'][a][k]:.4f}" for k in ks) + " \\\\"
            for a in arms)
        # The contrast P1 registers, with its paired interval on each population. Reporting only
        # the levels made the selection penalty a difference of point estimates; the intervals
        # are what say whether the two populations disagree.
        gaps = r["fusion_minus"]["product"]
        # the interval row at full size ran 112 pt past the page; it is set smaller and the
        # bounds are given to three places, which is all the width the column has
        def _b(x):
            # a positive bound carries no sign here: at this size twelve plus signs were the
            # difference between fitting the page and running 22 pt past it, and the caption
            # says so
            return f"{x:.3f}".replace("0.", ".").replace("-.", "$-$.")
        interval = " & ".join(
            "{\\scriptsize " + f"[{_b(gaps[k]['ci95'][0])},{_b(gaps[k]['ci95'][1])}]" + "}"
            for k in ks)
        gap_row = ("\\quad fusion $-$ product & "
                   + " & ".join(f"{gaps[k]['gap']:+.4f}" for k in ks) + " \\\\\n"
                   + "\\quad \\emph{interval} & " + interval + " \\\\")
        blocks.append(f"\\multicolumn{{{len(ks) + 1}}}{{l}}{{\\emph{{{pop}}}, "
                      f"$n = {r['population']['n']}$}} \\\\\n{rows}\n{gap_row}")
    head = " & ".join(f"${k}$" for k in ks)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'r' * len(ks)}}}\n\\toprule\n"
            f"ordering of one pool & \\multicolumn{{{len(ks)}}}{{c}}{{budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall when the same candidate pool is ordered five ways. The pool "
            "is built by the deployed configuration and capped by the generator's score before any arm sees "
            "it, so the pool, the matching rule and the budget are fixed and only the order "
            "varies. No model runs: the pools carry both component scores per candidate. The "
            "fusion-minus-product row is the contrast prediction P1 registers, given on both "
            "populations with its paired 95\\% interval, so the penalty for having selected the "
            "fusion rule on the comparison set can be read as two intervals rather than as a "
            "difference of point estimates. Interval bounds are unsigned when positive.}\n"
            "\\label{tab:si-ranking}\n\\end{table}\n")


def si_hyperparameters():
    """Every setting the two released checkpoints were trained under.

    Read from results/hyperparameters.json, which reads the config each run wrote beside its
    weights, so the table cannot drift from the checkpoints it describes.
    """
    d = art("hyperparameters.json")
    # The run each checkpoint came from, read from the artifact rather than described. Both are
    # one run; the caption used to say they were two.
    runs = d.get("runs", {})
    gen, fil = d["components"]["generator"], d["components"]["filter"]
    keys = [k for k in gen if k != "run"] + [k for k in fil if k != "run" and k not in gen]

    def cell(row, key):
        if key not in row:
            return "---"
        value = row[key]
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "---"
        text = str(value).replace("_", "\\_")
        return f"\\texttt{{{text}}}" if any(c in text for c in "[]") else text

    lines = ["\\begin{table}[h]", "\\centering\\footnotesize",
             "\\begin{tabular}{lll}", "\\toprule",
             " & generator & filter \\\\", "\\midrule"]
    for key in keys:
        lines.append(f"{key} & {cell(gen, key)} & {cell(fil, key)} \\\\")
    lines += ["\\midrule",
              f"training substrates & \\multicolumn{{2}}{{l}}{{{d['data']['training substrates']}}} \\\\",
              f"validation substrates & \\multicolumn{{2}}{{l}}{{{d['data']['validation substrates']}}} \\\\",
              f"sampling seed & \\multicolumn{{2}}{{l}}{{{d['data']['sampling seed']}}} \\\\",
              "\\bottomrule", "\\end{tabular}",
              "\\caption{The settings the two released checkpoints were trained under, read from "
              "the configuration each run recorded beside its weights. Both come from one run, "
              f"\\texttt{{{runs.get('generator', '').replace('artifacts/', '').replace('_', chr(92) + '_')}}}, "
              "which is established by reproduction rather than by recollection "
              "(Section~\\ref{sec:si-prov}); the two columns differ because the two stages were "
              "configured differently within it. None of these values was tuned for this paper: "
              "they are the configuration the run was launched with, carried forward from earlier "
              "development on the training and validation splits, and no search over them is "
              "reported here or held in this repository. That is a gap in a paper about selection "
              "discipline and it is named rather than dressed as a choice. The two decision "
              "thresholds are inert under the emission rule this work deploys, which returns the "
              "whole ranked pool and truncates by no threshold "
              "(Section~\\ref{MS-sec:methods}); they are printed because they are in the "
              "configuration the checkpoints carry.}",
              "\\label{tab:si-hyperparameters}", "\\end{table}", ""]
    return "\n".join(lines)


def si_chemistry():
    """Recall within each biotransformation class, for every arm.

    Classes are formula deltas, which is what the annotation determines; the producer's own
    record says what each name does not distinguish.
    """
    d = art("error_by_chemistry.json")
    arms = [("GRAIL exhaustive", "GRAIL exh."), ("GRAIL interactive", "GRAIL int."),
            ("metatox", "MetaTox"), ("sygma", "SyGMa"), ("metapredictor", "MetaPred."),
            ("biotransformer", "BioTrans.")]
    budget = "15"
    # A class with five references cannot support two decimals, and printing them there invites
    # an ordering to be read off noise. The small classes are marked rather than dropped, since
    # what they show -- a blind spot every arm shares -- is the reason the split is here at all.
    SMALL = 10
    rows = []
    for name, entry in d["classes"].items():
        cells = " & ".join(f"{entry['recall'][key][budget]:.2f}".lstrip("0")
                           for key, _ in arms)
        mark = "$^{\\dagger}$" if entry["references"] < SMALL else ""
        rows.append(f"{name}{mark} & {entry['references']} & {cells} \\\\")
    head = " & ".join(label for _, label in arms)
    n = d["population"]["references_classified"]
    n_small = sum(1 for e in d["classes"].values() if e["references"] < SMALL)
    # Which arms have nothing left to add at this budget, from the table that measures it.
    _dep = art("deployment_table.json")
    short = _dep["substrates_whose_list_is_shorter_than_the_budget"]["15"]
    n_subs = _dep["population"]["n"]
    # Every arm, and the count of those past half rather than an adjective. An earlier draft of
    # this caption said four of six run out "on most substrates" when two of the four run out on
    # 34 and 15 of 291, and left out the interactive arm, which runs out on 181.
    ORDER = [("whole bank", "the exhaustive arm"), ("trained budget", "the interactive arm"),
             ("metatox", "MetaTox"), ("sygma", "SyGMa"),
             ("metapredictor", "MetaPredictor"), ("biotransformer", "BioTransformer")]
    exhausted = ", ".join(f"{lab} on {short[k]}" for k, lab in ORDER if k in short)
    exhausted += f", of {n_subs}"
    WORDS = {0: "None", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    most_word = WORDS[sum(1 for k, _ in ORDER if k in short and short[k] * 2 > n_subs)]
    return ("\\begin{table}[h]\n\\centering\\scriptsize\n"
            "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
            f"transformation class & refs & {head} \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall at a budget of 15 within each transformation class, on the "
            f"comparison set's {n} annotated references; leading zeros are dropped. A class is a "
            "change in molecular formula, so it groups mechanisms the annotation does not "
            f"separate. $^{{\\dagger}}$ marks the {n_small} classes holding fewer than {SMALL} "
            "references, where one hit moves a cell by more than a tenth and no ordering across a "
            "row should be read. At this budget the arms differ in how often they have anything "
            f"left to emit, so a row is in part a comparison of list lengths: {exhausted} "
            f"(Table~\\ref{{tab:si-short}}). {most_word} of the six run out on more than half of "
            f"the {n_subs}. Table~\\ref{{tab:si-chemistry-matched}} repeats the split with each "
            "comparator\'s own length held instead of the budget.}\n"
            "\\label{tab:si-chemistry}\n\\end{table}\n")


def si_chemistry_matched():
    """The class split with list length held instead of the budget.

    Table S8 is read at a budget of fifteen, where four of the six arms have already emitted
    everything they are going to emit on most substrates. A row of that table is therefore partly
    a comparison of how long each list is, which is the same confound the pooled comparison
    answers with a matched-length control; this is that control applied per class. Each block cuts
    both of this work\'s arms to the comparator\'s own list length on each substrate, so the three
    cells of a block are read over the same slots.
    """
    d = art("error_by_chemistry.json")
    LABEL = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPred.",
             "biotransformer": "BioTrans."}
    comps = [c for c in LABEL if c in next(iter(d["classes"].values()))["recall_at_matched_length"]]
    if not comps:
        raise SystemExit("error_by_chemistry.json carries no matched-length block, so the class "
                         "split cannot be read with length held")
    SMALL = 10
    rows = []
    for name, entry in d["classes"].items():
        cells = []
        for c in comps:
            cell = entry["recall_at_matched_length"][c]
            cells += [f"{cell['GRAIL exhaustive']:.2f}".lstrip("0"),
                      f"{cell[c]:.2f}".lstrip("0")]
        mark = "$^{\\dagger}$" if entry["references"] < SMALL else ""
        rows.append(f"{name}{mark} & {entry['references']} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{LABEL[c]}}}" for c in comps)
    sub = " & ".join("ours & theirs" for _ in comps)
    n = d["population"]["references_classified"]
    return ("\\begin{table}[h]\n\\centering\\scriptsize\n"
            "\\begin{tabular}{@{}lr" + "rr" * len(comps) + "@{}}\n\\toprule\n"
            f"transformation class & refs & {head} \\\\\n"
            f" & & {sub} \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The class split of Table~\\ref{tab:si-chemistry} with list length held "
            "instead of the budget. Within each block both arms are cut, substrate by substrate, "
            "to the number of candidates that comparator returned there, under the same "
            "construction as Table~\\ref{MS-tab:si-matched}; \\emph{ours} is the exhaustive "
            f"arm. On the comparison set\'s {n} classified references; leading zeros are dropped, "
            "and $^{\\dagger}$ marks the classes under ten references, where no ordering should "
            "be read. A block\'s two cells are comparable with each other and not with another "
            "block\'s, since each block has its own slot count.}\n"
            "\\label{tab:si-chemistry-matched}\n\\end{table}\n")


def si_budget():
    """Recall and cost at each rule budget built, on validation.

    Budgets whose pool is still being written are excluded by the producer and named in its
    artifact; this prints what was measured.
    """
    d = art("budget_curve.json")
    ks = ["1", "5", "15", "30"]
    rows = []
    def cell_of(table, budget, key):
        c = table.get(budget)
        if c is None:
            return "---"
        star = "$^{*}$" if c["excludes_zero"] else ""
        return (("$-$" if c[key] < 0 else "+") + f"{abs(c[key]):.4f}".lstrip("0") + star)

    for budget in sorted(d["by_budget"], key=int):
        row = d["by_budget"][budget]
        gap = cell_of(d["against_the_deployed_budget_at_k15"], budget, "gap_at_15")
        gap30 = cell_of(d["against_the_deployed_budget_at_k30"], budget, "gap_at_30")
        marker = "\\textbf{" + budget + "}" if int(budget) == d["deployed_budget"] else budget
        recalls = " & ".join(f"{row['recall_micro'][k]:.4f}".lstrip("0") for k in ks)
        rows.append(f"{marker} & {row['mean_candidates']} & {recalls} & {gap} & {gap30} \\\\")
    n = d["population"]["n_substrates"]
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
            "rule budget & candidates & $r@1$ & $r@5$ & $r@15$ & $r@30$ & vs deployed at 15 "
            "& at 30 \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{What the rule budget buys, on the {n} validation substrates every budget "
            "holds. Candidates is the "
            "mean number returned after deduplication and the pool cap; leading zeros are dropped. "
            "The last two columns are the paired difference in micro recall against the deployed "
            "rule budget, in bold, read at output budgets of 15 and of 30, with $^{*}$ marking an "
            "interval that excludes zero. The two are reported because the paper reads this curve "
            "at both and they do not agree.}\n"
            "\\label{tab:si-budget}\n\\end{table}\n")


def si_counts():
    """Every candidate count the paper prints, with the population and the stage it belongs to.

    Six of them appear across the manuscript and the tables and they are different quantities. A
    referee counted them and could not tell which was which, which is a fair complaint about a
    paper whose argument is that a number without its configuration means nothing.
    """
    mt = art("mode_timings.json")
    dep = art("deployment_table.json")
    h10 = art("h10_verdict.json")
    ert = art("emission_rule_transfer.json")
    case = art("case_study.json")

    rows = [
        ("pool before deduplication", "interactive", "validation draw, 293",
         h10["mean_pool"]["trained"]),
        ("pool before deduplication", "exhaustive", "validation draw, 293",
         h10["mean_pool"]["whole_bank"]),
        ("pool the emission rule reads", "interactive",
         f"comparison set, {ert['population']['n']}", ert["mean_pool"]),
        ("list returned, deduplicated and capped", "interactive",
         f"validation draw, {mt['interactive']['candidates']['n']}",
         mt["interactive"]["candidates"]["mean"]),
        ("list returned, deduplicated and capped", "exhaustive",
         f"validation draw, {mt['exhaustive']['candidates']['n']}",
         mt["exhaustive"]["candidates"]["mean"]),
        ("list returned, deduplicated and capped", "interactive",
         "comparison set, 291", dep["mean_output_length"]["trained budget"]),
        ("list returned, deduplicated and capped", "exhaustive",
         "comparison set, 291", dep["mean_output_length"]["whole bank"]),
        ("list returned, one substrate", "interactive",
         "the worked example", case["n_candidates"]),
    ]

    # Every arm, not only this work's two. The table's own caption promises every candidate count
    # the paper reports, and a comparator's is a candidate count: MetaTox alone appears as three
    # different numbers across the manuscript and the supporting information, each a different
    # stage, none of them saying which. A count that names its stage cannot be mistaken for one
    # that does not, and that is the whole of the difference.
    ARMS = [("metatox", "MetaTox"), ("sygma", "SyGMa"),
            ("metapredictor", "MetaPredictor"), ("biotransformer", "BioTransformer"),
            ("gloryx", "GLORYx")]
    emitted = dep.get("mean_emitted_untruncated_2dp") or dep["mean_emitted_untruncated"]
    widest = max(int(k) for k in dep["recall_micro"]) if "recall_micro" in dep else 50
    for key, label in ARMS:
        if key in emitted:
            rows.append(("list as the comparator emits it", label, "comparison set, 291",
                         emitted[key]))
    for key, label in ARMS:
        if key in dep["mean_output_length"]:
            rows.append((f"the same list truncated at $k={widest}$", label,
                         "comparison set, 291", dep["mean_output_length"][key]))
    # The caption promises three quantities and this is the third. It used to read a top-level
    # "mean_slots" key that the artifact does not have, so the loop found nothing, added no rows,
    # and a bare except swallowed the silence: the table promised a quantity it never printed.
    # The slots live inside each contrast, and a missing one is now an error rather than a gap.
    matched = art("matched_length.json")
    slots = {}
    for name, c in matched["contrasts"].items():
        arm, _, comparator = name.partition(" - ")
        if arm == "whole bank" and "mean_slots" in c:
            slots[comparator] = c["mean_slots"]
    for key, label in ARMS:
        if key not in slots:
            raise SystemExit(f"matched_length.json carries no slot count for {key}, so the "
                             f"caption's third quantity cannot be printed for it")
        rows.append(("slots the matched-length control allows", label,
                     "comparison set, 291", round(float(slots[key]), 2)))

    body = "\n".join(
        f"{what} & {arm} & {pop} & {value} \\\\" for what, arm, pop, value in rows)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\begin{tabular}{llll}\n\\toprule\n"
            "quantity & mode & population & mean \\\\\n\\midrule\n"
            + body
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every candidate count this work reports, for every arm. They differ "
            "because they are measured at different stages of the pipeline and on different "
            "populations, not because any of them is an estimate of the others: a list as a "
            "system emits it, the same list truncated at the widest budget the sweep reads, and "
            "the slots the matched-length control allows are three quantities and the first is "
            "the one that describes what a user is handed.}\n"
            "\\label{tab:si-counts}\n\\end{table}\n")


def si_macro():
    """The whole comparison under the other aggregation, with paired intervals.

    The manuscript reports micro throughout and quoted one macro pair without an interval, at the
    budget where micro does not separate. Aggregation is a choice like the other two and is swept
    here rather than exercised once.
    """
    d = art("deployment_table.json")
    macro, con = d["recall_macro"], d["contrasts_macro"]
    # Which cells the aggregation actually moves, counted rather than asserted. The section said
    # the choice does not move a verdict where the paper claims one, and it moves five, one of
    # them a claimed lead. Only one contrast column is printed here, so a reader cannot check the
    # sentence against the table; the caption therefore carries the cells.
    micro = d.get("contrasts", {})
    moved = []
    for k, cells in con.items():
        for name, cell in cells.items():
            other = micro.get(k, {}).get(name)
            if other and other["excludes_zero"] != cell["excludes_zero"]:
                moved.append((int(k), name, other["excludes_zero"]))
    moved.sort()
    LONG = {"whole bank": "the exhaustive arm", "trained budget": "the interactive arm"}
    def phrase(k, name, micro_sep):
        a, b = [x.strip() for x in name.split(" - ")]
        who = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor",
               "biotransformer": "BioTransformer"}.get(b, b)
        return (f"{LONG.get(a, a)} against {who} at $k={k}$, which separates under "
                + ("micro and not macro" if micro_sep else "macro and not micro"))
    ks = sorted((int(k) for k in macro), key=int)
    # The fourth comparator arrived after this table was first written and the column was never
    # added, so a table captioned as the whole comparison under the other aggregation was five
    # sixths of it. The arms are filtered against the artifact so a later one cannot go missing
    # the same way.
    arms = [(key, label) for key, label in
            (("whole bank", "GRAIL exh."), ("trained budget", "GRAIL int."),
             ("metatox", "MetaTox"), ("sygma", "SyGMa"), ("metapredictor", "MetaPred."),
             ("biotransformer", "BioTrans."))
            if key in macro[str(ks[0])]]
    # One block of levels and two of contrasts, the same shape as the micro table so the two can
    # be read against each other. Printing one contrast of eight -- and the one this work wins --
    # is what made the robustness claim uncheckable from the page it was stated on.
    def trim(x):
        return ("$-$" if x < 0 else "+") + f"{abs(x):.3f}".lstrip("0")

    comps = [("metatox", "MetaTox"), ("sygma", "SyGMa"), ("metapredictor", "MetaPredictor"),
             ("biotransformer", "BioTransformer")]
    level_rows = []
    for k in ks:
        cells = " & ".join(f"{macro[str(k)][key]:.4f}".lstrip("0") for key, _ in arms
                           if key in macro[str(k)])
        level_rows.append(f"${k}$ & {cells} \\\\")
    contrast_blocks = []
    for arm, label in (("whole bank", "exhaustive"), ("trained budget", "interactive")):
        rows = []
        for k in ks:
            cells = []
            for key, _ in comps:
                c = con[str(k)].get(f"{arm} - {key}")
                if c is None:
                    cells.append("---")
                    continue
                star = "$^{*}$" if c["excludes_zero"] else "\\phantom{$^{*}$}"
                cells.append(f"{trim(c['gap'])}{star} [{trim(c['ci95'][0])}, "
                             f"{trim(c['ci95'][1])}]")
            rows.append(f"${k}$ & " + " & ".join(cells) + " \\\\")
        contrast_blocks.append(
            f"\\multicolumn{{{len(comps) + 1}}}{{l}}{{\\emph{{GRAIL {label}}} minus, under macro}}"
            " \\\\\n" + "\n".join(rows))
    head = " & ".join(label for _, label in arms)
    chead = " & ".join(lab for _, lab in comps)
    return ("\\begin{table}[h]\n\\centering\\footnotesize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            f"\\begin{{tabular}}{{r{'r' * len(arms)}}}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n"
            + "\n".join(level_rows)
            + "\n\\bottomrule\n\\end{tabular}\n\n"
            "\\vspace{4pt}\n\n"
            "\\begin{tabular}{r" + "l" * len(comps) + "}\n\\toprule\n"
            f"$k$ & {chead} \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(contrast_blocks)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{"
            + (f"The aggregation moves {len(moved)} verdicts of the "
               f"{sum(len(v) for v in con.values())} contrasts computed under both: "
               + "; ".join(phrase(*m) for m in moved) + ". " if moved else
               "No verdict differs between the two aggregations. ")
            + "The comparison under macro aggregation, the mean of per-substrate recall, "
            "on the same population and with the same conventions as Table~\\ref{MS-tab:sweep}; "
            "leading zeros are dropped. The last column is the paired difference against MetaTox, "
            "with $^{*}$ marking an interval excluding zero. Macro weights a substrate carrying one "
            "reference like one carrying twelve, so it answers a different question from micro and "
            "is not an estimate of it.}\n"
            "\\label{tab:si-macro}\n\\end{table}\n")


def si_matched():
    """Every arm against every comparator with the list length taken from the comparator."""
    d = art("matched_length.json")
    rows = []
    for pair, c in d["contrasts"].items():
        a, b = [x.strip() for x in pair.split(" - ")]
        arm = {"whole bank": "exhaustive", "trained budget": "interactive"}.get(a, a)
        # BioTransformer arrived after this map was written and printed lowercase for two
        # revisions; the corrected SyGMa arrived after that and did the same.
        name = {"metatox": "MetaTox", "sygma": "SyGMa",
                "metapredictor": "MetaPredictor", "biotransformer": "BioTransformer",
                "gloryx": "GLORYx",
                "sygma on the standardised drawing": "SyGMa, standardised"}.get(b)
        if name is None:
            raise SystemExit(f"matched_length.json names a comparator this table has no label "
                             f"for: {b}")
        star = "$^{*}$" if c["excludes_zero"] else ""
        gap = ("$-$" if c["gap"] < 0 else "+") + f"{abs(c['gap']):.4f}".lstrip("0")
        lo = ("$-$" if c["ci95"][0] < 0 else "+") + f"{abs(c['ci95'][0]):.4f}".lstrip("0")
        hi = ("$-$" if c["ci95"][1] < 0 else "+") + f"{abs(c['ci95'][1]):.4f}".lstrip("0")
        ours = f"{c['recall_ours']:.4f}".lstrip("0")
        thei = f"{c['recall_theirs']:.4f}".lstrip("0")
        rows.append(f"{arm} & {name} & {c['mean_slots']} & {ours} & {thei} & "
                    f"{gap}{star} [{lo}, {hi}] \\\\")
    n = d["population"]["n_substrates"]
    cap = d["cap_on_the_comparator_list"]
    # Where the cut actually binds, per comparator, so the caption names it rather than gesturing
    # at it. A comparator absent from this map has no substrate whose list runs past the cut.
    LABEL = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor",
             "biotransformer": "BioTransformer", "gloryx": "GLORYx",
             "sygma on the standardised drawing": "SyGMa on the standardised drawing"}
    bound = {}
    for name, c in d["contrasts"].items():
        arm, _, comp = name.partition(" - ")
        u = c.get("without_the_cap_on_the_comparator") or {}
        hit = u.get("substrates_the_cap_binds_on")
        if arm == "whole bank" and hit:
            bound[LABEL.get(comp, comp)] = hit
    if not bound:
        raise SystemExit("matched_length.json records no substrate where the comparator cut "
                         "binds, so the caption cannot say where it does")
    # Per comparator, and not added up: the counts are of different lists, and a substrate can be
    # past the cut for one comparator and inside it for another, so a total would not be a count
    # of anything.
    detail = " and ".join(f"{v} for {k}" for k, v in sorted(bound.items(), key=lambda kv: -kv[1]))
    return ("\\begin{table*}[t]\n\\centering\\small\n"
            "\\begin{tabular}{llrrrl}\n\\toprule\n"
            "arm & comparator & slots & ours & theirs & difference \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{The comparison with the output budget taken from the comparator instead of "
            f"the experiment. On each of the {n} substrates both arms are cut to the number of "
            "candidates that comparator returned there, so the two are read over the same slots on "
            "every substrate and not on average; slots is the mean of those lengths. That number "
            "is not the comparator\'s mean emission, because the comparator\'s list first has its "
            "duplicates removed, a prediction equal to the substrate dropped, and is then "
            f"truncated at {cap} candidates. The truncation is this work\'s choice and not the "
            f"comparator\'s: it sits five past this work\'s own pool cap, beyond which neither of "
            "its arms has a candidate to put in a slot, so a comparison there would measure that "
            f"cap rather than the ordering. It binds on {detail}, of the "
            f"{n} substrates, and on none for the other two, whose lists are shorter "
            "than the cut everywhere. What it costs is measured and not assumed: every contrast "
            "here is identical to four decimal places with the cut removed "
            "(Section~\\ref{SI-sec:si-matched}). The last two rows are the same control against "
            "SyGMa re-run on the drawing the declared standardiser produces rather than the one "
            "the corpus stores, which is the form in which that correction can be applied to a "
            "margin measured over slots. $^{*}$ marks "
            "an interval excluding zero. Leading zeros are dropped.}\n"
            "\\label{tab:si-matched}\n\\end{table*}\n")


def si_short():
    """How many of each arm's lists are shorter than the budget, at every budget.

    The main text gives these at k=50 only, and they bear directly on how the two claimed leads
    should be read: where a method has run out of candidates the budget has stopped measuring
    ranking and is measuring list length instead.
    """
    d = art("deployment_table.json")
    short = d["substrates_whose_list_is_shorter_than_the_budget"]
    ks = sorted(short, key=int)
    arms = list(short[ks[0]])
    label = {"whole bank": "GRAIL exh.", "trained budget": "GRAIL int.", "metatox": "MetaTox",
             "sygma": "SyGMa", "metapredictor": "MetaPred.", "biotransformer": "BioTrans."}
    rows = "\n".join(
        f"${k}$ & " + " & ".join(str(short[k][a]) for a in arms) + " \\\\" for k in ks)
    head = " & ".join(label.get(a, a) for a in arms)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{r{'r' * len(arms)}}}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n" + rows
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Substrates whose returned list is shorter than the budget, per arm and per "
            f"budget, of the {d['population']['n']} in the comparison set. Where a method has run "
            "out of candidates the budget has stopped measuring ranking and is measuring list "
            "length, so these counts bear directly on how the leads at the widest budgets should "
            "be read. The main text gives the $k=50$ row; the rest is here.}\n"
            "\\label{tab:si-short}\n\\end{table}\n")


def si_dialect():
    """What the substrate's drawing does to each arm, and to each verdict.

    Budgets are rows and arms are columns. Nine budgets across ran 69 pt past the page even with
    the leading zeros gone, which is the same reason the intervals table is transposed.
    """
    d = art("dialect_sweep.json")
    ks = [str(k) for k in d["budgets"]]
    arms = list(d["effect_on_each_arm"])

    def cell(v):
        body = f"{v['difference']:+.4f}".replace("0.", ".")
        lo, hi = (f"{x:+.3f}".replace("0.", ".") for x in v["ci95"])
        mark = "$^{*}$" if v["separates"] else "\\phantom{$^{*}$}"
        return f"{body}{mark} {{\\scriptsize [{lo},{hi}]}}"

    rows = "\n".join(
        f"${k}$ & " + " & ".join(cell(d["effect_on_each_arm"][a][k]) for a in arms) + " \\\\"
        for k in ks)
    cover = d["coverage_ceiling"]
    head = " & ".join(a.replace("GRAIL ", "") for a in arms)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{r{'l' * len(arms)}}}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n" + rows
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall with every substrate presented as the declared standardiser "
            "draws it, minus the same quantity with the substrate as the corpus stores it, on the "
            f"{d['n']} substrates of the comparison set, with the paired 95\\% interval. A "
            "positive value is recall the corpus's drawing was denying the arm; $^{*}$ marks an "
            "interval excluding zero, and leading zeros are dropped. The annotation is looked up "
            "under the corpus string in both runs, so the two are scored against one reference "
            "set and are paired substrate by substrate. On the same population the coverage "
            f"ceiling is {cover['stored']['coverage']:.4f} under the stored drawing and "
            f"{cover['standardised']['coverage']:.4f} under the standardiser's, a difference of "
            f"{cover['difference']['value']:+.4f} "
            f"$[{cover['difference']['ci95'][0]:+.4f}, {cover['difference']['ci95'][1]:+.4f}]$. "
            f"Of {d['verdict_cells']} arm-against-comparator verdicts, "
            f"{d['verdict_cells_that_move']} move, and none is a lead the paper claims. The three "
            "remaining comparators are held at the corpus drawing because MetaTox is a web "
            "service and MetaPredictor a frozen delivery; SyGMa, which can be re-run, is swept "
            "above.}\n"
            "\\label{tab:si-dialect}\n\\end{table}\n")


def si_intervals():
    """Every GRAIL-versus-comparator difference with its paired interval.

    One block per GRAIL arm, budgets as rows. The two transposes were 816 pt and 688 pt wide
    against a 470 pt page; three comparator columns fit.
    """
    d = art("deployment_table.json")
    mult = art("multiplicity.json")
    con = d["contrasts"]
    ks = sorted((int(k) for k in con), key=int)
    comps = [("metatox", "MetaTox"), ("sygma", "SyGMa"), ("metapredictor", "MetaPredictor"),
             ("biotransformer", "BioTransformer")]

    def trim(x):
        """+0.0496 -> +.050, so a cell is a number and not a paragraph."""
        return ("$-$" if x < 0 else "+") + f"{abs(x):.3f}".lstrip("0")

    blocks = []
    for arm, label in (("whole bank", "exhaustive"), ("trained budget", "interactive")):
        rows = []
        for k in ks:
            cells = []
            for key, _ in comps:
                c = con[str(k)].get(f"{arm} - {key}")
                if c is None:
                    cells.append("---"); continue
                # A second mark for the family-wise reading: * separates per comparison,
                # dagger says it does not survive Holm over the whole sweep.
                key_cell = f"{arm} - {key} @ {k}"
                holm = mult["cells"].get(key_cell, {})
                star = "$^{*}$" if c["excludes_zero"] else "\\phantom{$^{*}$}"
                if c["excludes_zero"] and holm.get("separates_after_holm") is False:
                    star += "$^{\\dagger}$"
                else:
                    star += "\\phantom{$^{\\dagger}$}"
                cells.append(f"{trim(c['gap'])}{star} [{trim(c['ci95'][0])}, {trim(c['ci95'][1])}]")
            rows.append(f"${k}$ & " + " & ".join(cells) + " \\\\")
        blocks.append(f"\\multicolumn{{{len(comps) + 1}}}{{l}}{{\\emph{{GRAIL {label}}} minus}}"
                      " \\\\\n" + "\n".join(rows))
    head = " & ".join(lab for _, lab in comps)
    return ("\\begin{table}[h]\n\\centering\\footnotesize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{r" + "l" * len(comps) + "}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every difference between a GRAIL arm and a comparator on the comparison "
            "set, in micro recall, with its paired bootstrap 95\\% interval; leading zeros are "
            "dropped. $^{*}$ marks an interval excluding zero, which is the condition under which "
            "the paper claims a lead or a trail, and the verdicts in the main text and in "
            "Figure~\\ref{MS-fig:sweep} are read from these and from nothing else. The estimator is "
            "Equation~\\ref{MS-eq:bootstrap} at "
            "$B = 10\\,000$ resamples, seed 0. $^{\\dagger}$ marks a cell that separates per "
            "comparison but not under Holm's correction over the whole sweep, a family of "
            f"{mult['n_tests']} tests.}}\n"
            "\\label{tab:si-intervals}\n\\end{table}\n")


def si_precision():
    d = art("precision_table.json")
    pr = d["precision_micro"]
    arms = list(pr)
    ks = sorted((int(k) for k in pr[arms[0]]), key=int)
    rows = [f"${k}$ & " + " & ".join(f"{pr[a][str(k)]:.4f}".lstrip("0") for a in arms)
            + " \\\\" for k in ks]
    head = " & ".join(a.replace("GRAIL ", "GRAIL\\ ") for a in arms)
    return ("\\begin{table}[h]\n\\centering\\scriptsize\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            f"\\begin{{tabular}}{{r{'r' * len(arms)}}}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro precision at each budget on the comparison set, under the parent-drop "
            "convention of Table~\\ref{MS-tab:sweep}. Precision under an incomplete annotation is a "
            "lower bound and is not used here to order systems.}"
            "\n\\label{tab:si-precision}\n\\end{table}\n")


def si_parentdrop():
    d = art("parent_drop_effect.json")
    eff, pres = d["effect"], d["parent_returned"]
    arms = list(eff)
    ks = sorted((int(k) for k in eff[arms[0]]), key=int)

    def trim(x):
        if abs(x) < 5e-5:
            return "$0$"
        return ("$-$" if x < 0 else "$+$") + f"{abs(x):.4f}".lstrip("0")

    rows = []
    # The caption asserts that no cell separates from zero, and until now the table printed only
    # the point effects, so the assertion rested on nothing a reader could see. The widest
    # interval and the largest effect are read off the artifact and printed instead.
    widest = max(((f"{a} at $k={k}$", eff[a][str(k)]["ci95"]) for a in arms for k in ks),
                 key=lambda row: row[1][1] - row[1][0])
    largest = max(abs(eff[a][str(k)]["effect"]) for a in arms for k in ks)
    for a in arms:
        cells = " & ".join(trim(eff[a][str(k)]["effect"]) for k in ks)
        rows.append(f"{a} & {pres[a]['substrates_returning_the_parent']} & {cells} \\\\")
    head = " & ".join(f"${k}$" for k in ks)
    med = ", ".join(f"{a.replace('GRAIL ', '')} {pres[a]['median_rank_when_returned']}"
                    for a in arms if pres[a]["median_rank_when_returned"])
    return ("\\begin{table}[h]\n\\centering\\scriptsize\n"
            f"\\begin{{tabular}}{{lr{'r' * len(ks)}}}\n\\toprule\n"
            f" & returns & \\multicolumn{{{len(ks)}}}{{c}}{{effect on micro recall at budget "
            f"$k$}} \\\\\n\\cmidrule(lr){{3-{len(ks) + 2}}}\n"
            f"arm & parent & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{What the parent-drop convention gives each arm. \\emph{Returns parent} is "
            f"the number of the {d['population']['n']} substrates whose returned list contains the "
            f"substrate itself; where it does, its median rank is {med}. The effect is recall with "
            "the convention minus recall without it, with leading zeros dropped, so a positive "
            "value is a hit promoted into the window and a negative one is the convention "
            f"discarding a reference: {d['substrates_whose_own_key_is_a_reference']} substrates "
            "carry their own key among their references. Every cell carries a paired 95\\% "
            "interval computed the same way as every other interval here and none of the "
            f"{len(arms) * len(ks)} excludes zero; printing them all would fill the table with "
            "the same statement, so the widest is given instead, "
            f"$[{widest[1][0]:+.4f}, {widest[1][1]:+.4f}]$ for {widest[0]}. The largest effect "
            f"anywhere in the grid is {largest:.4f}, which is "
            f"{round(largest * d['population']['n_references'])} hits in "
            f"{int(d['population']['n_references'])} references.}}\n"
            "\\label{tab:si-parentdrop}\n\\end{table}\n")


def si_case():
    d = art("case_study_exhaustive.json")
    i = art("case_study.json")
    # Atoms on the template's reactant side, so the claim that most of this list comes from very
    # small templates can be checked against the table it is made beside rather than taken. The
    # count is RDKit's over the parsed reactant templates, which is the definition the census
    # behind that claim uses; counting bracket atoms in the SMIRKS string instead gives a
    # different answer on this very list, 12 against 11.
    def reactant_atoms(smirks):
        from rdkit import RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        try:
            rxn = AllChem.ReactionFromSmarts(str(smirks))
        except Exception:
            return None
        if rxn is None or rxn.GetNumReactantTemplates() == 0:
            return None
        return sum(rxn.GetReactantTemplate(i).GetNumAtoms()
                   for i in range(rxn.GetNumReactantTemplates()))

    rows, small = [], 0
    for c in d["candidates"][:20]:
        star = "$\\star$" if c["is_reference"] else ""
        sites = ",".join(str(a) for a in c["firing_atoms"][:6]) or "---"
        n_at = reactant_atoms(c.get("rule", ""))
        small += bool(n_at and n_at <= 3)
        rows.append(f"{c['rank']} & {star} & {c['rule_id']} & {c['rule_source']} & "
                    f"{n_at if n_at else '---'} & {sites} & "
                    f"{c['generator']:.3f} & {c['filter']:.3f} \\\\")
    return ("\\begin{table}[h]\n\\centering\\footnotesize\n"
            "\\begin{tabular}{rlrlrlrr}\n\\toprule\n"
            "rank & ref & rule & source & atoms & sites & generator & filter \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The first twenty of the "
            f"{d['n_candidates']} candidates the exhaustive mode returns for the worked example, "
            "with the rule that produced each, whether that rule was curated or mined, and the "
            "number of atoms on its reactant side and the "
            "substrate atoms it fired on. $\\star$ marks an annotated metabolite. "
            f"{small} of the twenty come from a template of three reactant atoms or fewer, which "
            "is the count the manuscript quotes and which this column is here to let a reader "
            "check. The interactive "
            f"mode returns {i['n_candidates']} candidates for the same substrate. The complete "
            "ranked lists of both modes are in the released artifacts.}\n"
            "\\label{tab:si-case}\n\\end{table}\n")


def si_aggregation():
    """Recall under each aggregation rule the implementation offers, against the deployed one.

    The levels here are the deployed arm's own, which is the point: the deployed rule re-derived
    from per-template scores has to reproduce the published column or the re-run is on a different
    model, and the producer refuses to write the artifact when it does not.
    """
    d = art("aggregation_ablation.json")
    # Three is here because the manuscript quotes the maximum rule's extremum at it, the extremum
    # is a tie between three and five with different intervals, and until this column existed a
    # reader following the manuscript's own pointer found only the five cell and read the pair as
    # a contradiction. Every budget a document names has to be printed somewhere.
    ks = ["3", "5", "15", "30", "50"]
    order = ["noisy_or", "max", "mean", "hybrid"]
    NAME = {"noisy_or": "noisy-or", "max": "maximum", "mean": "mean", "hybrid": "hybrid"}
    rows = []
    for rule in order:
        row = d["by_rule"][rule]
        cells = " & ".join(f"{row['recall'][k]:.4f}".lstrip("0") for k in ks)
        # The difference is shown at a tight budget and a wide one, because that is where the
        # rules disagree: a rule that discounts duplication helps the head of the list and does
        # nothing measurable at the budgets this work reads its leads at.
        if rule == d["deployed"]:
            gaps = ["---", "---", "---"]
            label = f"\\textbf{{{NAME[rule]}}}"
        else:
            gaps = []
            for at in ("3", "5", "30"):
                cell = row["minus_noisy_or"][at]
                star = "$^{*}$" if cell["excludes_zero"] else ""
                gaps.append(("$-$" if cell["difference"] < 0 else "+")
                            + f"{abs(cell['difference']):.4f}"[1:] + star)
            label = NAME[rule]
        rows.append(f"{label} & {row['mean_ranked']:.1f} & {cells} & "
                    + " & ".join(gaps) + " \\\\")
    head = " & ".join(f"$r@{k}$" for k in ks)
    n = d["population"]["n_substrates"]
    refs = d["population"]["n_references"]
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n"
            f"aggregation & candidates & {head} & at 3 & at 5 & at 30 \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{Micro recall under each rule for combining the templates that reach one "
            f"candidate, on the {n} substrates of the comparison set carrying {refs} annotated "
            "metabolites; leading zeros are dropped. Candidates is the mean number ranked after "
            "deduplication and the pool cap. The deployed rule is in bold and reproduces the "
            "whole-bank column of the comparison table at every budget. The last two columns are "
            "the paired difference against it in micro recall at the two tight budgets a document of this work names and at a wide one, "
            "with $^{*}$ marking an interval that excludes zero; the rules disagree at the head "
            "of the list and not at the budgets this work reads its leads at.}\n"
            "\\label{tab:si-aggregation}\n\\end{table}\n")


def generators():
    """Every table this module writes, as (name, callable).

    Lifted out of the entry point so a checker can enumerate what the tables read. The macro
    generator was instrumented for that and this one was not, which left every artifact reaching
    the page through a table outside the guarantee.
    """
    return (("si_table_splits", si_splits), ("si_table_criteria", si_criteria),
                     ("si_table_oracle", si_oracle),
                     ("si_table_case", si_case),
                     ("si_table_ranking", si_ranking),
                     ("si_table_intervals", si_intervals),
                     ("si_table_precision", si_precision),
                     ("si_table_parentdrop", si_parentdrop),
                     ("si_table_dialect", si_dialect),
                     ("si_table_short", si_short),
                     ("si_table_hyperparameters", si_hyperparameters),
                     ("si_table_chemistry", si_chemistry),
                     ("si_table_chemistry_matched", si_chemistry_matched),
                     ("si_table_budget", si_budget),
                     ("si_table_counts", si_counts),
                     ("si_table_macro", si_macro),
                     ("si_table_matched", si_matched),
                     ("si_table_aggregation", si_aggregation),
                     # The verdict grid is printed in the manuscript rather than the Supporting
                     # Information: it is the evidence the title advertises. One generator, one
                     # label, two documents pointing at the same object.
                     ("si_table_criterion_levels", si_criterion_levels),
                     ("si_table_fusion_k", si_fusion_k),
                     ("si_table_applicability", si_applicability),
                     ("si_table_population_list", si_population_list),
                     ("si_table_equalised", si_drawing_equalised),
                     ("table_criterion", si_criterion_sweep))


if __name__ == "__main__":
    for name, fn in generators():
        try:
            (OUT / f"{name}.tex").write_text(fn())
            print(f"  wrote paper2/{name}.tex")
        except FileNotFoundError as e:
            # A refusal must not leave the previous table standing. Skipping and exiting zero
            # means a stale .tex survives, LaTeX compiles it, and the document carries numbers
            # from an artifact the generator has just declined to use. Removing it makes the
            # build fail on a missing \input, which is the loudest failure available here.
            stale = OUT / f"{name}.tex"
            if stale.exists():
                stale.unlink()
                print(f"  SKIP {name}, and removed the table it would have replaced: {e}")
            else:
                print(f"  SKIP {name}: {e}")
