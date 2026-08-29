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
    rows = []
    for c in crits:
        v = d["by_criterion"][c]["verdict_by_budget"]
        cells = " & ".join(mark[v[str(k)]] for k in ks)
        rows.append(f"{short.get(c, c)} & {cells} \\\\")
    head = " & ".join(f"${k}$" for k in ks)
    moved = d["n_budgets_moving"]
    worst = max(moved, key=lambda c: moved[c])
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'c' * len(ks)}}}\n\\toprule\n"
            f"criterion & \\multicolumn{{{len(ks)}}}{{c}}{{output budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The verdict of the comparison under each declared matching criterion. "
            "$+$ marks a budget where GRAIL's better arm leads the strongest comparator with the "
            "paired interval excluding zero, $-$ one where it trails on the same terms, and "
            "$\\cdot$ one where the interval covers zero. Every cell is read from the interval "
            "and never from the point estimate. Against the default criterion the verdict moves "
            f"at {moved[worst]} of {len(ks)} budgets under \\texttt{{{worst}}}.}}\n"
            "\\label{tab:si-criterion}\n\\end{table}\n")


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
    label = {"fusion": "fusion (deployed)", "filter": "pair filter alone",
             "generator": "rule score alone", "product": "their product",
             "random": "seeded permutation"}
    blocks = []
    for pop in d["by_population"]:
        r = d["by_population"][pop]
        rows = "\n".join(
            f"{label[a]} & " + " & ".join(f"{r['recall_micro'][a][k]:.4f}" for k in ks) + " \\\\"
            for a in arms)
        blocks.append(f"\\multicolumn{{{len(ks) + 1}}}{{l}}{{\\emph{{{pop}}}, "
                      f"$n = {r['population']['n']}$}} \\\\\n{rows}")
    head = " & ".join(f"${k}$" for k in ks)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'r' * len(ks)}}}\n\\toprule\n"
            f"ordering of one pool & \\multicolumn{{{len(ks)}}}{{c}}{{budget $k$}} \\\\\n"
            f"\\cmidrule(lr){{2-{len(ks) + 1}}}\n & {head} \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Micro recall when the same candidate pool is ordered five ways. The pool "
            "is built by the deployed configuration and capped by rule score before any arm sees "
            "it, so the pool, the matching rule and the budget are fixed and only the order "
            "varies. No model runs: the pools carry both component scores per candidate.}\n"
            "\\label{tab:si-ranking}\n\\end{table}\n")


def si_intervals():
    """Every GRAIL-versus-comparator difference with its paired interval.

    One block per GRAIL arm, budgets as rows. The two transposes were 816 pt and 688 pt wide
    against a 470 pt page; three comparator columns fit.
    """
    d = art("deployment_table.json")
    con = d["contrasts"]
    ks = sorted((int(k) for k in con), key=int)
    comps = [("metatox", "MetaTox"), ("sygma", "SyGMa"), ("metapredictor", "MetaPredictor")]

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
                star = "$^{*}$" if c["excludes_zero"] else "\\phantom{$^{*}$}"
                cells.append(f"{trim(c['gap'])}{star} [{trim(c['ci95'][0])}, {trim(c['ci95'][1])}]")
            rows.append(f"${k}$ & " + " & ".join(cells) + " \\\\")
        blocks.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{GRAIL {label}}} minus}} \\\\\n"
                      + "\n".join(rows))
    head = " & ".join(lab for _, lab in comps)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\begin{tabular}{rlll}\n\\toprule\n"
            f"$k$ & {head} \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every difference between a GRAIL arm and a comparator on the comparison "
            "set, in micro recall, with its paired bootstrap 95\\% interval; leading zeros are "
            "dropped. $^{*}$ marks an interval excluding zero, which is the condition under which "
            "the paper claims a lead or a trail, and the verdicts in the main text and in "
            "Figure~1 are read from these and from nothing else. The estimator is Equation~6 at "
            "$B = 10\\,000$ resamples, seed 0.}\n"
            "\\label{tab:si-intervals}\n\\end{table}\n")


def si_case():
    d = art("case_study_exhaustive.json")
    i = art("case_study.json")
    rows = []
    for c in d["candidates"][:20]:
        star = "$\\star$" if c["is_reference"] else ""
        sites = ",".join(str(a) for a in c["firing_atoms"][:6]) or "---"
        rows.append(f"{c['rank']} & {star} & {c['rule_id']} & {c['rule_source']} & {sites} & "
                    f"{c['generator']:.3f} & {c['filter']:.3f} \\\\")
    return ("\\begin{table}[h]\n\\centering\\footnotesize\n"
            "\\begin{tabular}{rlrlrrr}\n\\toprule\n"
            "rank & ref & rule & source & sites & generator & filter \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The first twenty of the "
            f"{d['n_candidates']} candidates the exhaustive mode returns for the worked example, "
            "with the rule that produced each, whether that rule was curated or mined, and the "
            "substrate atoms it fired on. $\\star$ marks an annotated metabolite. The interactive "
            f"mode returns {i['n_candidates']} candidates for the same substrate. The complete "
            "ranked lists of both modes are in the released artifacts.}\n"
            "\\label{tab:si-case}\n\\end{table}\n")


if __name__ == "__main__":
    for name, fn in (("si_table_splits", si_splits), ("si_table_criteria", si_criteria),
                     ("si_table_criterion", si_criterion_sweep), ("si_table_oracle", si_oracle),
                     ("si_table_case", si_case),
                     ("si_table_ranking", si_ranking),
                     ("si_table_intervals", si_intervals)):
        try:
            (OUT / f"{name}.tex").write_text(fn())
            print(f"  wrote paper2/{name}.tex")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
