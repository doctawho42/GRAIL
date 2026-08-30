"""The remaining manuscript tables, generated from artifacts.

Every table in this paper is produced from the file that measured it. A table typed by hand falls
behind the artifact the first time a run changes, and the comparison table in the first draft did
exactly that: it reported one comparator where the defining artifact carried three.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def thousands(x):
    return f"{x:,}".replace(",", "{,}")


def modes():
    mt = json.loads((ROOT / "results/mode_timings.json").read_text())
    i, e = mt["interactive"], mt["exhaustive"]
    return f"""\\begin{{table}}[t]
\\centering\\footnotesize
\\begin{{tabular}}{{lrr}}
\\toprule
 & interactive & exhaustive \\\\
\\midrule
rules applied & {i['top_k']} & {thousands(e['top_k'])} \\\\
candidates, mean & {i['candidates']['mean']} & {e['candidates']['mean']} \\\\
candidates, median & {i['candidates']['median']} & {e['candidates']['median']} \\\\
seconds, median & {i['median_s']} & {e['median_s']} \\\\
seconds, mean & {i['mean_s']} & --- \\\\
seconds, 90th pct & {i['p90_s']} & --- \\\\
seconds, slowest & {i['max_s']} & did not finish \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{The two operating modes: rules applied, candidates returned and wall-clock time. Every
figure is measured on the validation draw, {i['n']} substrates for the interactive mode and
{e['candidates']['n']} for the exhaustive one, which lacks a pool for one of them. Candidates are
what a caller receives: deduplicated by matching key and capped at
{i['candidates']['cap']}. Times cover everything before the filter.}}
\\label{{tab:modes}}
\\end{{table}}
"""


def grain():
    cen = json.loads((ROOT / "results/novel_type_census.json").read_text())
    names = ["multiset of changed bonds, with counts", "the same, counts dropped",
             "element pairs that take part", "number of bonds that change"]
    rows = []
    for nm, g in zip(names, cen["granularity_curve"]):
        yes = "yes" if g["determines_a_product"] else "\\textbf{no}"
        rows.append(f"{nm} & {g['types']} & {g['seen_once']} & "
                    f"{g['share_of_mass_in_singletons'] * 100:.1f}\\% & {yes} \\\\")
    return ("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lrrrl}\n\\toprule\n"
            "definition of a type & types & once & mass & names a \\\\\n"
            " & & & & transf. \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The novel-type gap against the definition of a type. The tail is present "
            "wherever a type still names a transformation and collapses only below that.}\n"
            "\\label{tab:grain}\n\\end{table*}\n")


def hypotheses():
    """The registered predictions, numbered in the order the paper reaches them.

    The register's own identifiers run H7 to H15 with H1 to H6 belonging to work this paper does
    not report, and they are not in the order a reader meets them. That numbering is a record of
    when each was written, which is information about us and not about the result, so the table
    numbers them P1 to P9 in reading order and carries the register's identifier beside each so
    the audit trail survives the rename.
    """
    n = json.loads((ROOT / "results/paper2_numbers.json").read_text())["numbers"]
    H = [("H7", "rank fusion instead of a product of scores", "$+0.05$", "h7.diff", "validation"),
         ("H9", "candidate pool capped at 100", "$+0.015$", "h9.diff", "validation"),
         ("H10", "rule budget of 30 from the checkpoint", "$\\le +0.05$", "h10.bought", "validation"),
         # the grid is five criteria by eleven budgets, 55 cells, which is what the text reports
         ("H11", "emit the whole pool, no threshold", "0 cells", "h11.lost", "grid $5\\times11$"),
         ("H12", "group score as a third ranking", "$+0.02$", "h12.diff", "comparison set$^{\\dagger}$"),
         ("H8", "group score, groups emitted as blocks", "$+0.05$", "h8.diff", "comparison set$^{\\dagger}$"),
         ("H14", "group score as a gate before fusion", "$+0.02$", "h14.diff", "comparison set$^{\\dagger}$"),
         ("H13", "standardise surviving candidates only", "$\\ge 10\\times$", "h13.factor", "validation"),
         ("H15", "survivors, tautomer budget 200", "$<10$ s", "h15.time", "validation"),
         ("H16", "composite share, the two instruments' union", "$\\ge 0.134$",
          "composite.unionshare", "mined bank")]
    rows = []
    for i, (h, what, thr, key, pop) in enumerate(H, 1):
        v = n[key]
        # a share is not a difference and must not carry a sign
        signed = h != "H16"
        val = (f"{v:+.4f}" if signed else f"{v:.4f}") if isinstance(v, float) and abs(v) < 1 \
            else str(v)
        rows.append(f"P{i} & {what} & {thr} & ${val}$ & {pop} & {h} \\\\")
    return ("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llllll}\n\\toprule\n"
            " & what was fixed & threshold & measured & tested on & register \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every deployed choice as a prediction fixed before it was checked: what "
            "was fixed, the threshold, the value measured, the population it was checked on, and "
            "the identifier it carries in the released register. P8's figure is a speed-up "
            "factor, P9's a median in seconds and P10's a share of the mined bank; the rest are "
            "differences in micro recall at a budget of 15. $^{\\dagger}$ marks the three whose "
            "threshold was fixed in advance but whose population was not: they were adjudicated "
            "on the comparison set, which was recorded afterwards.}"
            "\n\\label{tab:hyp}\n\\end{table*}\n")


# The four annotated metabolites of the worked example, named so the table reads as chemistry
# rather than as keys. The names are the corpus's own annotation read back, not an assignment.
CASE_NAMES = {
    "FIRDBEQIJQERSE-UHFFFAOYSA-N": "dFdU, deamination",
    "KNTREFQOVSMROS-UHFFFAOYSA-N": "dFdCMP, 5$^\\prime$-monophosphate",
    "FRQISCZGNNXEMD-UHFFFAOYSA-N": "dFdCDP, 5$^\\prime$-diphosphate",
    "YMOXEIOKAJSRQX-UHFFFAOYSA-N": "dFdCTP, 5$^\\prime$-triphosphate",
}


def case_study():
    """The worked example: one substrate, both modes, the rule and site behind every hit."""
    inter = json.loads((ROOT / "results/case_study.json").read_text())
    exh = json.loads((ROOT / "results/case_study_exhaustive.json").read_text())

    def hits(d):
        return {c["key"]: c for c in d["candidates"] if c["is_reference"]}

    hi, he = hits(inter), hits(exh)
    rows = []
    for key, name in CASE_NAMES.items():
        ci, ce = hi.get(key), he.get(key)
        rank_i = str(ci["rank"]) if ci else "---"
        rule_i = str(ci["rule_id"]) if ci else "not produced"
        rank_e = str(ce["rank"]) if ce else "---"
        rule_e = str(ce["rule_id"]) if ce else "not produced"
        rows.append(f"{name} & {rank_i} & {rule_i} & {rank_e} & {rule_e}")
    body = " \\\\\n".join(rows)

    return (
        "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{interactive} & \\multicolumn{2}{c}{exhaustive} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "annotated metabolite & rank & rule & rank & rule \\\\\n\\midrule\n"
        f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{The four annotated metabolites of gemcitabine and the rank each is returned "
        "at by the two operating modes, with the bank template that produced it. Rule is the "
        "index into the deployed bank. The substrate is drawn as the corpus stores it.}\n"
        "\\label{tab:case}\n\\end{table*}\n")


if __name__ == "__main__":
    for name, fn in (("table_modes", modes), ("table_grain", grain),
                     ("table_hypotheses", hypotheses), ("table_case", case_study)):
        (ROOT / f"paper2/{name}.tex").write_text(fn())
        print(f"  wrote paper2/{name}.tex")
