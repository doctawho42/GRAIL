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
\\centering\\small
\\begin{{tabular}}{{lrr}}
\\toprule
 & interactive & exhaustive \\\\
\\midrule
rules applied & {i['top_k']} & {thousands(e['top_k'])} \\\\
mean candidates emitted & \\numOutputTrained & \\numOutputBank \\\\
median s per substrate & {i['median_s']} & {e['median_s']} \\\\
mean s per substrate & {i['mean_s']} & --- \\\\
90th percentile & {i['p90_s']} & --- \\\\
slowest substrate & {i['max_s']} & did not finish \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{The two operating modes. Both medians are over the {i['n']} validation substrates and
measure the same boundary, everything before the filter. The interactive mode's mean is six times
its median because of one substrate, a 291-heavy-atom peptide taking {i['max_s']}~s that the
exhaustive mode has never finished at any budget. A service must publish the tail as well as the
median.}}
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
    return ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lrrrl}\n\\toprule\n"
            "definition of a type & types & once & mass & names a \\\\\n"
            " & & & & transf. \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{The novel-type gap against the definition of a type. The tail is present "
            "wherever a type still names a transformation and collapses only below that.}\n"
            "\\label{tab:grain}\n\\end{table}\n")


def hypotheses():
    n = json.loads((ROOT / "results/paper2_numbers.json").read_text())["numbers"]
    H = [("H7", "rank fusion instead of product", "$+0.05$", "h7.diff", "validation"),
         ("H9", "pool cap 100 by generator", "$+0.015$", "h9.diff", "validation"),
         ("H10", "rule budget 30 from checkpoint", "$\\le +0.05$", "h10.bought", "validation"),
         ("H11", "emission = whole pool", "0 cells", "h11.lost", "grid $5\\times11$"),
         ("H12", "group score as third rank", "$+0.02$", "h12.diff", "comparison set"),
         ("H8", "group score, blocked", "$+0.05$", "h8.diff", "comparison set"),
         ("H14", "group score as a gate", "$+0.02$", "h14.diff", "comparison set"),
         ("H13", "standardise survivors only", "$\\ge 10\\times$", "h13.factor", "validation"),
         ("H15", "survivors + tautomer budget 200", "$<10$ s", "h15.time", "validation")]
    rows = []
    for h, what, thr, key, pop in H:
        v = n[key]
        val = f"{v:+.4f}" if isinstance(v, float) and abs(v) < 1 else str(v)
        rows.append(f"{h} & {what} & {thr} & ${val}$ & {pop} \\\\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lllll}\n\\toprule\n"
            " & what was fixed & threshold & measured & tested on \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every deployed choice as a registered hypothesis, with the population it "
            "was checked on. H13's figure is a factor and H15's a median in seconds; the rest are "
            "differences in micro recall@15.}\n\\label{tab:hyp}\n\\end{table}\n")


if __name__ == "__main__":
    for name, fn in (("table_modes", modes), ("table_grain", grain),
                     ("table_hypotheses", hypotheses)):
        (ROOT / f"paper2/{name}.tex").write_text(fn())
        print(f"  wrote paper2/{name}.tex")
