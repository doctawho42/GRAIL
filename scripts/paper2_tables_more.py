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
    # The bank holds 7,581 templates and 7,580 parse; what the mode applies is the second, which
    # the reactant-size census counts on the same file.
    rs = json.loads((ROOT / "results/reactant_size_census.json").read_text())
    parseable = rs["templates_parsed"]
    ce = json.loads((ROOT / "results/cost_envelope.json").read_text())
    sampled = ce["n_done"]
    unfinished = sum(1 for r in ce["rows"] if not r.get("finished"))
    # The exhaustive mode's mean, ninetieth percentile and slowest were printed as em-dashes,
    # which tells a reader nothing about a cost the paper asks them to weigh. They are censored
    # statistics, so they are printed as such: taken over the substrates that finished, marked,
    # and with the censoring rate in the caption.
    _fin = sorted(r["t_generate"] for r in ce["rows"] if r.get("finished"))
    env = {"deadline": ce["deadline_s"], "n_finished": len(_fin),
           "mean_finished": round(sum(_fin) / max(len(_fin), 1), 2),
           "p90_finished": round(_fin[int(0.9 * (len(_fin) - 1))], 2) if _fin else None,
           # The per cent sign has to reach LaTeX escaped, or it comments out the rest of the
           # caption line including the closing brace, and the document stops compiling.
           "censored_pct": f"{unfinished / max(sampled, 1):.1%}".replace("%", "\\%")}
    # The timing sweep is not a draw from the validation set: it orders substrates by heavy-atom
    # count, takes every nth, and adds the largest twelve outright, because that is where the
    # non-completions live. Two rows of this table therefore have a different population from the
    # rest, and a caption saying "every figure" over both is wrong at the row that matters most to
    # a deployer. The bias and the population rate it implies are printed rather than left to S32.
    every = int(ce["sample_every"])
    tail = 12
    _rows = sorted(ce["rows"], key=lambda r: r["heavy"])
    tail_unfinished = sum(1 for r in _rows[-tail:] if not r.get("finished"))
    rest = _rows[:-tail]
    rest_unfinished = sum(1 for r in rest if not r.get("finished"))
    population = tail_unfinished + every * rest_unfinished
    # The draw's own size, read from the artifact that defines it rather than typed, and required
    # to reproduce the sample size the timing artifact recorded before it is used.
    n_draw = json.loads(
        (ROOT / "results/val_pools.json").read_text())["population"]["declared_n"]
    overlap = sum(1 for _i in range(n_draw - tail, n_draw) if _i % every == 0)
    if -(-n_draw // every) + tail - overlap != sampled:
        raise SystemExit(f"the timing sample of {sampled} is not what a draw of every {every}th "
                         f"of {n_draw} plus the largest {tail} produces")
    share = f"{population / n_draw:.1%}".replace("%", "\\%")
    env["bias"] = (f"every {every}rd substrate of that draw by heavy-atom count plus the largest "
                   f"{tail} outright, so it over-represents the sizes where the deadline bites: "
                   f"{tail_unfinished} of the {unfinished} non-completions are among those {tail}, "
                   f"and the rate the whole draw would show is about {population} in {n_draw}, "
                   f"{share}")
    return f"""\\begin{{table}}[t]
\\centering\\footnotesize
\\begin{{tabular}}{{lrr}}
\\toprule
 & interactive & exhaustive \\\\
\\midrule
rules applied & {i['top_k']} & {thousands(parseable)} \\\\
candidates, mean & {i['candidates']['mean']} & {e['candidates']['mean']} \\\\
candidates, median & {i['candidates']['median']} & {e['candidates']['median']} \\\\
seconds, median & {i['median_s']} & {e['median_s']} \\\\
seconds, mean & {i['mean_s']} & {env['mean_finished']}$^{{\\dagger}}$ \\\\
seconds, 90th pct & {i['p90_s']} & {env['p90_finished']}$^{{\\dagger}}$ \\\\
seconds, slowest & {i['max_s']} & $>{int(env['deadline'])}$$^{{\\dagger}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{The two operating modes: rules applied, candidates returned and wall-clock time. Every
row but the two marked $^{{\\dagger}}$ is measured on the validation draw, {i['n']} substrates for
the interactive mode and {e['candidates']['n']} for the exhaustive one, which lacks a pool for one
of them; the median seconds are over those populations. Candidates are
what a caller receives: deduplicated by matching key and capped at
{i['candidates']['cap']}. Times cover everything before the filter. $^{{\\dagger}}$ marks a statistic that is both censored and measured on a different population: a sampled timing sweep of {sampled} substrates, on which the exhaustive mode exceeds a {int(env['deadline'])}-second deadline for {unfinished}, {env['censored_pct']}, so its mean and ninetieth percentile are taken over the {env['n_finished']} that finished. Those two are lower bounds against that sweep, where censoring removes the slowest; against the whole draw the sweep's design pushes the other way, since it over-represents large substrates on purpose, and the net of the two is not established here. That sweep is drawn as {env['bias']}. Its slowest substrate is one of the censored ones, which is why that cell carries the mark as well: a deadline is defined only for the sweep, so a $>$ entry can come from nowhere else, and the two arms' slowest cells are therefore not on the same substrates. On the test split, where no deadline is imposed, it fails on none. Every time here is measured on an unloaded machine and one substrate at a time; the exhaustive median carried to the load of a second arm is larger, and Section~\\ref{{SI-sec:si-runtime}} gives it, derives it from this median rather than measuring it again, and says between which arms the load ratio was taken. The interactive mode's own slowest substrate, {i['max_s']}~s, is far above its median, so a service answering a form should impose a deadline and fall back rather than assume the median.}}
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
    H = [("H7", "rank fusion instead of a score product", "$+0.05$", "h7.diff", "validation"),
         ("H9", "candidate pool capped at 100", "$+0.015$", "h9.diff", "validation"),
         ("H10", "rule budget of 30", "$\\le +0.05$", "h10.bought", "validation"),
         # the grid is five criteria by eleven budgets, 55 cells, which is what the text reports
         ("H11", "emit the whole pool, no threshold", "0 cells", "h11.lost", "grid $5\\times11$"),
         ("H12", "group score as a third ranking", "$+0.02$", "h12.diff", "comparison set$^{\\dagger}$"),
         ("H8", "group score, groups emitted as blocks", "$+0.05$", "h8.diff", "comparison set$^{\\dagger}$"),
         ("H14", "group score as a gate before fusion", "$+0.02$", "h14.diff", "comparison set$^{\\dagger}$"),
         ("H13", "standardise survivors only", "$\\ge 10\\times$", "h13.factor", "validation"),
         ("H15", "survivors, tautomer budget 200", "$<10$ s", "h15.time", "validation"),
         ("H16", "composite share, both instruments", "$\\ge 0.134$",
          "composite.unionshare", "mined bank")]
    # The verdict, which the reader previously had to assemble from three sections: whether the
    # measurement clears the threshold, and for the confirmations whether the whole interval does
    # or only the point estimate. Held in one place so the table says what happened.
    # The verdict says what happened, in the cell, rather than deferring it to a symbol. Three
    # of the seven confirmations are clean; three establish only that the effect exceeds zero,
    # because the registered threshold falls inside the interval; one holds on the population it
    # was registered against and not on the other. A reader who reads only this column should
    # reach the same reading as one who reads the footnotes.
    VERDICT = {
        "H7": "confirmed; threshold inside interval",
        "H9": "confirmed; threshold inside interval",
        "H10": "confirmed on validation; not on the comparison set",
        "H11": "confirmed",
        # P5/H12 is the one confirmation whose population was fixed after the result was visible:
        # it was adjudicated on the comparison set, which is derived from the test split, and S25
        # says so. A confirmation chosen that way is a measurement, and the table says the weaker
        # of the two words rather than leaving a reader to find the qualification three sections
        # later. The threshold also falls inside the interval, which is the second weakening.
        "H12": "measured, not adjudicated in advance; threshold inside interval",
        "H8": "failed", "H14": "failed", "H13": "failed",
        "H15": "confirmed", "H16": "confirmed",
    }
    rows = []
    for i, (h, what, thr, key, pop) in enumerate(H, 1):
        v = n[key]
        # a share is not a difference and must not carry a sign
        signed = h != "H16"
        val = (f"{v:+.4f}" if signed else f"{v:.4f}") if isinstance(v, float) and abs(v) < 1 \
            else str(v)
        rows.append(f"P{i} & {what} & {thr} & ${val}$ & {pop} & {VERDICT[h]} & {h} \\\\")
    return ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\begin{tabular}{@{}lllllp{0.125\\textwidth}l@{}}\n\\toprule\n"
            " & what was fixed & threshold & measured & tested on & verdict & register "
            "\\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every deployed choice as a prediction fixed before it was checked: what "
            "was fixed, the threshold, the value measured, the population it was checked on, and "
            "what the check returned. P8's figure is a speed-up "
            "factor, P9's a median in seconds and P10's a share of the mined bank; the rest are "
            "differences in micro recall at a budget of 15. $^{\\dagger}$ marks the three whose "
            "threshold was fixed in advance but whose population was not: they were adjudicated "
            "on the comparison set, which was recorded afterwards. Where the verdict says the "
            "threshold falls inside the interval, what the data establish is that the effect "
            "exceeds zero rather than that it clears the bar. P3 is confirmed on the validation "
            "population it was registered against; on the comparison set the same quantity is "
            "$+\\numHOneZeroComparison$, past its ceiling, and its effect is "
            "$\\numSpreadPThreeInsd\\times$ the interactive arm\'s retraining spread at the "
            "configuration reported here, which is indicative rather than a test because the "
            "effect is measured on validation and the spread on the comparison set. Each row "
            "carries its identifier in the released register in "
            "the last column, which numbers them in the order they were written rather than the "
            "order they are read; the register itself is Supporting Information "
            "Section~\\ref{SI-sec:si-register}.}"
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
        "index into the deployed bank. The substrate is drawn as the corpus stores it; the same "
        "run on the drawing a chemist would submit returns different ranks, and both drawings "
        "are given in Section~\\ref{SI-sec:si-case}.}\n"
        "\\label{tab:case}\n\\end{table*}\n")


if __name__ == "__main__":
    for name, fn in (("table_modes", modes), ("table_grain", grain),
                     ("table_hypotheses", hypotheses), ("table_case", case_study)):
        (ROOT / f"paper2/{name}.tex").write_text(fn())
        print(f"  wrote paper2/{name}.tex")
