r"""Checks for paper/app/xdomain.tex -- the cross-domain appendix.

This appendix carries the paper's four printed interaction tables (external criterion ladder,
five-method aggregates, four-method budget sweep, seven-system retrosynthesis top-1), the Holm
family, the normalised-headroom robustness column, and the two-curator agreement measurement. Two
hundred and twenty-nine of its numerals were read by no check.

Every table here is bound row by row and column by column rather than by spot-checking a cell,
because the defect this file exists to catch is a single row going stale while its neighbours stay
correct. Where the prose quotes a difference between two printed rows, the difference is recomputed
from those rows, so that moving either row breaks the sentence that reads them.

Figures with no artifact behind them are NOT invented a source here; they are listed in the report
that accompanies this module.
"""
from __future__ import annotations

import itertools
import re

# a bracketed 95% interval as this appendix prints it: $[+0.000,+0.020]$ or $[-0.020,-0.011]$
CI = r"\$\[\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*\]\$"
N = r"([-+]?\d[\d.]*)"
DN = r"\$([-+]?\d[\d.]*)\$"


def _cell(ctx, name, printed, computed, note=""):
    """``ctx.check`` at the printed precision, then one concession to the artifacts' own precision.

    Most of these artifacts store four decimals and the manuscript prints three, so a stored
    ``0.3735`` sits exactly on the boundary: both ``0.373`` and ``0.374`` are roundings of some value
    that rounds to ``0.3735``, and which one ``round`` returns is decided by the binary
    representation rather than by anything about the measurement. Where the strict comparison fails
    only by that half-ulp the verdict is relaxed and the note says so; a printed value a whole
    last-digit step away -- another method's figure, a stale digit -- still fails.
    """
    ctx.check(name, printed, computed, note)
    ok, *_ = ctx.checks[-1]
    if ok or not isinstance(printed, str) or "." not in printed:
        return
    try:
        pv, cv = float(printed), float(computed)
    except (TypeError, ValueError):
        return
    places = len(printed.split(".")[1])
    body = repr(cv)
    stored = len(body.split(".")[1]) if "." in body and "e" not in body else 0
    if stored <= places:
        return  # the artifact is exact at the printed precision: the strict verdict stands
    if abs(pv - cv) < 0.5 * 10 ** -places + 0.5 * 10 ** -stored:
        ctx.checks[-1] = (True, name, printed, computed,
                          f"{note}; on the artifact's own last-digit boundary".lstrip("; "))


def _macro(ctx, name):
    """The manuscript's own definition of a count macro, so a row can be read at its face value."""
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{((?:[^{}]|\{,\})*)\}", ctx.flat)
    return m.group(1) if m else None


def _blk(ctx, name, pattern, flags=0):
    """Match a span of the manuscript, recording a failure when the passage has been reworded."""
    m = re.search(pattern, ctx.flat, flags)
    if m is None:
        ctx.checks.append((False, f"xdomain, {name} is present", "present", "no match",
                           "paper/app/xdomain.tex"))
    return m


def _count(ctx, name, pattern, expect, note="", whole=False):
    """How many times a figure is printed. A figure printed twice can go stale once.

    Counted inside this appendix unless ``whole``, since the same three digits elsewhere in the
    manuscript are usually a different quantity.
    """
    hay = ctx.flat if whole else re.sub(r"\s+", " ", ctx.tex.get("xdomain.tex", ""))
    n = len(re.findall(pattern, hay))
    ctx.checks.append((n == expect, f"xdomain, {name}", str(expect), f"{n} printings", note))


def register(ctx) -> None:  # noqa: C901 -- one appendix, one function, read top to bottom
    _external_ladder(ctx)
    _matched_emission(ctx)
    _loses_on_recall(ctx)
    _population_axis(ctx)
    _provenance_table(ctx)
    _five_method(ctx)
    _ordering_stability(ctx)
    _four_method(ctx)
    _holm_table(ctx)
    _normalised_headroom(ctx)
    _factorised(ctx)
    _retro_groups(ctx)
    _retro_leaderboard(ctx)
    _other_domains(ctx)
    _contamination(ctx)
    _curators(ctx)


# --------------------------------------------------------------------------------------------
# Recall by criterion on the external set (tab:matchsensext)
# --------------------------------------------------------------------------------------------
def _external_ladder(ctx) -> None:
    grid = ctx.art("gloryx_criterion_grid.json")
    audit = ctx.art("external_overlap_audit.json")
    clean = ctx.art("gloryx_clean_subset.json")
    if grid is None or audit is None or clean is None:
        return

    cap = _blk(ctx, "the external table's footnote",
               r"GRAIL trained on \$(\d+)\$ and validated on \$(\d+)\$ of the \$(\d+)\$ drugs, "
               r"so its row is computed on the \$(\d+)\$ it has not seen")
    if cap:
        tr, va, n_all, unseen = (int(x) for x in cap.groups())
        seen = audit["GLORYx external set"]["in_train_or_val"]
        _cell(ctx, "xdomain, external footnote, train and validation sum to the overlap",
              str(tr + va), seen, "results/external_overlap_audit.json in_train_or_val")
        _cell(ctx, "xdomain, external footnote, the GLORYx set size", str(n_all),
              audit["GLORYx external set"]["n"], "results/external_overlap_audit.json")
        _cell(ctx, "xdomain, external footnote, the unseen drugs", str(unseen), clean["n_clean"],
              "results/gloryx_clean_subset.json n_clean")
        _cell(ctx, "xdomain, external footnote, unseen is the complement", str(n_all - seen),
              clean["n_clean"], "37 - 24 = 13")

    blk = _blk(ctx, "the external criterion table",
               r"\\label\{tab:matchsensext\}(.{0,900}?)\\bottomrule")
    if not blk:
        return
    rows = re.findall(r"(?:\\textbf\{)?(SyGMa|BioTransformer|MetaPredictor|GRAIL)\}?"
                      r"(?:\$\^\\dagger\$)? & " + N + r" & " + N + r" & " + N + r" & " + DN
                      + r" " + CI, blk.group(1))
    ctx.checks.append((len(rows) == 4, "xdomain, the external table has four rows", "4",
                       f"{len(rows)} parsed", "paper/app/xdomain.tex"))
    for meth, ik, ns, taut, step, lo, hi in rows:
        # GRAIL's row is the thirteen unseen drugs; the other three are the whole set
        pop, art = ("clean13", clean) if meth == "GRAIL" else ("all37", grid)
        rec = art["recall"][meth] if meth == "GRAIL" else grid["recall"]["all37"][meth]
        st = (clean["steps"][meth] if meth == "GRAIL"
              else grid["stereo_step"]["all37"][meth])
        src = ("results/gloryx_clean_subset.json" if meth == "GRAIL"
               else f"results/gloryx_criterion_grid.json {pop}")
        _cell(ctx, f"xdomain, external table, {meth} InChIKey", ik, rec["inchikey"], src)
        _cell(ctx, f"xdomain, external table, {meth} no-stereo", ns, rec["inchi_no_stereo"], src)
        _cell(ctx, f"xdomain, external table, {meth} tautomer", taut, rec["inchikey_tautomer"], src)
        gain = st["stereo"] if meth == "GRAIL" else st["gain"]
        _cell(ctx, f"xdomain, external table, {meth} stereo step", step, gain, src)
        _cell(ctx, f"xdomain, external table, {meth} stereo step lo", lo, st["ci95"][0], src)
        _cell(ctx, f"xdomain, external table, {meth} stereo step hi", hi, st["ci95"][1], src)

    # the sentence the recommended-criterion subsection reads off the GRAIL row
    m = _blk(ctx, "the stereo step read off the GRAIL row",
             r"it rises from \$([\d.]+)\$ to \$([\d.]+)\$ when stereochemistry is disregarded, "
             r"against a further \$([\d.]+)\$ from tautomer canonicalisation\. "
             r"\(The corresponding figures on all \$(\d+)\$, \$([\d.]+)\$ to \$([\d.]+)\$")
    if m:
        c13, a37 = clean["recall"]["GRAIL"], grid["recall"]["all37"]["GRAIL"]
        _cell(ctx, "xdomain, unseen GRAIL under strict InChIKey", m.group(1), c13["inchikey"],
              "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, unseen GRAIL stereo-blind", m.group(2), c13["inchi_no_stereo"],
              "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, unseen GRAIL tautomer step", m.group(3),
              c13["inchikey_tautomer"] - c13["inchi_no_stereo"],
              "results/gloryx_clean_subset.json, tautomer minus no-stereo")
        _cell(ctx, "xdomain, the contaminated figures are on all 37", m.group(4),
              grid["populations"]["all37"]["n"], "results/gloryx_criterion_grid.json")
        _cell(ctx, "xdomain, all-37 GRAIL under strict InChIKey", m.group(5), a37["inchikey"],
              "results/gloryx_criterion_grid.json all37")
        _cell(ctx, "xdomain, all-37 GRAIL stereo-blind", m.group(6), a37["inchi_no_stereo"],
              "results/gloryx_criterion_grid.json all37")


# --------------------------------------------------------------------------------------------
# The criterion at a matched output size
# --------------------------------------------------------------------------------------------
def _matched_emission(ctx) -> None:
    art = ctx.art("criterion_within_method.json")
    if art is None:
        return
    me = art["matched_emission"]
    at8 = me["at_matched_emission_8"]

    m = _blk(ctx, "the matched-emission sentence",
             r"GRAIL reaches \$k=(\d+)\$ and the two comparators \$k=(\d+)\$, giving \$([\d.]+)\$, "
             r"\$([\d.]+)\$ and \$([\d.]+)\$ emitted; the gains from canonical to tautomer matching "
             r"there are " + DN + r" " + CI + r", " + DN + r" " + CI + r" and " + DN + r" " + CI)
    if m:
        g = m.groups()
        _cell(ctx, "xdomain, matched emission, GRAIL's budget", g[0], at8["GRAIL"]["k"],
              "results/criterion_within_method.json")
        _cell(ctx, "xdomain, matched emission, the comparators' budget", g[1], at8["SyGMa"]["k"],
              "results/criterion_within_method.json")
        ctx.checks.append((at8["SyGMa"]["k"] == at8["MetaPredictor"]["k"],
                           "xdomain, matched emission, one budget serves both comparators",
                           str(at8["SyGMa"]["k"]), str(at8["MetaPredictor"]["k"]),
                           "results/criterion_within_method.json"))
        for i, meth in enumerate(("GRAIL", "SyGMa", "MetaPredictor")):
            _cell(ctx, f"xdomain, matched emission, {meth} emits", g[2 + i], at8[meth]["emitted"],
                  "results/criterion_within_method.json")
            _cell(ctx, f"xdomain, matched emission, {meth} gain", g[5 + 3 * i], at8[meth]["gain"],
                  "results/criterion_within_method.json")
            _cell(ctx, f"xdomain, matched emission, {meth} gain lo", g[6 + 3 * i],
                  at8[meth]["ci95"][0], "results/criterion_within_method.json")
            _cell(ctx, f"xdomain, matched emission, {meth} gain hi", g[7 + 3 * i],
                  at8[meth]["ci95"][1], "results/criterion_within_method.json")

    m = _blk(ctx, "the matched-emission paired differences",
             r"the paired differences against GRAIL are " + DN + r" " + CI
             + r" against SyGMa and " + DN + r" " + CI + r" against MetaPredictor")
    if m:
        pd = me["paired_differences_at_matched_emission"]
        for i, key in enumerate(("GRAIL-SyGMa", "GRAIL-MetaPredictor")):
            _cell(ctx, f"xdomain, matched emission, paired {key}", m.group(1 + 3 * i),
                  pd[key]["delta"], "results/criterion_within_method.json")
            _cell(ctx, f"xdomain, matched emission, paired {key} lo", m.group(2 + 3 * i),
                  pd[key]["ci95"][0], "results/criterion_within_method.json")
            _cell(ctx, f"xdomain, matched emission, paired {key} hi", m.group(3 + 3 * i),
                  pd[key]["ci95"][1], "results/criterion_within_method.json")

    m = _blk(ctx, "the uncapped-count mismatch sentence",
             r"at \$k=15\$ SyGMa emits \$([\d.]+)\$ of the \$([\d.]+)\$ it would emit uncapped")
    if m:
        row = me["as_the_paper_compared_them"]["by_method"]["SyGMa"]
        _cell(ctx, "xdomain, SyGMa emitted at k=15", m.group(1), row["emitted_at_k15"],
              "results/criterion_within_method.json")
        _cell(ctx, "xdomain, SyGMa emitted uncapped", m.group(2), row["emitted_uncapped"],
              "results/criterion_within_method.json")


# --------------------------------------------------------------------------------------------
# GRAIL does not win on recall
# --------------------------------------------------------------------------------------------
def _loses_on_recall(ctx) -> None:
    art = ctx.art("anchor_certification.json")
    if art is None:
        return
    m = _blk(ctx, "the paired margin against SyGMa",
             r"a paired difference of " + DN + r" in mean per-substrate recall, 95\\% confidence "
             r"interval " + CI)
    if m:
        d = art["delta_mean_recall"]
        _cell(ctx, "xdomain, GRAIL minus SyGMa mean recall", m.group(1), d["point"],
              "results/anchor_certification.json delta_mean_recall")
        _cell(ctx, "xdomain, GRAIL minus SyGMa lo", m.group(2), d["lo"],
              "results/anchor_certification.json")
        _cell(ctx, "xdomain, GRAIL minus SyGMa hi", m.group(3), d["hi"],
              "results/anchor_certification.json")


# --------------------------------------------------------------------------------------------
# The population axis
# --------------------------------------------------------------------------------------------
def _population_axis(ctx) -> None:
    pop = ctx.art("population_axis.json")
    retro = ctx.art("retro_population_axis.json")
    paired = ctx.art("match_sensitivity_fulln_paired.json")
    flip = ctx.art("rank_flip_ci.json")
    c0 = ctx.art("retro_leaderboard_cluster0.json")

    if pop is not None:
        m = _blk(ctx, "the MetaPredictor-SyGMa reversal sentence",
                 r"MetaPredictor against SyGMa on recall reverses under every one of the five "
                 r"criteria, ahead by about \$([\d.]+)\$ on the shared subset and behind by about "
                 r"\$([\d.]+)\$ on the full split")
        rows = [r for r in pop["rows"]
                if r["pair"] == "MetaPredictor vs SyGMa" and r["metric"] == "recall"]
        ctx.checks.append((len(rows) == 5 and all(r["reordered"] for r in rows),
                           "xdomain, the pair reverses under all five criteria", "5 of 5",
                           f"{sum(1 for r in rows if r['reordered'])} of {len(rows)}",
                           "results/population_axis.json"))
        if m and rows:
            _cell(ctx, "xdomain, the pair's lead on the shared subset", m.group(1),
                  sum(r["gap_shared150"] for r in rows) / len(rows),
                  "results/population_axis.json, mean over the five criteria")
            _cell(ctx, "xdomain, the pair's deficit on the full split", m.group(2),
                  -sum(r["gap_full1170"] for r in rows) / len(rows),
                  "results/population_axis.json, mean over the five criteria")

    if retro is not None:
        m = _blk(ctx, "the agreeing shared reactions",
                 r"the two clusters share \$(\d+)\$ products, \$([\d,{}]+)\$ of which agree on the "
                 r"recorded reaction; those \$([\d,{}]+)\$ are nested")
        if m:
            _cell(ctx, "xdomain, products in both clusters", m.group(1), retro["products_in_both"],
                  "results/retro_population_axis.json")
            for i, g in enumerate(m.groups()[1:], start=1):
                _cell(ctx, f"xdomain, agreeing reactions, printing {i}", g.replace("{,}", ""),
                      retro["agreeing_reactions"], "results/retro_population_axis.json")
        m = _blk(ctx, "the released-file chance rate",
                 r"On the released files \$(\d+)\$ of \$(\d+)\$ have intervals excluding zero, "
                 r"which is about what \$(\d+)\$ tests produce by chance")
        if m:
            _cell(ctx, "xdomain, released-file intervals excluding zero", m.group(1),
                  retro["interactions_excluding_zero"], "results/retro_population_axis.json")
            for i in (2, 3):
                _cell(ctx, f"xdomain, released-file comparisons, printing {i - 1}", m.group(i),
                      retro["comparisons"], "results/retro_population_axis.json")

        m = _blk(ctx, "the joint ten-system ordering",
                 r"On the \$(\d+)\$ they share, all ten are comparable at once: "
                 r"\\texttt\{gln\} leads at \$([\d.]+)\$ top-1 under canonical matching, ahead of "
                 r"\\texttt\{chemformer\} at \$([\d.]+)\$ and \\texttt\{gta\} at \$([\d.]+)\$")
        if m:
            jl = retro["joint_leaderboard"]
            _cell(ctx, "xdomain, the joint leaderboard's reactions", m.group(1),
                  retro["agreeing_reactions"], "results/retro_population_axis.json")
            ctx.checks.append((len(jl) == 10, "xdomain, ten systems are jointly comparable", "10",
                               str(len(jl)), "results/retro_population_axis.json"))
            for i, sysname in enumerate(("gln", "chemformer", "gta"), start=2):
                _cell(ctx, f"xdomain, joint leaderboard top-1, {sysname}", m.group(i),
                      jl[sysname]["canonical"]["1"],
                      "results/retro_population_axis.json joint_leaderboard")
            order = sorted(jl, key=lambda s: -jl[s]["canonical"]["1"])
            ctx.checks.append((order[:3] == ["gln", "chemformer", "gta"],
                               "xdomain, the joint leaderboard's leading three", "gln chemformer gta",
                               " ".join(order[:3]), "results/retro_population_axis.json"))

    if c0 is not None:
        m = _blk(ctx, "gln on its own cluster",
                 r"\\texttt\{gln\} is fifth of seven at \$([\d.]+)\$")
        if m:
            acc = c0["accuracy"]
            _cell(ctx, "xdomain, gln top-1 on its own cluster", m.group(1),
                  acc["gln"]["canonical"]["top1"],
                  "results/retro_leaderboard_cluster0.json accuracy/gln/canonical/top1")
            rank = c0["orderings"]["top1"]["canonical"].index("gln") + 1
            ctx.checks.append((rank == 5 and len(acc) == 7,
                               "xdomain, gln's rank on its own cluster", "fifth of seven",
                               f"{rank} of {len(acc)}", "results/retro_leaderboard_cluster0.json"))

    if paired is not None and flip is not None:
        m = _blk(ctx, "the criterion differentials the null is measured against",
                 r"criterion differentials that survive correction, \$([\d.]+)\$ internally and "
                 r"\$([\d.]+)\$ on the shared subset")
        if m:
            internal = [p for p in paired["pairwise"] if p["pair"] == "GRAIL_vs_MetaPredictor"][0]
            _cell(ctx, "xdomain, the certified internal criterion differential", m.group(1),
                  internal["interaction_b_minus_a"],
                  "results/match_sensitivity_fulln_paired.json GRAIL_vs_MetaPredictor")
            _cell(ctx, "xdomain, the certified shared-subset differential", m.group(2),
                  flip["interaction_B_extra_gain_from_normalization"]["mean"],
                  "results/rank_flip_ci.json")

    if c0 is not None:
        m = _blk(ctx, "the cross-domain family quoted in the population section",
                 r"same estimator and \$(\d+)\$ of its \$(\d+)\$ tests survive the same correction")
        if m:
            _cell(ctx, "xdomain, cross-domain Holm survivors, population section", m.group(1),
                  len(c0["holm_survivors"]), "results/retro_leaderboard_cluster0.json")
            _cell(ctx, "xdomain, cross-domain family size, population section", m.group(2),
                  c0["n_interaction_tests"], "results/retro_leaderboard_cluster0.json")


# --------------------------------------------------------------------------------------------
# Which number is which (tab:provenance)
# --------------------------------------------------------------------------------------------
def _provenance_table(ctx) -> None:
    five = ctx.art("match_sensitivity_5method.json")
    fulln = ctx.art("match_sensitivity_fulln_paired.json")
    setm = ctx.art("set_metrics_by_criterion.json")
    anchor = ctx.art("anchor_certification.json")
    if not (five and fulln and setm and anchor):
        return

    blk = _blk(ctx, "the provenance table",
               r"\\label\{tab:provenance\}(.{0,2600}?)\\bottomrule")
    if not blk:
        return
    body = blk.group(1)
    rows = re.findall(r"(GRAIL|SyGMa|MetaPredictor) (\$[^&]+?\$) *& ([^&]+?) & (\$?[^&]+?\$?) & "
                      r"([a-z, 3]+) & (\$k\{=\}15\$|none|either) & (yes|no) \\\\", body)
    ctx.checks.append((len(rows) == 12, "xdomain, the provenance table has twelve rows", "12",
                       f"{len(rows)} parsed", "paper/app/xdomain.tex"))

    # rows are keyed by the method and the place the row points at, never by the figure itself:
    # a row identified by its own value cannot be caught going stale
    printed = {}
    for meth, fig, where, n, agg, trunc, collapsed in rows:
        tag = re.search(r"\\ref\{([^}]+)\}|\\texttt\{([^}]+)\}", where)
        tag = (tag.group(1) or tag.group(2)).replace("\\", "") if tag else where.strip()
        printed[(meth, tag)] = {"figure": fig.strip("$"), "n": n.strip(), "aggregation": agg.strip(),
                                "truncation": trunc, "collapsed": collapsed}

    taut5 = {m: five["by_method"][m]["inchikey_tautomer"]["recall@15"] for m in five["by_method"]}
    full = fulln["recall_at_15"]
    uncapped = setm["populations"]["full1170"]["by_mode"]["inchikey_tautomer"]

    # every row that carries a literal figure, bound to the one leaf that produces it
    want = {
        ("GRAIL", "tab:matchsens"): (full["GRAIL"]["inchikey_tautomer"],
                                     "results/match_sensitivity_fulln_paired.json GRAIL tautomer"),
        ("GRAIL", "tab:matchsens-subset"):
            (taut5["GRAIL"], "results/match_sensitivity_5method.json GRAIL tautomer recall@15"),
        ("SyGMa", "set_metrics_by_criterion"):
            (uncapped["SyGMa"]["recall"]["point"],
             "results/set_metrics_by_criterion.json full1170 SyGMa recall"),
        ("SyGMa", "tab:matchsens"): (full["SyGMa"]["inchikey_tautomer"],
                                     "results/match_sensitivity_fulln_paired.json SyGMa tautomer"),
        ("SyGMa", "sec:xmc"): (anchor["mean_recall_SyGMa"],
                               "results/anchor_certification.json mean_recall_SyGMa"),
        ("SyGMa", "tab:matchsens-subset"):
            (taut5["SyGMa"], "results/match_sensitivity_5method.json SyGMa tautomer recall@15"),
        ("MetaPredictor", "tab:matchsens"):
            (full["MetaPredictor"]["inchikey_tautomer"],
             "results/match_sensitivity_fulln_paired.json MetaPredictor tautomer"),
        ("MetaPredictor", "tab:matchsens-subset"):
            (taut5["MetaPredictor"],
             "results/match_sensitivity_5method.json MetaPredictor tautomer recall@15"),
    }
    # the population, truncation and collapse columns are the whole point of the table: a figure
    # here is only distinguished from its neighbours by the footing its row declares
    footing = {
        "tab:matchsens": ("\\ntest", "macro", "$k{=}15$", "yes",
                          fulln["n_substrates"], "results/match_sensitivity_fulln_paired.json"),
        "tab:matchsens-subset": ("\\nshared", "macro", "$k{=}15$", "yes",
                                 five["n_substrates"], "results/match_sensitivity_5method.json"),
        "set_metrics_by_criterion": ("\\ntest", "macro", "none", "yes",
                                     setm["populations"]["full1170"]["n_substrates"],
                                     "results/set_metrics_by_criterion.json full1170"),
        "sec:xmc": (None, "macro", "$k{=}15$", "yes", anchor["common_n"],
                    "results/anchor_certification.json common_n"),
    }
    for key, (computed, note) in want.items():
        row = printed.get(key)
        ctx.checks.append((row is not None,
                           f"xdomain, provenance row {key[0]} at {key[1]} is printed", "present",
                           "present" if row else "missing", "paper/app/xdomain.tex"))
        if not row:
            continue
        _cell(ctx, f"xdomain, provenance figure, {key[0]} at {key[1]}", row["figure"],
              computed, note)
        macro, agg, trunc, collapsed, n_art, n_note = footing[key[1]]
        printed_n = row["n"].strip("$")
        if macro:
            ctx.checks.append((printed_n == macro,
                               f"xdomain, provenance population, {key[0]} at {key[1]}", macro,
                               printed_n, "paper/app/xdomain.tex"))
            printed_n = _macro(ctx, macro.lstrip("\\"))
        _cell(ctx, f"xdomain, provenance population size, {key[0]} at {key[1]}",
              (printed_n or "").replace("{,}", "").split()[0], n_art, n_note)
        # MetaPredictor's row reads "either" because the budget does not bind it, which is the
        # separate check below; every other row names the truncation it was scored under
        trunc = "either" if key == ("MetaPredictor", "tab:matchsens") else trunc
        ctx.checks.append((row["aggregation"] == agg and row["truncation"] == trunc
                           and row["collapsed"] == collapsed,
                           f"xdomain, provenance footing, {key[0]} at {key[1]}",
                           f"{agg} / {trunc} / {collapsed}",
                           f"{row['aggregation']} / {row['truncation']} / {row['collapsed']}",
                           "paper/app/xdomain.tex"))

    # the one row whose figure is printed only in another appendix, tied to both printings
    budget = printed.get(("GRAIL", "app:budget"))
    m = _blk(ctx, "the budget appendix's reading of the same two figures",
             r"GRAIL reads \$([\d.]+)\$ here against \$([\d.]+)\$ there on the same substrates")
    if m and budget:
        _cell(ctx, "xdomain, provenance figure, GRAIL at app:budget", budget["figure"],
              float(m.group(1)),
              "the figure Appendix~\\ref{app:budget} prints, which this row points at")
        subset = printed.get(("GRAIL", "tab:matchsens-subset"))
        if subset:
            _cell(ctx, "xdomain, the subset figure the budget appendix reads back", m.group(2),
                  float(subset["figure"]),
                  "the provenance table's own GRAIL row at tab:matchsens-subset")

    # the row that names the common set carries its size
    m = re.search(r"SyGMa \$0\.572\$ & \\S\\ref\{sec:xmc\} & \$([\d,{}]+)\$ common", body)
    ctx.checks.append((bool(m), "xdomain, the 0.572 row names its common set", "present",
                       "present" if m else "missing", "paper/app/xdomain.tex"))
    if m:
        _cell(ctx, "xdomain, the common set behind SyGMa 0.572", m.group(1).replace("{,}", ""),
              anchor["common_n"], "results/anchor_certification.json common_n")

    # MetaPredictor's row claims truncation does not bind it: capped and uncapped must agree
    _cell(ctx, "xdomain, MetaPredictor is unaffected by the budget",
          f"{full['MetaPredictor']['inchikey_tautomer']:.3f}",
          uncapped["MetaPredictor"]["recall"]["point"],
          "k=15 and untruncated agree, which is what the row's 'either' asserts")

    # the prose reads a gap off two of those rows
    m = _blk(ctx, "the SyGMa-MetaPredictor gap between the two rows",
             r"SyGMa ahead by \$([\d.]+)\$ untruncated and by \$([\d.]+)\$ at \$k\{=\}15\$")
    mp = printed.get(("MetaPredictor", "tab:matchsens"))
    if m and mp:
        for i, (label, key) in enumerate(
                (("untruncated", ("SyGMa", "set_metrics_by_criterion")),
                 ("k=15", ("SyGMa", "tab:matchsens")))):
            row = printed.get(key)
            if row is None:
                continue
            _cell(ctx, f"xdomain, the {label} gap between the printed rows", m.group(i + 1),
                  float(row["figure"]) - float(mp["figure"]),
                  f"the provenance table's own {key[0]} row at {key[1]} and MetaPredictor's at "
                  f"tab:matchsens")

    # 0.340 is printed twice: in the table and in the subsection that reads it
    _count(ctx, "GRAIL's collapsed full-split recall is printed twice", r"\$0\.340\$", 2,
           "tab:provenance and the subsection below it")
    m = _blk(ctx, "the sentence that defines the collapsed figure",
             r"On the full split, \$([\d.]+)\$ is the mean of per-substrate recalls after duplicate "
             r"predictions are collapsed")
    if m:
        _cell(ctx, "xdomain, the collapsed full-split figure", m.group(1),
              full["GRAIL"]["inchikey_tautomer"],
              "results/match_sensitivity_fulln_paired.json GRAIL tautomer")


# --------------------------------------------------------------------------------------------
# The five-method comparison (tab:matchsens-subset and tab:setmetrics)
# --------------------------------------------------------------------------------------------
def _five_method(ctx) -> None:
    five = ctx.art("match_sensitivity_5method.json")
    setm = ctx.art("set_metrics_by_criterion.json")
    mp = ctx.art("metapredictor_1170.json")

    if five is not None:
        blk = _blk(ctx, "the five-method criterion table",
                   r"\\label\{tab:matchsens-subset\}(.{0,1400}?)\\bottomrule")
        if blk:
            rows = re.findall(r"(?:\\textbf\{)?(GRAIL|BioTransformer|SyGMa|MetaTrans|MetaPredictor)"
                              r"\}? *& " + N + r" & " + N + r" & " + N + r" & " + N + r" & " + N
                              + r" & (?:\\phantom\{0\})?([\d.]+) \\\\", blk.group(1))
            ctx.checks.append((len(rows) == 5, "xdomain, the five-method table has five rows", "5",
                               f"{len(rows)} parsed", "paper/app/xdomain.tex"))
            modes = ("canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer")
            for row in rows:
                meth, cells, out = row[0], row[1:6], row[6]
                by = five["by_method"][meth]
                for mode, cell in zip(modes, cells):
                    _cell(ctx, f"xdomain, five-method table, {meth} {mode}", cell,
                          by[mode]["recall@15"],
                          "results/match_sensitivity_5method.json")
                _cell(ctx, f"xdomain, five-method table, {meth} mean outputs", out,
                      by["inchikey_tautomer"]["mean_output"],
                      "results/match_sensitivity_5method.json mean_output")

    if mp is not None:
        m = _blk(ctx, "the MetaPredictor full-split rerun",
                 r"MetaPredictor is also scored on \$([\d,{}]+)\$ of the test substrates, at "
                 r"recall@15 of ([\d.]+)")
        if m:
            _cell(ctx, "xdomain, MetaPredictor's full-split substrates",
                  m.group(1).replace("{,}", ""), mp["n_substrates"],
                  "results/metapredictor_1170.json")
            _cell(ctx, "xdomain, MetaPredictor's full-split recall", m.group(2),
                  mp["recall_at"]["15"], "results/metapredictor_1170.json")

    if setm is None:
        return
    sh = setm["populations"]["shared150"]["by_mode"]["inchikey_tautomer"]
    blk = _blk(ctx, "the four-aggregate table",
               r"\\label\{tab:setmetrics\}(.{0,1400}?)\\bottomrule")
    if blk:
        rows = re.findall(r"(?:\\textbf\{)?(MetaTrans|MetaPredictor|BioTransformer|GRAIL|SyGMa)\}?"
                          r" *& (?:\\textbf\{)?([\d.]+)\}? & (?:\\textbf\{)?([\d.]+)\}? & "
                          r"(?:\\textbf\{)?([\d.]+)\}? & (?:\\textbf\{)?([\d.]+)\}? & "
                          r"(?:\\phantom\{0\})?([\d.]+) & (\d) & (\d) \\\\", blk.group(1))
        ctx.checks.append((len(rows) == 5, "xdomain, the aggregate table has five rows", "5",
                           f"{len(rows)} parsed", "paper/app/xdomain.tex"))
        rk_r = sh["_ranking"]["recall"]
        rk_f = sh["_ranking"]["f1"]
        for meth, prec, rec, f1, jac, out, rr, rf in rows:
            src = "results/set_metrics_by_criterion.json shared150 inchikey_tautomer"
            _cell(ctx, f"xdomain, aggregate table, {meth} precision", prec,
                  sh[meth]["precision"]["point"], src)
            _cell(ctx, f"xdomain, aggregate table, {meth} recall", rec,
                  sh[meth]["recall"]["point"], src)
            _cell(ctx, f"xdomain, aggregate table, {meth} F1", f1, sh[meth]["f1"]["point"], src)
            _cell(ctx, f"xdomain, aggregate table, {meth} Jaccard", jac,
                  sh[meth]["jaccard"]["point"], src)
            _cell(ctx, f"xdomain, aggregate table, {meth} outputs", out, sh[meth]["mean_output"],
                  src)
            _cell(ctx, f"xdomain, aggregate table, {meth} rank by recall", rr,
                  rk_r.index(meth) + 1, src)
            _cell(ctx, f"xdomain, aggregate table, {meth} rank by F1", rf, rk_f.index(meth) + 1, src)

    pd = sh["_paired_diffs"]
    m = _blk(ctx, "the widest F1 separation among the selecting methods",
             r"the widest separation being GRAIL against MetaTrans at " + DN + r" " + CI)
    if m:
        src = "results/set_metrics_by_criterion.json shared150 _paired_diffs GRAIL-MetaTrans|f1"
        _cell(ctx, "xdomain, GRAIL against MetaTrans on F1", m.group(1),
              pd["GRAIL-MetaTrans|f1"]["point"], src)
        _cell(ctx, "xdomain, GRAIL against MetaTrans on F1, lo", m.group(2),
              pd["GRAIL-MetaTrans|f1"]["ci95"][0], src)
        _cell(ctx, "xdomain, GRAIL against MetaTrans on F1, hi", m.group(3),
              pd["GRAIL-MetaTrans|f1"]["ci95"][1], src)

    m = _blk(ctx, "SyGMa's position on F1 and on recall",
             r"SyGMa is below all four, " + DN + r" " + CI + r" against GRAIL rising to " + DN
             + r" " + CI + r" against MetaTrans\. On recall it is above two of them, " + DN + r" "
             + CI + r" against BioTransformer and " + DN + r" " + CI + r" against GRAIL")
    if m:
        g = m.groups()
        for i, key in enumerate(("GRAIL-SyGMa|f1", "MetaTrans-SyGMa|f1",
                                 "BioTransformer-SyGMa|recall", "GRAIL-SyGMa|recall")):
            src = f"results/set_metrics_by_criterion.json shared150 _paired_diffs {key}"
            _cell(ctx, f"xdomain, SyGMa against the field, {key}", g[3 * i], pd[key]["point"], src)
            _cell(ctx, f"xdomain, SyGMa against the field, {key} lo", g[3 * i + 1],
                  pd[key]["ci95"][0], src)
            _cell(ctx, f"xdomain, SyGMa against the field, {key} hi", g[3 * i + 2],
                  pd[key]["ci95"][1], src)

    m = _blk(ctx, "the width of the widest F1 interval among the selecting methods",
             r"the widest of those intervals still admits a difference of \$([\d.]+)\$ at "
             r"\$n=\\nshared\$")
    if m:
        sel = ("GRAIL", "BioTransformer", "MetaTrans", "MetaPredictor")
        widest = max(max(abs(v) for v in pd[k]["ci95"])
                     for k in pd
                     if k.endswith("|f1") and all(p in sel for p in k[:-3].split("-")))
        _cell(ctx, "xdomain, the widest F1 difference still admitted", m.group(1), widest,
              "results/set_metrics_by_criterion.json, widest |CI| over selecting-method F1 pairs")


# --------------------------------------------------------------------------------------------
# How much of the ordering is real
# --------------------------------------------------------------------------------------------
def _ordering_stability(ctx) -> None:
    art = ctx.art("ordering_stability.json")
    if art is None:
        return
    blk = _blk(ctx, "the ordering-stability table",
               r"what is allowed to vary & cells & orderings & pairs changing sign & separated both "
               r"ways \\\\(.{0,700}?)\\bottomrule")
    if not blk:
        return
    rows = re.findall(r"(criterion and aggregate|criterion alone, at recall|criterion alone, at F1|"
                      r"aggregate alone, at one criterion) *& \$(\d+)\$ & \$(\d+)\$ & \$(\d+)/(\d+)\$"
                      r" & \$(\d+)/(\d+)\$ \\\\", blk.group(1))
    ctx.checks.append((len(rows) == 4, "xdomain, the ordering-stability table has four rows", "4",
                       f"{len(rows)} parsed", "paper/app/xdomain.tex"))
    slug = {"criterion and aggregate": "criterion_and_aggregate",
            "criterion alone, at recall": "criterion_alone_at_recall",
            "criterion alone, at F1": "criterion_alone_at_f1",
            "aggregate alone, at one criterion": "aggregate_alone_at_inchikey_tautomer"}
    for label, cells, orders, sign, sign_d, cert, cert_d in rows:
        s = art["slices"][slug[label]]
        src = f"results/ordering_stability.json slices/{slug[label]}"
        _cell(ctx, f"xdomain, ordering stability, {label}, cells", cells, s["cells"], src)
        _cell(ctx, f"xdomain, ordering stability, {label}, orderings", orders,
              s["distinct_orderings"], src)
        _cell(ctx, f"xdomain, ordering stability, {label}, pairs changing sign", sign,
              s["pairs_changing_sign"], src)
        _cell(ctx, f"xdomain, ordering stability, {label}, pairs the sign count is over",
              sign_d, s["pairs"], src)
        _cell(ctx, f"xdomain, ordering stability, {label}, separated both ways", cert,
              s["pairs_reversing_certified"], src)
        _cell(ctx, f"xdomain, ordering stability, {label}, pairs the certified count is over",
              cert_d, s["pairs"], src)
        ctx.checks.append((len(s["certified"]) == s["pairs_reversing_certified"],
                           f"xdomain, ordering stability, {label}, certified list matches its count",
                           str(s["pairs_reversing_certified"]), str(len(s["certified"])), src))


# --------------------------------------------------------------------------------------------
# Four methods, one population
# --------------------------------------------------------------------------------------------
def _four_method(ctx) -> None:
    art = ctx.art("four_method_291.json")
    parent = ctx.art("parent_sensitivity.json")
    if art is None:
        return

    m = _blk(ctx, "the MetaTox population sentence",
             r"MetaTox supplies predictions for \$(\d+)\$ test substrates")
    if m:
        _cell(ctx, "xdomain, the four-method population", m.group(1), art["config"]["n_substrates"],
              "results/four_method_291.json config/n_substrates")

    blk = _blk(ctx, "the four-method budget table",
               r"method & emitted & r@\$5\$ & r@\$10\$ & r@\$15\$ & r@\$30\$ & r@\$50\$ "
               r"\\\\(.{0,700}?)\\bottomrule")
    if blk:
        rows = re.findall(r"(SyGMa|MetaTox|MetaPredictor|GRAIL) *& \$([\d.]+)\$ & \$([\d.]+)\$ & "
                          r"\$([\d.]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$ \\\\",
                          blk.group(1))
        ctx.checks.append((len(rows) == 4, "xdomain, the four-method table has four rows", "4",
                           f"{len(rows)} parsed", "paper/app/xdomain.tex"))
        for meth, emitted, *cells in rows:
            pm = art["per_method"][meth]
            src = f"results/four_method_291.json per_method/{meth}"
            _cell(ctx, f"xdomain, four-method table, {meth} emitted", emitted,
                  pm["mean_emitted_uncapped"], src)
            for k, cell in zip(("5", "10", "15", "30", "50"), cells):
                _cell(ctx, f"xdomain, four-method table, {meth} r@{k}", cell, pm["recall"][k], src)
        ctx.checks.append((len(art["orderings"]) == 3,
                           "xdomain, the sweep takes three orderings", "3",
                           str(len(art["orderings"])), "results/four_method_291.json"))

    m = _blk(ctx, "MetaTox's move across the sweep",
             r"MetaTox is last of the four at every budget up to \$(\d+)\$ and first of them from "
             r"\$(\d+)\$ on")
    if m:
        ks = sorted(art["config"]["k_sweep"])
        rec = {k: {meth: art["per_method"][meth]["recall"][str(k)] for meth in art["per_method"]}
               for k in ks}
        last = [k for k in ks if min(rec[k], key=rec[k].get) == "MetaTox"]
        first = [k for k in ks if max(rec[k], key=rec[k].get) == "MetaTox"]
        # the largest budget up to which it is last at every budget, and the smallest from which
        # it leads at every budget: a run of budgets, not a single cell that happens to agree
        last_upto = max(k for k in last if set(last) >= {j for j in ks if j <= k})
        first_from = min(k for k in first if set(first) >= {j for j in ks if j >= k})
        _cell(ctx, "xdomain, the last budget at which MetaTox is last", m.group(1), last_upto,
              "results/four_method_291.json per_method recalls")
        _cell(ctx, "xdomain, the first budget at which MetaTox leads", m.group(2), first_from,
              "results/four_method_291.json per_method recalls")

    m = _blk(ctx, "the parent-return counts",
             r"GRAIL returns the substrate itself on \$(\d+)\$ of the \$(\d+)\$, MetaPredictor on "
             r"\$(\d+)\$ and MetaTox on \$(\d+)\$")
    if m:
        pr = art["parent_returned_by_method"]
        src = "results/four_method_291.json parent_returned_by_method"
        _cell(ctx, "xdomain, GRAIL returns the parent", m.group(1), pr["GRAIL"], src)
        _cell(ctx, "xdomain, the population the parents are counted over", m.group(2),
              art["config"]["n_substrates"], "results/four_method_291.json")
        _cell(ctx, "xdomain, MetaPredictor returns the parent", m.group(3), pr["MetaPredictor"], src)
        _cell(ctx, "xdomain, MetaTox returns the parent", m.group(4), pr["MetaTox"], src)
        ctx.checks.append((pr["SyGMa"] == 0, "xdomain, SyGMa's parent count is zero", "zero",
                           str(pr["SyGMa"]), src))

    if parent is not None:
        m = _blk(ctx, "the parent-kept sensitivity",
                 r"GRAIL's recall@\$15\$ moves from \$([\d.]+)\$ to \$([\d.]+)\$")
        if m:
            _cell(ctx, "xdomain, GRAIL at k=15 with the parent dropped", m.group(1),
                  parent["parent_dropped"]["recall"]["GRAIL"]["15"],
                  "results/parent_sensitivity.json parent_dropped")
            _cell(ctx, "xdomain, GRAIL at k=15 with the parent kept", m.group(2),
                  parent["parent_kept"]["recall"]["GRAIL"]["15"],
                  "results/parent_sensitivity.json parent_kept")
            ctx.checks.append((parent["parent_dropped"]["orderings"]
                               == parent["parent_kept"]["orderings"],
                               "xdomain, the orderings survive keeping the parent",
                               "the same three orderings at the same budgets",
                               "same" if parent["parent_dropped"]["orderings"]
                               == parent["parent_kept"]["orderings"] else "different",
                               "results/parent_sensitivity.json"))


# --------------------------------------------------------------------------------------------
# The multiplicity correction (tab:holm)
# --------------------------------------------------------------------------------------------
def _holm_table(ctx) -> None:
    art = ctx.art("multiplicity_holm.json")
    c0 = ctx.art("retro_leaderboard_cluster0.json")
    if art is None:
        return

    cap = _blk(ctx, "the Holm table's caption",
               r"the last row is the second confirmatory comparison, on \$(\d+)\$ test substrates\."
               r".{0,120}?the \$(\d+)\$ paired cross-domain interactions are corrected together in "
               r"\\S\\ref\{app:xdomainflip\}, where \$(\d+)\$ survive")
    if cap:
        est = art["selection_confirmatory"]["estimand"]
        n = int(re.search(r"n=(\d+)", est).group(1))
        _cell(ctx, "xdomain, the second confirmatory comparison's substrates", cap.group(1), n,
              "results/multiplicity_holm.json selection_confirmatory estimand")
        if c0 is not None:
            _cell(ctx, "xdomain, the cross-domain family in the Holm caption", cap.group(2),
                  c0["n_interaction_tests"], "results/retro_leaderboard_cluster0.json")
            _cell(ctx, "xdomain, the cross-domain survivors in the Holm caption", cap.group(3),
                  len(c0["holm_survivors"]), "results/retro_leaderboard_cluster0.json")

    blk = _blk(ctx, "the Holm table", r"\\label\{tab:holm\}(.{0,2600}?)\\bottomrule")
    if not blk:
        return
    body = blk.group(1)

    sizes = re.findall(r"(external|internal|selection) \(\$m\{=\}(\d+)\$\)", body)
    ctx.checks.append((len(sizes) == 3, "xdomain, the Holm table declares three families", "3",
                       f"{len(sizes)} parsed", "paper/app/xdomain.tex"))
    fam_art = {"external": "external_grid", "internal": "internal_grid",
               "selection": "selection_confirmatory"}
    for fam, m_ in sizes:
        _cell(ctx, f"xdomain, Holm family size, {fam}", m_, art[fam_art[fam]]["family_size"],
              f"results/multiplicity_holm.json {fam_art[fam]}")

    rows = re.findall(r"(MetaPredictor vs SyGMa|GRAIL vs SyGMa|BioTransformer vs GRAIL|"
                      r"BioTransformer vs MetaPredictor|BioTransformer vs SyGMa|"
                      r"MetaPredictor vs GRAIL|GRAIL vs BioTransformer|MetaTrans vs SyGMa|"
                      r"learned selector vs frequency prior) *& "
                      r"\$([\d.]+)(?:\\times10\^\{(-\d+)\})?\$ & "
                      r"\$([\d.]+)(?:\\times10\^\{(-\d+)\})?\$ & \$([\d.]+)\$ & "
                      r"(rejected|not rejected) \\\\", body)
    ctx.checks.append((len(rows) == 9, "xdomain, the Holm table has nine rows", "9",
                       f"{len(rows)} parsed", "paper/app/xdomain.tex"))

    lookup = {}
    for fam, key in fam_art.items():
        for r in art[key]["result"]:
            lookup[r["pair"].replace("_", " ")] = (key, r)
    for pair, p_m, p_e, h_m, h_e, thr, decision in rows:
        if pair not in lookup:
            ctx.checks.append((False, f"xdomain, Holm row {pair} has an artifact row", "present",
                               "missing", "results/multiplicity_holm.json"))
            continue
        key, r = lookup[pair]
        src = f"results/multiplicity_holm.json {key}"
        p_printed = float(p_m) * (10 ** int(p_e) if p_e else 1)
        h_printed = float(h_m) * (10 ** int(h_e) if h_e else 1)
        digits = len(p_m.split(".")[1]) if "." in p_m else 0
        _cell(ctx, f"xdomain, Holm raw p, {pair}", f"{p_printed:.{digits + (-int(p_e) if p_e else 0)}f}",
              r["p"], src)
        digits = len(h_m.split(".")[1]) if "." in h_m else 0
        _cell(ctx, f"xdomain, Holm adjusted p, {pair}",
              f"{h_printed:.{digits + (-int(h_e) if h_e else 0)}f}", r["p_holm"], src)
        _cell(ctx, f"xdomain, Holm threshold, {pair}", thr, r["threshold"], src)
        ctx.checks.append(((decision == "rejected") == r["rejected"],
                           f"xdomain, Holm decision, {pair}", decision,
                           "rejected" if r["rejected"] else "not rejected", src))

    m = _blk(ctx, "the interaction that misses its threshold",
             r"MetaTrans--SyGMa misses its threshold of \$([\d.]+)\$ at \$p=([\d.]+)\$")
    if m:
        r = [x for x in art["internal_grid"]["result"] if x["pair"] == "MetaTrans_vs_SyGMa"][0]
        _cell(ctx, "xdomain, the missed threshold", m.group(1), r["threshold"],
              "results/multiplicity_holm.json internal_grid")
        _cell(ctx, "xdomain, the p that misses it", m.group(2), r["p"],
              "results/multiplicity_holm.json internal_grid")

    m = _blk(ctx, "the selection comparison quoted in prose",
             r"rule-selection gap of \\S\\ref\{sec:xmc\}, " + DN + r" " + CI
             + r" macro at \$k=15\$ on \$(\d+)\$ test substrates")
    if m:
        sc = art["selection_confirmatory"]
        _cell(ctx, "xdomain, the learned-versus-prior gap", m.group(1), sc["delta"],
              "results/multiplicity_holm.json selection_confirmatory delta")
        _cell(ctx, "xdomain, the learned-versus-prior gap lo", m.group(2), sc["ci95"][0],
              "results/multiplicity_holm.json selection_confirmatory")
        _cell(ctx, "xdomain, the learned-versus-prior gap hi", m.group(3), sc["ci95"][1],
              "results/multiplicity_holm.json selection_confirmatory")
        _cell(ctx, "xdomain, the substrates behind that gap", m.group(4),
              int(re.search(r"n=(\d+)", sc["estimand"]).group(1)),
              "results/multiplicity_holm.json selection_confirmatory estimand")

    m = _blk(ctx, "the alpha the selection comparison is judged at",
             r"at \$p=2\.0\\times10\^\{-8\}\$ against \$([\d.]+)\$ the correction changes nothing")
    if m:
        _cell(ctx, "xdomain, the alpha of the selection family", m.group(1), art["alpha"],
              "results/multiplicity_holm.json alpha")


# --------------------------------------------------------------------------------------------
# The same endpoint on a scale-free footing
# --------------------------------------------------------------------------------------------
def _normalised_headroom(ctx) -> None:
    art = ctx.art("headroom_normalised.json")
    if art is None:
        return

    m = _blk(ctx, "the baselines the external arms enter from",
             r"MetaPredictor entering at \$([\d.]+)\$ against SyGMa's \$([\d.]+)\$ and gaining the "
             r"more")
    if m:
        ext = art["external"]["per_method"]
        _cell(ctx, "xdomain, MetaPredictor's external strict baseline", m.group(1),
              ext["MetaPredictor"]["recall_strict"], "results/headroom_normalised.json")
        _cell(ctx, "xdomain, SyGMa's external strict baseline", m.group(2),
              ext["SyGMa"]["recall_strict"], "results/headroom_normalised.json")

    blk = _blk(ctx, "the normalised-gain table",
               r"method & recall \(strict\) & raw gain & normalised gain \(95\\% CI\) "
               r"\\\\(.{0,1500}?)\\bottomrule")
    if blk:
        body = blk.group(1)
        panels = re.split(r"\\midrule", body)
        seen = 0
        for panel in panels:
            arm = ("external" if "external" in panel else
                   "internal" if "internal" in panel else None)
            if arm is None:
                continue
            rows = re.findall(r"(SyGMa|BioTransformer|MetaPredictor|GRAIL) *& ([\d.]+) & " + DN
                              + r" & " + DN + r" " + CI, panel)
            per = art[arm]["per_method"]
            ctx.checks.append((len(rows) == len(per),
                               f"xdomain, the normalised table's {arm} panel has every method",
                               str(len(per)), f"{len(rows)} parsed", "paper/app/xdomain.tex"))
            seen += len(rows)
            for meth, strict, raw, norm, lo, hi in rows:
                src = f"results/headroom_normalised.json {arm}/per_method/{meth}"
                _cell(ctx, f"xdomain, normalised table, {arm} {meth} strict", strict,
                      per[meth]["recall_strict"], src)
                _cell(ctx, f"xdomain, normalised table, {arm} {meth} raw gain", raw,
                      per[meth]["raw_gain"], src)
                _cell(ctx, f"xdomain, normalised table, {arm} {meth} normalised", norm,
                      per[meth]["normalised_gain"], src)
                _cell(ctx, f"xdomain, normalised table, {arm} {meth} normalised lo", lo,
                      per[meth]["normalised_gain_ci95"][0], src)
                _cell(ctx, f"xdomain, normalised table, {arm} {meth} normalised hi", hi,
                      per[meth]["normalised_gain_ci95"][1], src)
        ctx.checks.append((seen == 6, "xdomain, the normalised table has six rows", "6",
                           str(seen), "paper/app/xdomain.tex"))

    m = _blk(ctx, "the normalised interactions",
             r"MetaPredictor--SyGMa " + DN + r" " + CI + r", GRAIL--SyGMa " + DN + r" " + CI
             + r", BioTransformer--GRAIL " + DN + r" " + CI + r", BioTransformer--MetaPredictor "
             + DN + r" " + CI + r", and internally GRAIL--BioTransformer " + DN + r" " + CI)
    if m:
        g = m.groups()
        keys = [("external", "MetaPredictor_vs_SyGMa"), ("external", "GRAIL_vs_SyGMa"),
                ("external", "BioTransformer_vs_GRAIL"),
                ("external", "BioTransformer_vs_MetaPredictor"),
                ("internal", "GRAIL_vs_BioTransformer")]
        for i, (arm, key) in enumerate(keys):
            d = art[arm]["pairwise_normalised_differential"][key]
            src = f"results/headroom_normalised.json {arm}/pairwise_normalised_differential/{key}"
            _cell(ctx, f"xdomain, normalised interaction, {key}", g[3 * i], d["b_minus_a"], src)
            _cell(ctx, f"xdomain, normalised interaction, {key} lo", g[3 * i + 1], d["ci95"][0], src)
            _cell(ctx, f"xdomain, normalised interaction, {key} hi", g[3 * i + 2], d["ci95"][1], src)

    m = _blk(ctx, "the two normalised interactions that stay indistinguishable",
             r"BioTransformer--SyGMa " + DN + r" " + CI + r" and MetaPredictor--GRAIL " + DN + r" "
             + CI)
    if m:
        g = m.groups()
        for i, key in enumerate(("BioTransformer_vs_SyGMa", "MetaPredictor_vs_GRAIL")):
            d = art["external"]["pairwise_normalised_differential"][key]
            src = f"results/headroom_normalised.json external {key}"
            _cell(ctx, f"xdomain, normalised interaction, {key}", g[3 * i], d["b_minus_a"], src)
            _cell(ctx, f"xdomain, normalised interaction, {key} lo", g[3 * i + 1], d["ci95"][0], src)
            _cell(ctx, f"xdomain, normalised interaction, {key} hi", g[3 * i + 2], d["ci95"][1], src)

    m = _blk(ctx, "the note that the normalised GRAIL row is contaminated",
             r"is not the \$(\d+)\$-drug figure \\S\\ref\{sec:rankflip\} reports")
    if m:
        clean = ctx.art("gloryx_clean_subset.json")
        if clean is not None:
            _cell(ctx, "xdomain, the clean-drug count named beside the normalised row", m.group(1),
                  clean["n_clean"], "results/gloryx_clean_subset.json n_clean")


# --------------------------------------------------------------------------------------------
# The factorised generator
# --------------------------------------------------------------------------------------------
def _factorised(ctx) -> None:
    fac = ctx.art("factorized_eval_matched.json")
    rr = ctx.art("hybrid_rerank_full1170.json")
    fulln = ctx.art("match_sensitivity_fulln.json")
    m = _blk(ctx, "the factorised generator's headline",
             r"reaches recall@15 of " + DN + r" " + CI + r" on the full test split against the "
             r"deployed \$\\grailmacro\$, at \$([\d.]+)\$ candidates per substrate against "
             r"\$([\d.]+)\$")
    if m and fac is not None:
        ci = fac["recall@15_bootstrap_ci"]
        _cell(ctx, "xdomain, the factorised recall@15", m.group(1), fac["recall@15"],
              "results/factorized_eval_matched.json recall@15")
        _cell(ctx, "xdomain, the factorised recall lo", m.group(2), ci["lo"],
              "results/factorized_eval_matched.json")
        _cell(ctx, "xdomain, the factorised recall hi", m.group(3), ci["hi"],
              "results/factorized_eval_matched.json")
        _cell(ctx, "xdomain, the factorised candidates per substrate", m.group(4),
              fac["mean_output"], "results/factorized_eval_matched.json mean_output")
        if fulln is not None:
            _cell(ctx, "xdomain, the deployed candidates per substrate", m.group(5),
                  fulln["mean_output"], "results/match_sensitivity_fulln.json mean_output")

    m = _blk(ctx, "the re-ranking gain",
             r"re-ranking gain of " + DN + r", 95\\% confidence interval " + CI)
    if m and rr is not None:
        d = rr["paired_delta_recall15"]["c_minus_a"]
        _cell(ctx, "xdomain, the re-ranking gain", m.group(1), d["point"],
              "results/hybrid_rerank_full1170.json c_minus_a")
        _cell(ctx, "xdomain, the re-ranking gain lo", m.group(2), d["lo"],
              "results/hybrid_rerank_full1170.json")
        _cell(ctx, "xdomain, the re-ranking gain hi", m.group(3), d["hi"],
              "results/hybrid_rerank_full1170.json")


# --------------------------------------------------------------------------------------------
# The eleven released files are three populations
# --------------------------------------------------------------------------------------------
def _retro_groups(ctx) -> None:
    art = ctx.art("evalretro_ingest.json")
    if art is None:
        return
    cl = art["clusters"]

    m = _blk(ctx, "the clustering overlaps",
             r"the within-cluster overlap is \$([\d.]+)\$ and the between-cluster overlap is "
             r"\$([\d.]+)\$")
    if m:
        between = max(v["share_of_smaller"] for v in art["population_overlap"].values()
                      if "share_of_smaller" in v)
        _cell(ctx, "xdomain, the between-cluster overlap", m.group(2), between,
              "results/evalretro_ingest.json population_overlap, largest share_of_smaller")

    blk = _blk(ctx, "the three-population table",
               r"group & systems & reactions agreed & share of its products in our split "
               r"\\\\(.{0,500}?)\\bottomrule")
    if blk:
        rows = re.findall(r"(seven|three|one)-system *& (\d+) & \$([\d,{}]+)\$ & \$([\d.]+)\$ \\\\",
                          blk.group(1))
        ctx.checks.append((len(rows) == 3, "xdomain, the population table has three rows", "3",
                           f"{len(rows)} parsed", "paper/app/xdomain.tex"))
        want = {"seven": "cluster0", "three": "cluster1", "one": "cluster2"}
        for name, nsys, nrxn, share in rows:
            c = cl[want[name]]
            src = f"results/evalretro_ingest.json clusters/{want[name]}"
            _cell(ctx, f"xdomain, {name}-system group, systems", nsys, len(c["systems"]), src)
            _cell(ctx, f"xdomain, {name}-system group, reactions agreed",
                  nrxn.replace("{,}", ""), c["reactions"], src)
            _cell(ctx, f"xdomain, {name}-system group, share of its products in our split", share,
                  c["share_of_products_in_repo_test_split"], src)
        ctx.checks.append((sum(len(c["systems"]) for c in cl.values()) == art["n_systems"],
                           "xdomain, the three groups hold every released system",
                           str(art["n_systems"]),
                           str(sum(len(c["systems"]) for c in cl.values())),
                           "results/evalretro_ingest.json n_systems"))
        ctx.checks.append((len(cl) == art["n_test_sets"],
                           "xdomain, the groups are the declared number of test sets",
                           str(art["n_test_sets"]), str(len(cl)),
                           "results/evalretro_ingest.json n_test_sets"))

    _split_sizes(ctx)


def _split_sizes(ctx) -> None:
    """The size of this repository's USPTO-50k copy, which is the deviation the section reports.

    No artifact records the per-split count directly. What one does record is the training split
    the transfer probe was fit against, and the paper's own claim is that the three splits total the
    figure it prints; so the total is checked against that leaf plus the per-split figure the same
    sentence prints, which makes moving either of them fail. The count of test reactions the
    seven-system group carries, and the size the name is conventionally understood to denote, are
    figures about the literature rather than about a run here and are left unbound deliberately.
    """
    tr = ctx.art("retro_transfer.json")
    if tr is None:
        return
    m = _blk(ctx, "the size of our USPTO-50k copy",
             r"The three-system group carries \$([\d,{}]+)\$, and so does the copy of "
             r"``USPTO-50k'' in this repository, whose three splits total \$([\d,{}]+)\$ reactions")
    if not m:
        return
    per_split = int(m.group(1).replace("{,}", ""))
    total = int(m.group(2).replace("{,}", ""))
    _cell(ctx, "xdomain, our copy's three splits total what it prints", str(total),
          tr["n_train"] + 2 * per_split,
          "results/retro_transfer.json n_train plus the test and validation splits printed here")

    # the same per-split figure is printed three more times in this appendix and must not drift
    m2 = _blk(ctx, "the single-model probe's population",
              r"is on our own \$([\d,{}]+)\$-reaction copy")
    if m2:
        _cell(ctx, "xdomain, the probe's copy is the same size", m2.group(1).replace("{,}", ""),
              per_split, "the same split size the sentence above prints")
    m3 = _blk(ctx, "the structural reading verified against our own answers",
              r"is our own recorded reactants on \$([\d,{}]+)\$ of its \$([\d,{}]+)\$ reactions, "
              r"for all three of its systems")
    if m3:
        _cell(ctx, "xdomain, the reactions the structural reading is verified on",
              m3.group(1).replace("{,}", ""), per_split,
              "the same split size the sentence above prints")
        _cell(ctx, "xdomain, the reactions it is verified against", m3.group(2).replace("{,}", ""),
              per_split, "the same split size the sentence above prints")
        ctx.checks.append((m3.group(1) == m3.group(2),
                           "xdomain, the structural reading holds on every one of them",
                           m3.group(2), m3.group(1), "paper/app/xdomain.tex"))
    _count(ctx, "our copy's split size is printed four times", r"\$5\{,\}004\$", 4,
           "the deviation sentence, the probe's population, and both halves of the reading check")


# --------------------------------------------------------------------------------------------
# The seven-system run, and the three-system replication
# --------------------------------------------------------------------------------------------
def _retro_leaderboard(ctx) -> None:
    c0 = ctx.art("retro_leaderboard_cluster0.json")
    c1 = ctx.art("retro_leaderboard_cluster1.json")
    if c0 is None:
        return
    acc = c0["accuracy"]
    slug = {"Graph2SMILES": "graph2smiles", "GraphRetro": "graphretro",
            "Retroformer": "retroformer", "LocalRetro": "localretro", "GLN": "gln",
            "G2Retro": "g2retro", "RetroXpert": "retroxpert"}

    # the exchange stated in words at the top of the appendix, which prints four of the table's
    # cells a second time under each of the two conventions it contrasts
    m = _blk(ctx, "the exchange written out in full",
             r"Under canonical matching the leaders run GraphRetro \$([\d.]+)\$, LocalRetro "
             r"\$([\d.]+)\$, Retroformer \$([\d.]+)\$, Graph2SMILES \$([\d.]+)\$; disregard "
             r"stereochemistry and they run Graph2SMILES \$([\d.]+)\$, Retroformer \$([\d.]+)\$, "
             r"GraphRetro \$([\d.]+)\$, LocalRetro \$([\d.]+)\$, so fourth becomes first")
    if m:
        order = [("canonical", ["GraphRetro", "LocalRetro", "Retroformer", "Graph2SMILES"]),
                 ("nostereo", ["Graph2SMILES", "Retroformer", "GraphRetro", "LocalRetro"])]
        i = 0
        for mode, names in order:
            for name in names:
                _cell(ctx, f"xdomain, the exchange in words, {name} under {mode}", m.group(i + 1),
                      acc[slug[name]][mode]["top1"],
                      f"results/retro_leaderboard_cluster0.json accuracy/{slug[name]}/{mode}")
                i += 1
            top4 = c0["orderings"]["top1"][mode][:4]
            ctx.checks.append(([slug[n] for n in names] == top4,
                               f"xdomain, the exchange in words, the {mode} leading four",
                               " ".join(slug[n] for n in names), " ".join(top4),
                               "results/retro_leaderboard_cluster0.json orderings"))

    m = _blk(ctx, "the cell count the family is drawn from",
             r"The family is twenty-one system pairs by six pairs drawn from the four conventions "
             r"by four budgets, which is \$(\d+)\$ cells less the \$(\d+)\$ in which two "
             r"conventions return identical results and no test exists")
    if m:
        pairs = len(list(itertools.combinations(c0["config"]["systems"], 2)))
        conv = len(list(itertools.combinations(c0["config"]["modes"], 2)))
        budgets = len(c0["config"]["ks"])
        _cell(ctx, "xdomain, the cells the family is drawn from", m.group(1),
              pairs * conv * budgets,
              "results/retro_leaderboard_cluster0.json config: systems, modes and ks")
        _cell(ctx, "xdomain, the cells in which no test exists", m.group(2),
              pairs * conv * budgets - c0["n_interaction_tests"],
              "the cells above less results/retro_leaderboard_cluster0.json n_interaction_tests")

    _count(ctx, "the seven-system reaction count is printed five times", r"\$4\{,\}999\$", 5,
           "twice in the body, and in this appendix's group table, run paragraph and figure caption",
           whole=True)
    for i, m in enumerate(re.finditer(
            r"Seven systems, \$([\d,{}]+)\$ reactions, frozen predictions|"
            r"criteria, on the \$([\d,{}]+)\$ reactions their released prediction files share",
            ctx.flat), start=1):
        _cell(ctx, f"xdomain, the seven-system reactions, printing {i}",
              (m.group(1) or m.group(2)).replace("{,}", ""), c0["config"]["n_reactions"],
              "results/retro_leaderboard_cluster0.json")

    blk = _blk(ctx, "the seven-system top-1 table",
               r"system & canonical & \\textsc\{inchikey\} & no-stereo & tautomer "
               r"\\\\(.{0,1200}?)\\bottomrule")
    if blk:
        rows = re.findall(r"(Graph2SMILES|GraphRetro|Retroformer|LocalRetro|GLN|G2Retro|RetroXpert)"
                          r" *& (?:\\textbf\{)?([\d.]+)\}? & (?:\\textbf\{)?([\d.]+)\}? & "
                          r"(?:\\textbf\{)?([\d.]+)\}? & (?:\\textbf\{)?([\d.]+)\}? \\\\",
                          blk.group(1))
        ctx.checks.append((len(rows) == 7, "xdomain, the seven-system table has seven rows", "7",
                           f"{len(rows)} parsed", "paper/app/xdomain.tex"))
        for name, canon, ik, ns, taut in rows:
            a = acc[slug[name]]
            src = f"results/retro_leaderboard_cluster0.json accuracy/{slug[name]}"
            _cell(ctx, f"xdomain, seven-system table, {name} canonical", canon,
                  a["canonical"]["top1"], src)
            _cell(ctx, f"xdomain, seven-system table, {name} InChIKey", ik, a["inchikey"]["top1"],
                  src)
            _cell(ctx, f"xdomain, seven-system table, {name} no-stereo", ns, a["nostereo"]["top1"],
                  src)
            _cell(ctx, f"xdomain, seven-system table, {name} tautomer", taut, a["tautomer"]["top1"],
                  src)

    m = _blk(ctx, "the cross-domain certification counts",
             r"Of \$(\d+)\$ paired interaction tests \(every pair of systems, every pair of "
             r"conventions, every \$k\$\) \$(\d+)\$ have intervals excluding zero and "
             r"\\textbf\{\$(\d+)\$ survive Holm at \$0\.05\$ across the whole family\}")
    if m:
        src = "results/retro_leaderboard_cluster0.json"
        _cell(ctx, "xdomain, the cross-domain family size", m.group(1), c0["n_interaction_tests"],
              src)
        _cell(ctx, "xdomain, cross-domain intervals excluding zero", m.group(2),
              len(c0["certified_interactions"]), src)
        _cell(ctx, "xdomain, cross-domain Holm survivors", m.group(3), len(c0["holm_survivors"]),
              src)

    m = _blk(ctx, "the largest cross-domain interaction",
             r"The largest is Graph2SMILES against GraphRetro between canonical and stereo-blind "
             r"matching, " + DN + r" " + CI)
    if m:
        key = "top1|graph2smiles vs graphretro|canonical vs nostereo"
        hs = c0["holm_survivors"]
        largest = max(hs, key=lambda k: abs(hs[k]["delta"]))
        ctx.checks.append((largest == key, "xdomain, the largest survivor is the one named",
                           key, largest, "results/retro_leaderboard_cluster0.json"))
        src = f"results/retro_leaderboard_cluster0.json holm_survivors/{key}"
        _cell(ctx, "xdomain, the largest interaction", m.group(1), hs[key]["delta"], src)
        _cell(ctx, "xdomain, the largest interaction lo", m.group(2), hs[key]["ci95"][0], src)
        _cell(ctx, "xdomain, the largest interaction hi", m.group(3), hs[key]["ci95"][1], src)

    m = _blk(ctx, "the budgets at which nothing exchanges",
             r"at \$k=(\d+)\$, \$(\d+)\$ and \$(\d+)\$ the seven take a single ordering under all "
             r"four conventions and no pair exchanges")
    if m:
        for g in m.groups():
            tag = f"top{g}"
            orders = c0["orderings"].get(tag)
            if orders is None:
                ctx.checks.append((False, f"xdomain, k={g} is a budget the run evaluated",
                                   f"k={g}", "the run carries no such budget",
                                   "results/retro_leaderboard_cluster0.json orderings"))
                continue
            ctx.checks.append((len({tuple(v) for v in orders.values()}) == 1
                               and not c0["pairs_that_exchange"][tag],
                               f"xdomain, one ordering and no exchange at k={g}",
                               "one ordering, no exchange",
                               f"{len({tuple(v) for v in orders.values()})} orderings, "
                               f"{len(c0['pairs_that_exchange'][tag])} exchanges",
                               "results/retro_leaderboard_cluster0.json"))
        ctx.checks.append((len({tuple(v) for v in c0["orderings"]["top1"].values()}) > 1,
                           "xdomain, top-1 is the level that does move", "more than one ordering",
                           f"{len({tuple(v) for v in c0['orderings']['top1'].values()})} orderings",
                           "results/retro_leaderboard_cluster0.json"))

    if c1 is not None:
        m = _blk(ctx, "the three-system replication's family",
                 r"of its \$(\d+)\$ paired interactions survive Holm")
        if m:
            _cell(ctx, "xdomain, the three-system family size", m.group(1),
                  c1["n_interaction_tests"], "results/retro_leaderboard_cluster1.json")
        m = _blk(ctx, "the two largest differentials compared",
                 r"a largest differential of \$([\d.]+)\$ against \$([\d.]+)\$")
        if m:
            for i, (art, tag) in enumerate(((c1, "three-system"), (c0, "seven-system"))):
                hs = art["holm_survivors"]
                _cell(ctx, f"xdomain, the largest surviving differential, {tag}", m.group(i + 1),
                      max(abs(v["delta"]) for v in hs.values()),
                      f"results/retro_leaderboard_cluster{i ^ 1}.json holm_survivors")
        ctx.checks.append((not c1["pairs_that_exchange"]["top1"]
                           and len({tuple(v) for v in c1["orderings"]["top1"].values()}) == 1,
                           "xdomain, the three-system group does not reorder at top-1",
                           "one ordering, no exchange",
                           f"{len({tuple(v) for v in c1['orderings']['top1'].values()})} orderings, "
                           f"{len(c1['pairs_that_exchange']['top1'])} exchanges",
                           "results/retro_leaderboard_cluster1.json"))


# --------------------------------------------------------------------------------------------
# The other two domains, and the packing distribution
# --------------------------------------------------------------------------------------------
def _other_domains(ctx) -> None:
    pk = ctx.art("packing_vs_differential.json")
    mo = ctx.art("moses_uniqueness.json")
    tr = ctx.art("xdomain_retro_protocol.json")

    if pk is not None:
        per = pk["per_leaderboard"]
        m = _blk(ctx, "generation's median movement",
                 r"Generation is next at \$1\.4\\%\$, because its median movement is \$([\d.]+)\$")
        if m:
            _cell(ctx, "xdomain, generation's median movement", m.group(1),
                  per["molecular generation, MOSES"]["median_differential"],
                  "results/packing_vs_differential.json")
        m = _blk(ctx, "the two widest translation medians",
                 r"a median of \$([\d.]+)\$ and \$([\d.]+)\$, and are exposed at")
        if m:
            _cell(ctx, "xdomain, the en-de median gap", m.group(1),
                  per["translation, WMT24 en-de"]["median_gap"],
                  "results/packing_vs_differential.json")
            _cell(ctx, "xdomain, the ja-zh median gap", m.group(2),
                  per["translation, WMT24 ja-zh"]["median_gap"],
                  "results/packing_vs_differential.json")

    if mo is not None:
        m = _blk(ctx, "the MOSES family size",
                 r"certified across a family of \$(\d+)\$ interactions and it is entirely "
                 r"stereochemical")
        if m:
            _cell(ctx, "xdomain, the MOSES family size", m.group(1), mo["n_interactions"],
                  "results/moses_uniqueness.json n_interactions")
        m = _blk(ctx, "the one non-zero tautomer difference on MOSES",
                 r"the single non-zero difference being JT-VAE's \$([\d.]+)\$ against \$([\d.]+)\$")
        if m:
            _cell(ctx, "xdomain, JT-VAE under canonical matching", m.group(1),
                  mo["uniqueness"]["canonical"]["jtn"]["unique@10000"],
                  "results/moses_uniqueness.json uniqueness/canonical/jtn")
            _cell(ctx, "xdomain, JT-VAE under tautomer matching", m.group(2),
                  mo["uniqueness"]["inchikey_tautomer"]["jtn"]["unique@10000"],
                  "results/moses_uniqueness.json uniqueness/inchikey_tautomer/jtn")
        m = _blk(ctx, "the combinatorial baseline's null movement",
                 r"the combinatorial baseline moves by exactly \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$")
        if m:
            d = mo["criterion_effect"]["inchi_no_stereo"]["combinatorial"]
            src = "results/moses_uniqueness.json criterion_effect/inchi_no_stereo/combinatorial"
            _cell(ctx, "xdomain, the combinatorial baseline's movement", m.group(1), d["delta"], src)
            _cell(ctx, "xdomain, the combinatorial baseline's movement lo", m.group(2),
                  d["ci95"][0], src)
            _cell(ctx, "xdomain, the combinatorial baseline's movement hi", m.group(3),
                  d["ci95"][1], src)
        m = _blk(ctx, "the MOSES rank exchange",
                 r"LatentGAN past the combinatorial baseline, an interaction of " + DN + r" " + CI)
        if m:
            d = mo["interactions"]["inchi_no_stereo"]["combinatorial vs latent_gan"]
            src = "results/moses_uniqueness.json interactions/inchi_no_stereo"
            _cell(ctx, "xdomain, the MOSES interaction", m.group(1), d["delta"], src)
            _cell(ctx, "xdomain, the MOSES interaction lo", m.group(2), d["ci95"][0], src)
            _cell(ctx, "xdomain, the MOSES interaction hi", m.group(3), d["ci95"][1], src)
            ctx.checks.append((sorted(mo["models_that_moved"]) == ["combinatorial", "latent_gan"],
                               "xdomain, the two models the criterion moves",
                               "combinatorial and latent_gan",
                               ", ".join(sorted(mo["models_that_moved"])),
                               "results/moses_uniqueness.json models_that_moved"))
        m = _blk(ctx, "the packing condition on MOSES",
                 r"Seven of the eight sit at \$([\d.]+)\$ or above")
        if m:
            vals = sorted(v["unique@10000"] for v in mo["uniqueness"]["canonical"].values())
            _cell(ctx, "xdomain, the level seven of the eight sit at or above", m.group(1), vals[1],
                  "results/moses_uniqueness.json uniqueness/canonical, second smallest")

    if tr is not None:
        m = _blk(ctx, "the single-model retrosynthesis probe",
                 r"Scoring ReactionT5v2 on (\d+) substrates.{0,400}?move only about ([\d.]+) in top-1 "
                 r"accuracy across canonical, stereo-stripped, InChIKey and tautomer matching, "
                 r"between ([\d.]+) and ([\d.]+)")
        if m:
            tops = {k: v["top1"] for k, v in tr["accuracy_by_mode"].items()}
            _cell(ctx, "xdomain, the probe's substrates", m.group(1), tr["n"],
                  "results/xdomain_retro_protocol.json n")
            _cell(ctx, "xdomain, the probe's spread", m.group(2), tr["spread_across_modes"]["top1"],
                  "results/xdomain_retro_protocol.json spread_across_modes")
            _cell(ctx, "xdomain, the probe's lowest top-1", m.group(3), min(tops.values()),
                  "results/xdomain_retro_protocol.json accuracy_by_mode")
            _cell(ctx, "xdomain, the probe's highest top-1", m.group(4), max(tops.values()),
                  "results/xdomain_retro_protocol.json accuracy_by_mode")
        m = _blk(ctx, "the probe's movement quoted again as a bound",
                 r"the movement here is about \$([\d.]+)\$, the seven published systems")
        if m:
            _cell(ctx, "xdomain, the probe's spread, second printing", m.group(1),
                  tr["spread_across_modes"]["top1"],
                  "results/xdomain_retro_protocol.json spread_across_modes")


# --------------------------------------------------------------------------------------------
# The external set is not external to GRAIL
# --------------------------------------------------------------------------------------------
def _contamination(ctx) -> None:
    audit = ctx.art("external_overlap_audit.json")
    grid = ctx.art("gloryx_criterion_grid.json")
    clean = ctx.art("gloryx_clean_subset.json")
    flip = ctx.art("gloryx_rank_flip_ci.json")
    if not (audit and grid and clean):
        return

    m = _blk(ctx, "the overlap sentence",
             r"\$(\d+)\$ of those \$(\d+)\$ parents are substrates in GRAIL's training split and "
             r"\$(\d+)\$ more are in validation")
    if m:
        _cell(ctx, "xdomain, the overlap sentence's GLORYx size", m.group(2),
              audit["GLORYx external set"]["n"], "results/external_overlap_audit.json")
        _cell(ctx, "xdomain, train and validation sum to the overlap, in prose",
              str(int(m.group(1)) + int(m.group(3))),
              audit["GLORYx external set"]["in_train_or_val"],
              "results/external_overlap_audit.json in_train_or_val")

    m = _blk(ctx, "the memorisation-signature sentence",
             r"MetaPredictor enters that step at \$([\d.]+)\$, below the rule engine's \$([\d.]+)\$")
    if m:
        a37 = grid["recall"]["all37"]
        _cell(ctx, "xdomain, MetaPredictor's strict external recall", m.group(1),
              a37["MetaPredictor"]["inchikey"], "results/gloryx_criterion_grid.json all37")
        _cell(ctx, "xdomain, SyGMa's strict external recall", m.group(2),
              a37["SyGMa"]["inchikey"], "results/gloryx_criterion_grid.json all37")

    m = _blk(ctx, "the recomputation on the thirteen",
             r"falls by up to a factor of \$([\d.]+)\$: \$([\d.]+) \\to ([\d.]+)\$ under strict "
             r"\\textsc\{inchikey\}, \$([\d.]+) \\to ([\d.]+)\$ once stereochemistry is disregarded, "
             r"\$([\d.]+) \\to ([\d.]+)\$ under the tautomer-aware default")
    if m:
        g = m.groups()
        a, c = grid["recall"]["all37"]["GRAIL"], grid["recall"]["clean13"]["GRAIL"]
        _cell(ctx, "xdomain, the largest ratio between the two populations", g[0],
              max(a[k] / c[k] for k in a if k in ("inchikey", "inchi_no_stereo",
                                                  "inchikey_tautomer")),
              "results/gloryx_criterion_grid.json, all37 over clean13")
        for i, mode in enumerate(("inchikey", "inchi_no_stereo", "inchikey_tautomer")):
            _cell(ctx, f"xdomain, GRAIL on all 37 under {mode}", g[1 + 2 * i], a[mode],
                  "results/gloryx_criterion_grid.json all37")
            _cell(ctx, f"xdomain, GRAIL on the thirteen under {mode}", g[2 + 2 * i], c[mode],
                  "results/gloryx_criterion_grid.json clean13")

    m = _blk(ctx, "the bound on the other three methods' movement",
             r"The other three methods move by less than \$([\d.]+)\$")
    if m:
        moved = max(abs(grid["recall"]["all37"][meth][mode] - grid["recall"]["clean13"][meth][mode])
                    for meth in ("BioTransformer", "MetaPredictor", "SyGMa")
                    for mode in ("inchikey", "inchi_no_stereo", "inchikey_tautomer"))
        printed = float(m.group(1))
        step = 10 ** -len(m.group(1).split(".")[1])
        ctx.checks.append((printed - step < moved <= printed,
                           "xdomain, the bound on the other three methods is the tight one",
                           m.group(1), f"{moved:.4f}",
                           "results/gloryx_criterion_grid.json, largest all37-clean13 movement"))

    m = _blk(ctx, "GRAIL's stereo step on the clean subset",
             r"GRAIL's stereo step remains significant at " + DN + r" " + CI + r" against " + DN
             + r" on the full set")
    if m:
        _cell(ctx, "xdomain, GRAIL's clean stereo step", m.group(1),
              clean["steps"]["GRAIL"]["stereo"], "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, GRAIL's clean stereo step lo", m.group(2),
              clean["steps"]["GRAIL"]["ci95"][0], "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, GRAIL's clean stereo step hi", m.group(3),
              clean["steps"]["GRAIL"]["ci95"][1], "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, GRAIL's stereo step on the full set", m.group(4),
              grid["stereo_step"]["all37"]["GRAIL"]["gain"],
              "results/gloryx_criterion_grid.json all37")

    m = _blk(ctx, "the interaction that is the same measurement",
             r"so the interaction is numerically the same measurement, " + DN + r" " + CI)
    if m:
        d = clean["pairwise"]["GRAIL_vs_SyGMa"]
        _cell(ctx, "xdomain, GRAIL against SyGMa on the thirteen", m.group(1), d["interaction"],
              "results/gloryx_clean_subset.json pairwise/GRAIL_vs_SyGMa")
        _cell(ctx, "xdomain, GRAIL against SyGMa on the thirteen, lo", m.group(2), d["ci95"][0],
              "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, GRAIL against SyGMa on the thirteen, hi", m.group(3), d["ci95"][1],
              "results/gloryx_clean_subset.json")
        ctx.checks.append((clean["steps"]["SyGMa"]["stereo"] == 0.0
                           and clean["steps"]["SyGMa"]["ci95"] == [0.0, 0.0],
                           "xdomain, SyGMa's stereo step on the thirteen is exactly zero", "0 [0,0]",
                           f"{clean['steps']['SyGMa']['stereo']} "
                           f"{clean['steps']['SyGMa']['ci95']}",
                           "results/gloryx_clean_subset.json"))

    m = _blk(ctx, "the interaction that does not survive",
             r"The interaction between BioTransformer and GRAIL does not: " + DN + r" " + CI
             + r" on all \$\\nglory\$, it is " + DN + r" " + CI + r" on the unseen drugs")
    if m and flip is not None:
        full = flip["pairwise"]["BioTransformer_vs_GRAIL"]["interaction_b_extra_gain"]
        cl = clean["pairwise"]["BioTransformer_vs_GRAIL"]
        _cell(ctx, "xdomain, BioTransformer against GRAIL on all 37", m.group(1), full["mean"],
              "results/gloryx_rank_flip_ci.json")
        _cell(ctx, "xdomain, BioTransformer against GRAIL on all 37, lo", m.group(2),
              full["ci95"][0], "results/gloryx_rank_flip_ci.json")
        _cell(ctx, "xdomain, BioTransformer against GRAIL on all 37, hi", m.group(3),
              full["ci95"][1], "results/gloryx_rank_flip_ci.json")
        _cell(ctx, "xdomain, BioTransformer against GRAIL on the thirteen", m.group(4),
              cl["interaction"], "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, BioTransformer against GRAIL on the thirteen, lo", m.group(5),
              cl["ci95"][0], "results/gloryx_clean_subset.json")
        _cell(ctx, "xdomain, BioTransformer against GRAIL on the thirteen, hi", m.group(6),
              cl["ci95"][1], "results/gloryx_clean_subset.json")
        n_full = sum(1 for v in flip["pairwise"].values()
                     if isinstance(v, dict) and "interaction_b_extra_gain" in v
                     and v["interaction_b_extra_gain"]["verdict"] == "SIGNIFICANT")
        n_clean = sum(1 for v in clean["pairwise"].values() if v["excludes_zero"])
        ctx.checks.append((n_full == 4 and n_clean == 2,
                           "xdomain, four significant on the full set and two on the clean one",
                           "four and two", f"{n_full} and {n_clean}",
                           "results/gloryx_rank_flip_ci.json, results/gloryx_clean_subset.json"))

    m = _blk(ctx, "the headline external result",
             r"SyGMa leads MetaPredictor by " + DN + r" " + CI + r" and one rung later the "
             r"advantage is gone")
    if m:
        d = grid["pairwise_delta"]["all37"]["MetaPredictor_vs_SyGMa"]["inchikey"]
        src = "results/gloryx_criterion_grid.json all37 MetaPredictor_vs_SyGMa inchikey"
        _cell(ctx, "xdomain, SyGMa's strict lead over MetaPredictor", m.group(1),
              d["delta_a_minus_b"], src)
        _cell(ctx, "xdomain, SyGMa's strict lead lo", m.group(2), d["ci95"][0], src)
        _cell(ctx, "xdomain, SyGMa's strict lead hi", m.group(3), d["ci95"][1], src)


# --------------------------------------------------------------------------------------------
# What two independent curations agree on
# --------------------------------------------------------------------------------------------
def _curators(ctx) -> None:
    agree = ctx.art("annotation_agreement.json")
    ci = ctx.art("annotation_agreement_ci.json")
    cvm = ctx.art("curator_vs_model.json")
    card = ctx.art("cardinality_crossfit.json")
    refs = ctx.art("test_references.json")

    if agree is not None:
        m = _blk(ctx, "the criterion ladder of curator agreement",
                 r"two curations of the same drug reach a Jaccard of \$([\d.]+)\$; under canonical "
                 r"SMILES, which this harness keys stereo-blind, \$([\d.]+)\$; under "
                 r"Tanimoto\$=\$1, \$([\d.]+)\$; stereo-blind \\textsc\{inchikey\}, \$([\d.]+)\$; "
                 r"and under the tautomer-aware default \$([\d.]+)\$")
        if m:
            modes = ("inchikey", "canonical", "tanimoto1", "inchi_no_stereo", "inchikey_tautomer")
            for i, mode in enumerate(modes):
                _cell(ctx, f"xdomain, curator Jaccard under {mode}", m.group(i + 1),
                      agree["by_mode"][mode]["jaccard"],
                      f"results/annotation_agreement.json by_mode/{mode}/jaccard")
            vals = [float(m.group(i + 1)) for i in range(5)]
            ctx.checks.append((vals[0] == min(vals) and vals[4] == max(vals),
                               "xdomain, strict InChIKey is the floor and tautomer the ceiling",
                               "strictest lowest, most tolerant highest",
                               f"min {min(vals)} max {max(vals)}",
                               "results/annotation_agreement.json"))

        m = _blk(ctx, "the references each curation carries",
                 r"they carry \$([\d.]+)\$ to \$([\d.]+)\$ references each against \$([\d.]+)\$ "
                 r"across the corpus")
        if m:
            by = agree["by_mode"]["inchikey_tautomer"]
            _cell(ctx, "xdomain, the GLORYx curation's set size", m.group(1),
                  by["mean_gloryx_set"], "results/annotation_agreement.json")
            _cell(ctx, "xdomain, the corpus curation's set size", m.group(2),
                  by["mean_corpus_set"], "results/annotation_agreement.json")
            if refs is not None:
                _cell(ctx, "xdomain, references per substrate across the corpus", m.group(3),
                      sum(len(v) for v in refs.values()) / len(refs),
                      "results/test_references.json, mean list length")

    if ci is not None:
        m = _blk(ctx, "the one-sided recovery figures",
                 r"the corpus recovers \$([\d.]+)\$ " + CI + r" of the GLORYx list and GLORYx "
                 r"\$([\d.]+)\$ " + CI + r" of the corpus list")
        if m:
            g = ci["gloryx_recovered_by_corpus"]
            c = ci["corpus_recovered_by_gloryx"]
            _cell(ctx, "xdomain, the corpus recovers of GLORYx", m.group(1), g["mean"],
                  "results/annotation_agreement_ci.json gloryx_recovered_by_corpus")
            _cell(ctx, "xdomain, that recovery's lo", m.group(2), g["ci95"][0],
                  "results/annotation_agreement_ci.json")
            _cell(ctx, "xdomain, that recovery's hi", m.group(3), g["ci95"][1],
                  "results/annotation_agreement_ci.json")
            _cell(ctx, "xdomain, GLORYx recovers of the corpus", m.group(4), c["mean"],
                  "results/annotation_agreement_ci.json corpus_recovered_by_gloryx")
            _cell(ctx, "xdomain, that recovery's lo, other direction", m.group(5), c["ci95"][0],
                  "results/annotation_agreement_ci.json")
            _cell(ctx, "xdomain, that recovery's hi, other direction", m.group(6), c["ci95"][1],
                  "results/annotation_agreement_ci.json")
        m = _blk(ctx, "the two levels quoted again as a bound",
                 r"of the same chemistry coincide: \$([\d.]+)\$ one way, \$([\d.]+)\$ the other")
        if m:
            _cell(ctx, "xdomain, the corpus-side level, second printing", m.group(1),
                  ci["corpus_recovered_by_gloryx"]["mean"],
                  "results/annotation_agreement_ci.json")
            _cell(ctx, "xdomain, the GLORYx-side level, second printing", m.group(2),
                  ci["gloryx_recovered_by_corpus"]["mean"],
                  "results/annotation_agreement_ci.json")
        m = _blk(ctx, "the caveat that the agreement level is an upper bound",
                 r"so \$([\d.]+)\$ is more likely an upper bound on inter-source agreement")
        if m:
            _cell(ctx, "xdomain, the corpus-side level, upper-bound caveat", m.group(1),
                  ci["corpus_recovered_by_gloryx"]["mean"],
                  "results/annotation_agreement_ci.json corpus_recovered_by_gloryx")
        _count(ctx, "the corpus-side agreement level is printed five times", r"\$0\.539\$", 5,
               "the one-sided sentence, the bound, the budget comparison, the plain restatement "
               "and the upper-bound caveat")

    if cvm is None:
        return
    arm = cvm["arms"]["corpus"]
    m = _blk(ctx, "the uncapped comparison against the curation",
             r"MetaPredictor recovers \$([\d.]+)\$ " + CI + r", SyGMa \$([\d.]+)\$, "
             r"BioTransformer \$([\d.]+)\$, GRAIL \$([\d.]+)\$; MetaPredictor's paired gap to the "
             r"curation is " + DN + r" " + CI)
    if m:
        src = "results/curator_vs_model.json arms/corpus"
        _cell(ctx, "xdomain, MetaPredictor uncapped", m.group(1), arm["MetaPredictor"]["uncapped"],
              src)
        _cell(ctx, "xdomain, MetaPredictor uncapped lo", m.group(2),
              arm["MetaPredictor"]["ci95"][0], src)
        _cell(ctx, "xdomain, MetaPredictor uncapped hi", m.group(3),
              arm["MetaPredictor"]["ci95"][1], src)
        _cell(ctx, "xdomain, SyGMa uncapped", m.group(4), arm["SyGMa"]["uncapped"], src)
        _cell(ctx, "xdomain, BioTransformer uncapped", m.group(5),
              arm["BioTransformer"]["uncapped"], src)
        _cell(ctx, "xdomain, GRAIL uncapped", m.group(6), arm["GRAIL"]["uncapped"], src)
        _cell(ctx, "xdomain, MetaPredictor's uncapped gap to the curation", m.group(7),
              arm["MetaPredictor"]["gap_uncapped"], src)
        _cell(ctx, "xdomain, the uncapped gap's lo", m.group(8),
              arm["MetaPredictor"]["gap_uncapped_ci95"][0], src)
        _cell(ctx, "xdomain, the uncapped gap's hi", m.group(9),
              arm["MetaPredictor"]["gap_uncapped_ci95"][1], src)

    m = _blk(ctx, "what the uncapped recall is bought with",
             r"bought with \$([\d.]+)\$ metabolites per drug against the curation's \$([\d.]+)\$, "
             r"SyGMa's near-tie with \$([\d.]+)\$")
    if m:
        src = "results/curator_vs_model.json arms/corpus mean_output"
        _cell(ctx, "xdomain, MetaPredictor's output per drug", m.group(1),
              arm["MetaPredictor"]["mean_output"], src)
        _cell(ctx, "xdomain, the curation's output per drug", m.group(2),
              arm["GLORYx curation"]["mean_output"], src)
        _cell(ctx, "xdomain, SyGMa's output per drug", m.group(3), arm["SyGMa"]["mean_output"], src)

    m = _blk(ctx, "the budget-matched comparison",
             r"its \$([\d.]+)\$ stands against MetaPredictor's \$([\d.]+)\$, SyGMa's \$([\d.]+)\$, "
             r"GRAIL's \$([\d.]+)\$ and BioTransformer's \$([\d.]+)\$, and every paired gap "
             r"excludes zero \(MetaPredictor " + DN + r" " + CI + r", the smallest of the four\)")
    if m:
        src = "results/curator_vs_model.json arms/corpus budget_matched"
        order = ("GLORYx curation", "MetaPredictor", "SyGMa", "GRAIL", "BioTransformer")
        for i, meth in enumerate(order):
            _cell(ctx, f"xdomain, budget-matched recall, {meth}", m.group(i + 1),
                  arm[meth]["budget_matched"], src)
        _cell(ctx, "xdomain, MetaPredictor's budget-matched gap", m.group(6),
              arm["MetaPredictor"]["gap_budget_matched"], src)
        _cell(ctx, "xdomain, the budget-matched gap's lo", m.group(7),
              arm["MetaPredictor"]["gap_budget_matched_ci95"][0], src)
        _cell(ctx, "xdomain, the budget-matched gap's hi", m.group(8),
              arm["MetaPredictor"]["gap_budget_matched_ci95"][1], src)
        four = {k: arm[k]["gap_budget_matched"] for k in order[1:]}
        ctx.checks.append((min(four, key=four.get) == "MetaPredictor"
                           and all(arm[k]["gap_budget_matched_ci95"][0] > 0 for k in order[1:]),
                           "xdomain, MetaPredictor's is the smallest of four gaps that exclude zero",
                           "smallest, all excluding zero",
                           f"smallest is {min(four, key=four.get)}", src))

    m = _blk(ctx, "the symmetric k=6 match",
             r"A fixed \$k\{=\}6\$, the curation's mean output, truncates the curation as well and "
             r"points the same way without reaching significance: \$([\d.]+)\$ against \$([\d.]+)\$,"
             r" gap " + DN + r" " + CI)
    if m:
        src = "results/curator_vs_model.json arms/corpus at_k6"
        _cell(ctx, "xdomain, the curation at k=6", m.group(1), arm["GLORYx curation"]["at_k6"], src)
        _cell(ctx, "xdomain, MetaPredictor at k=6", m.group(2), arm["MetaPredictor"]["at_k6"], src)
        _cell(ctx, "xdomain, the k=6 gap", m.group(3), arm["MetaPredictor"]["gap_at_k6"], src)
        _cell(ctx, "xdomain, the k=6 gap lo", m.group(4),
              arm["MetaPredictor"]["gap_at_k6_ci95"][0], src)
        _cell(ctx, "xdomain, the k=6 gap hi", m.group(5),
              arm["MetaPredictor"]["gap_at_k6_ci95"][1], src)

    m = _blk(ctx, "the plain restatement of the budget-matched result",
             r"the curator retrieves \$([\d.]+)\$ of the other source's list and the best model "
             r"\$([\d.]+)\$")
    if m:
        _cell(ctx, "xdomain, the curator's budget-matched recall, restated", m.group(1),
              arm["GLORYx curation"]["budget_matched"], "results/curator_vs_model.json")
        _cell(ctx, "xdomain, the best model's budget-matched recall, restated", m.group(2),
              max(arm[k]["budget_matched"] for k in
                  ("MetaPredictor", "SyGMa", "GRAIL", "BioTransformer")),
              "results/curator_vs_model.json, best of the four models")

    if card is not None:
        m = _blk(ctx, "the cardinality oracle beside the cap",
                 r"the oracle cutoff lifts macro-F1 from \$([\d.]+)\$ at the best constant \$k\$ to "
                 r"\$([\d.]+)\$, and a head trained to predict that cutoff recovers \$([\d.]+)\$ of "
                 r"the \$([\d.]+)\$ available")
        if m:
            mp = card["methods"]["MetaPredictor"]
            src = "results/cardinality_crossfit.json methods/MetaPredictor"
            _cell(ctx, "xdomain, the best constant k", m.group(1),
                  mp["macro_f1"]["best_constant_oos"], src)
            _cell(ctx, "xdomain, the oracle cutoff", m.group(2), mp["macro_f1"]["oracle_kstar"], src)
            _cell(ctx, "xdomain, what the head recovers", m.group(3),
                  mp["gain_over_constant"]["predicted_kstar"], src)
            _cell(ctx, "xdomain, what is available", m.group(4),
                  mp["gain_over_constant"]["oracle"], src)
