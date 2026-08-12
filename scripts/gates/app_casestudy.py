"""Checks for paper/app/casestudy.tex and paper/app/emission.tex.

The case study is where the four choices were measured, and the emission appendix is the one arm
the pool claim rests on. Between them they printed seventy-three numerals that no check read: the
whole recall-by-criterion table, the whole emission-policy table, every figure the external
replication paragraph reasons from, and both printings of the coverage bound BioTransformer is
quoted at. A number nothing reads is a number that can go stale silently, which is how one figure
came to be printed three times with three different values.

Every check below names one leaf of one artifact. Where a figure is printed twice --- the pool
sentence, the family size for the budget sweep, BioTransformer's best single setting --- both
printings are bound and the number of printings is asserted, so a copy that drifts from its twin
fails rather than hides.
"""
from __future__ import annotations

import re

CS = "scripts/gates/app_casestudy.py"


def _miss(ctx, name, note=""):
    """A sentence the checks below hang off is gone: that is a failure, not a skip."""
    ctx.checks.append((False, name, "the sentence", "not matched in the manuscript", note or CS))


def _places(printed: str) -> int:
    printed = printed.strip()
    return len(printed.split(".")[1]) if "." in printed else 0


def _rounds_to(printed: str, value) -> bool:
    """Is ``printed`` a rounding of ``value`` at the precision it is printed to?

    ``ctx.check`` answers this with round-half-even, which is right except when the artifact holds
    the halfway value itself: 0.0875 stored to four places is printed 0.088 by a round-half-up
    convention and 0.087 by Python's. Both are roundings of the number the artifact records, so
    both are accepted here and everything else is still rejected at half a unit of the last place.
    """
    q = 10.0 ** (-_places(printed))
    return abs(float(printed) - float(value)) <= q / 2 + 1e-9


def register(ctx) -> None:
    _protocol_families(ctx)
    _engine_reach(ctx)
    _dispatch_bound(ctx)
    _retro_library(ctx)
    _decomposition(ctx)
    _match_sensitivity_table(ctx)
    _external_replication(ctx)
    _aggregate_and_budget(ctx)
    _pool_sentence(ctx)
    _emission_table(ctx)
    _emission_sweep(ctx)
    _emission_crossfit(ctx)
    _emission_absolute_rules(ctx)
    _emission_populations(ctx)
    register_curator_agreement(ctx)


# --------------------------------------------------------------------------------------------
# casestudy.tex
# --------------------------------------------------------------------------------------------
def _protocol_families(ctx) -> None:
    """The three family sizes the protocol paragraph declares, against the runs that fixed them."""
    L = ctx.art("retro_leaderboard_cluster0.json")
    Q = ctx.art("four_method_291.json")
    RPA = ctx.art("retro_population_axis.json")
    PA = ctx.art("population_axis.json")
    if L is None or Q is None or RPA is None or PA is None:
        return
    m = re.search(r"the cross-domain interaction, whose \$(\d+)\$ tests form one family", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the confirmatory-family sentence")
    else:
        ctx.check("casestudy, the cross-domain family size", m.group(1), L["n_interaction_tests"],
                  "results/retro_leaderboard_cluster0.json")
    m = re.search(r"the \$(\d+)\$-cell budget sweep and the \$(\d+)\$ population interactions",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the corrected-but-not-confirmatory sentence")
    else:
        ctx.check("casestudy, the budget sweep's family size", m.group(1),
                  Q["holm"]["family_size"], "results/four_method_291.json")
        ctx.check("casestudy, the population family size", m.group(2),
                  RPA["comparisons"] + PA["comparisons"],
                  "results/retro_population_axis.json + results/population_axis.json")
    # the same family size is printed again where the sweep is described; both printings are the
    # artifact's, and there are exactly two of them in the manuscript
    m2 = re.search(r"survive Holm over the \$(\d+)\$-cell grid that axis admits", ctx.flat)
    if not m2:
        _miss(ctx, "casestudy, the sweep's grid sentence")
    else:
        ctx.check("casestudy, the sweep's family size, second printing", m2.group(1),
                  Q["holm"]["family_size"], "results/four_method_291.json")
    n = len(re.findall(r"\$54\$", ctx.flat))
    ctx.checks.append((n == 2, "casestudy, the sweep's family size is printed twice", "2 printings",
                       f"{n} printings", CS))


def _engine_reach(ctx) -> None:
    """The two arms of the hydrogen-presentation knob and the overlap they are measured on."""
    B = ctx.art("bank_overlap_sygma__clean_test.json")
    K = ctx.art("engine_knobs__clean_test.json")
    R = ctx.art("reach_engine_vs_bank__clean_test.json")
    if B is None or K is None or R is None:
        return
    m = re.search(r"\$(\d+)\$ of SyGMa's \$(\d+)\$ rules sit verbatim in ours", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the verbatim-overlap sentence")
    else:
        ctx.check("casestudy, SyGMa rules inside our bank", m.group(1),
                  B["containment"]["in_grail_bank"], "results/bank_overlap_sygma__clean_test.json")
        ctx.check("casestudy, SyGMa's rule count", m.group(2), B["containment"]["sygma_rules"],
                  "results/bank_overlap_sygma__clean_test.json")
    m = re.search(r"those two arms reach \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ and \$([\d.]+)\$ "
                  r"\$\[([\d.]+),([\d.]+)\]\$, a paired \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the two-arm reach sentence")
        return
    hs = K["one_knob_at_a_time"]["explicit_hydrogens"]
    ctx.check("casestudy, the expanded arm's reach", m.group(1), K["default_reach"],
              "results/engine_knobs__clean_test.json")
    ctx.check("casestudy, the expanded arm's interval, lower", m.group(2),
              R["arms"]["A_grail_engine_152_rules"]["ci95"][0],
              "results/reach_engine_vs_bank__clean_test.json")
    ctx.check("casestudy, the expanded arm's interval, upper", m.group(3),
              R["arms"]["A_grail_engine_152_rules"]["ci95"][1],
              "results/reach_engine_vs_bank__clean_test.json")
    ctx.check("casestudy, the unexpanded arm's reach", m.group(4), hs["reach"],
              "results/engine_knobs__clean_test.json")
    ctx.check("casestudy, the unexpanded arm's interval, lower", m.group(5),
              K["all_configurations"]["add_hs=False,norm=standardize,drop_invalid=True"]["ci95"][0],
              "results/engine_knobs__clean_test.json")
    ctx.check("casestudy, the unexpanded arm's interval, upper", m.group(6),
              K["all_configurations"]["add_hs=False,norm=standardize,drop_invalid=True"]["ci95"][1],
              "results/engine_knobs__clean_test.json")
    ctx.check("casestudy, the paired knob difference", m.group(7),
              hs["paired_vs_default"]["delta"], "results/engine_knobs__clean_test.json")
    ctx.check("casestudy, the paired difference, lower", m.group(8),
              hs["paired_vs_default"]["ci95"][0], "results/engine_knobs__clean_test.json")
    # 0.2075 is exactly halfway at three places, so 0.207 and 0.208 are both roundings of it
    ctx.checks.append((_rounds_to(m.group(9), hs["paired_vs_default"]["ci95"][1]),
                       "casestudy, the paired difference, upper", m.group(9),
                       hs["paired_vs_default"]["ci95"][1],
                       "results/engine_knobs__clean_test.json"))
    # the arms are the same measurement seen twice: one knob's two settings, and the difference
    # between the reaches the sentence prints has to be the paired delta it prints beside them
    gap = hs["reach"] - K["default_reach"]
    ctx.checks.append((abs(gap - hs["paired_vs_default"]["delta"]) < 1e-3,
                       "casestudy, the two arms differ by the paired delta", round(gap, 4),
                       hs["paired_vs_default"]["delta"], "results/engine_knobs__clean_test.json"))


def _dispatch_bound(ctx) -> None:
    """The bank-wide bound, its best single setting, and the null bank the design needs."""
    H = ctx.art("hydrogen_dispatch__clean_test.json")
    if H is None:
        return
    ctx.checks.append((H["config"]["population"] == "clean_test",
                       "casestudy, the dispatch measurement is on the clean split", "clean_test",
                       H["config"]["population"], "results/hydrogen_dispatch__clean_test.json"))
    sy = H["banks"]["sygma_175"]
    m = re.search(r"none of whose \$(\d+)\$ templates names one", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the structural-null sentence")
    else:
        ctx.check("casestudy, SyGMa's templates, dispatch null", m.group(1), sy["n_rules"],
                  "results/hydrogen_dispatch__clean_test.json")
        ctx.checks.append((sy["dispatched_to_expanded"] == 0,
                           "casestudy, no SyGMa template names a hydrogen atom", "none",
                           f"{sy['dispatched_to_expanded']} do",
                           "results/hydrogen_dispatch__clean_test.json"))
    bt = H["banks"]["biotransformer"]
    m = re.search(r"For BioTransformer that is \$([\d.]+)\$ against the \$([\d.]+)\$ its best "
                  r"single setting reaches", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the guaranteed-reach sentence")
    else:
        ctx.check("casestudy, BioTransformer's guaranteed reach", m.group(1),
                  bt["guaranteed_reach"], "results/hydrogen_dispatch__clean_test.json")
        ctx.check("casestudy, BioTransformer's best single setting", m.group(2),
                  bt["best_global"], "results/hydrogen_dispatch__clean_test.json")
        ratio = bt["best_global"] / bt["guaranteed_reach"]
        ctx.checks.append((4.0 <= ratio < 5.0, "casestudy, the bound is four times the guarantee",
                           "four times", round(ratio, 2),
                           "results/hydrogen_dispatch__clean_test.json"))
    m = re.search(r"and \$([\d.]+)\$ is our measurement of that bank", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the not-published-by-its-authors clause")
    else:
        ctx.check("casestudy, that setting again, second printing", m.group(1), bt["best_global"],
                  "results/hydrogen_dispatch__clean_test.json")
    n = len(re.findall(r"\$0\.4963\$", ctx.flat))
    ctx.checks.append((n == 2, "casestudy, the best single setting is printed twice", "2 printings",
                       f"{n} printings", CS))


def _retro_library(ctx) -> None:
    """The retrosynthesis library the hazard generalises to, and that every pattern is exposed."""
    T = ctx.art("retro_template_convention.json")
    if T is None:
        return
    m = re.search(r"every one of the \$([\d.{},]+)\$ patterns in the standard retrosynthesis",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the retrosynthesis-library sentence")
        return
    ctx.check("casestudy, the retrosynthesis template count", m.group(1).replace("{,}", ""),
              T["convention_census"]["templates"], "results/retro_template_convention.json")
    c = T["convention_census"]
    ctx.checks.append((c["with_a_degree_primitive"] == c["templates"],
                       "casestudy, every one of them pins a connection count",
                       "every one", f"{c['with_a_degree_primitive']} of {c['templates']}",
                       "results/retro_template_convention.json"))


def _decomposition(ctx) -> None:
    """The ranking factor and its interval, and SyGMa's selection factor of one."""
    F = ctx.art("recall_factorization.json")
    S = ctx.art("decompose_sygma.json")
    if F is None or S is None:
        return
    m = re.search(r"Ranking is close to lossless at \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the ranking-factor sentence")
    else:
        r = F["factors"]["ranking_conversion"]
        ctx.check("casestudy, the ranking factor", m.group(1), r["point"],
                  "results/recall_factorization.json")
        ctx.check("casestudy, the ranking factor, lower", m.group(2), r["lo"],
                  "results/recall_factorization.json")
        ctx.check("casestudy, the ranking factor, upper", m.group(3), r["hi"],
                  "results/recall_factorization.json")
    m = re.search(r"SyGMa's selection factor is \$(\d+)\$", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the no-selection-stage sentence")
    else:
        ctx.check("casestudy, SyGMa's selection factor", m.group(1),
                  S["factors"]["selection_retention"]["point"], "results/decompose_sygma.json")


def _match_sensitivity_table(ctx) -> None:
    """Table 'recall by matching criterion': every row, every column, and the gain beside them."""
    P = ctx.art("match_sensitivity_fulln_paired.json")
    W = ctx.art("criterion_within_method.json")
    if P is None or W is None:
        return
    modes = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
    rows = (("GRAIL", r"\\textbf\{GRAIL\} & "), ("MetaPredictor", r"MetaPredictor & "),
            ("SyGMa", r"SyGMa & "))
    for method, lead in rows:
        m = re.search(lead + r"([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & "
                             r"\$\+([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$", ctx.flat)
        if not m:
            _miss(ctx, f"the criterion table, {method}'s row")
            continue
        for i, mode in enumerate(modes):
            ctx.check(f"the criterion table, {method} under {mode}", m.group(i + 1),
                      P["recall_at_15"][method][mode],
                      "results/match_sensitivity_fulln_paired.json")
        ctx.check(f"the criterion table, {method}'s gain", m.group(6),
                  W["methods"][method]["at"]["15"]["gain"],
                  "results/criterion_within_method.json")
        ctx.check(f"the criterion table, {method}'s gain, lower", m.group(7),
                  W["methods"][method]["at"]["15"]["ci95"][0],
                  "results/criterion_within_method.json")
        ctx.check(f"the criterion table, {method}'s gain, upper", m.group(8),
                  W["methods"][method]["at"]["15"]["ci95"][1],
                  "results/criterion_within_method.json")
        # the gain column is the row's own two ends: the caption says it is a paired difference
        # rather than the difference of the rounded columns, which is a claim about how close
        # they must be, not a licence for the column to drift from the row it sits beside
        span = (P["recall_at_15"][method]["inchikey_tautomer"]
                - P["recall_at_15"][method]["canonical"])
        ctx.checks.append((abs(span - W["methods"][method]["at"]["15"]["gain"]) < 1e-3,
                           f"the criterion table, {method}'s gain is its row's own span",
                           round(span, 4), W["methods"][method]["at"]["15"]["gain"],
                           "results/match_sensitivity_fulln_paired.json"))
    m = re.search(r"paired bootstrap over substrates with \$([\d.{},]+)\$ resamples", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the resample-count sentence")
    else:
        ctx.check("casestudy, the resamples behind that table", m.group(1).replace("{,}", ""),
                  P["n_boot"], "results/match_sensitivity_fulln_paired.json")


def _external_replication(ctx) -> None:
    """The GLORYx paragraph: the contamination audit, the strict gap, the interaction, the bound."""
    A = ctx.art("external_overlap_audit.json")
    C = ctx.art("gloryx_clean_subset.json")
    G = ctx.art("gloryx_criterion_grid.json")
    if A is None or C is None or G is None:
        return
    m = re.search(r"\$(\d+)\$ of the \$(\d+)\$ sit in GRAIL's own train or validation split, so "
                  r"its row uses the \$(\d+)\$ unseen", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the contamination sentence")
    else:
        ext = A["GLORYx external set"]
        ctx.check("casestudy, GLORYx drugs inside train or val", m.group(1),
                  ext["in_train_or_val"], "results/external_overlap_audit.json")
        ctx.check("casestudy, the GLORYx set's size", m.group(2), ext["n"],
                  "results/external_overlap_audit.json")
        ctx.check("casestudy, the uncontaminated remainder", m.group(3), C["n_clean"],
                  "results/gloryx_clean_subset.json")
        ctx.checks.append((ext["in_train_or_val"] + C["n_clean"] == ext["n"],
                           "casestudy, seen and unseen account for the whole set",
                           f"{ext['in_train_or_val']} + {C['n_clean']}", ext["n"],
                           "results/external_overlap_audit.json + gloryx_clean_subset.json"))
    pair = G["pairwise_delta"]["all37"]["MetaPredictor_vs_SyGMa"]
    m = re.search(r"matching MetaPredictor trails SyGMa by \$([\d.]+)\$", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the strict-criterion gap sentence")
    else:
        ctx.check("casestudy, MetaPredictor's strict deficit to SyGMa", m.group(1),
                  abs(pair["inchikey"]["delta_a_minus_b"]), "results/gloryx_criterion_grid.json")
    m = re.search(r"MetaPredictor gains \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ more than "
                  r"SyGMa across the stereo step", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the external interaction sentence")
    else:
        it = G["stereo_interaction"]["all37"]["MetaPredictor_extra_gain_over_SyGMa"]
        ctx.check("casestudy, the external interaction", m.group(1), it["mean"],
                  "results/gloryx_criterion_grid.json")
        ctx.check("casestudy, the external interaction, lower", m.group(2), it["ci95"][0],
                  "results/gloryx_criterion_grid.json")
        ctx.check("casestudy, the external interaction, upper", m.group(3), it["ci95"][1],
                  "results/gloryx_criterion_grid.json")
    m = re.search(r"while still admitting SyGMa ahead by \$([\d.]+)\$", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the relaxed-gap sentence")
    else:
        rel = pair["inchi_no_stereo"]
        ctx.check("casestudy, how far ahead the relaxed interval still admits SyGMa", m.group(1),
                  abs(rel["ci95"][0]), "results/gloryx_criterion_grid.json")
        ctx.checks.append((rel["ci95"][0] < 0 < rel["ci95"][1],
                           "casestudy, the relaxed gap covers zero", "covers zero",
                           f"[{rel['ci95'][0]},{rel['ci95'][1]}]",
                           "results/gloryx_criterion_grid.json"))


def _aggregate_and_budget(ctx) -> None:
    """The aggregate disagreement, the budget sweep, the fourth method, and the like-emitting null."""
    M = ctx.art("set_metrics_by_criterion.json")
    B = ctx.art("budget_curves.json")
    Q = ctx.art("four_method_291.json")
    A = ctx.art("aggregate_vs_output_size.json")
    W = ctx.art("criterion_within_method.json")
    F = ctx.art("recall_factorization.json")
    if any(x is None for x in (M, B, Q, A, W, F)):
        return
    full = M["populations"]["full1170"]["by_mode"]
    taut = full["inchikey_tautomer"]["_paired_diffs"]
    m = re.search(r"SyGMa leading on recall by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ and sitting "
                  r"\$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ below on F1", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the two-aggregates sentence")
    else:
        rec, f1 = taut["GRAIL-SyGMa|recall"], taut["GRAIL-SyGMa|f1"]
        ctx.check("casestudy, SyGMa's recall lead", m.group(1), abs(rec["point"]),
                  "results/set_metrics_by_criterion.json")
        ctx.check("casestudy, that lead's lower bound", m.group(2), abs(rec["ci95"][1]),
                  "results/set_metrics_by_criterion.json")
        ctx.check("casestudy, that lead's upper bound", m.group(3), abs(rec["ci95"][0]),
                  "results/set_metrics_by_criterion.json")
        # 0.0875 is exactly halfway at three places, so both neighbours are roundings of it
        ctx.checks.append((_rounds_to(m.group(4), f1["point"]), "casestudy, SyGMa's F1 deficit",
                           m.group(4), f1["point"], "results/set_metrics_by_criterion.json"))
        ctx.check("casestudy, that deficit's lower bound", m.group(5), f1["ci95"][0],
                  "results/set_metrics_by_criterion.json")
        ctx.check("casestudy, that deficit's upper bound", m.group(6), f1["ci95"][1],
                  "results/set_metrics_by_criterion.json")
    # the budget sweep: where the order changes, and the margin at the budget fixed in advance
    m = re.search(r"SyGMa sits above GRAIL at every budget up to \$(\d+)\$, separated from \$k=1\$ "
                  r"to \$k=(\d+)\$ and by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ at \$k=(\d+)\$, "
                  r"and below from \$(\d+)\$ on, where the field's \$(\d+)\$ sits", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the budget-sweep sentence")
    else:
        g, s = B["macro_f1_by_k"]["GRAIL"], B["macro_f1_by_k"]["SyGMa"]
        above = [k for k in range(1, len(g) + 1) if s[k - 1] > g[k - 1]]
        ctx.check("casestudy, the last budget SyGMa leads at", m.group(1), max(above),
                  "results/budget_curves.json")
        ctx.check("casestudy, the first budget SyGMa trails at", m.group(7), max(above) + 1,
                  "results/budget_curves.json")
        # the sentence claims every budget in [1, said] separates, which is what is tested: the
        # artifact's run in fact reaches one budget further, so the printed end is conservative
        # rather than wrong, and a claim extending past the run is what has to fail
        said = int(m.group(2))
        sep = [k for k in range(1, len(g) + 1)
               if B["paired_by_k"][str(k)]["GRAIL-SyGMa"]["excludes_zero"]]
        run = 1
        while run + 1 in sep:
            run += 1
        ctx.checks.append((all(k in sep for k in range(1, said + 1)),
                           "casestudy, that margin separates at every budget it is claimed to",
                           f"1 to {said}", f"the run reaches {run}", "results/budget_curves.json"))
        k5 = B["paired_at_k_fixed"]["GRAIL-SyGMa"]
        ctx.check("casestudy, the margin at the budget fixed in advance", m.group(3),
                  abs(k5["delta"]), "results/budget_curves.json")
        ctx.check("casestudy, that margin's lower bound", m.group(4), abs(k5["ci95"][1]),
                  "results/budget_curves.json")
        ctx.check("casestudy, that margin's upper bound", m.group(5), abs(k5["ci95"][0]),
                  "results/budget_curves.json")
        ctx.check("casestudy, the budget that margin is quoted at", m.group(6),
                  B["k_fixed_in_advance"], "results/budget_curves.json")
        ctx.check("casestudy, the field's budget", m.group(8), F["k"],
                  "results/recall_factorization.json")
    m = re.search(r"On \$(\d+)\$ substrates carrying all four, MetaTox at \$([\d.]+)\$ candidates",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the fourth-method sentence")
    else:
        ctx.check("casestudy, the four-method population", m.group(1), Q["config"]["n_substrates"],
                  "results/four_method_291.json")
        ctx.check("casestudy, MetaTox's emitted count", m.group(2),
                  Q["per_method"]["MetaTox"]["mean_emitted_uncapped"],
                  "results/four_method_291.json")
    m = re.search(r"it is second at \$([\d.]+)\$ behind MetaPredictor's \$([\d.]+)\$ and last at "
                  r"\$k=(\d+)\$", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the lead-is-volume sentence")
    else:
        by = Q["macro_f1_by_budget"]
        ctx.check("casestudy, MetaTox's macro F1 at the field's budget", m.group(1),
                  by["MetaTox"]["15"], "results/four_method_291.json")
        ctx.check("casestudy, MetaPredictor's macro F1 there", m.group(2),
                  by["MetaPredictor"]["15"], "results/four_method_291.json")
        order = sorted(by, key=lambda k: -by[k]["15"])
        ctx.checks.append((order[1] == "MetaTox", "casestudy, MetaTox is second on macro F1 there",
                           "second", f"{order.index('MetaTox') + 1} of {len(order)}",
                           "results/four_method_291.json"))
        low = m.group(3)
        ctx.checks.append((min(by, key=lambda k: by[k][low]) == "MetaTox",
                           f"casestudy, MetaTox is last on macro F1 at k={low}", "last",
                           min(by, key=lambda k: by[k][low]), "results/four_method_291.json"))
    m = re.search(r"those pairs leave intervals up to \$([\d.]+)\$ open", ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the failure-to-detect sentence")
    else:
        ctx.check("casestudy, the widest interval the like-emitting null leaves open", m.group(1),
                  A["power"]["widest_interval_among_comparable_pairs"],
                  "results/aggregate_vs_output_size.json")
    m = re.search(r"separated under four of the five criteria at \$(-[\d.]+)\$ "
                  r"\$\[(-[\d.]+),(-[\d.]+)\]\$ and indistinguishable from zero under plain",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the criterion-erases sentence")
    else:
        key = "GRAIL-MetaPredictor|f1"
        gap = full["inchikey_tautomer"]["_paired_diffs"][key]
        ctx.check("casestudy, the GRAIL-MetaPredictor F1 gap", m.group(1), gap["point"],
                  "results/set_metrics_by_criterion.json")
        ctx.check("casestudy, that gap's lower bound", m.group(2), gap["ci95"][0],
                  "results/set_metrics_by_criterion.json")
        ctx.check("casestudy, that gap's upper bound", m.group(3), gap["ci95"][1],
                  "results/set_metrics_by_criterion.json")
        sep = [mode for mode in full if full[mode]["_paired_diffs"][key]["excludes_zero"]]
        ctx.checks.append((len(sep) == 4 and "inchikey" not in sep,
                           "casestudy, that gap separates under four criteria and not under "
                           "plain inchikey", "four, not inchikey",
                           f"{len(sep)}: {','.join(sorted(sep))}",
                           "results/set_metrics_by_criterion.json"))
    m = re.search(r"SyGMa emitting \$(\d+)\$ candidates per substrate against GRAIL's \$([\d.]+)\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "casestudy, the output-size confound sentence")
    else:
        ctx.check("casestudy, SyGMa's emitted count", m.group(1),
                  round(W["methods"]["SyGMa"]["mean_emitted_uncapped"]),
                  "results/criterion_within_method.json")
        ctx.check("casestudy, GRAIL's emitted count", m.group(2),
                  W["methods"]["GRAIL"]["mean_emitted_uncapped"],
                  "results/criterion_within_method.json")


# --------------------------------------------------------------------------------------------
# emission.tex, and the paragraph in casestudy.tex that summarises it
# --------------------------------------------------------------------------------------------
def _pool_sentence(ctx) -> None:
    """The pool's shape, printed once in the case study and once in the emission appendix."""
    F = ctx.art("recall_factorization.json")
    if F is None:
        return
    rows = F["per_substrate"]
    refs = sorted(r["U"] for r in rows)
    mean_refs = sum(refs) / len(refs)
    median = refs[len(refs) // 2] if len(refs) % 2 else (refs[len(refs) // 2 - 1]
                                                        + refs[len(refs) // 2]) / 2
    emitted = [len(r.get("deployed_top15") or []) for r in rows]
    mean_emitted = sum(emitted) / len(emitted)
    found = list(re.finditer(r"median substrate [^.]*?has one and the mean \$([\d.]+)\$,[^.]*?"
                             r"emits \$([\d.]+)\$ for every substrate alike", ctx.flat))
    ctx.checks.append((len(found) == 2, "the pool sentence is printed in both places",
                       "2 printings", f"{len(found)} printings", CS))
    if not found:
        _miss(ctx, "the pool sentence")
        return
    for i, m in enumerate(found):
        ctx.check(f"the pool sentence {i + 1}, references per substrate", m.group(1), mean_refs,
                  "results/recall_factorization.json, mean of per_substrate.U")
        ctx.check(f"the pool sentence {i + 1}, what the deployed policy emits", m.group(2),
                  mean_emitted,
                  "results/recall_factorization.json, mean size of per_substrate.deployed_top15")
    ctx.checks.append((median == 1, "the pool sentence, the median substrate has one reference",
                       "one", median, "results/recall_factorization.json"))


def _emission_table(ctx) -> None:
    """The emission-policy table: every row, every column, and the arm each row is measured against."""
    H = ctx.art("setsize_headroom.json")
    if H is None:
        return
    arms = H["arms"]
    best = H["config"]["best_global_constant"]
    ctx.checks.append((best == "fixed k=1" and arms[best]["f1_gain_over_best_constant"] == 0.0,
                       "the emission table, the comparator is the best constant", "fixed k=1",
                       best, "results/setsize_headroom.json"))
    fixed = {k: v["f1"] for k, v in arms.items() if k.startswith("fixed ")}
    ctx.checks.append((max(fixed, key=fixed.get) == best,
                       "the emission table, no constant beats the one called best", best,
                       max(fixed, key=fixed.get), "results/setsize_headroom.json"))
    m = re.search(r"deployed budget, \$k=(\d+)\$ & \$([\d.]+)\$ & \$(-[\d.]+)\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission table, the deployed row")
    else:
        arm = arms[f"fixed k={m.group(1)}"]
        ctx.check("the emission table, the deployed budget's F1", m.group(2), arm["f1"],
                  "results/setsize_headroom.json")
        ctx.check("the emission table, the deployed budget against the constant", m.group(3),
                  arm["f1_gain_over_best_constant"], "results/setsize_headroom.json")
    m = re.search(r"best global constant, \$k=(\d+)\$ & \$([\d.]+)\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission table, the constant row")
    else:
        ctx.checks.append((f"fixed k={m.group(1)}" == best,
                           "the emission table, the constant row is the artifact's best", best,
                           f"fixed k={m.group(1)}", "results/setsize_headroom.json"))
        ctx.check("the emission table, the best constant's F1", m.group(2), arms[best]["f1"],
                  "results/setsize_headroom.json")
    printed_alphas = []
    for alpha in ("0.8", "0.75", "0.6", "0.5", "0.4", "0.3"):
        bold = alpha == "0.5"
        pat = (r"\\textbf\{relative rule, \$\\alpha=0\.5\$\} & \$\\mathbf\{([\d.]+)\}\$ & "
               r"\$\\mathbf\{\+([\d.]+)\}\$ & \$\[\+([\d.]+),\+([\d.]+)\]\$ & (\w+)" if bold else
               r"relative rule, \$\\alpha=" + alpha.replace(".", r"\.") +
               r"\$ & \$([\d.]+)\$ & \$\+([\d.]+)\$ & \$\[\+([\d.]+),\+([\d.]+)\]\$ & (\w+)")
        m = re.search(pat, ctx.flat)
        if not m:
            _miss(ctx, f"the emission table, the row for alpha={alpha}")
            continue
        printed_alphas.append(float(alpha))
        arm = arms[f"gap rule a={alpha}"]
        ctx.check(f"the emission table, alpha={alpha}, F1", m.group(1), arm["f1"],
                  "results/setsize_headroom.json")
        ctx.check(f"the emission table, alpha={alpha}, gain", m.group(2),
                  arm["f1_gain_over_best_constant"], "results/setsize_headroom.json")
        ctx.check(f"the emission table, alpha={alpha}, lower", m.group(3),
                  arm["ci95_vs_best_constant"][0], "results/setsize_headroom.json")
        ctx.check(f"the emission table, alpha={alpha}, upper", m.group(4),
                  arm["ci95_vs_best_constant"][1], "results/setsize_headroom.json")
        ctx.checks.append(((m.group(5) == "yes") == arm["separated_vs_best_constant"],
                           f"the emission table, alpha={alpha}, separated", m.group(5),
                           arm["separated_vs_best_constant"], "results/setsize_headroom.json"))
    for label, arm_name, sign in (("oracle: each substrate's own count", "oracle count", r"\+"),
                                  ("forecast of that count", "predicted count", "-")):
        pat = (re.escape(label).replace(r"\ ", " ") +
               r" & \$([\d.]+)\$ & \$" + sign + r"([\d.]+)\$ & \$\[([+-][\d.]+),([+-][\d.]+)\]\$")
        m = re.search(pat, ctx.flat)
        if not m:
            _miss(ctx, f"the emission table, the {arm_name} row")
            continue
        arm = arms[arm_name]
        ctx.check(f"the emission table, {arm_name}, F1", m.group(1), arm["f1"],
                  "results/setsize_headroom.json")
        ctx.check(f"the emission table, {arm_name}, gain", ("-" if sign == "-" else "")
                  + m.group(2), arm["f1_gain_over_best_constant"], "results/setsize_headroom.json")
        ctx.check(f"the emission table, {arm_name}, lower", m.group(3),
                  arm["ci95_vs_best_constant"][0], "results/setsize_headroom.json")
        ctx.check(f"the emission table, {arm_name}, upper", m.group(4),
                  arm["ci95_vs_best_constant"][1], "results/setsize_headroom.json")
    # the third column is the first column minus the constant's: a row that goes stale on its own
    # leaves a table whose gain column no longer subtracts from its own baseline
    off = [k for k, v in arms.items()
           if abs((v["f1"] - arms[best]["f1"]) - v["f1_gain_over_best_constant"]) > 5e-4]
    ctx.checks.append((not off, "the emission table, every gain is that row minus the constant",
                       "every row", f"{len(off)} do not: {','.join(sorted(off))}",
                       "results/setsize_headroom.json"))
    ctx.checks.append((sorted(printed_alphas, reverse=True) == printed_alphas
                       and len(printed_alphas) == 6,
                       "the emission table, the six relative rows are printed in order",
                       "six, descending", f"{len(printed_alphas)}: {printed_alphas}", CS))


def _emission_sweep(ctx) -> None:
    """The plateau: the sweep's ends, the six that separate, the argmax, and the constant-of-one."""
    H = ctx.art("setsize_headroom.json")
    C = ctx.art("emission_crossfit.json")
    if H is None or C is None:
        return
    alphas = C["config"]["alphas"]
    m = re.search(r"Sweeping \$\\alpha\$ over ten values from \$([\d.]+)\$ to \$([\d.]+)\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "the emission sweep sentence")
    else:
        ctx.check("the emission sweep, the loosest threshold", m.group(1), alphas[0],
                  "results/emission_crossfit.json")
        ctx.check("the emission sweep, the tightest threshold", m.group(2), alphas[-1],
                  "results/emission_crossfit.json")
        ctx.checks.append((len(alphas) == 10, "the emission sweep, ten values", "ten",
                           len(alphas), "results/emission_crossfit.json"))
    gaps = {float(k.split("=")[1]): v for k, v in H["arms"].items() if k.startswith("gap rule a=")}
    sep = sorted((a for a, v in gaps.items() if v["separated_vs_best_constant"]), reverse=True)
    m = re.search(r"six consecutive values from \$([\d.]+)\$ down to \$([\d.]+)\$ beat the best "
                  r"constant", ctx.flat)
    if not m:
        _miss(ctx, "the emission plateau sentence")
    else:
        ctx.check("the emission plateau, its loose end", m.group(1), max(sep) if sep else None,
                  "results/setsize_headroom.json")
        ctx.check("the emission plateau, its tight end", m.group(2), min(sep) if sep else None,
                  "results/setsize_headroom.json")
        run = [a for a in sorted(gaps, reverse=True) if sep and a <= max(sep)][:len(sep)]
        ctx.checks.append((len(sep) == 6 and run == sep,
                           "the emission plateau, six consecutive values and no others",
                           "six consecutive", f"{len(sep)}: {sep}",
                           "results/setsize_headroom.json"))
    m = re.search(r"At \$\\alpha=([\d.]+)\$ the rule is a constant of one and gains "
                  r"\$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission constant-of-one sentence")
    else:
        arm = H["arms"][f"gap rule a={m.group(1)}"]
        ctx.check("the emission sweep, what the constant-of-one arm gains", m.group(2),
                  arm["f1_gain_over_best_constant"], "results/setsize_headroom.json")
        ctx.check("the emission sweep, that arm's lower bound", m.group(3),
                  arm["ci95_vs_best_constant"][0], "results/setsize_headroom.json")
        ctx.check("the emission sweep, that arm's upper bound", m.group(4),
                  arm["ci95_vs_best_constant"][1], "results/setsize_headroom.json")
        ctx.checks.append((not arm["separated_vs_best_constant"],
                           "the emission sweep, the loose extreme does not separate",
                           "not separated", arm["separated_vs_best_constant"],
                           "results/setsize_headroom.json"))
    m = re.search(r"But \$([\d.]+)\$ is an argmax over the sweep", ctx.flat)
    if not m:
        _miss(ctx, "the emission argmax sentence")
    else:
        top = max(gaps, key=lambda a: gaps[a]["f1_gain_over_best_constant"])
        ctx.check("the emission sweep, the argmax threshold", m.group(1), top,
                  "results/setsize_headroom.json")


def _emission_crossfit(ctx) -> None:
    """The cross-fit: the split count, the reproduced estimate, the unanimity, the held-out spread."""
    C = ctx.art("emission_crossfit.json")
    if C is None:
        return
    m = re.search(r"over \$(\d+)\$ random splits, reproduces the point estimate at \$\+([\d.]+)\$ "
                  r"and selects \$\\alpha=([\d.]+)\$ as the training-part argmax in all \$(\d+)\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "the emission cross-fit sentence")
    else:
        ctx.check("the cross-fit, the number of splits", m.group(1), C["config"]["splits"],
                  "results/emission_crossfit.json")
        ctx.check("the cross-fit, the held-out estimate", m.group(2), C["held_out_gain_mean"],
                  "results/emission_crossfit.json")
        chosen = C["alpha_chosen"]
        ctx.check("the cross-fit, splits choosing that threshold", m.group(4),
                  chosen.get(m.group(3)), "results/emission_crossfit.json")
        ctx.checks.append((len(chosen) == 1, "the cross-fit, no split chose another threshold",
                           "one threshold", f"{len(chosen)}: {','.join(sorted(chosen))}",
                           "results/emission_crossfit.json"))
    m = re.search(r"The spread across held-out fifths is wide, \$\[(-[\d.]+),\+([\d.]+)\]\$",
                  ctx.flat)
    if not m:
        _miss(ctx, "the emission held-out-spread sentence")
    else:
        ctx.check("the cross-fit, the spread's lower end", m.group(1),
                  C["held_out_gain_ci95"][0], "results/emission_crossfit.json")
        ctx.check("the cross-fit, the spread's upper end", m.group(2),
                  C["held_out_gain_ci95"][1], "results/emission_crossfit.json")
    m = re.search(r"gains \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ in macro F1, but that is the "
                  r"weak arm", ctx.flat)
    H = ctx.art("setsize_headroom.json")
    if not m:
        _miss(ctx, "the emission weak-arm sentence")
    elif H is not None:
        arm = H["arms"]["gap rule a=0.5"]
        ctx.check("the emission weak arm, the gain over the deployed budget", m.group(1),
                  arm["f1_gain_over_k15"], "results/setsize_headroom.json")
        ctx.check("the emission weak arm, its lower bound", m.group(2), arm["ci95"][0],
                  "results/setsize_headroom.json")
        ctx.check("the emission weak arm, its upper bound", m.group(3), arm["ci95"][1],
                  "results/setsize_headroom.json")
    # 'four fifths of the oracle': both printings of the reachable share, and of the oracle's own
    if H is not None:
        avail = list(re.finditer(r"of an available \$\+([\d.]+)\$", ctx.flat))
        ctx.checks.append((len(avail) == 2, "the available headroom is printed in both places",
                           "2 printings", f"{len(avail)} printings", CS))
        for i, mm in enumerate(avail):
            ctx.check(f"the oracle's headroom, printing {i + 1}", mm.group(1),
                      H["arms"]["oracle count"]["f1_gain_over_best_constant"],
                      "results/setsize_headroom.json")
        m = re.search(r"The relative rule reaches \$\+([\d.]+)\$ of an available", ctx.flat)
        if not m:
            _miss(ctx, "the emission four-fifths sentence")
        else:
            ctx.check("the share of the oracle the rule reaches", m.group(1),
                      H["arms"]["gap rule a=0.5"]["f1_gain_over_best_constant"],
                      "results/setsize_headroom.json")


def _emission_absolute_rules(ctx) -> None:
    """The released negative result: two absolute rules that do not separate, and one that does."""
    S = ctx.art("stopping_rule.json")
    if S is None:
        return
    m = re.search(r"gains \$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ and calibrating it "
                  r"\$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$, neither separated from zero",
                  ctx.flat)
    if not m:
        _miss(ctx, "the emission absolute-threshold sentence")
    else:
        for label, key, base in (("the absolute threshold", "threshold", 1),
                                 ("the calibrated threshold", "calibrated", 4)):
            p = S["paired_vs_constant"][key]
            ctx.check(f"{label}'s gain over the constant", m.group(base), p["delta"],
                      "results/stopping_rule.json")
            ctx.check(f"{label}'s lower bound", m.group(base + 1), p["ci95"][0],
                      "results/stopping_rule.json")
            ctx.check(f"{label}'s upper bound", m.group(base + 2), p["ci95"][1],
                      "results/stopping_rule.json")
            ctx.checks.append((not p["excludes_zero"], f"{label} does not separate from zero",
                               "not separated", p["excludes_zero"], "results/stopping_rule.json"))
    m = re.search(r"an expected-F1 rule at \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission expected-F1 sentence")
    else:
        p = S["paired_vs_constant"]["expected_f1"]
        ctx.check("the expected-F1 rule's gain", m.group(1), p["delta"],
                  "results/stopping_rule.json")
        ctx.check("the expected-F1 rule's lower bound", m.group(2), p["ci95"][0],
                  "results/stopping_rule.json")
        ctx.check("the expected-F1 rule's upper bound", m.group(3), p["ci95"][1],
                  "results/stopping_rule.json")
        ctx.checks.append((p["excludes_zero"], "the expected-F1 rule does separate", "separated",
                           p["excludes_zero"], "results/stopping_rule.json"))


def _emission_populations(ctx) -> None:
    """Where the rule does not hold: the two strata, their sizes, and the strict-criterion rerun."""
    P = ctx.art("setsize_headroom__tautomer_parents.json")
    M = ctx.art("setsize_headroom__tautomer_metabolites.json")
    I = ctx.art("setsize_headroom__inchikey_all.json")
    A = ctx.art("setsize_headroom.json")
    if any(x is None for x in (P, M, I, A)):
        return
    m = re.search(r"On the \$(\d+)\$ parent drugs the rule beats the best constant by "
                  r"\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$; on the \$(\d+)\$ substrates that "
                  r"are themselves metabolites of another substrate here it does not, at "
                  r"\$(-[\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission stratum sentence")
    else:
        par, met = P["arms"]["gap rule a=0.5"], M["arms"]["gap rule a=0.5"]
        ctx.check("the parent stratum's size", m.group(1), P["config"]["n_substrates"],
                  "results/setsize_headroom__tautomer_parents.json")
        ctx.check("the parent stratum's margin", m.group(2),
                  par["f1_gain_over_best_constant"],
                  "results/setsize_headroom__tautomer_parents.json")
        ctx.check("the parent stratum's lower bound", m.group(3),
                  par["ci95_vs_best_constant"][0],
                  "results/setsize_headroom__tautomer_parents.json")
        ctx.check("the parent stratum's upper bound", m.group(4),
                  par["ci95_vs_best_constant"][1],
                  "results/setsize_headroom__tautomer_parents.json")
        ctx.check("the metabolite stratum's size", m.group(5), M["config"]["n_substrates"],
                  "results/setsize_headroom__tautomer_metabolites.json")
        ctx.check("the metabolite stratum's margin", m.group(6),
                  met["f1_gain_over_best_constant"],
                  "results/setsize_headroom__tautomer_metabolites.json")
        ctx.check("the metabolite stratum's lower bound", m.group(7),
                  met["ci95_vs_best_constant"][0],
                  "results/setsize_headroom__tautomer_metabolites.json")
        ctx.check("the metabolite stratum's upper bound", m.group(8),
                  met["ci95_vs_best_constant"][1],
                  "results/setsize_headroom__tautomer_metabolites.json")
        total = P["config"]["n_substrates"] + M["config"]["n_substrates"]
        ctx.checks.append((total == A["config"]["n_substrates"],
                           "the two strata partition the split",
                           f"{P['config']['n_substrates']} + {M['config']['n_substrates']}",
                           A["config"]["n_substrates"],
                           "results/setsize_headroom.json and its two population reruns"))
        ctx.checks.append((not met["separated_vs_best_constant"],
                           "the metabolite stratum does not separate", "not separated",
                           met["separated_vs_best_constant"],
                           "results/setsize_headroom__tautomer_metabolites.json"))
    m = re.search(r"than the tautomer-aware default the margin is \$\+([\d.]+)\$ "
                  r"\$\[\+([\d.]+),\+([\d.]+)\]\$", ctx.flat)
    if not m:
        _miss(ctx, "the emission strict-criterion sentence")
    else:
        arm = I["arms"]["gap rule a=0.5"]
        ctx.checks.append((I["config"]["match"] == "inchikey",
                           "the strict-criterion rerun is under plain inchikey", "inchikey",
                           I["config"]["match"], "results/setsize_headroom__inchikey_all.json"))
        ctx.check("the strict-criterion margin", m.group(1),
                  arm["f1_gain_over_best_constant"],
                  "results/setsize_headroom__inchikey_all.json")
        ctx.check("the strict-criterion margin, lower", m.group(2),
                  arm["ci95_vs_best_constant"][0],
                  "results/setsize_headroom__inchikey_all.json")
        ctx.check("the strict-criterion margin, upper", m.group(3),
                  arm["ci95_vs_best_constant"][1],
                  "results/setsize_headroom__inchikey_all.json")


def register_curator_agreement(ctx) -> None:
    """The two-curator Jaccard the case study quotes, which no pattern reached.

    The sentence is one of the paper's most quotable --- two independent curations of the same
    twenty-five drugs agree at $0.145$ under strict matching and $0.406$ under the tautomer-aware
    default --- and it was carried by nothing. Both readings come out of the criterion audit's own
    ladder, and the claim that follows them, that a third of the apparent disagreement is notation,
    is checked against the pair rather than left as prose.
    """
    import re as _re

    a = ctx.art("annotation_agreement_criterion_audit.json")
    if a is None:
        return
    m = _re.search(r"agree at a Jaccard of \$([\d.]+)\$ under strict\s*\\textsc\{inchikey\} "
                   r"matching and \$([\d.]+)\$ under the tautomer-aware default, a third of what "
                   r"reads as\s*disagreement", ctx.flat)
    ctx.checks.append((bool(m), "casestudy, the curator-agreement sentence is present", "present",
                       "matched" if m else "not matched",
                       "results/annotation_agreement_criterion_audit.json"))
    if not m:
        return
    ladder = a["criterion_consistency_defect"]["ladder"]
    ctx.check("casestudy, curator agreement under strict matching", m.group(1),
              ladder["inchikey"]["as_published"],
              "results/annotation_agreement_criterion_audit.json")
    ctx.check("casestudy, curator agreement under the tautomer default", m.group(2),
              ladder["inchikey_tautomer"]["as_published"],
              "results/annotation_agreement_criterion_audit.json")
    # "a third of what reads as disagreement": the share of the strict-matching disagreement that
    # the tautomer-aware reading recovers
    lo, hi = ladder["inchikey"]["as_published"], ladder["inchikey_tautomer"]["as_published"]
    share = (hi - lo) / (1.0 - lo)
    ctx.checks.append((0.28 <= share <= 0.36,
                       "casestudy, a third of the apparent disagreement is notation",
                       "between 0.28 and 0.36", f"{share:.3f}",
                       "results/annotation_agreement_criterion_audit.json"))
