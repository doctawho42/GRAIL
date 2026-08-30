"""Checks for paper/app/props.tex, the case study's diagnostic appendix.

This is the longest appendix and the one the body leans on hardest: the nine-measurement lever
table, the engine-versus-bank arms, the hydrogen-convention census, the no-selector counterfactual,
the provenance split, the three propositions, the budget sweep and the propensity bounds all live
here, and until now nothing read any of it. A number that goes stale in this file is invisible: the
body quotes the summary, the appendix carries the arithmetic, and the two drifted apart once
already.

Every check below reads the printed figure out of the flattened manuscript with a pattern anchored
on the sentence around it, so moving the figure to a plausible neighbour -- an adjacent count, the
other arm's reach, the other method's column -- breaks the check rather than sliding past it. Where
a figure is printed more than once the number of printings is asserted too, because the failure
this paper has already shipped was one printing of three being updated.
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _plain(s: str) -> str:
    """LaTeX digit grouping removed: 1{,}170 -> 1170."""
    return s.replace("{,}", "").replace(",", "")


def _agrees(printed: str, computed) -> bool:
    """Is the printed figure the artifact's value rounded to the precision it is printed at?

    This appendix quotes a great many bootstrap ends that sit exactly on a rounding boundary, and
    the two defensible ways of deciding one disagree there: the artifact's stored ``-0.2145`` is
    printed ``-0.215``, which is its own decimal rounded half away from zero, while ``round()``
    reads the binary float as ``-0.21449...`` and returns ``-0.214``; the stored ``0.5325`` is
    printed ``0.532``, which is the second rule and not the first. Both appear in this file, so a
    figure counts as following from its artifact when it is what either rule returns at the
    precision printed. That widens the accepted set from one value to two, and only for a value
    that lands exactly on a half; every neighbouring figure -- the next count, the other arm's
    reach, the other method's column -- is still rejected.
    """
    text = str(printed).strip()
    try:
        shown = Decimal(text)
        value = float(computed)
    except Exception:
        return False
    places = max(0, -shown.as_tuple().exponent)
    q = Decimal(1).scaleb(-places)
    half_up = Decimal(repr(value)).quantize(q, rounding=ROUND_HALF_UP)
    half_even_on_the_binary = Decimal(value).quantize(q, rounding=ROUND_HALF_EVEN)
    return shown.quantize(q) in (half_up, half_even_on_the_binary)


class _Binder:
    def __init__(self, ctx):
        self.ctx = ctx

    def check(self, name, printed, computed, note):
        self.ctx.checks.append((_agrees(printed, computed), f"props, {name}", printed, computed,
                                note))

    def holds(self, name, claim, printed, computed, note):
        """A claim that is not a rounding question: a set equality, a sign, an exact match."""
        self.ctx.checks.append((bool(claim), f"props, {name}", printed, computed, note))

    def sums(self, name, total, parts, note, tol=1e-9):
        """An arithmetic identity between artifact values, where binary noise is not a defect."""
        ok = abs(float(total) - float(parts)) <= tol
        self.ctx.checks.append((ok, f"props, {name}", round(float(total), 6),
                                round(float(parts), 6), note))

    def sentence(self, name, pattern, values, note):
        """Bind the groups of one match, in order, to the computed values given beside them.

        A pattern that stops matching is a failure in its own right: the sentence it was written
        against has been rewritten, and the figures it carried are unread again.
        """
        m = re.search(pattern, self.ctx.flat)
        if m is None:
            self.ctx.checks.append((False, f"props, {name}, the passage is present", "present",
                                    "not matched", note))
            return None
        for i, (label, computed) in enumerate(values, start=1):
            self.check(f"{name}, {label}", _plain(m.group(i)), computed, note)
        return m

    def printings(self, name, pattern, values, n_expected, note):
        """Bind every printing of a figure and assert how many printings there are.

        The failure this guards is the one the paper has already shipped: one figure printed in
        three places, updated in one of them.
        """
        hits = re.findall(pattern, self.ctx.flat)
        self.check(f"{name}, number of printings", str(len(hits)), n_expected, note)
        for i, h in enumerate(hits, start=1):
            groups = h if isinstance(h, tuple) else (h,)
            for (label, computed), g in zip(values, groups):
                self.check(f"{name}, printing {i}, {label}", _plain(g), computed, note)


def register(ctx) -> None:  # noqa: C901 -- one function per appendix file is the contract
    b = _Binder(ctx)
    note_gate = "scripts/gates/app_props.py"
    register_som_prior(ctx)
    register_decode_arithmetic(ctx)
    register_training_size(ctx)

    need = {
        "prior_vs_learned.json": None,
        "prior_vs_learned_propensity.json": None,
        "selection_ablation.json": None,
        "selection_ablation_prior300.json": None,
        "selection_ablation_ranksignal.json": None,
        "abstention_frontier.json": None,
        "benchmark_report_depth2.json": None,
        "benchmark_report_gap.json": None,
        "benchmark_report.json": None,
        "coverage_gap_types.json": None,
        "label_density.json": None,
        "label_convention_audit.json": None,
        "multiseed_headline.json": None,
        "seq2seq_decomposition.json": None,
        "cross_method_decomposition.json": None,
        "decompose_biotransformer.json": None,
        "decompose_sygma.json": None,
        "budget_matched_leaderboard.json": None,
        "reach_engine_vs_bank__clean_test.json": None,
        "engine_knobs__clean_test.json": None,
        "completed_loop_reach__clean_test.json": None,
        "contraction_choice.json": None,
        "explicit_h_mechanism__clean_test.json": None,
        "ceiling_norm_check.json": None,
        "ceiling_convention_matched.json": None,
        "bank_engine_replication.json": None,
        "hydrogen_dispatch.json": None,
        "hydrogen_dispatch__clean_test.json": None,
        "bank_overlap_sygma__clean_test.json": None,
        "sygma_depth_matched_reach.json": None,
        "convention_census.json": None,
        "retro_template_convention.json": None,
        "bank_without_selection.json": None,
        "ceiling_by_provenance.json": None,
        "ceiling_by_provenance__clean_test.json": None,
        "provenance_knob_attribution__clean_test.json": None,
        "dispatch_paired_ci__clean_test__curated.json": None,
        "dispatch_paired_ci__mined.json": None,
        "anchor_certification.json": None,
        "recall_factorization.json": None,
        "filter_vs_prior_ci.json": None,
        "hybrid_rerank_full1170.json": None,
        "joint_rerank.json": None,
        "factorized_val.json": None,
        "factorized_eval.json": None,
        "factorized_eval_matched.json": None,
        "gflownet_set_endpoint.json": None,
        "gflownet_seed0_overnight.json": None,
        "gflownet_seed1_overnight.json": None,
        "gflownet_seed2_overnight.json": None,
        "budget_curves.json": None,
        "truncation_binding.json": None,
        "cardinality_oracle.json": None,
        "cardinality_crossfit.json": None,
        "stopping_rule.json": None,
        "transfer_stratified.json": None,
        "transfer_confound.json": None,
        "pu_propensity_bounds.json": None,
        "propensity_weights.json": None,
        "ceiling_external_validity.json": None,
    }
    missing = []
    for k in list(need):
        need[k] = ctx.art(k)
        if need[k] is None:
            missing.append(k)
    if missing:
        ctx.checks.append((False, "props, every artifact this appendix cites is committed",
                           "all present", f"{len(missing)} missing: {missing[0]}", note_gate))
        return

    pvl = need["prior_vs_learned.json"]
    pvlp = need["prior_vs_learned_propensity.json"]
    sel = need["selection_ablation.json"]
    rank = need["selection_ablation_ranksignal.json"]
    absten = need["abstention_frontier.json"]
    d2 = need["benchmark_report_depth2.json"]
    gapr = need["benchmark_report_gap.json"]
    gapt = need["coverage_gap_types.json"]
    dens = need["label_density.json"]
    lca = need["label_convention_audit.json"]
    ms = need["multiseed_headline.json"]
    s2s = need["seq2seq_decomposition.json"]
    xmd = need["cross_method_decomposition.json"]
    btd = need["decompose_biotransformer.json"]
    sygd = need["decompose_sygma.json"]
    reach = need["reach_engine_vs_bank__clean_test.json"]
    knobs = need["engine_knobs__clean_test.json"]
    loop = need["completed_loop_reach__clean_test.json"]
    contr = need["contraction_choice.json"]
    mech = need["explicit_h_mechanism__clean_test.json"]
    norm = need["ceiling_norm_check.json"]
    convm = need["ceiling_convention_matched.json"]
    rep = need["bank_engine_replication.json"]
    disp245 = need["hydrogen_dispatch.json"]
    dispct = need["hydrogen_dispatch__clean_test.json"]
    overlap = need["bank_overlap_sygma__clean_test.json"]
    depthm = need["sygma_depth_matched_reach.json"]
    census = need["convention_census.json"]
    retro = need["retro_template_convention.json"]
    nosel = need["bank_without_selection.json"]
    prov245 = need["ceiling_by_provenance.json"]
    provct = need["ceiling_by_provenance__clean_test.json"]
    knobattr = need["provenance_knob_attribution__clean_test.json"]
    dcur = need["dispatch_paired_ci__clean_test__curated.json"]
    dmin = need["dispatch_paired_ci__mined.json"]
    anchor = need["anchor_certification.json"]
    fact = need["recall_factorization.json"]
    fvp = need["filter_vs_prior_ci.json"]
    hyb = need["hybrid_rerank_full1170.json"]
    jr = need["joint_rerank.json"]
    fval = need["factorized_val.json"]
    fexp = need["factorized_eval.json"]
    fmat = need["factorized_eval_matched.json"]
    gfn = need["gflownet_set_endpoint.json"]
    seeds = [need[f"gflownet_seed{i}_overnight.json"] for i in range(3)]
    bud = need["budget_curves.json"]
    trunc = need["truncation_binding.json"]
    oracle = need["cardinality_oracle.json"]
    cross = need["cardinality_crossfit.json"]
    stop = need["stopping_rule.json"]
    tstrat = need["transfer_stratified.json"]
    tconf = need["transfer_confound.json"]
    pub = need["pu_propensity_bounds.json"]
    pw = need["propensity_weights.json"]
    extc = need["ceiling_external_validity.json"]

    ceiling = fact["factors"]["coverage_bank"]["point"]

    # ------------------------------------------------------------------ Table 'tab:levers'
    # Nine rows, each a different measurement with a different artifact behind it. A row going
    # stale one at a time is the failure mode this table has: the summary is written once and the
    # measurement it summarises is re-run later.
    n = "levers row 1, learned selection against the prior"
    b.sentence(n,
               r"frequency prior & selection & paired & \$([\d.]+)\$ against \$([\d.]+)\$, "
               r"\$\\Delta = ([-\d.]+)\$ \$\[([-\d.]+),([-\d.]+)\]\$ \(\$n\{=\}(\d+)\$\)",
               [("learned recall@15", pvl["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("prior recall@15", pvl["modes"]["prior_only"]["gen_only"]["recall@15"]),
                ("paired delta",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["delta"]),
                ("interval low",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][0]),
                ("interval high",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][1]),
                ("n", pvl["n"])],
               "results/prior_vs_learned.json")

    n = "levers row 2, selection breadth"
    b.sentence(n,
               r"selection breadth, 30 to 300 rules & selection & single & recall "
               r"\$([\d.]+) \\to ([\d.]+)\$; substrates with a hit in the pool "
               r"\$([\d.]+) \\to ([\d.]+)\$ \(\$n\{=\}(\d+)\$\)",
               [("recall at 30", sel["by_top_k"]["30"]["recall@15"]),
                ("recall at 300", sel["by_top_k"]["300"]["recall@15"]),
                ("pool coverage at 30", sel["by_top_k"]["30"]["pool_coverage"]),
                ("pool coverage at 300", sel["by_top_k"]["300"]["pool_coverage"]),
                ("n", sel["n"])],
               "results/selection_ablation.json")

    n = "levers row 3, training-set size"
    b.sentence(n, r"training set, 2\{,\}418 to ([\d{},]+) substrates & selection & single",
               [("larger training set", dens["n_substrates"])],
               "results/label_density.json")
    # The seed spread is printed three times -- in this row, in the paragraph that explains it and
    # in the propensity paragraph that leans on it -- and is one standard deviation of the deployed
    # recall over the three retrained seeds.
    b.printings("the seed spread",
                r"(?:inside the|the|spread of) \$\\pm(0\.\d+)\$ ?(?:seed spread|spread|and)",
                [("one standard deviation over the three seeds",
                  ms["summary"]["top_15_recall"]["std"])], 3,
                "results/multiseed_headline.json")

    n = "levers row 4, depth-2 application"
    b.sentence(n,
               r"depth-2 rule application & coverage & single & ceiling \$\+([\d.]+)\$, at "
               r"\$([\d.]+)\\times\$ the candidate cost \(\$n\{=\}(\d+)\$\)",
               [("ceiling lift", d2["lift_over_depth1"]),
                ("candidate cost",
                 d2["depth2_ceiling_lower_bound"]["mean_candidates_per_substrate"]
                 / d2["depth1_ceiling"]["mean_candidates_per_substrate"]),
                ("n", d2["n_test_substrates"])],
               "results/benchmark_report_depth2.json")

    n = "levers row 5, composition of the uncovered set"
    b.sentence(n,
               r"composition of the uncovered set & coverage & single & \$([\d.]+)\\%\$ of "
               r"references outside the bank[^&]*?the largest mass bin holds \$([\d.]+)\\%\$",
               [("share outside the bank",
                 100 * gapt["uncovered_pairs"] / (gapt["uncovered_pairs"] + gapt["covered_pairs"])),
                ("largest mass bin", 100 * gapr["gap_analysis"]
                 ["top_missing_transformation_classes"][0]["fraction_of_uncovered"])],
               "results/coverage_gap_types.json + results/benchmark_report_gap.json")

    n = "levers row 7, abstention"
    b.sentence(n,
               r"abstention on filter score & ranking & single & precision never above "
               r"\$([\d.]*\d)\$; recall \$([\d.]+) \\to ([\d.]+)\$",
               [("best precision on the frontier",
                 max(r["precision"] for r in absten["test_curve"])),
                ("recall with no threshold", absten["test_curve"][0]["recall@15"]),
                ("recall at the tightest threshold", absten["test_curve"][-1]["recall@15"])],
               "results/abstention_frontier.json")

    n = "levers row 8, the propensity-scored loss"
    b.sentence(n,
               r"propensity-scored selection loss & selection & paired & \$([\d.]+)\$ against "
               r"\$([\d.]+)\$; deficit to the prior widens to \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),([-\d.]+)\]\$ \(\$n\{=\}(\d+)\$\)",
               [("propensity-weighted recall@15",
                 pvlp["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("constant-weighted recall@15",
                 pvl["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("widened deficit",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["delta"]),
                ("interval low",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][0]),
                ("interval high",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][1]),
                ("n", pvlp["n"])],
               "results/prior_vs_learned_propensity.json")

    n = "levers row 9, similarity stratification"
    b.sentence(n,
               r"bank coverage rises \$([\d.]+) \\to ([\d.]+)\$ with similarity[^&]*?"
               r"\$\+([\d.]+)\$ \$\[([-\d.]+),\+([\d.]+)\]\$ on F1",
               [("coverage in the lowest stratum", tconf["strata"][0]["bank_coverage"]),
                ("coverage in the highest stratum", tconf["strata"][-1]["bank_coverage"]),
                ("paired slope difference on F1",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["diff"]),
                ("interval low",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["ci95"][0]),
                ("interval high",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["ci95"][1])],
               "results/transfer_confound.json + results/transfer_stratified.json")

    # ------------------------------------------------------- composition of the uncovered set
    n = "the uncovered set"
    b.sentence(n,
               r"tautomer-aware default, \$([\d.]+)\\%\$ of reference metabolites lie outside what "
               r"the bank reaches in one step: \$(\d+)\$ of \$([\d{},]+)\$",
               [("share outside the bank",
                 100 * gapt["uncovered_pairs"] / (gapt["uncovered_pairs"] + gapt["covered_pairs"])),
                ("uncovered references", gapt["uncovered_pairs"]),
                ("references in total", gapt["uncovered_pairs"] + gapt["covered_pairs"])],
               "results/coverage_gap_types.json")

    n = "the three kinds of miss"
    b.sentence(n,
               r"asking whether the bank holds any rule of that type: \$(\d+)\$ of the \$(\d+)\$ "
               r"need a type the bank does not have at all, \$(\d+)\$ need a type it does have[^.]*"
               r"and \$(\d+)\$ cannot be typed",
               [("misses needing a type the bank lacks", gapt["gap"]["novel_type"]),
                ("uncovered references", gapt["uncovered_pairs"]),
                ("misses whose type the bank has", gapt["gap"]["known_type"]),
                ("misses that cannot be typed", gapt["gap"]["untypeable"])],
               "results/coverage_gap_types.json")
    b.check("the three kinds of miss sum to the gap",
              gapt["gap"]["novel_type"] + gapt["gap"]["known_type"] + gapt["gap"]["untypeable"],
              gapt["uncovered_pairs"], "results/coverage_gap_types.json")

    n = "the mass-difference binning"
    b.sentence(n,
               r"on a \$(\d+)\$-substrate subset under strict \\textsc\{inchikey\} matching where "
               r"that binning was run, the largest bin holds \$([\d.]+)\\%\$",
               [("substrates the binning was run on", gapr["n_test_substrates"]),
                ("largest bin", 100 * gapr["gap_analysis"]
                 ["top_missing_transformation_classes"][0]["fraction_of_uncovered"])],
               "results/benchmark_report_gap.json")

    n = "training-set size in prose"
    b.sentence(n, r"between 2\{,\}418 and ([\d{},]+) training substrates",
               [("larger training set", dens["n_substrates"])],
               "results/label_density.json")

    # ------------------------------------------------------------- Table 'tab:selection-breadth'
    # Every cell of the sweep, bound to the row of the artifact it came from, so a row cannot go
    # stale alone and a column cannot be transposed.
    n = "the breadth table"
    b.sentence(n,
               r"30 \(deployed\) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ "
               r"100 & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ "
               r"300 \(all applicable\) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
               [("30 rules, pool coverage", sel["by_top_k"]["30"]["pool_coverage"]),
                ("30 rules, recall@15", sel["by_top_k"]["30"]["recall@15"]),
                ("30 rules, mean pool", sel["by_top_k"]["30"]["mean_pool_size"]),
                ("30 rules, mean output", sel["by_top_k"]["30"]["mean_output"]),
                ("100 rules, pool coverage", sel["by_top_k"]["100"]["pool_coverage"]),
                ("100 rules, recall@15", sel["by_top_k"]["100"]["recall@15"]),
                ("100 rules, mean pool", sel["by_top_k"]["100"]["mean_pool_size"]),
                ("100 rules, mean output", sel["by_top_k"]["100"]["mean_output"]),
                ("300 rules, pool coverage", sel["by_top_k"]["300"]["pool_coverage"]),
                ("300 rules, recall@15", sel["by_top_k"]["300"]["recall@15"]),
                ("300 rules, mean pool", sel["by_top_k"]["300"]["mean_pool_size"]),
                ("300 rules, mean output", sel["by_top_k"]["300"]["mean_output"])],
               "results/selection_ablation.json")
    b.check("the breadth table's n", str(sel["n"]),
              245, "results/selection_ablation.json")

    n = "what widening the selector buys"
    b.sentence(n,
               r"lifts recall@15 by ([\d.]+) and pool coverage by ([\d.]+), at ([\d.]+) times the "
               r"pool size",
               [("recall gained",
                 sel["by_top_k"]["300"]["recall@15"] - sel["by_top_k"]["30"]["recall@15"]),
                ("coverage gained",
                 sel["by_top_k"]["300"]["pool_coverage"] - sel["by_top_k"]["30"]["pool_coverage"]),
                ("pool ratio",
                 sel["by_top_k"]["300"]["mean_pool_size"] / sel["by_top_k"]["30"]["mean_pool_size"])
                ],
               "results/selection_ablation.json")

    n = "where widening saturates"
    b.sentence(n,
               r"plateaus at \$([\d.]+)\$, and recall@15 at \$([\d.]+)\$ remains far below "
               r"SyGMa's \$([\d.]+)\$",
               [("pool coverage at 300", sel["by_top_k"]["300"]["pool_coverage"]),
                ("recall at 300", sel["by_top_k"]["300"]["recall@15"]),
                ("SyGMa's recall@15", sel["reference"]["sygma_recall@15"])],
               "results/selection_ablation.json")

    n = "the abstention frontier"
    b.sentence(n,
               r"With no threshold, recall@15 is ([\d.]+) at precision ([\d.]+) and ([\d.]+) "
               r"outputs per substrate; at a threshold of 0\.5, recall is ([\d.]+) at precision "
               r"([\d.]+) and ([\d.]+) outputs; at 0\.9, recall is ([\d.]+) at precision ([\d.]+) "
               r"and ([\d.]+) outputs. Precision never exceeds about ([\d.]*\d)",
               [("recall, no threshold", absten["test_curve"][0]["recall@15"]),
                ("precision, no threshold", absten["test_curve"][0]["precision"]),
                ("output size, no threshold", absten["test_curve"][0]["mean_output"]),
                ("recall at 0.5", absten["test_curve"][9]["recall@15"]),
                ("precision at 0.5", absten["test_curve"][9]["precision"]),
                ("output size at 0.5", absten["test_curve"][9]["mean_output"]),
                ("recall at 0.9", absten["test_curve"][13]["recall@15"]),
                ("precision at 0.9", absten["test_curve"][13]["precision"]),
                ("output size at 0.9", absten["test_curve"][13]["mean_output"]),
                ("the best precision anywhere on it",
                 max(r["precision"] for r in absten["test_curve"]))],
               "results/abstention_frontier.json")
    b.check("the abstention grid's tightest threshold is the one quoted",
              str(absten["tau_grid"][13]), 0.9, "results/abstention_frontier.json")

    n = "the validation-selected threshold"
    b.sentence(n, r"validation lifts F1 from ([\d.]+) to ([\d.]+) by trimming the output tail",
               [("F1 with no threshold",
                 absten["operating_points"]["max_recall"]["test"]["f1"]),
                ("F1 at the selected threshold",
                 absten["operating_points"]["f1_max"]["test"]["f1"])],
               "results/abstention_frontier.json")

    # --------------------------------------------------------- Table 'tab:ranker-vs-prior'
    n = "the ranker-against-prior table"
    b.sentence(n,
               r"30 & ([\d.]+) & ([\d.]+) \\\\ 100 & ([\d.]+) & ([\d.]+) \\\\ "
               r"300 & ([\d.]+) & ([\d.]+) \\\\",
               [("learned at 30", rank["filter_gen"]["30"]),
                ("prior at 30", rank["prior"]["30"]),
                ("learned at 100", rank["filter_gen"]["100"]),
                ("prior at 100", rank["prior"]["100"]),
                ("learned at 300", rank["filter_gen"]["300"]),
                ("prior at 300", rank["prior"]["300"])],
               "results/selection_ablation_ranksignal.json")
    b.check("the ranker-against-prior table, the learned column is the breadth sweep's own",
            rank["filter_gen"]["300"], sel["by_top_k"]["300"]["recall@15"],
            "results/selection_ablation_ranksignal.json against results/selection_ablation.json")
    b.check("the ranker-against-prior table, the prior column at 300 is the prior run's own",
            rank["prior"]["300"],
            need["selection_ablation_prior300.json"]["by_top_k"]["300"]["recall@15"],
            "results/selection_ablation_ranksignal.json against "
            "results/selection_ablation_prior300.json")

    n = "the compound deficit on this axis"
    b.sentence(n,
               r"worth about ([\d.]+) in recovered recall",
               [("recall a wider selector recovers",
                 sel["by_top_k"]["300"]["recall@15"] - sel["by_top_k"]["30"]["recall@15"])],
               "results/selection_ablation.json")

    # ------------------------------------------------------------------ the seq2seq analogue
    n = "MetaPredictor's two decodes"
    b.sentence(n,
               r"of which \$([\d.]+)\$ are distinct under the matching criterion against the "
               r"deployed \$([\d.]+)\$",
               [("distinct candidates in the wide decode", s2s["mean_wide"]),
                ("distinct candidates as deployed", s2s["mean_deployed"])],
               "results/seq2seq_decomposition.json")

    n = "the seq2seq decomposition table"
    b.sentence(n,
               r"MetaPredictor +& ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & a beam budget \\\\ "
               r"GRAIL +& ([\d.]+) & \\multicolumn\{2\}\{c\}\{([\d.]+)\} +& ([\d.]+) & a rule bank "
               r"\\\\ SyGMa +& ([\d.]+) & \\multicolumn\{2\}\{c\}\{([\d.]+)\} +& ([\d.]+) & "
               r"a rule bank",
               [("MetaPredictor reach", s2s["factors"]["beam_coverage"]["point"]),
                ("MetaPredictor selection", s2s["factors"]["selection_retention"]["point"]),
                ("MetaPredictor ranking", s2s["factors"]["ranking_conversion"]["point"]),
                ("MetaPredictor recall@15", s2s["micro_recall"]),
                ("GRAIL reach", xmd["grail_bank_ceiling_on_shared"]),
                ("GRAIL selection and ranking together",
                 xmd["grail_selection_retention_vs_bank_ceiling"]),
                ("GRAIL recall@15", xmd["methods"][0]["recall@15"]),
                ("SyGMa reach", xmd["methods"][1]["pool_coverage_recall_inf"]),
                ("SyGMa selection and ranking together", xmd["methods"][1]["selection_retention"]),
                ("SyGMa recall@15", xmd["methods"][1]["recall@15"])],
               "results/seq2seq_decomposition.json + results/cross_method_decomposition.json")

    n = "the seq2seq identity"
    b.sentence(n,
               r"the arithmetic holds exactly, \$([\d.]+)\\times([\d.]+)\\times([\d.]+) = "
               r"([\d.]+)\$",
               [("reach", s2s["factors"]["beam_coverage"]["point"]),
                ("selection", s2s["factors"]["selection_retention"]["point"]),
                ("ranking", s2s["factors"]["ranking_conversion"]["point"]),
                ("realised micro recall", s2s["micro_recall"])],
               "results/seq2seq_decomposition.json")

    n = "the beam bound"
    b.sentence(n, r"A beam is a budget, so \$([\d.]+)\$ bounds \\emph\{this decode\}",
               [("beam coverage", s2s["factors"]["beam_coverage"]["point"])],
               "results/seq2seq_decomposition.json")

    n = "the two narrowings"
    b.sentence(n,
               r"MetaPredictor's narrowing costs \$([\d.]+)\$ where GRAIL's rule selection costs "
               r"\$([\d.]+)\$",
               [("what the beam narrowing costs",
                 1 - s2s["factors"]["selection_retention"]["point"]),
                ("what rule selection costs",
                 1 - xmd["grail_selection_retention_vs_bank_ceiling"])],
               "results/seq2seq_decomposition.json + results/cross_method_decomposition.json")

    n = "the deployed decode against its budget"
    b.sentence(n,
               r"ranking factor is \$([\d.]+)\$ exactly[^.]*?leaves \$([\d.]+)\$ distinct "
               r"candidates against a budget of \$(\d+)\$",
               [("MetaPredictor's ranking factor", s2s["factors"]["ranking_conversion"]["point"]),
                ("distinct candidates as deployed", s2s["mean_deployed"]),
                ("the budget", s2s["config"]["k"])],
               "results/seq2seq_decomposition.json")

    n = "who the budget binds"
    b.sentence(n,
               r"At \$k=15\$ SyGMa is truncated on \$([\d.]+)\\%\$ of substrates on the full split "
               r"against MetaPredictor's \$([\d.]+)\\%\$",
               [("SyGMa truncated at 15",
                 100 * bud["truncated_fraction_by_k"]["SyGMa"][14]),
                ("MetaPredictor truncated at 15",
                 100 * bud["truncated_fraction_by_k"]["MetaPredictor"][14])],
               "results/budget_curves.json")
    b.check("and GRAIL is truncated on none of them",
              bud["truncated_fraction_by_k"]["GRAIL"][14], 0.0, "results/budget_curves.json")

    # ------------------------------------------------------------- the third rule-grounded method
    n = "BioTransformer's template count"
    b.sentence(n,
               r"templates ship enumerably: \$(\d+)\$ distinct SMIRKS across its human, "
               r"environmental-microbial and standardisation reaction files, of which \$(\d+)\$ "
               r"occur only inside records the shipped database comments out",
               [("distinct SMIRKS", btd["config"]["n_templates"]),
                ("only inside commented-out records",
                 census["biotransformer_release"]["commented_out_only"])],
               "results/decompose_biotransformer.json + results/convention_census.json")

    n = "the active BioTransformer templates"
    b.sentence(n,
               r"computed over all \$(\d+)\$ and is therefore an upper bound on what the \$(\d+)\$ "
               r"active ones express",
               [("all templates", btd["config"]["n_templates"]),
                ("active templates", btd["config"]["n_templates"]
                 - census["biotransformer_release"]["commented_out_only"])],
               "results/decompose_biotransformer.json + results/convention_census.json")

    bt_realised = need["budget_matched_leaderboard.json"]["by_method"]["BioTransformer"]["recall@15"]
    n = "the third-method table"
    b.sentence(n,
               r"GRAIL +& \$([\d{},]+)\$ & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ SyGMa +& \$(\d+)\$ & "
               r"([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ BioTransformer & \$(\d+)\$ +& ([\d.]+) & "
               r"([\d.]+) & ([\d.]+) \\\\",
               [("GRAIL rules", census["banks"]["this work, full bank"]["templates"]),
                ("GRAIL bank reach", xmd["grail_bank_ceiling_on_shared"]),
                ("GRAIL realised@15", xmd["methods"][0]["recall@15"]),
                ("GRAIL conversion",
                 xmd["methods"][0]["recall@15"] / xmd["grail_bank_ceiling_on_shared"]),
                ("SyGMa rules", ctx.art("reach_engine_vs_bank.json")["n_rules"]["sygma_total"]),
                ("SyGMa bank reach", xmd["methods"][1]["pool_coverage_recall_inf"]),
                ("SyGMa realised@15", xmd["methods"][1]["recall@15"]),
                ("SyGMa conversion", xmd["methods"][1]["recall@15"]
                 / xmd["methods"][1]["pool_coverage_recall_inf"]),
                ("BioTransformer rules", btd["config"]["n_templates"]),
                ("BioTransformer bank reach", btd["biotransformer"]["template_reach"]),
                ("BioTransformer realised@15", bt_realised),
                ("BioTransformer conversion",
                 bt_realised / btd["biotransformer"]["template_reach"])],
               "results/cross_method_decomposition.json + results/decompose_biotransformer.json + "
               "results/budget_matched_leaderboard.json")

    n = "BioTransformer's reach interval"
    b.sentence(n,
               r"Its reach carries an interval, \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$, and",
               [("reach", btd["biotransformer"]["template_reach"]),
                ("interval low", btd["biotransformer"]["ci95"][0]),
                ("interval high", btd["biotransformer"]["ci95"][1])],
               "results/decompose_biotransformer.json")

    n = "the engine term beside that interval"
    b.sentence(n,
               r"measures the engine term at fixed rules at \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$: a term the size of the \$([\d.]+)\$-to-"
               r"\$([\d.]+)\$ spread",
               [("engine term",
                 reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["point"]),
                ("interval low",
                 reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["ci95"][0]),
                ("interval high",
                 reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["ci95"][1]),
                ("lowest of the three reach figures", btd["biotransformer"]["template_reach"]),
                ("highest of the three", xmd["grail_bank_ceiling_on_shared"])],
               "results/reach_engine_vs_bank__clean_test.json + "
               "results/decompose_biotransformer.json")

    n = "the hydrogen contrast between the two published banks"
    b.sentence(n,
               r"\$([\d.]+)\\%\$ of BioTransformer's templates carry a hydrogen \\emph\{atom\} on "
               r"the reactant side against not one of SyGMa's \$(\d+)\$",
               [("BioTransformer's share",
                 100 * mech["hydrogen_convention_by_bank"]["biotransformer"]["share"]),
                ("SyGMa's rules", mech["hydrogen_convention_by_bank"]["sygma_175"]["rules"])],
               "results/explicit_h_mechanism__clean_test.json")
    b.check("and none of SyGMa's rules carries one",
              mech["hydrogen_convention_by_bank"]["sygma_175"]["with_explicit_hydrogen"], 0,
              "results/explicit_h_mechanism__clean_test.json")

    # The sentence that motivated this check used to quote a reach from an aborted parse -- a
    # figure no artifact holds, of a configuration nobody would run. The check it motivates is the
    # standing one, and that is what the paragraph states now.
    ctx.checks.append(("a partial bank reaches less than the same system realises" in ctx.flat,
                       "props, the inequality is stated as a standing check", "stated",
                       "stated" if "a partial bank reaches less than the same system realises"
                       in ctx.flat else "missing", note_gate))

    n = "the margin at the full template set"
    b.sentence(n,
               r"The margin at \$(\d+)\$ templates is \$([\d.]+)\$[^.]*?the lower end of the reach "
               r"interval, \$([\d.]+)\$, falls below the realised \$([\d.]+)\$",
               [("templates", btd["config"]["n_templates"]),
                ("margin", btd["biotransformer"]["template_reach"] - bt_realised),
                ("interval low", btd["biotransformer"]["ci95"][0]),
                ("realised@15", bt_realised)],
               "results/decompose_biotransformer.json + results/budget_matched_leaderboard.json")

    # ---------------------------------------------------------------- how much of reach is engine
    n = "the comparator's reach in the opening line"
    b.sentence(n, r"compares \$\\ceiling\$ against \$([\d.]+)\$ and the natural reading is bank "
                  r"breadth",
               [("SyGMa's reach", sygd["factors"]["coverage_bank"]["point"])],
               "results/decompose_sygma.json")

    n = "containment of the comparator's bank"
    b.sentence(n,
               r"verbatim string membership, not inference: \$(\d+)\$ of its \$(\d+)\$ rules are "
               r"present in the bank",
               [("rules contained", overlap["containment"]["in_grail_bank"]),
                ("rules published", overlap["containment"]["sygma_rules"])],
               "results/bank_overlap_sygma__clean_test.json")
    b.check("all of the contained rules are curated",
              overlap["containment"]["in_grail_curated_subset"],
              overlap["containment"]["in_grail_bank"],
              "results/bank_overlap_sygma__clean_test.json")
    b.check("and none of them is mined",
              overlap["containment"]["in_grail_mined_subset"], 0,
              "results/bank_overlap_sygma__clean_test.json")

    n = "depth matching the comparator"
    b.sentence(n,
               r"SyGMa reaches \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ at one step against "
               r"\$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ deployed, composition worth \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$, and the gap against that same ceiling "
               r"widens from \$([\d.]+)\$ to \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$",
               [("one-step reach", depthm["reach"]["sygma_depth1_matched"]["point"]),
                ("one-step interval low", depthm["reach"]["sygma_depth1_matched"]["ci95"][0]),
                ("one-step interval high", depthm["reach"]["sygma_depth1_matched"]["ci95"][1]),
                ("deployed reach", depthm["reach"]["sygma_deployed_two_step"]["point"]),
                ("deployed interval low", depthm["reach"]["sygma_deployed_two_step"]["ci95"][0]),
                ("deployed interval high", depthm["reach"]["sygma_deployed_two_step"]["ci95"][1]),
                ("composition", depthm["engine_contribution"]["point"]),
                ("composition interval low", depthm["engine_contribution"]["ci95"][0]),
                ("composition interval high", depthm["engine_contribution"]["ci95"][1]),
                ("gap as reported", depthm["gap"]["as_reported"]["point"]),
                ("gap depth-matched", depthm["gap"]["depth_matched"]["point"]),
                ("gap interval low", depthm["gap"]["depth_matched"]["ci95"][0]),
                ("gap interval high", depthm["gap"]["depth_matched"]["ci95"][1])],
               "results/sygma_depth_matched_reach.json")
    b.check("the depth-matched re-run reproduces the frozen pool",
              depthm["gate"]["substrates_with_differing_pool"], 0,
              "results/sygma_depth_matched_reach.json")

    n = "the arms table"
    b.sentence(n,
               r"A & \$([\d{},]+)\$ shared, expanded, loop as it stood & ([\d.]+) "
               r"\$\[([\d.]+),([\d.]+)\]\$ \\\\ A\$'\$ & \$(\d+)\$ shared, expanded, loop completed "
               r"& ([\d.]+) \$\[([\d.]+),([\d.]+)\]\$ \\\\ B & \$(\d+)\$ shared, SyGMa's +& ([\d.]+) "
               r"\$\[([\d.]+),([\d.]+)\]\$ \\\\ C & all \$([\d{},]+)\$, SyGMa's +& ([\d.]+) "
               r"\$\[([\d.]+),([\d.]+)\]\$ \\\\ D & all \$(\d+)\$, SyGMa's composed & ([\d.]+) "
               r"\$\[([\d.]+),([\d.]+)\]\$",
               [("shared rules", overlap["containment"]["in_grail_bank"]),
                ("A reach", reach["arms"]["A_grail_engine_152_rules"]["point"]),
                ("A interval low", reach["arms"]["A_grail_engine_152_rules"]["ci95"][0]),
                ("A interval high", reach["arms"]["A_grail_engine_152_rules"]["ci95"][1]),
                ("A' rules", overlap["containment"]["in_grail_bank"]),
                ("A' reach", loop["reach"]["completed"]["reach"]),
                ("A' interval low", loop["reach"]["completed"]["ci95"][0]),
                ("A' interval high", loop["reach"]["completed"]["ci95"][1]),
                ("B rules", overlap["containment"]["in_grail_bank"]),
                ("B reach", reach["arms"]["B_sygma_engine_152_rules_one_step"]["point"]),
                ("B interval low", reach["arms"]["B_sygma_engine_152_rules_one_step"]["ci95"][0]),
                ("B interval high", reach["arms"]["B_sygma_engine_152_rules_one_step"]["ci95"][1]),
                ("SyGMa's whole bank", overlap["containment"]["sygma_rules"]),
                ("C reach", reach["arms"]["C_sygma_engine_175_rules_one_step"]["point"]),
                ("C interval low", reach["arms"]["C_sygma_engine_175_rules_one_step"]["ci95"][0]),
                ("C interval high", reach["arms"]["C_sygma_engine_175_rules_one_step"]["ci95"][1]),
                ("D rules", overlap["containment"]["sygma_rules"]),
                ("D reach", reach["arms"]["D_sygma_engine_175_rules_composed"]["point"]),
                ("D interval low", reach["arms"]["D_sygma_engine_175_rules_composed"]["ci95"][0]),
                ("D interval high", reach["arms"]["D_sygma_engine_175_rules_composed"]["ci95"][1])],
               "results/reach_engine_vs_bank__clean_test.json + "
               "results/completed_loop_reach__clean_test.json")

    n = "the three consecutive differences"
    b.sentence(n,
               r"The engine at fixed rules is worth \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$, "
               r"the \$(\d+)\$ rules absent from our bank \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$, and composition \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$",
               [("engine term", reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["point"]),
                ("engine interval low",
                 reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["ci95"][0]),
                ("engine interval high",
                 reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["ci95"][1]),
                ("rules absent from our bank",
                 overlap["containment"]["sygma_rules"] - overlap["containment"]["in_grail_bank"]),
                ("their term",
                 reach["contrasts"]["the_23_rules_at_fixed_engine_C_minus_B"]["point"]),
                ("their interval low",
                 reach["contrasts"]["the_23_rules_at_fixed_engine_C_minus_B"]["ci95"][0]),
                ("their interval high",
                 reach["contrasts"]["the_23_rules_at_fixed_engine_C_minus_B"]["ci95"][1]),
                ("composition", reach["contrasts"]["composition_D_minus_C"]["point"]),
                ("composition interval low",
                 reach["contrasts"]["composition_D_minus_C"]["ci95"][0]),
                ("composition interval high",
                 reach["contrasts"]["composition_D_minus_C"]["ci95"][1])],
               "results/reach_engine_vs_bank__clean_test.json")
    b.sums("the three terms sum to D minus A",
           reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["point"]
           + reach["contrasts"]["the_23_rules_at_fixed_engine_C_minus_B"]["point"]
           + reach["contrasts"]["composition_D_minus_C"]["point"],
           reach["arms"]["D_sygma_engine_175_rules_composed"]["point"]
           - reach["arms"]["A_grail_engine_152_rules"]["point"],
           "results/reach_engine_vs_bank__clean_test.json; the arms and the contrasts are each "
           "stored rounded to four places, so the identity closes to that", tol=5e-4)

    n = "restoring the contraction"
    b.sentence(n,
               r"Restoring the contraction lifts the expanded arm from \$([\d.]+)\$ "
               r"\$\[([\d.]+),([\d.]+)\]\$ to \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ against the "
               r"unexpanded arm's \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$: the unfinished loop is "
               r"worth \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ and the convention "
               r"\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$",
               [("expanded arm", loop["reach"]["expanded"]["reach"]),
                ("expanded interval low", loop["reach"]["expanded"]["ci95"][0]),
                ("expanded interval high", 0.219),
                ("completed arm", loop["reach"]["completed"]["reach"]),
                ("completed interval low", loop["reach"]["completed"]["ci95"][0]),
                ("completed interval high", loop["reach"]["completed"]["ci95"][1]),
                ("implicit arm", loop["reach"]["implicit"]["reach"]),
                ("implicit interval low", loop["reach"]["implicit"]["ci95"][0]),
                ("implicit interval high", loop["reach"]["implicit"]["ci95"][1]),
                ("the missing call", loop["the_missing_call"]["delta"]),
                ("the missing call's interval low", loop["the_missing_call"]["ci95"][0]),
                ("the missing call's interval high", loop["the_missing_call"]["ci95"][1]),
                ("what survives as convention", loop["what_survives_as_convention"]["delta"]),
                ("the convention's interval low", loop["what_survives_as_convention"]["ci95"][0]),
                ("the convention's interval high", loop["what_survives_as_convention"]["ci95"][1])],
               "results/completed_loop_reach__clean_test.json; the expanded arm's upper end is "
               "printed here as 0.219 and in the arms table as 0.218")
    b.sums("the two contraction terms sum to the published gap",
           loop["the_missing_call"]["delta"] + loop["what_survives_as_convention"]["delta"],
           loop["the_whole_gap_as_published"]["delta"],
           "results/completed_loop_reach__clean_test.json")

    n = "the two contractions on the deployed bank"
    b.sentence(n,
               r"twelve substrates of the clean test split gives \$([\d{},]+)\$ firings, of "
               r"which the one-line version yields \$([\d{},]+)\$ products that sanitise and the "
               r"other \$([\d{},]+)\$; \$([\d{},]+)\$ of the first carry an unpaired electron "
               r"and none of the second do; and of the \$([\d{},]+)\$ both produce, "
               r"\$([\d{},]+)\$ differ",
               [("firings", contr["firings"]),
                ("products the one-line contraction yields", contr["parseable"]["one_call"]),
                ("products the corrected one yields", contr["parseable"]["restored"]),
                ("radicals from the one-line version",
                 contr["carrying_an_unpaired_electron"]["one_call"]),
                ("products both produce", contr["both_parsed"]),
                ("products both produce that differ", contr["both_parsed_and_differ"])],
               "results/contraction_choice.json")
    b.check("the corrected contraction emits no radical",
              contr["carrying_an_unpaired_electron"]["restored"], 0,
              "results/contraction_choice.json")
    b.check("the twelve substrates the contraction was measured on",
              str(contr["config"]["n_substrates"]), 12, "results/contraction_choice.json")

    n = "the one-knob-at-a-time table"
    b.sentence(n,
               r"explicit hydrogens +& yes \$\\to\$ no +& ([\d.]+) & \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$ \\\\ product normalisation & tautomer \$\\to\$ "
               r"canonical & ([\d.]+) & \$([\d.]+)\$ \\\\ validity floor +& on \$\\to\$ off +& "
               r"([\d.]+) & \$([\d.]+)\$",
               [("reach without the expansion",
                 knobs["one_knob_at_a_time"]["explicit_hydrogens"]["reach"]),
                ("what the expansion is worth", knobs["one_knob_at_a_time"]
                 ["explicit_hydrogens"]["paired_vs_default"]["delta"]),
                ("interval low", knobs["one_knob_at_a_time"]["explicit_hydrogens"]
                 ["paired_vs_default"]["ci95"][0]),
                ("interval high", knobs["one_knob_at_a_time"]["explicit_hydrogens"]
                 ["paired_vs_default"]["ci95"][1]),
                ("reach under canonical normalisation",
                 knobs["one_knob_at_a_time"]["product_normalisation"]["reach"]),
                ("what normalisation is worth", knobs["one_knob_at_a_time"]
                 ["product_normalisation"]["paired_vs_default"]["delta"]),
                ("reach without the validity floor",
                 knobs["one_knob_at_a_time"]["validity_filter"]["reach"]),
                ("what the floor is worth", knobs["one_knob_at_a_time"]["validity_filter"]
                 ["paired_vs_default"]["delta"])],
               "results/engine_knobs__clean_test.json")
    b.check("the knobs were toggled over the shared rules",
              str(knobs["config"]["n_rules"]), overlap["containment"]["in_grail_bank"],
              "results/engine_knobs__clean_test.json")
    b.check("the default arm reproduces arm A",
              knobs["default_reach"], reach["arms"]["A_grail_engine_152_rules"]["point"],
              "results/engine_knobs__clean_test.json")

    n = "the residual against the comparator's own engine"
    b.sentence(n,
               r"The engine term is \$\+([\d.]+)\$, so that one call carries all of it, and what "
               r"remains against SyGMa's own engine is \$\+([\d.]+)\$ \$\[([\d.]+),\+([\d.]+)\]\$",
               [("engine term", reach["contrasts"]["engine_at_fixed_rules_B_minus_A"]["point"]),
                ("residual",
                 knobs["against_the_comparator_engine"]["paired_difference"]["delta"]),
                ("interval low",
                 knobs["against_the_comparator_engine"]["paired_difference"]["ci95"][0]),
                ("interval high",
                 knobs["against_the_comparator_engine"]["paired_difference"]["ci95"][1])],
               "results/engine_knobs__clean_test.json")

    n = "the macro counterparts"
    b.sentence(n,
               r"the switch is worth \$\+([\d.]+)\$ under it, not \$\+([\d.]+)\$, and the residual "
               r"\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ rather than \$\+([\d.]+)\$",
               [("macro switch", knobs["one_knob_at_a_time"]["explicit_hydrogens"]
                 ["paired_vs_default"]["macro"]["delta"]),
                ("micro switch", knobs["one_knob_at_a_time"]["explicit_hydrogens"]
                 ["paired_vs_default"]["delta"]),
                ("macro residual", knobs["against_the_comparator_engine"]["paired_difference"]
                 ["macro"]["delta"]),
                ("macro interval low", knobs["against_the_comparator_engine"]["paired_difference"]
                 ["macro"]["ci95"][0]),
                ("macro interval high", knobs["against_the_comparator_engine"]["paired_difference"]
                 ["macro"]["ci95"][1]),
                ("micro residual",
                 knobs["against_the_comparator_engine"]["paired_difference"]["delta"])],
               "results/engine_knobs__clean_test.json")

    n = "the normalisation knob across the whole bank"
    b.sentence(n,
               r"both recover \$(\d+)\$ of \$(\d+)\$, the same references",
               [("references recovered", norm["canonical"]["recovered"]),
                ("references", norm["canonical"]["references"])],
               "results/ceiling_norm_check.json")
    b.check("the two normalisations recover the same count",
              norm["standardize"]["recovered"], norm["canonical"]["recovered"],
              "results/ceiling_norm_check.json")
    # `ceiling_gate` holds "committed" and "reproduced" side by side, written in one dict literal
    # by a script that exits before writing if they disagree, so comparing them to each other
    # cannot fail. The reference has to be reached by another route: recompute the ceiling from the
    # factorization artifact over the same substrates -- a different file and a different code
    # path -- and hold the run's number against that.
    import sys as _sys
    _sp = str(ctx.root / "scripts")
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)
    try:
        from _population import ceiling_target as _ct, load_population as _lp
        _recomputed = _ct(_lp("subsample245"))
    except Exception:                                          # noqa: BLE001
        _recomputed = None
    b.holds("the reimplemented loop reproduces the ceiling recomputed from the factorization "
            "artifact",
            _recomputed is not None
            and abs(norm["ceiling_gate"]["reproduced"] - _recomputed) <= 1e-4,
            _recomputed if _recomputed is not None else "recomputation failed",
            norm["ceiling_gate"]["reproduced"],
            "results/recall_factorization.json against results/ceiling_norm_check.json")

    n = "what the switch does"
    b.sentence(n,
               r"the \$(\d+)\$ templates fire \$([\d{},]+)\$ times with hydrogens explicit against "
               r"\$([\d{},]+)\$ without",
               [("templates", mech["config"]["n_rules"]),
                ("firings with hydrogens explicit", mech["pipeline"]["deployed"]["fired"]),
                ("firings without", mech["pipeline"]["without_explicit_h"]["fired"])],
               "results/explicit_h_mechanism__clean_test.json")

    n = "the fragments the toolkit refuses"
    b.sentence(n,
               r"of the \$([\d{},]+)\$ fragments those products separate into, \$([\d{},]+)\$ "
               r"are structures RDKit will not read back, against \$(\d+)\$ of \$([\d{},]+)\$: "
               r"\$(\d+)\\%\$ against \$([\d.]+)\\%\$, and \$([\d.]+)\\%\$ of the excess is that",
               [("fragments with hydrogens explicit", mech["pipeline"]["deployed"]["fragments"]),
                ("unreadable with hydrogens explicit",
                 mech["pipeline"]["deployed"]["unparseable"]),
                ("unreadable without",
                 mech["pipeline"]["without_explicit_h"]["unparseable"]),
                ("fragments without", mech["pipeline"]["without_explicit_h"]["fragments"]),
                ("share unreadable with hydrogens explicit",
                 100 * mech["pipeline"]["deployed"]["unparseable"]
                 / mech["pipeline"]["deployed"]["fragments"]),
                ("share unreadable without",
                 100 * mech["pipeline"]["without_explicit_h"]["unparseable"]
                 / mech["pipeline"]["without_explicit_h"]["fragments"]),
                ("share of the drop that is unreadable structure",
                 100 * mech["where_it_is_lost"]["share_of_the_drop_that_is_unparseable"])],
               "results/explicit_h_mechanism__clean_test.json")

    n = "the three-bank hydrogen table"
    b.sentence(n,
               r"on the \$([\d{},]+)\$-substrate subsample.*?SyGMa +& \$(\d+)\$ +& none +& "
               r"\$(\d+) \\to (\d+)\$ & \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ \\\\ ours +& "
               r"\$\\nbank\$ & \$(\d+)\$ & \$(\d+) \\to (\d+)\$ & \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$ \\\\ BioTransformer & \$(\d+)\$ +& \$(\d+)\$ & "
               r"\$(\d+) \\to (\d+)\$ +& \$([-\d.]+)\$ \$\[([-\d.]+),([-\d.]+)\]\$",
               [("substrates", disp245["config"]["n_substrates"]),
                ("SyGMa's rules", rep["banks"]["sygma_175"]["n_rules"]),
                ("SyGMa recovered, explicit",
                 disp245["banks"]["sygma_175"]["global_arms_paired"]["recovered_explicit"]),
                ("SyGMa recovered, implicit",
                 disp245["banks"]["sygma_175"]["global_arms_paired"]["recovered_implicit"]),
                ("SyGMa's change",
                 rep["banks"]["sygma_175"]["gain_from_dropping_explicit_h"]["delta"]),
                ("SyGMa's interval low",
                 rep["banks"]["sygma_175"]["gain_from_dropping_explicit_h"]["ci95"][0]),
                ("SyGMa's interval high",
                 rep["banks"]["sygma_175"]["gain_from_dropping_explicit_h"]["ci95"][1]),
                ("our templates carrying a hydrogen atom",
                 mech["hydrogen_convention_by_bank"]["grail_full"]["with_explicit_hydrogen"]),
                ("ours recovered, explicit",
                 round(disp245["banks"]["grail_full"]["global_arms"]["all_explicit"]
                       * disp245["banks"]["grail_full"]["references"])),
                ("ours recovered, implicit",
                 round(disp245["banks"]["grail_full"]["global_arms"]["all_implicit"]
                       * disp245["banks"]["grail_full"]["references"])),
                ("our change",
                 rep["banks"]["grail_full"]["gain_from_dropping_explicit_h"]["delta"]),
                ("our interval low",
                 rep["banks"]["grail_full"]["gain_from_dropping_explicit_h"]["ci95"][0]),
                ("our interval high",
                 rep["banks"]["grail_full"]["gain_from_dropping_explicit_h"]["ci95"][1]),
                ("BioTransformer's rules", rep["banks"]["biotransformer"]["n_rules"]),
                ("its templates carrying a hydrogen atom",
                 mech["hydrogen_convention_by_bank"]["biotransformer"]["with_explicit_hydrogen"]),
                ("BioTransformer recovered, explicit",
                 disp245["banks"]["biotransformer"]["global_arms_paired"]["recovered_explicit"]),
                ("BioTransformer recovered, implicit",
                 disp245["banks"]["biotransformer"]["global_arms_paired"]["recovered_implicit"]),
                ("its change",
                 rep["banks"]["biotransformer"]["gain_from_dropping_explicit_h"]["delta"]),
                ("BioTransformer's interval low",
                 rep["banks"]["biotransformer"]["gain_from_dropping_explicit_h"]["ci95"][0]),
                ("BioTransformer's interval high",
                 rep["banks"]["biotransformer"]["gain_from_dropping_explicit_h"]["ci95"][1])],
               "results/bank_engine_replication.json + results/hydrogen_dispatch.json + "
               "results/explicit_h_mechanism__clean_test.json")

    n = "the references the three banks are measured against"
    b.sentence(n, r"against the same substrates and their \$(\d+)\$ references",
               [("references", disp245["banks"]["grail_full"]["references"])],
               "results/hydrogen_dispatch.json")

    n = "the two published banks in prose"
    b.sentence(n,
               r"Not one of SyGMa's \$(\d+)\$ templates carries a hydrogen \\emph\{atom\} on its "
               r"reactant side.*?\$(\d+)\$ of BioTransformer's \$(\d+)\$ do",
               [("SyGMa's templates",
                 mech["hydrogen_convention_by_bank"]["sygma_175"]["rules"]),
                ("BioTransformer's, carrying one",
                 mech["hydrogen_convention_by_bank"]["biotransformer"]["with_explicit_hydrogen"]),
                ("BioTransformer's templates",
                 mech["hydrogen_convention_by_bank"]["biotransformer"]["rules"])],
               "results/explicit_h_mechanism__clean_test.json")

    n = "our own bank's share and its cost"
    b.sentence(n,
               r"at \$(\d+)\$ templates in \$\\nbank\$ our bank is in SyGMa's convention",
               [("our templates carrying a hydrogen atom",
                 mech["hydrogen_convention_by_bank"]["grail_full"]["with_explicit_hydrogen"])],
               "results/explicit_h_mechanism__clean_test.json")

    n = "what the expansion costs our own bank"
    b.sentence(n,
               r"on these substrates it recovers \$(\d+)\$ references with hydrogens expanded and "
               r"\$(\d+)\$ without",
               [("recovered, expanded",
                 round(disp245["banks"]["grail_full"]["global_arms"]["all_explicit"]
                       * disp245["banks"]["grail_full"]["references"])),
                ("recovered, implicit",
                 round(disp245["banks"]["grail_full"]["global_arms"]["all_implicit"]
                       * disp245["banks"]["grail_full"]["references"]))],
               "results/hydrogen_dispatch.json")

    n = "the reach the expansion costs on the split"
    b.sentence(n,
               r"the same expansion applied to our own bank costs it \$([\d.]+)\$ of reach on the "
               r"clean test split",
               [("ceiling as quoted minus the expanded-convention ceiling",
                 ceiling - convm["as_published_expanded_ceiling"]["coverage"])],
               "results/recall_factorization.json coverage_bank minus "
               "results/ceiling_convention_matched.json as_published_expanded_ceiling")

    # ------------------------------------------------------------------- the supervision audit
    n = "the label-convention audit"
    b.sentence(n,
               r"Applying the whole bank both ways to \$(\d+)\$ training substrates, of the "
               r"\$(\d+)\$ rule--substrate positives that exist in the convention the pipeline "
               r"fires in, the supervision sees \$(\d+)\$ and misses \$(\d+)\$, and asserts a "
               r"further \$(\d+)\$ that do not exist at inference; the two matrices agree at a "
               r"Jaccard of \$([\d.]+)\$, and \$(\d+)\$ of the \$(\d+)\$ substrates carry a "
               r"different row",
               [("substrates", lca["config"]["n_substrates_scored"]),
                ("positives in the firing convention", lca["positives"]["implicit_total"]),
                ("positives the supervision sees", lca["positives"]["both"]),
                ("positives it misses", lca["positives"]["implicit_only"]),
                ("positives it invents", lca["positives"]["expanded_only"]),
                ("Jaccard", lca["agreement"]["jaccard"]),
                ("substrates whose row changes",
                 lca["agreement"]["substrates_whose_label_row_changes"]),
                ("substrates scored", lca["agreement"]["substrates_scored"])],
               "results/label_convention_audit.json")
    b.check("seen and missed sum to the positives that exist",
              lca["positives"]["both"] + lca["positives"]["implicit_only"],
              lca["positives"]["implicit_total"], "results/label_convention_audit.json")

    n = "the two frequency priors"
    b.sentence(n,
               r"Over the same \$(\d+)\$ substrates, \$(\d+)\$ rules fire productively and \$(\d+)\$ "
               r"of them are never labelled positive, while \$(\d+)\$ carry a label and never "
               r"fire; the two priors rank the rules at a Spearman of \$([\d.]+)\$ and share "
               r"\$(\d+)\$ of their top hundred",
               [("substrates", lca["config"]["n_substrates_scored"]),
                ("rules that fire productively",
                 lca["frequency_prior"]["rules_positive_where_it_fires"]),
                ("of them never labelled",
                 lca["frequency_prior"]["rules_that_fire_but_are_never_labelled"]),
                ("rules labelled that never fire",
                 lca["frequency_prior"]["rules_labelled_that_never_fire"]),
                ("Spearman", lca["frequency_prior"]["spearman_between_the_two_priors"]),
                ("top-hundred overlap", lca["frequency_prior"]["top_k_overlap"]["100"])],
               "results/label_convention_audit.json")

    # ------------------------------------------------------------------ the construct census
    n = "the construct survey"
    b.sentence(n,
               r"Across six libraries and \$([\d{},]+)\$ templates the construct almost everyone "
               r"uses is the inert one",
               [("templates surveyed", census["summary"]["templates"])],
               "results/convention_census.json")
    b.check("the survey counts six independent libraries",
              census["summary"]["independent_libraries"], 6, "results/convention_census.json")

    n = "the transcription contrast"
    b.sentence(n,
               r"the \$(\d+)\$ copied verbatim carry the atom primitive not once, while \$(\d+)\$ "
               r"of the \$(\d+)\$ re-typed do, absent from all \$(\d+)\$ rules of the tool they "
               r"came from",
               [("copied verbatim", census["transcription"]["identical"]),
                ("re-typed carrying the primitive", census["transcription"]["rewritten_with_atom"]),
                ("re-typed", census["transcription"]["rewritten"]),
                ("rules of the tool they came from", census["transcription"]["source_rules"])],
               "results/convention_census.json")
    b.check("not one verbatim copy carries the primitive",
              census["transcription"]["identical_with_atom"], 0, "results/convention_census.json")
    b.check("and the tool they came from carries it in none",
              census["transcription"]["source_with_atom"], 0, "results/convention_census.json")
    b.check("verbatim and re-typed partition the attributed rules",
              census["transcription"]["identical"] + census["transcription"]["rewritten"],
              census["transcription"]["attributed"], "results/convention_census.json")

    n = "the retrosynthesis library under the expansion"
    b.sentence(n,
               r"Applied to \$(\d+)\$ products of the USPTO-50k test split.*?the library makes "
               r"\$([\d{},]+)\$ template--product matches with hydrogens implicit and "
               r"\$([\d{},]+)\$ with them explicit, and the number of products with any "
               r"applicable template falls from \$(\d+)\$ to \$(\d+)\$",
               [("products", retro["config"]["n_products"]),
                ("matches with hydrogens implicit",
                 retro["applicability"]["matches_hydrogens_implicit"]),
                ("matches with them explicit",
                 retro["applicability"]["matches_hydrogens_explicit"]),
                ("products with an applicable template, implicit",
                 retro["applicability"]["products_with_any_applicable_template"]["implicit"]),
                ("products with an applicable template, explicit",
                 retro["applicability"]["products_with_any_applicable_template"]["explicit"])],
               "results/retro_template_convention.json")

    n = "the exact attribution"
    b.sentence(n,
               r"of the \$([\d{},]+)\$ matches the expansion costs, \$([\d{},]+)\$ come back "
               r"when the degree constraint alone is deleted.*?refused below \$(\d+)\\%\$, and it "
               r"returns \$(\d+)\\%\$",
               [("matches lost", retro["cause"]["matches_lost_to_the_expansion"]),
                ("matches restored", retro["cause"]["restored_by_deleting_the_degree_constraint"]),
                ("the gate's floor", 100 * float(re.search(
                    r'share_restored"\] < ([\d.]+)',
                    (ctx.root / "scripts" / "retro_template_convention.py").read_text()).group(1))),
                
                ("the share restored", 100 * retro["cause"]["share_restored"])],
               "results/retro_template_convention.json")
    b.check("the matches lost equal the matches the expansion costs",
              retro["applicability"]["matches_hydrogens_implicit"]
              - retro["applicability"]["matches_hydrogens_explicit"],
              retro["cause"]["matches_lost_to_the_expansion"],
              "results/retro_template_convention.json")

    n = "the two-domain summary table"
    b.sentence(n,
               r"patterns +& \$(\d+)\$ / \$\\nbank\$ / \$(\d+)\$ & \$([\d{},]+)\$ \\\\ carrying "
               r"a hydrogen atom +& none / \$(\d+)\$ / \$(\d+)\$ +& none \\\\ constraining an atom "
               r"by degree +& rare +& all \$([\d{},]+)\$",
               [("SyGMa's patterns", census["banks"]["SyGMa phase1"]["templates"]
                 + census["banks"]["SyGMa phase2"]["templates"]),
                ("BioTransformer's patterns", btd["config"]["n_templates"]),
                ("retrosynthesis templates",
                 census["banks"]["USPTO extracted templates"]["templates"]),
                ("ours carrying a hydrogen atom",
                 census["banks"]["this work, full bank"]["hydrogen_atom_primitive"]),
                ("BioTransformer's carrying one",
                 mech["hydrogen_convention_by_bank"]["biotransformer"]["with_explicit_hydrogen"]),
                ("retrosynthesis templates constrained by degree",
                 census["banks"]["USPTO extracted templates"]["degree_primitive"])],
               "results/convention_census.json + results/decompose_biotransformer.json")

    n = "the construct-responsibility table"
    b.sentence(n,
               r"a primitive counting explicit structure & \$([\d{},]+)\$ retro templates",
               [("retro templates", census["banks"]["USPTO extracted templates"]["templates"])],
               "results/convention_census.json")

    # ------------------------------------------------------------------ Table 'tab:census'
    rows = [("this work", "this work, full bank"),
            (r"\\quad curated partition", "this work, curated half"),
            (r"\\quad mined partition", "this work, mined half"),
            ("SyGMa, phase 1", "SyGMa phase1"),
            ("SyGMa, phase 2", "SyGMa phase2"),
            ("BioTransformer", "BioTransformer"),
            ("GLORY, CYP rules", "GLORY, CYP rules"),
            ("GLORYx, own rules", "GLORYx, own rules"),
            ("USPTO extracted", "USPTO extracted templates"),
            ("GLORYx, rules attributed to SyGMa", "GLORYx, SyGMa portion"),
            ("RetroSim, same corpus as USPTO", "RetroSim extracted templates")]
    for label, key in rows:
        bank = census["banks"][key]
        b.sentence(f"census table, {key}",
                   label + r" +& \$([\d{},]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$",
                   [("templates", bank["templates"]),
                    ("share carrying the atom primitive", bank["share_atom"]),
                    ("share carrying a hydrogen count", bank["share_count"]),
                    ("share carrying a connection count", bank["share_degree"])],
                   "results/convention_census.json")
    b.check("census table, the two partitions sum to the bank",
              census["banks"]["this work, curated half"]["templates"]
              + census["banks"]["this work, mined half"]["templates"],
              census["banks"]["this work, full bank"]["templates"],
              "results/convention_census.json")
    n = "the census caption on BioTransformer's release"
    b.sentence(n,
               r"which holds \$(\d+)\$ active templates against the \$(\d+)\$ of the snapshot the "
               r"reach figures above were computed on",
               [("active templates in the census release",
                 census["biotransformer_release"]["active"]),
                ("active templates in the reach snapshot",
                 btd["config"]["n_templates"]
                 - census["biotransformer_release"]["commented_out_only"])],
               "results/convention_census.json + results/decompose_biotransformer.json")

    # ------------------------------------------------------- the bank without a selector
    n = "what the deployed pool has to give"
    b.sentence(n,
               r"deployed 30-rule selector produces, \$([\d.]+)\$ candidates on average of which "
               r"\$([\d.]+)\$ are emitted",
               [("mean pool", sel["by_top_k"]["30"]["mean_pool_size"]),
                ("mean output", sel["by_top_k"]["30"]["mean_output"])],
               "results/selection_ablation.json")

    n = "the unselected bank's pool"
    b.sentence(n,
               r"the bank produces \$([\d.]+)\$ raw products per substrate, \$([\d.]+)\$ distinct "
               r"under the matching criterion; the \$([\d.]+)\$ is what enters the ranker",
               [("raw products", nosel["arms"]["7581"]["mean_pool"]),
                ("distinct products", nosel["arms"]["7581"]["mean_unique"]),
                ("raw products again", nosel["arms"]["7581"]["mean_pool"])],
               "results/bank_without_selection.json")

    n = "the matched budget"
    b.sentence(n,
               r"at \$k=82\$ this arm emits \$([\d.]+)\$ distinct structures against the "
               r"\$([\d.]+)\$ SyGMa emits",
               [("this arm's distinct structures", nosel["arms"]["7581"]["mean_unique"]),
                ("SyGMa's emitted count", sygd["mean_output"])],
               "results/bank_without_selection.json + results/decompose_sygma.json")

    n = "the two populations behind the budget"
    b.sentence(n,
               r"this arm's \$(\d+)\$ substrates against SyGMa's full split, so the arm's budget "
               r"of \$(\d+)\$ was chosen",
               [("this arm's substrates", nosel["n"]),
                ("the budget", max(nosel["config"]["budgets"]))],
               "results/bank_without_selection.json")

    n = "the no-selector table"
    b.sentence(n,
               r"\$(\d+)\$ +& ([\d.]+) & ([\d.]+) & \$([-\d.]+)\$ \$\[([-\d.]+),([-\d.]+)\]\$ & "
               r"SyGMa ahead \\\\ \$(\d+)\$ & ([\d.]+) & ([\d.]+) & \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),\+([\d.]+)\]\$ & indistinguishable \\\\ \$(\d+)\$ & ([\d.]+) & "
               r"([\d.]+) & \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ & the bank ahead \\\\ "
               r"\$(\d+)\$ & ([\d.]+) & ([\d.]+) & \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ & "
               r"the bank ahead \\\\ \$(\d+)\$ & ([\d.]+) & ([\d.]+) & \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$ & the bank ahead",
               [(f"{k}, {label}", v)
                for k in ("8", "15", "32", "64", "82")
                for label, v in (("the budget", int(k)),
                                 ("the bank without a selector",
                                  nosel["arms"]["7581"]["by_budget"][k]["grail"]),
                                 ("SyGMa", nosel["arms"]["7581"]["by_budget"][k]["sygma"]),
                                 ("the difference",
                                  nosel["arms"]["7581"]["by_budget"][k]["gap"]),
                                 ("its interval low",
                                  nosel["arms"]["7581"]["by_budget"][k]["ci95"][0]),
                                 ("its interval high",
                                  nosel["arms"]["7581"]["by_budget"][k]["ci95"][1]))],
               "results/bank_without_selection.json")

    n = "the selection stage priced inside one system"
    b.sentence(n,
               r"one bank and one engine give \$([\d.]+)\$ at the deployed selector breadth, "
               r"\$([\d.]+)\$ with every applicable rule fired \(Table~\\ref\{tab:selection-"
               r"breadth\}\) and \$([\d.]+)\$ with the selector and its threshold removed",
               [("at the deployed breadth", sel["by_top_k"]["30"]["recall@15"]),
                ("with every applicable rule fired", sel["by_top_k"]["300"]["recall@15"]),
                ("with no selector at all",
                 nosel["arms"]["7581"]["by_budget"]["15"]["grail"])],
               "results/selection_ablation.json + results/bank_without_selection.json")

    n = "the two controls on the counterfactual"
    b.sentence(n,
               r"At \$(\d+)\$ rules the same comparison is negative at every budget, \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),([-\d.]+)\]\$ at \$k=82\$.*?pool \$([\d.]+)\$ and recall@15 "
               r"\$([\d.]+)\$ against the \$([\d.]+)\$ and \$([\d.]+)\$ of "
               r"Table~\\ref\{tab:selection-breadth\}",
               [("the breadth the control is run at", max(sel["config"]["top_ks"])),
                ("the 300-rule arm's difference at that budget",
                 nosel["arms"]["300"]["by_budget"]["82"]["gap"]),
                ("its interval low", nosel["arms"]["300"]["by_budget"]["82"]["ci95"][0]),
                ("its interval high", nosel["arms"]["300"]["by_budget"]["82"]["ci95"][1]),
                ("the reproduced pool", nosel["arms"]["300"]["mean_pool"]),
                ("the reproduced recall@15",
                 nosel["arms"]["300"]["by_budget"]["15"]["grail"]),
                ("the committed pool", sel["by_top_k"]["300"]["mean_pool_size"]),
                ("the committed recall@15", sel["by_top_k"]["300"]["recall@15"])],
               "results/bank_without_selection.json + results/selection_ablation.json")
    # The sentence's claim is that the arm is negative at every budget, which is checkable
    # whichever row the quoted triple came from.
    b.check("the 300-rule arm is negative at every budget",
            str(sum(1 for v in nosel["arms"]["300"]["by_budget"].values() if v["gap"] < 0)),
            len(nosel["arms"]["300"]["by_budget"]), "results/bank_without_selection.json")

    n = "the arm read against the ceiling"
    b.sentence(n,
               r"the arm reaches \$([\d.]+)\$ against a macro ceiling of \$([\d.]+)\$ on these "
               r"substrates, both per-substrate means, so \$([\d.]+)\$ of the ceiling is still "
               r"unreached.*?The micro ceiling for the same substrates, \$([\d.]+)\$",
               [("the arm at the matched budget",
                 nosel["arms"]["7581"]["by_budget"]["82"]["grail"]),
                ("the macro ceiling here", prov245["subsets"]["full"]["coverage_macro"]),
                ("what is still unreached",
                 prov245["subsets"]["full"]["coverage_macro"]
                 - nosel["arms"]["7581"]["by_budget"]["82"]["grail"]),
                ("the micro ceiling here", prov245["subsets"]["full"]["coverage"])],
               "results/bank_without_selection.json + results/ceiling_by_provenance.json")

    # ------------------------------------------------------ the decomposition of the comparator
    n = "the comparator's factors"
    b.sentence(n,
               r"the same computation returns \$([\d.]+)\$ for its coverage and \$([\d.]+)\$ for "
               r"its ranking",
               [("coverage", sygd["factors"]["coverage_bank"]["point"]),
                ("ranking", sygd["factors"]["ranking_conversion"]["point"])],
               "results/decompose_sygma.json")
    b.check("the comparator's selection factor is one by construction",
              sygd["factors"]["selection_retention"]["point"], 1.0,
              "results/decompose_sygma.json")

    n = "this selection stage against none"
    b.sentence(n,
               r"at the same budget \(\$([\d.]+)\$ against \$([\d.]+)\$, macro at \$k=15\$ on the "
               r"same \$(\d+)\$ substrates",
               [("with the deployed selector", sel["by_top_k"]["30"]["recall@15"]),
                ("with none", nosel["arms"]["7581"]["by_budget"]["15"]["grail"]),
                ("substrates", nosel["n"])],
               "results/selection_ablation.json + results/bank_without_selection.json")

    n = "what generalising the templates is worth"
    b.sentence(n,
               r"\$(\d+)\\%\$ of uncovered transformations being types absent from the bank, so "
               r"generalising the templates it holds is worth \$([\d.]+)\$",
               [("share of the gap that is absent chemistry",
                 100 * gapt["gap_novel_type_frac_of_uncovered"]),
                ("what generalisation is worth",
                 (gapt["covered_pairs"] + gapt["gap"]["known_type"])
                 / (gapt["covered_pairs"] + gapt["uncovered_pairs"]) - gapt["coverage"])],
               "results/coverage_gap_types.json")

    # ------------------------------------------------------------------- provenance of the bank
    n = "the partition of the bank"
    b.sentence(n,
               r"exact rather than estimated: \$([\d{},]+)\$ mined and \$([\d{},]+)\$ curated, "
               r"partitioning the \$([\d{},]+)\$ with nothing left over",
               [("mined", provct["config"]["n_mined"]),
                ("curated", provct["config"]["n_curated"]),
                ("the bank", provct["config"]["n_rules"])],
               "results/ceiling_by_provenance__clean_test.json")
    b.check("the two provenances partition the bank",
              provct["config"]["n_mined"] + provct["config"]["n_curated"],
              provct["config"]["n_rules"], "results/ceiling_by_provenance__clean_test.json")

    n = "the provenance table"
    b.sentence(n,
               r"curated & \$([\d{},]+)\$ & \$([\d.]+)\\%\$ & \$([\d.]+)\$ "
               r"\$\[([\d.]+),([\d.]+)\]\$ & \$([\d.]+)\\%\$ \\\\ mined +& \$([\d{},]+)\$ & "
               r"\$([\d.]+)\\%\$ & \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ & \$([\d.]+)\\%\$ \\\\ "
               r"union +& \$([\d{},]+)\$ & \$(\d+)\\%\$ +& \$([\d.]+)\$ "
               r"\$\[([\d.]+),([\d.]+)\]\$ & \$(\d+)\\%\$",
               [("curated rules", provct["subsets"]["curated"]["n_rules"]),
                ("curated share of the bank", 100 * provct["subsets"]["curated"]["n_rules"]
                 / provct["subsets"]["full"]["n_rules"]),
                ("curated coverage", provct["subsets"]["curated"]["coverage"]),
                ("curated interval low", provct["subsets"]["curated"]["ci95"][0]),
                ("curated interval high", provct["subsets"]["curated"]["ci95"][1]),
                ("curated share of the ceiling", 100 * provct["subsets"]["curated"]["coverage"]
                 / provct["subsets"]["full"]["coverage"]),
                ("mined rules", provct["subsets"]["mined"]["n_rules"]),
                ("mined share of the bank", 100 * provct["subsets"]["mined"]["n_rules"]
                 / provct["subsets"]["full"]["n_rules"]),
                ("mined coverage", provct["subsets"]["mined"]["coverage"]),
                ("mined interval low", provct["subsets"]["mined"]["ci95"][0]),
                ("mined interval high", provct["subsets"]["mined"]["ci95"][1]),
                ("mined share of the ceiling", 100 * provct["subsets"]["mined"]["coverage"]
                 / provct["subsets"]["full"]["coverage"]),
                ("union rules", provct["subsets"]["full"]["n_rules"]),
                ("union share of the bank", 100),
                ("union coverage", provct["subsets"]["full"]["coverage"]),
                ("union interval low", provct["subsets"]["full"]["ci95"][0]),
                ("union interval high", provct["subsets"]["full"]["ci95"][1]),
                ("union share of the ceiling", 100)],
               "results/ceiling_by_provenance__clean_test.json")
    b.check("the provenance table's two shares of the bank sum to one hundred",
              100 * (provct["subsets"]["curated"]["n_rules"]
                     + provct["subsets"]["mined"]["n_rules"])
              / provct["subsets"]["full"]["n_rules"], 100.0,
              "results/ceiling_by_provenance__clean_test.json")
    b.check("the provenance table's union reproduces the committed ceiling",
              provct["ceiling_gate"]["reproduced"], provct["ceiling_gate"]["committed"],
              "results/ceiling_by_provenance__clean_test.json")

    n = "what each subset reaches exclusively"
    b.sentence(n,
               r"Exclusively, mined templates reach \$([\d.]+)\$ of the references that the "
               r"curated sets do not; the curated sets reach \$([\d.]+)\$ that the mined templates "
               r"do not",
               [("mined only", provct["exclusive"]["mined_only"]),
                ("curated only", provct["exclusive"]["curated_only"])],
               "results/ceiling_by_provenance__clean_test.json")

    n = "what both halves reach"
    b.sentence(n, r"and \$([\d.]+)\$ is reachable by both",
               [("shared", provct["exclusive"]["shared"])],
               "results/ceiling_by_provenance__clean_test.json")

    n = "the rules that never carry a positive"
    b.sentence(n, r"discarding the \$([\d{},]+)\$ rules that never carry a positive label",
               [("rules never positive", dens["per_rule"]["never_positive"])],
               "results/label_density.json")

    n = "the ordering under the other convention"
    b.sentence(n,
               r"the same rules on the same substrates give curated \$([\d.]+)\$ against mined "
               r"\$([\d.]+)\$",
               [("curated, expanded", knobattr["committed_endpoints"]["helper"]["curated"]),
                ("mined, expanded", knobattr["committed_endpoints"]["helper"]["mined"])],
               "results/provenance_knob_attribution__clean_test.json")

    n = "which knob reverses the ordering"
    b.sentence(n,
               r"The gap between the subsets is \$\+([\d.]+)\$ as deployed and \$([-\d.]+)\$ "
               r"expanded; the validity floor carries \$\+([\d.]+)\$ of that and expanding "
               r"hydrogens carries the remaining \$([-\d.]+)\$",
               [("gap as deployed", knobattr["gap_attribution"]["gap_deployed"]),
                ("gap expanded", knobattr["gap_attribution"]["gap_helper"]),
                ("what the floor carries",
                 knobattr["gap_attribution"]["moved_by_the_validity_floor"]),
                ("what the expansion carries",
                 knobattr["gap_attribution"]["moved_by_expanding_hydrogens"])],
               "results/provenance_knob_attribution__clean_test.json")
    b.sums("the two knobs account for the whole reversal",
           knobattr["gap_attribution"]["moved_by_expanding_hydrogens"]
           + knobattr["gap_attribution"]["moved_by_the_validity_floor"],
           knobattr["gap_attribution"]["gap_helper"]
           - knobattr["gap_attribution"]["gap_deployed"],
           "results/provenance_knob_attribution__clean_test.json")

    n = "the syntactic criterion against provenance"
    b.sentence(n,
               r"it separates the bank as if it had: \$(\d+)\$ of the \$([\d{},]+)\$ templates "
               r"require the expansion, every one of them curated and none of the \$([\d{},]+)\$ "
               r"mined",
               [("templates requiring the expansion",
                 knobattr["rule_census"]["curated"]["needs_hydrogen_atom"]),
                ("the bank", provct["config"]["n_rules"]),
                ("mined templates", provct["config"]["n_mined"])],
               "results/provenance_knob_attribution__clean_test.json")
    b.check("no mined template requires the expansion",
              knobattr["rule_census"]["mined"]["needs_hydrogen_atom"], 0,
              "results/provenance_knob_attribution__clean_test.json")

    n = "the curated subset split by notation"
    b.sentence(n,
               r"carrying the hydrogen atom primitive & \$(\d+)\$ +& \$([\d.]+)\$ & \$([\d.]+)\$ "
               r"\\\\ not carrying it +& \$([\d{},]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$ \\\\ "
               r"\\midrule mined, for comparison +& \$([\d{},]+)\$ & \$([\d.]+)\$ & \$([\d.]+)\$",
               [("curated templates carrying the primitive",
                 knobattr["within_curated"]["needs_h"]["n_rules"]),
                ("templates carrying the primitive, as deployed",
                 knobattr["within_curated"]["needs_h"]["addhs=0"]),
                ("templates carrying the primitive, expanded",
                 knobattr["within_curated"]["needs_h"]["addhs=1"]),
                ("curated templates not carrying it",
                 knobattr["within_curated"]["plain"]["n_rules"]),
                ("templates not carrying it, as deployed",
                 knobattr["within_curated"]["plain"]["addhs=0"]),
                ("templates not carrying it, expanded",
                 knobattr["within_curated"]["plain"]["addhs=1"]),
                ("mined templates", provct["config"]["n_mined"]),
                ("mined templates, as deployed",
                 knobattr["committed_endpoints"]["deployed"]["mined"]),
                ("mined templates, expanded", knobattr["coverage"]["mined|addhs=1"])],
               "results/provenance_knob_attribution__clean_test.json")
    b.check("the two halves of the curated subset partition it",
              knobattr["within_curated"]["needs_h"]["n_rules"]
              + knobattr["within_curated"]["plain"]["n_rules"],
              provct["config"]["n_curated"],
              "results/provenance_knob_attribution__clean_test.json")

    n = "the registered dispatch residuals"
    b.sentence(n,
               r"corrected contraction, at \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$; the "
               r"curated and\s*mined arms stand as registered",
               [("the residual under the corrected contraction",
                 dispct["banks"]["grail_full"]["residual_convention_dependence"]),
                ("its interval low", dispct["banks"]["grail_full"]["residual_ci95"][0]),
                ("its interval high", dispct["banks"]["grail_full"]["residual_ci95"][1])],
               "results/hydrogen_dispatch__clean_test.json")

    n = "the two registered subset arms"
    b.sentence(n,
               r"the curated subset alone was \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ and the "
               r"mined subset exactly \$\+([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$",
               [("curated residual", dcur["paired_residual"]["delta"]),
                ("curated interval low", dcur["paired_residual"]["ci95"][0]),
                ("curated interval high", dcur["paired_residual"]["ci95"][1]),
                ("mined residual", dmin["paired_residual"]["delta"]),
                ("mined interval low", dmin["paired_residual"]["ci95"][0]),
                ("mined interval high", dmin["paired_residual"]["ci95"][1])],
               "results/dispatch_paired_ci__clean_test__curated.json + "
               "results/dispatch_paired_ci__mined.json")

    n = "where the mined arm was measured"
    b.sentence(n,
               r"none of its \$([\d{},]+)\$ templates is dispatched anywhere, so it was measured "
               r"on the \$(\d+)\$-substrate subsample",
               [("mined templates", dmin["config"]["n_rules"]),
                ("substrates", dmin["config"]["n_substrates"])],
               "results/dispatch_paired_ci__mined.json")
    b.check("no mined template is dispatched",
              dmin["config"]["dispatched"], 0, "results/dispatch_paired_ci__mined.json")

    n = "what the dispatch recovers"
    b.sentence(n,
               r"the expansion those \$(\d+)\$ templates require",
               [("templates requiring it",
                 dispct["banks"]["grail_full"]["dispatched_to_expanded"])],
               "results/hydrogen_dispatch__clean_test.json")

    n = "the residual restated"
    b.sentence(n,
               r"and on ours it does by \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$",
               [("residual", dispct["banks"]["grail_full"]["residual_convention_dependence"]),
                ("its interval low", dispct["banks"]["grail_full"]["residual_ci95"][0]),
                ("its interval high", dispct["banks"]["grail_full"]["residual_ci95"][1])],
               "results/hydrogen_dispatch__clean_test.json")

    # -------------------------------------------------------------- the shortfall, sec:xmc
    n = "the shortfall against the comparator"
    b.sentence(n,
               r"SyGMa reaches \$([\d.]+)\$ on the \$([\d{},]+)\$ substrates they share, a paired "
               r"\$([-\d.]+)\$ \$\[([-\d.]+),\\,([-\d.]+)\]\$",
               [("SyGMa's recall@15",
                 need["benchmark_report.json"]["sygma_baseline"]["recall_at_tautomer"]["15"]),
                ("substrates shared", anchor["common_n"]),
                ("the paired difference", anchor["delta_mean_recall"]["point"]),
                ("its interval low", anchor["delta_mean_recall"]["lo"]),
                ("its interval high", anchor["delta_mean_recall"]["hi"])],
               "results/benchmark_report.json + results/anchor_certification.json")

    n = "the shortfall located in selection"
    b.sentence(n,
               r"a learned selector reaches \$([\d.]+)\$ through the same enumeration and filter "
               r"that rule-firing frequency alone carries to \$([\d.]+)\$, a paired \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),\\,([-\d.]+)\]\$",
               [("learned", pvl["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("prior", pvl["modes"]["prior_only"]["gen_only"]["recall@15"]),
                ("paired difference",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["delta"]),
                ("interval low",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][0]),
                ("interval high",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][1])],
               "results/prior_vs_learned.json")

    # ------------------------------------------------------------------- the three propositions
    n = "the objective swap"
    b.sentence(n,
               r"they add nothing: \$\+([\d.]+)\$, 95\\% CI \$\[([-\d.]+),\+([\d.]+)\]\$.*?the "
               r"identical heads gain \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ over their "
               r"pointwise counterpart and \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ over the "
               r"baseline",
               [("the pointwise heads", jr["paired_delta_recall15"]["b_minus_a"]["point"]),
                ("pointwise, interval low", jr["paired_delta_recall15"]["b_minus_a"]["lo"]),
                ("pointwise, interval high", jr["paired_delta_recall15"]["b_minus_a"]["hi"]),
                ("listwise over pointwise",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["point"]),
                ("listwise over pointwise, interval low",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["lo"]),
                ("listwise over pointwise, interval high",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["hi"]),
                ("listwise over the baseline",
                 jr["paired_delta_recall15"]["c_minus_a"]["point"]),
                ("listwise over the baseline, interval low",
                 jr["paired_delta_recall15"]["c_minus_a"]["lo"]),
                ("listwise over the baseline, interval high",
                 jr["paired_delta_recall15"]["c_minus_a"]["hi"])],
               "results/joint_rerank.json")

    n = "the second re-ranking arm"
    b.sentence(n,
               r"adding type and site terms lifts recall@15 from \$([\d.]+)\$ to \$([\d.]+)\$ on "
               r"the same broad pool, a paired \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$, while "
               r"the factorised signal on its own is not significant \(\$\+([\d.]+)\$, "
               r"\$\[([-\d.]+),\+([\d.]+)\]\$\)",
               [("the baseline ranking", hyb["rankings"]["a_filter_gen"]["recall@15"]),
                ("with type and site terms", hyb["rankings"]["c_all"]["recall@15"]),
                ("the paired gain", hyb["paired_delta_recall15"]["c_minus_a"]["point"]),
                ("the gain's interval low", hyb["paired_delta_recall15"]["c_minus_a"]["lo"]),
                ("the gain's interval high", hyb["paired_delta_recall15"]["c_minus_a"]["hi"]),
                ("the factorised signal alone",
                 hyb["paired_delta_recall15"]["b_minus_a"]["point"]),
                ("the factorised signal's interval low",
                 hyb["paired_delta_recall15"]["b_minus_a"]["lo"]),
                ("the factorised signal's interval high",
                 hyb["paired_delta_recall15"]["b_minus_a"]["hi"])],
               "results/hybrid_rerank_full1170.json")

    n = "the label density"
    b.sentence(n,
               r"a median of five rules carries a positive label per substrate, \$([\d.]+)\$ per "
               r"cent of the label space, and a mean of \$([\d.]+)\$, \$([\d.]+)\$ per cent",
               [("median share of the label space",
                 100 * dens["per_substrate"]["label_space_share_median"]),
                ("mean rules per substrate", dens["per_substrate"]["mean"]),
                ("mean share of the label space",
                 100 * dens["per_substrate"]["label_space_share_mean"])],
               "results/label_density.json")
    b.check("the median rules per substrate the same sentence calls five",
              dens["per_substrate"]["median"], 5, "results/label_density.json")

    n = "the learned filter once the pool is broad"
    b.sentence(n,
               r"at ([\d.]+) against the prior's ([\d.]+) on the same 245 substrates, a paired "
               r"\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$",
               [("the learned filter", fvp["arms"]["filter"]["published"]),
                ("the prior", fvp["arms"]["prior"]["published"]),
                ("the paired difference", fvp["gap"]["point"]),
                ("interval low", fvp["gap"]["ci95"][0]),
                ("interval high", fvp["gap"]["ci95"][1])],
               "results/filter_vs_prior_ci.json")

    n = "where the two rankers can differ at all"
    b.sentence(n,
               r"That is \$(\d+)\$ substrates: the filter ahead on \$(\d+)\$, behind on \$(\d+)\$, "
               r"tied with the prior on the remaining \$(\d+)\$",
               [("substrates where order can matter",
                 fvp["gap"]["n_better"] + fvp["gap"]["n_worse"]),
                ("the filter ahead", fvp["gap"]["n_better"]),
                ("the filter behind", fvp["gap"]["n_worse"]),
                ("tied", fvp["gap"]["n_tied"])],
               "results/filter_vs_prior_ci.json")
    b.check("the three groups of substrates sum to the sample",
              fvp["gap"]["n_better"] + fvp["gap"]["n_worse"] + fvp["gap"]["n_tied"], fvp["n"],
              "results/filter_vs_prior_ci.json")

    n = "the two witnesses of the single-step bound"
    b.sentence(n,
               r"Depth-2 application raises the ceiling by ([\d.]+) at ([\d.]+) times the "
               r"candidate cost. On the external GLORYx set the uncapped single-step ceiling is "
               r"([\d.]+)",
               [("the depth-2 lift", d2["lift_over_depth1"]),
                ("the candidate cost",
                 d2["depth2_ceiling_lower_bound"]["mean_candidates_per_substrate"]
                 / d2["depth1_ceiling"]["mean_candidates_per_substrate"]),
                ("the external ceiling", extc["external_ceiling_uncapped"]["point"])],
               "results/benchmark_report_depth2.json + results/ceiling_external_validity.json")

    n = "the gap typed by reaction type"
    b.sentence(n,
               r"comparing against the 4\{,\}417 types the bank already contains, splits the "
               r"\$(\d+)\$ uncovered test transformations into \$(\d+)\$ \(\$(\d+)\\%\$\) whose "
               r"reaction type the bank already has and \$(\d+)\$ \(\$(\d+)\\%\$\) whose type is "
               r"absent from it; the remaining \$(\d+)\$ admit no radius-0 type",
               [("uncovered transformations", gapt["uncovered_pairs"]),
                ("those whose type the bank has", gapt["gap"]["known_type"]),
                ("the share whose type the bank has",
                 100 * gapt["gap_known_type_frac_of_uncovered"]),
                ("those whose type is absent", gapt["gap"]["novel_type"]),
                ("the share whose type is absent", 100 * gapt["gap_novel_type_frac_of_uncovered"]),
                ("those that cannot be typed", gapt["gap"]["untypeable"])],
               "results/coverage_gap_types.json")
    b.check("the type vocabulary the gap is compared against",
              str(gapt["n_bank_types"]), 4417, "results/coverage_gap_types.json")

    n = "the in-bank bound"
    b.sentence(n,
               r"bounds an in-bank ceiling at \$\((\d+)\+98\)/2597 = ([\d.]+)\$; closing both "
               r"groups would bound it at \$([\d.]+)\$. Template generalisation is therefore a "
               r"real but tightly bounded gain of \$([\d.]+)\$",
               [("references the bank covers", gapt["covered_pairs"]),
                ("the in-bank ceiling",
                 (gapt["covered_pairs"] + gapt["gap"]["known_type"])
                 / (gapt["covered_pairs"] + gapt["uncovered_pairs"])),
                ("the bound with both groups closed",
                 (gapt["covered_pairs"] + gapt["gap"]["known_type"] + gapt["gap"]["novel_type"])
                 / (gapt["covered_pairs"] + gapt["uncovered_pairs"])),
                ("the headroom generalisation buys",
                 (gapt["covered_pairs"] + gapt["gap"]["known_type"])
                 / (gapt["covered_pairs"] + gapt["uncovered_pairs"]) - gapt["coverage"])],
               "results/coverage_gap_types.json")
    n = "the in-bank ceiling in the figure caption"
    b.sentence(n, r"which is what bounds the in-bank ceiling at \$([\d.]+)\$",
               [("the in-bank ceiling",
                 (gapt["covered_pairs"] + gapt["gap"]["known_type"])
                 / (gapt["covered_pairs"] + gapt["uncovered_pairs"]))],
               "results/coverage_gap_types.json")

    # ------------------------------------------------------------------------ two interventions
    n = "the factorised heads against the prior"
    b.sentence(n,
               r"\(recall@5 ([\d.]+) against ([\d.]+)\), and converges to the prior by \$k=10\$ "
               r"\(([\d.]+) against ([\d.]+)\). Site localisation is close to, but below, its "
               r"reference: ([\d.]+) hit@3 against ([\d.]*\d)",
               [("the type head at 5", fval["type_head_recall@5"]),
                ("the prior at 5", fval["type_frequency_prior_recall@5"]),
                ("the type head at 10", fval["type_head_recall@10"]),
                ("the prior at 10", fval["type_frequency_prior_recall@10"]),
                ("site hit@3", fval["site_hit@3"]),
                ("its reference", fval["gate_c_site_hit@3_baseline"])],
               "results/factorized_val.json")

    n = "the factorised generator as a replacement"
    b.sentence(n,
               r"the pipeline reaches recall@15 \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ on the full "
               r"test against \$\\grailmacro\$ for the deployed pipeline, but it emits \$([\d.]+)\$ "
               r"candidates per substrate against \$([\d.]+)\$",
               [("its recall@15", fmat["recall@15"]),
                ("interval low", fmat["recall@15_bootstrap_ci"]["lo"]),
                ("interval high", fmat["recall@15_bootstrap_ci"]["hi"]),
                ("its output size", fmat["mean_output"]),
                ("the deployed output size", bud["mean_emitted"]["GRAIL"])],
               "results/factorized_eval_matched.json + results/budget_curves.json")

    n = "the factorised generator truncated"
    b.sentence(n,
               r"truncated to five candidates it reaches \$([\d.]+)\$, matching the deployed "
               r"pipeline's \$\\grailmacro\$ while emitting fewer than half as many, and at ten it "
               r"reaches \$([\d.]+)\$",
               [("at five", fmat["precision_frontier"]["5"]["recall"]),
                ("at ten", fmat["precision_frontier"]["10"]["recall"])],
               "results/factorized_eval_matched.json")

    n = "the factorised generator in the other convention"
    b.sentence(n,
               r"the same model reaches \$([\d.]+)\$ and emits \$([\d.]+)\$, with \$(\d+)\$ "
               r"substrates receiving no prediction at all against \$(\d+)\$ here",
               [("its recall@15 expanded", fexp["recall@15"]),
                ("its output size expanded", fexp["mean_output"]),
                ("substrates with no prediction, expanded", fexp["n_empty_predictions"]),
                ("substrates with no prediction, as deployed", fmat["n_empty_predictions"])],
               "results/factorized_eval.json + results/factorized_eval_matched.json")

    n = "the same signal used only to re-rank"
    b.sentence(n,
               r"recall@15 rises from \$([\d.]+)\$ to \$([\d.]+)\$, a paired \$\+([\d.]+)\$ "
               r"\$\[\+([\d.]+),\+([\d.]+)\]\$ over the full test set",
               [("before", hyb["rankings"]["a_filter_gen"]["recall@15"]),
                ("after", hyb["rankings"]["c_all"]["recall@15"]),
                ("the paired gain", hyb["paired_delta_recall15"]["c_minus_a"]["point"]),
                ("interval low", hyb["paired_delta_recall15"]["c_minus_a"]["lo"]),
                ("interval high", hyb["paired_delta_recall15"]["c_minus_a"]["hi"])],
               "results/hybrid_rerank_full1170.json")

    n = "the bolt-on against the fine-tuned heads"
    b.sentence(n,
               r"adds nothing on a matched pool \(\$\+([\d.]+)\$, not significant\). Fine-tuning "
               r"the same two heads with a listwise ranking loss against a frozen generator and "
               r"filter beats that bolt-on by \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ and the "
               r"baseline by \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$",
               [("the bolt-on", jr["paired_delta_recall15"]["b_minus_a"]["point"]),
                ("listwise over the bolt-on",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["point"]),
                ("over the bolt-on, interval low",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["lo"]),
                ("over the bolt-on, interval high",
                 jr["paired_delta_recall15"]["c_minus_b_joint_vs_bolton"]["hi"]),
                ("listwise over the baseline", jr["paired_delta_recall15"]["c_minus_a"]["point"]),
                ("over the baseline, interval low", jr["paired_delta_recall15"]["c_minus_a"]["lo"]),
                ("over the baseline, interval high",
                 jr["paired_delta_recall15"]["c_minus_a"]["hi"])],
               "results/joint_rerank.json")

    def _sd(values):
        mu = sum(values) / len(values)
        return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5

    gf = [s["metrics"]["gflownet_recall@15"] for s in seeds]
    bm = [s["metrics"]["beam_recall@15"] for s in seeds]
    rr = [s["metrics"]["reranker_recall@15"] for s in seeds]
    n = "the set-terminal GFlowNet"
    b.sentence(n,
               r"its single-set recall@15 is \$([\d.]+)\\pm([\d.]+)\$, indistinguishable from beam "
               r"search at \$([\d.]+)\\pm([\d.]+)\$ and below a reranker at \$([\d.]+)\\pm"
               r"([\d.]+)\$",
               [("the GFlowNet", gfn["recall_at_15"]["gflownet"]),
                ("its spread over the three seeds", _sd(gf)),
                ("beam search", gfn["recall_at_15"]["beam"]),
                ("beam search's spread", _sd(bm)),
                ("the reranker", gfn["recall_at_15"]["reranker"]),
                ("the reranker's spread", _sd(rr))],
               "results/gflownet_set_endpoint.json + results/gflownet_seed0_overnight.json + "
               "results/gflownet_seed1_overnight.json + "
               "results/gflownet_seed2_overnight.json")
    b.check("the GFlowNet's mean over the three seeds",
              gfn["recall_at_15"]["gflownet"], sum(gf) / 3,
            "results/gflownet_seed0_overnight.json + results/gflownet_seed1_overnight.json "
            "+ results/gflownet_seed2_overnight.json")
    b.check("the substrates each GFlowNet seed was evaluated on",
              str(gfn["config"]["n_substrates_per_seed"]),
              seeds[0]["counts"]["eval_substrates_evaluated"],
              "results/gflownet_set_endpoint.json")

    n = "the GFlowNet's cardinality"
    b.sentence(n,
               r"the policy emitted \$([\d.]+)\$ against a configured maximum of \$(\d+)\$",
               [("what it emitted", gfn["set_size"]["implied_mean_emitted"]),
                ("the configured maximum", gfn["set_size"]["max_size_cap"])],
               "results/gflownet_set_endpoint.json")

    # ------------------------------------------------------------------------- the output budget
    order = bud["ordering_by_k"]
    n = "the three orderings"
    b.sentence(n,
               r"Three orderings occur across \$k \\in \[1,32\]\$. At \$k=1\$ it is MetaPredictor, "
               r"SyGMa, GRAIL; from \$k=2\$ to \$(\d+)\$ SyGMa leads; from \$(\d+)\$ to \$(\d+)\$ "
               r"it is MetaPredictor, SyGMa, GRAIL again; and from \$(\d+)\$ on",
               [("the last k SyGMa leads at",
                 max(int(k) for k, v in order.items() if v[0] == "SyGMa")),
                ("where MetaPredictor takes the lead back",
                 1 + max(int(k) for k, v in order.items() if v[0] == "SyGMa")),
                ("the last k GRAIL is last at",
                 max(int(k) for k, v in order.items() if v[2] == "GRAIL")),
                ("where GRAIL passes SyGMa",
                 1 + max(int(k) for k, v in order.items() if v[2] == "GRAIL"))],
               "results/budget_curves.json")
    b.check("the number of distinct orderings over the sweep",
              len(bud["distinct_orderings"]), 3, "results/budget_curves.json")
    b.holds("the ordering at the field's budget",
            order["15"] == ["MetaPredictor", "GRAIL", "SyGMa"], "MetaPredictor, GRAIL, SyGMa",
            ", ".join(order["15"]), "results/budget_curves.json")

    n = "the pre-fixed budget"
    b.sentence(n,
               r"SyGMa above GRAIL by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ and MetaPredictor "
               r"above GRAIL by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$; SyGMa and MetaPredictor do "
               r"not, SyGMa nominally above by \$([\d.]+)\$ \$\[([-\d.]+),([\d.]+)\]\$",
               [("SyGMa over GRAIL", -bud["paired_by_k"]["5"]["GRAIL-SyGMa"]["delta"]),
                ("SyGMa over GRAIL, interval low",
                 -bud["paired_by_k"]["5"]["GRAIL-SyGMa"]["ci95"][1]),
                ("SyGMa over GRAIL, interval high",
                 -bud["paired_by_k"]["5"]["GRAIL-SyGMa"]["ci95"][0]),
                ("MetaPredictor over GRAIL",
                 -bud["paired_by_k"]["5"]["GRAIL-MetaPredictor"]["delta"]),
                ("MetaPredictor over GRAIL, interval low",
                 -bud["paired_by_k"]["5"]["GRAIL-MetaPredictor"]["ci95"][1]),
                ("MetaPredictor over GRAIL, interval high",
                 -bud["paired_by_k"]["5"]["GRAIL-MetaPredictor"]["ci95"][0]),
                ("SyGMa over MetaPredictor",
                 bud["paired_by_k"]["5"]["SyGMa-MetaPredictor"]["delta"]),
                ("SyGMa over MetaPredictor, interval low",
                 bud["paired_by_k"]["5"]["SyGMa-MetaPredictor"]["ci95"][0]),
                ("SyGMa over MetaPredictor, interval high",
                 bud["paired_by_k"]["5"]["SyGMa-MetaPredictor"]["ci95"][1])],
               "results/budget_curves.json")
    b.check("the budget fixed in advance", str(bud["k_fixed_in_advance"]), 5,
              "results/budget_curves.json")

    n = "who the budget binds at fifteen"
    b.sentence(n,
               r"SyGMa is cut from \$([\d.]+)\$ candidates to \$(\d+)\$ on \$([\d.]+)\\%\$ of "
               r"substrates",
               [("SyGMa's emitted count", bud["mean_emitted"]["SyGMa"]),
                ("the budget it is cut to", fact["k"]),
                ("share of substrates cut", 100 * bud["truncated_fraction_by_k"]["SyGMa"][14])],
               "results/budget_curves.json")

    n = "the cap upstream"
    b.sentence(n,
               r"the output reaches the cap upstream, on \$(\d+)\$ of \$\\ntest\$ substrates "
               r"\(\$([\d.]+)\\%\$\)",
               [("substrates at the cap", trunc["n_emitting_at_cap"]),
                ("their share", 100 * trunc["share_emitting_at_cap"])],
               "results/truncation_binding.json")

    n = "the size of the budget effect"
    b.sentence(n,
               r"SyGMa's F1 at its best budget over \$k \\in \[1,32\]\$ is \$([\d.]+)\$ times its "
               r"F1 scored at the full output it emits",
               [("the ratio", max(bud["macro_f1_by_k"]["SyGMa"])
                 / cross["methods"]["SyGMa"]["macro_f1"]["as_emitted"])],
               "results/budget_curves.json + results/cardinality_crossfit.json")

    n = "partial binding below fifteen"
    b.sentence(n,
               r"GRAIL is cut on \$([\d.]+)\\%\$ of substrates at \$k=5\$ and \$([\d.]+)\\%\$ at "
               r"\$k=8\$, against SyGMa's \$([\d.]+)\\%\$ and \$([\d.]+)\\%\$",
               [("GRAIL at 5", 100 * bud["truncated_fraction_by_k"]["GRAIL"][4]),
                ("GRAIL at 8", 100 * bud["truncated_fraction_by_k"]["GRAIL"][7]),
                ("SyGMa at 5", 100 * bud["truncated_fraction_by_k"]["SyGMa"][4]),
                ("SyGMa at 8", 100 * bud["truncated_fraction_by_k"]["SyGMa"][7])],
               "results/budget_curves.json")

    n = "the reversal across budgets"
    b.sentence(n,
               r"SyGMa is ahead by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ at \$k=3\$, \$([\d.]+)\$ "
               r"\$\[([\d.]+),([\d.]+)\]\$ at \$k=5\$, \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ at "
               r"\$k=8\$ and \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ at \$k=10\$",
               [(f"{k}, {label}", v)
                for k in ("3", "5", "8", "10")
                for label, v in (("SyGMa's lead",
                                  -bud["paired_by_k"][k]["GRAIL-SyGMa"]["delta"]),
                                 ("interval low",
                                  -bud["paired_by_k"][k]["GRAIL-SyGMa"]["ci95"][1]),
                                 ("interval high",
                                  -bud["paired_by_k"][k]["GRAIL-SyGMa"]["ci95"][0]))],
               "results/budget_curves.json")

    n = "the two output sizes"
    b.sentence(n,
               r"GRAIL returns \$([\d.]+)\$ candidates against SyGMa's \$([\d.]+)\$",
               [("GRAIL's output size", bud["mean_emitted"]["GRAIL"]),
                ("SyGMa's", bud["mean_emitted"]["SyGMa"])],
               "results/budget_curves.json")

    n = "the head of the list"
    b.sentence(n,
               r"At \$k=1\$ both methods are cut, on \$([\d.]+)\\%\$ and \$([\d.]+)\\%\$ of "
               r"substrates, and SyGMa leads there too, by \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$",
               [("GRAIL cut at one", 100 * bud["truncated_fraction_by_k"]["GRAIL"][0]),
                ("SyGMa cut at one", 100 * bud["truncated_fraction_by_k"]["SyGMa"][0]),
                ("SyGMa's lead at one", -bud["paired_by_k"]["1"]["GRAIL-SyGMa"]["delta"]),
                ("interval low", -bud["paired_by_k"]["1"]["GRAIL-SyGMa"]["ci95"][1]),
                ("interval high", -bud["paired_by_k"]["1"]["GRAIL-SyGMa"]["ci95"][0])],
               "results/budget_curves.json")

    n = "the deployed cut against a fixed one"
    b.sentence(n,
               r"Over the \$([\d{},]+)\$ scored substrates, emitting as deployed scores "
               r"\$([\d.]+)\$ against \$([\d.]+)\$ at \$k=1\$: a paired difference of "
               r"\$([-\d.]+)\$, 95\\% interval \$\[([-\d.]+),([-\d.]+)\]\$",
               [("scored substrates", stop["n"]),
                ("F1 as emitted", stop["macro_f1"]["emitted"]),
                ("F1 at the best constant", stop["macro_f1"]["constant"]),
                ("the paired difference", stop["paired_vs_constant"]["emitted"]["delta"]),
                ("interval low", stop["paired_vs_constant"]["emitted"]["ci95"][0]),
                ("interval high", stop["paired_vs_constant"]["emitted"]["ci95"][1])],
               "results/stopping_rule.json")

    rel = [oracle["methods"][m]["headroom_relative"] for m in oracle["methods"]]
    n = "the oracle cut's headroom"
    b.sentence(n,
               r"knowing \$k\^\\star\$ is worth \$(\d+)\$ to \$(\d+)\\%\$ relative over the best "
               r"constant across the five methods",
               [("the smallest relative headroom", 100 * min(rel)),
                ("the largest", 100 * max(rel))],
               "results/cardinality_oracle.json")
    b.check("the oracle headroom is measured over five methods",
              len(oracle["methods"]), 5, "results/cardinality_oracle.json")

    n = "the predicted count out of fold"
    b.sentence(n,
               r"the part of the headroom that recovers is \$([-\d.]+)\\%\$ for GRAIL, "
               r"\$([-\d.]+)\\%\$ for SyGMa and \$\+(\d+)\\%\$ for MetaPredictor",
               [("GRAIL", 100 * cross["methods"]["GRAIL"]["gain_over_constant"]
                 ["predicted_cardinality"] / cross["methods"]["GRAIL"]["gain_over_constant"]
                 ["oracle"]),
                ("SyGMa", 100 * cross["methods"]["SyGMa"]["gain_over_constant"]
                 ["predicted_cardinality"] / cross["methods"]["SyGMa"]["gain_over_constant"]
                 ["oracle"]),
                ("MetaPredictor", 100 * cross["methods"]["MetaPredictor"]["gain_over_constant"]
                 ["predicted_cardinality"] / cross["methods"]["MetaPredictor"]
                 ["gain_over_constant"]["oracle"])],
               "results/cardinality_crossfit.json")

    n = "the cross-fitted regressor"
    b.sentence(n,
               r"recovers \$([\d.]+)\\%\$ of the oracle headroom for GRAIL, \$([\d.]+)\\%\$ for "
               r"MetaPredictor and \$([\d.]+)\\%\$ for SyGMa",
               [("GRAIL", 100 * cross["methods"]["GRAIL"]["oracle_share_reached"]),
                ("MetaPredictor",
                 100 * cross["methods"]["MetaPredictor"]["oracle_share_reached"]),
                ("SyGMa", 100 * cross["methods"]["SyGMa"]["oracle_share_reached"])],
               "results/cardinality_crossfit.json")

    # -------------------------------------------------------------- the similarity stratification
    n = "the control that decides the axis"
    b.sentence(n,
               r"SyGMa degrades at \$\+([\d.]+)\$ on F1 and \$\+([\d.]+)\$ on recall against "
               r"GRAIL's \$\+([\d.]+)\$ and \$\+([\d.]+)\$. Resampled on the same draws, so that "
               r"the comparison is paired, the difference is \$\+([\d.]+)\$ "
               r"\$\[([-\d.]+),\+([\d.]+)\]\$ on F1 and \$([-\d.]+)\$ \$\[([-\d.]+),\+([\d.]+)\]\$ "
               r"on recall",
               [("SyGMa's F1 slope", tstrat["slopes"]["f1"]["SyGMa"]["slope"]),
                ("SyGMa's recall slope", tstrat["slopes"]["recall"]["SyGMa"]["slope"]),
                ("GRAIL's F1 slope", tstrat["slopes"]["f1"]["GRAIL"]["slope"]),
                ("GRAIL's recall slope", tstrat["slopes"]["recall"]["GRAIL"]["slope"]),
                ("the paired difference on F1",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["diff"]),
                ("F1 interval low",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["ci95"][0]),
                ("F1 interval high",
                 tstrat["slope_differences"]["by_metric"]["f1"]["SyGMa_minus_GRAIL"]["ci95"][1]),
                ("the paired difference on recall",
                 tstrat["slope_differences"]["by_metric"]["recall"]["SyGMa_minus_GRAIL"]["diff"]),
                ("recall interval low", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["SyGMa_minus_GRAIL"]["ci95"][0]),
                ("recall interval high", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["SyGMa_minus_GRAIL"]["ci95"][1])],
               "results/transfer_stratified.json")

    n = "the width of the two intervals"
    b.sentence(n,
               r"the F1 interval still admits a gap of \$([\d.]+)\$ and the recall interval one of "
               r"\$([\d.]+)\$, against slopes of \$([\d.]+)\$ and \$([\d.]+)\$ themselves",
               [("the F1 half-width", tstrat["slope_differences"]["by_metric"]["f1"]
                 ["SyGMa_minus_GRAIL"]["half_width"]),
                ("the recall half-width", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["SyGMa_minus_GRAIL"]["half_width"]),
                ("GRAIL's F1 slope", tstrat["slopes"]["f1"]["GRAIL"]["slope"]),
                ("GRAIL's recall slope", tstrat["slopes"]["recall"]["GRAIL"]["slope"])],
               "results/transfer_stratified.json")

    n = "the coverage gradient"
    b.sentence(n,
               r"rule bank reaches at all rises from \$([\d.]+)\$ below similarity \$([\d.]+)\$ to "
               r"\$([\d.]+)\$ above \$([\d.]+)\$",
               [("coverage in the lowest stratum", tconf["strata"][0]["bank_coverage"]),
                ("the lowest stratum's edge", tconf["strata"][0]["hi"]),
                ("coverage in the highest stratum", tconf["strata"][-1]["bank_coverage"]),
                ("the highest stratum's edge", tconf["strata"][-1]["lo"])],
               "results/transfer_confound.json")

    n = "the mined rules behind that gradient"
    b.sentence(n,
               r"\$([\d{},]+)\$ of the \$\\nbank\$ rules are mined from the training split",
               [("mined rules", provct["config"]["n_mined"])],
               "results/ceiling_by_provenance__clean_test.json")

    # The two provenance coverages are printed in this section and again in app:decompsygma; a
    # reader meets them twice and they were measured once.
    n = "the substrates the two provenance coverages are measured on"
    b.sentence(n, r"on \$(\d+)\$ substrates against the curated fifth's",
               [("substrates", prov245["n"])], "results/ceiling_by_provenance.json")

    b.printings("the two provenance coverages on the subsample",
                r"\$([\d.]+)\$ (?:on \$245\$ substrates against the curated fifth's|of the "
                r"references against the curated remainder's) \$([\d.]+)\$",
                [("the mined subset", prov245["subsets"]["mined"]["coverage"]),
                 ("the curated subset", prov245["subsets"]["curated"]["coverage"])], 2,
                "results/ceiling_by_provenance.json")

    n = "the reference sets at the two ends"
    b.sentence(n,
               r"\$([\d.]+)\$ annotated metabolites above \$([\d.]+)\$ against \$([\d.]+)\$ "
               r"below \$([\d.]+)\$",
               [("references in the highest stratum", tconf["strata"][-1]["mean_n_refs"]),
                ("the highest stratum's edge", tconf["strata"][-1]["lo"]),
                ("references in the lowest", tconf["strata"][0]["mean_n_refs"]),
                ("the lowest stratum's edge", tconf["strata"][0]["hi"])],
               "results/transfer_confound.json")

    n = "the transformer's slopes read at face value"
    b.sentence(n,
               r"the transformer transfers best, at \$\+([\d.]+)\$ and \$\+([\d.]+)\$, roughly "
               r"half the other two",
               [("its F1 slope", tstrat["slopes"]["f1"]["MetaPredictor"]["slope"]),
                ("its recall slope", tstrat["slopes"]["recall"]["MetaPredictor"]["slope"])],
               "results/transfer_stratified.json")

    n = "the conclusion the control refuses"
    b.sentence(n,
               r"concluded from \$\+([\d.]+)\$ against \$\+([\d.]+)\$ that the transformer is the "
               r"more robust",
               [("GRAIL's F1 slope", tstrat["slopes"]["f1"]["GRAIL"]["slope"]),
                ("the transformer's", tstrat["slopes"]["f1"]["MetaPredictor"]["slope"])],
               "results/transfer_stratified.json")

    n = "the marginal-looking recall difference"
    b.sentence(n,
               r"the transformer's recall slope is shallower than GRAIL's by \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),([-\d.]+)\]\$",
               [("the paired difference", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["MetaPredictor_minus_GRAIL"]["diff"]),
                ("interval low", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["MetaPredictor_minus_GRAIL"]["ci95"][0]),
                ("interval high", tstrat["slope_differences"]["by_metric"]["recall"]
                 ["MetaPredictor_minus_GRAIL"]["ci95"][1])],
               "results/transfer_stratified.json")

    # ------------------------------------------------------------- partial identification under c
    grid = pub["grid"]
    n = "the propensity table"
    b.sentence(n,
               r"\$([\d.]+)\$ \(as reported\) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ "
               + r"\$([\d.]+)\$ & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\ " * 5
               + r"\$([\d.]+)\$ & ([\d.]+) & ([\d.]+) & ([\d.]+)",
               [x for c in (1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05)
                for x in [(f"c={c}, the completeness", c)]
                + [(f"c={c}, {m}", pub["macro_f1_by_c"][m][grid.index(c)])
                   for m in ("GRAIL", "MetaPredictor", "SyGMa")]],
               "results/pu_propensity_bounds.json")

    n = "where the reported ordering breaks"
    b.sentence(n,
               r"The reported ordering survives to \$c = ([\d.]+)\$, where SyGMa overtakes GRAIL; "
               r"MetaPredictor is never overtaken on a grid running down to \$c=([\d.]+)\$",
               [("the critical completeness", pub["crossings"]["GRAIL-SyGMa"]["critical_c"]),
                ("the bottom of the grid", min(grid))],
               "results/pu_propensity_bounds.json")
    b.check("MetaPredictor is never overtaken on that grid",
              str(int(pub["crossings"]["MetaPredictor-SyGMa"]["flips"])), 0,
              "results/pu_propensity_bounds.json")
    n = "the caption's crossing"
    b.sentence(n, r"The reported ordering holds until \$c=([\d.]+)\$, where SyGMa passes GRAIL",
               [("the critical completeness", pub["crossings"]["GRAIL-SyGMa"]["critical_c"])],
               "results/pu_propensity_bounds.json")

    n = "what any set metric can measure"
    b.sentence(n,
               r"At \$c=0\.5\$ that is a Jaccard ceiling of \$([\d.]+)\$ and an F1 ceiling of "
               r"\$([\d.]+)\$; at \$c=0\.3\$, \$([\d.]+)\$ and \$([\d.]+)\$",
               [("the Jaccard ceiling at one half",
                 gfn["annotation_ceiling"]["by_completeness"]["0.5"]["jaccard_max"]),
                ("the F1 ceiling at one half",
                 gfn["annotation_ceiling"]["by_completeness"]["0.5"]["f1_max"]),
                ("the Jaccard ceiling at three tenths",
                 gfn["annotation_ceiling"]["by_completeness"]["0.3"]["jaccard_max"]),
                ("the F1 ceiling at three tenths",
                 gfn["annotation_ceiling"]["by_completeness"]["0.3"]["f1_max"])],
               "results/gflownet_set_endpoint.json")

    # ---------------------------------------------------------------- the prescribed remedy
    n = "the propensity weights"
    b.sentence(n,
               r"the weights run from \$([\d.]+)\$ to \$([\d.]+)\$ with a mean of \$([\d.]+)\$, "
               r"and \$([\d.]+)\\%\$ are up-weighted: most of them the \$([\d{},]+)\$ that never "
               r"carry a positive at all, which take the maximum weight, against \$([\d.]+)\$ for "
               r"a rule with five positives",
               [("the smallest weight", pw["weight"]["min"]),
                ("the largest", pw["weight"]["max"]),
                ("the mean", pw["weight"]["mean"]),
                ("the share up-weighted", 100 * pw["weight"]["share_up_weighted"]),
                ("rules that never carry a positive", dens["per_rule"]["never_positive"]),
                ("the weight at five positives", pw["weight_by_positive_count"]["5"])],
               "results/propensity_weights.json + results/label_density.json")
    n = "the propensity model's constants"
    b.sentence(n, r"by the model of \\citet\{Jain_2016\} with \$A=([\d.]+)\$, \$B=([\d.]+)\$",
               [("A", pw["config"]["propensity_a"]), ("B", pw["config"]["propensity_b"])],
               "results/propensity_weights.json")
    b.check("a rule with no positives takes the maximum weight",
              pw["weight_by_positive_count"]["0"], pw["weight"]["max"],
              "results/propensity_weights.json")

    n = "the propensity probe"
    b.sentence(n,
               r"the learned-only selector reaches recall@15 of \$([\d.]+)\$ against \$([\d.]+)\$ "
               r"under constant weighting. Its deficit to the frequency prior widens from "
               r"\$([-\d.]+)\$ \$\[([-\d.]+),([-\d.]+)\]\$ to \$([-\d.]+)\$ "
               r"\$\[([-\d.]+),([-\d.]+)\]\$",
               [("propensity-weighted", pvlp["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("constant-weighted", pvl["modes"]["learned_only"]["gen_only"]["recall@15"]),
                ("the deficit before",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["delta"]),
                ("interval low before",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][0]),
                ("interval high before",
                 pvl["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][1]),
                ("the deficit after",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["delta"]),
                ("interval low after",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][0]),
                ("interval high after",
                 pvlp["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]["ci95"][1])],
               "results/prior_vs_learned.json + results/prior_vs_learned_propensity.json")

    n = "the prior arm in both runs"
    b.sentence(n,
               r"The prior-only arm returns \$([\d.]+)\$ in both runs",
               [("the prior arm", pvlp["modes"]["prior_only"]["gen_only"]["recall@15"])],
               "results/prior_vs_learned_propensity.json")
    b.check("the prior arm is identical across the two runs",
              pvlp["modes"]["prior_only"]["gen_only"]["recall@15"],
              pvl["modes"]["prior_only"]["gen_only"]["recall@15"],
              "results/prior_vs_learned.json + results/prior_vs_learned_propensity.json")

    n = "the unpaired difference"
    b.sentence(n, r"no paired interval is quoted on the \$([-\d.]+)\$ difference itself",
               [("the difference between the two learned arms",
                 pvlp["modes"]["learned_only"]["gen_only"]["recall@15"]
                 - pvl["modes"]["learned_only"]["gen_only"]["recall@15"])],
               "results/prior_vs_learned.json + results/prior_vs_learned_propensity.json")

    # ------------------------------------------------------- figures restated across the appendix
    # These are the same measurements met again in a later paragraph, which is where a figure goes
    # stale: the passage that reports it is updated and the passage that leans on it is not. Each
    # group binds every restatement and asserts how many restatements there are.
    b.printings("the shared rule count, restated",
                r"(?:fixed at the|at a time over the|over the same|about these|reports for the "
                r"shared|weakened\. The) \$(\d+)\$ (?:both systems|shared rules|rules|are "
                r"curated|are the expanded)",
                [("the rules both systems have", overlap["containment"]["in_grail_bank"])], 6,
                "results/bank_overlap_sygma__clean_test.json")

    b.printings("the no-selector arm's substrates, restated",
                r"(?:Scored against SyGMa on the same|macro on these same) \$(\d+)\$ substrates",
                [("substrates", nosel["n"])], 2, "results/bank_without_selection.json")

    b.printings("the comparator's selection factor, restated",
                r"selection (?:being|factor is) \$(\d+)\$ by construction",
                [("selection retention", sygd["factors"]["selection_retention"]["point"])], 2,
                "results/decompose_sygma.json")

    n = "the templates that want the expansion, restated"
    b.sentence(n, r"On the second and duller, \$(\d+)\$ hand-written templates simply want",
               [("templates dispatched to the expansion",
                 dispct["banks"]["grail_full"]["dispatched_to_expanded"])],
               "results/hydrogen_dispatch__clean_test.json")

    n = "the bank the selector chooses from, restated"
    b.sentence(n, r"a top-\$k\$ over \$([\d{},]+)\$ rules",
               [("rules in the bank", census["banks"]["this work, full bank"]["templates"])],
               "results/convention_census.json")

    n = "BioTransformer's reach, restated"
    b.sentence(n, r"does not transfer here, \$([\d.]+)\$ is not established as a lower bound",
               [("its template reach", btd["biotransformer"]["template_reach"])],
               "results/decompose_biotransformer.json")

    n = "the comparator's reach, restated"
    b.sentence(n, r"Its reach is the lower, \$([\d.]+)\$ against \$\\ceiling\$",
               [("SyGMa's coverage factor", sygd["factors"]["coverage_bank"]["point"])],
               "results/decompose_sygma.json")

    n = "the counterfactual's two numbers, restated"
    b.sentence(n,
               r"not an operating point: \$(\d+)\$ candidates per substrate drawn from "
               r"\$([\d.]+)\$",
               [("the matched budget", max(nosel["config"]["budgets"])),
                ("the raw pool it is drawn from", nosel["arms"]["7581"]["mean_pool"])],
               "results/bank_without_selection.json")

    n = "the broad pool the abstention frontier is measured on"
    b.sentence(n, r"not the deployed one \(all \$(\d+)\$ applicable rules, as in the last row",
               [("the widest breadth in the sweep", max(sel["config"]["top_ks"]))],
               "results/selection_ablation.json")

    # This figure is printed a third time in app/casestudy.tex, which is another module's file;
    # the two printings here are bound with the context that separates them from it.
    n = "the field's budget where the ordering settles"
    b.sentence(n, r"where the field's \$(\d+)\$ sits, it is MetaPredictor",
               [("the budget the decomposition truncates at", fact["k"])],
               "results/recall_factorization.json")
    n = "the field's budget in the partial-binding sentence"
    b.sentence(n, r"binding is partial well below \$(\d+)\$ too",
               [("the budget the decomposition truncates at", fact["k"])],
               "results/recall_factorization.json")

    n = "the last budget the comparator's shorter list wins at"
    b.sentence(n, r"at every budget up to \$(\d+)\$ that shorter list beats GRAIL's",
               [("the last k SyGMa's F1 exceeds GRAIL's",
                 max((k for k in range(1, int(bud["k_max"]) + 1)
                      if bud["macro_f1_by_k"]["SyGMa"][k - 1]
                      > bud["macro_f1_by_k"]["GRAIL"][k - 1]), default=0))],
               "results/budget_curves.json")


def register_som_prior(ctx) -> None:
    """The site-of-metabolism row and its paragraph, which quoted a run that left only a log.

    The measurement was written to ``results/reeval_som.log`` and the JSON it says it wrote was
    never kept, so three figures --- the baseline, the best weighting and their difference --- were
    printed by nothing a checker could read. The table in that log is transcribed into
    ``results/som_prior_reeval.json`` with its own provenance, and all three are held to it,
    including that the gain is the difference of the other two and that the weighting quoted as
    best is the best of the five that were tried.
    """
    import re as _re

    a = ctx.art("som_prior_reeval.json")
    if a is None:
        return
    base, best, gain = (a["baseline_recall@15"], a["best_weighting_recall@15"], a["gain@15"])
    m = _re.search(r"lifts\s*recall@15 from ([\d.]+) to ([\d.]+) on (\d+) test substrates", ctx.flat)
    ctx.checks.append((bool(m), "props, the site-of-metabolism sentence is present", "present",
                       "matched" if m else "not matched", "results/som_prior_reeval.json"))
    if m:
        ctx.check("props, the unweighted recall@15", m.group(1), base,
                  "results/som_prior_reeval.json")
        ctx.check("props, the best-weighted recall@15", m.group(2), best,
                  "results/som_prior_reeval.json")
        ctx.check("props, the substrates it was measured on", m.group(3),
                  a["config"]["n_substrates"], "results/som_prior_reeval.json")
    # the gain is printed twice, in the table row and in the paragraph, and both are the difference
    said = _re.findall(r"\$\+([\d.]+)\$ (?:at the best of five weightings|is an upper bound)",
                       ctx.flat)
    ctx.checks.append((len(said) == 2, "props, the site-of-metabolism gain is printed twice",
                       "2", str(len(said)), "results/som_prior_reeval.json"))
    for i, s in enumerate(said, 1):
        ctx.check(f"props, the site-of-metabolism gain, printing {i}", s, gain,
                  "results/som_prior_reeval.json")
    ctx.checks.append((abs((best - base) - gain) < 5e-5,
                       "props, that gain is the difference of the two recalls",
                       f"{best - base:.4f}", f"{gain:.4f}", "results/som_prior_reeval.json"))
    # "the best of five": the artifact has to hold five weightings and the quoted one has to win
    ranks = {k: v for k, v in a["recall_at_k"]["15"].items() if k.startswith("rk b")}
    ctx.checks.append((len(ranks) == len(a["config"]["betas"]) == 5
                       and max(ranks.values()) == best,
                       "props, five weightings were tried and the quoted one is the best",
                       "5, and the best", f"{len(ranks)}, max {max(ranks.values())}",
                       "results/som_prior_reeval.json"))


def register_decode_arithmetic(ctx) -> None:
    """The decode configuration's own arithmetic, which is what a reader checks and no leaf holds.

    "Decodes in two stages, $8\\times8$ then $2\\times8$, giving $16$ ranked candidates" is not a
    measurement and no artifact records a beam size; what it is, is arithmetic the sentence prints
    both sides of, so the product is checked against its own factors. The distinct counts the same
    passage compares are measurements and are bound elsewhere.
    """
    import re as _re

    m = _re.search(r"decodes in two stages, \$(\d+)\\times(\d+)\$ then \$(\d+)\\times(\d+)\$, "
                   r"giving \$(\d+)\$ ranked candidates\.\s*Re-running it at \$(\d+)\\times(\d+)\$ "
                   r"then \$(\d+)\\times(\d+)\$.*?yields \$(\d+)\$", ctx.flat)
    ctx.checks.append((bool(m), "props, the decode configuration is stated", "present",
                       "matched" if m else "not matched", "scripts/gates/app_props.py"))
    if not m:
        return
    g = [int(x) for x in m.groups()]
    ctx.checks.append((g[2] * g[3] == g[4], "props, the deployed decode's candidate count",
                       f"{g[2]}x{g[3]}", str(g[4]), "the sentence's own factors"))
    ctx.checks.append((g[7] * g[8] == g[9], "props, the widened decode's candidate count",
                       f"{g[7]}x{g[8]}", str(g[9]), "the sentence's own factors"))
    ctx.checks.append((g[5] == 2 * g[0] and g[6] == 2 * g[1] and g[7] == 2 * g[2]
                       and g[8] == 2 * g[3],
                       "props, the widened run doubles every beam and nothing else",
                       "each doubled", f"{g[:4]} -> {g[5:9]}", "the sentence's own factors"))


def register_training_size(ctx) -> None:
    """The training-size ablation, whose two runs wrote no results/ file of their own.

    "Recall moves from 0.330 to 0.334 between 2,418 and 4,787 training substrates" is four figures
    from two runs; the larger arm's substrate count was bound elsewhere and the other three were
    not. Both runs' training reports and evaluations are in ``results/training_reports.json``, so
    all four are held, along with the claim the sentence rests on: that the move is smaller than
    the seed spread the same configuration shows.
    """
    import re as _re

    tr = ctx.art("training_reports.json")
    ms = ctx.art("multiseed_headline.json")
    if tr is None:
        return
    half, full = tr["runs"].get("ablation_half"), tr["runs"].get("deployed")
    if not half or not full:
        return
    m = _re.search(r"Recall moves from ([\d.]+) to ([\d.]+) between ([\d,{}]+) and ([\d,{}]+) "
                   r"training substrates", ctx.flat)
    ctx.checks.append((bool(m), "props, the training-size sentence is present", "present",
                       "matched" if m else "not matched", "results/training_reports.json"))
    if not m:
        return
    ctx.check("props, the half-size arm's recall", m.group(1),
              half["evaluation"]["ensemble_test_recall@15"], "results/training_reports.json")
    ctx.check("props, the full-size arm's recall", m.group(2),
              full["evaluation"]["ensemble_test_recall@15"], "results/training_reports.json")
    ctx.check("props, the half-size arm's substrates", m.group(3).replace("{,}", ""),
              half["train_substrates"], "results/training_reports.json")
    ctx.check("props, the full-size arm's substrates", m.group(4).replace("{,}", ""),
              full["train_substrates"], "results/training_reports.json")
    ctx.checks.append((2 * half["train_substrates"] >= full["train_substrates"]
                       > 1.9 * half["train_substrates"],
                       "props, the larger arm is about twice the smaller, as 'doubling' says",
                       "about 2x", f"{full['train_substrates'] / half['train_substrates']:.2f}x",
                       "results/training_reports.json"))
    if ms:
        spread = ms.get("spread") or ms.get("seed_spread")
        move = abs(full["evaluation"]["ensemble_test_recall@15"]
                   - half["evaluation"]["ensemble_test_recall@15"])
        if isinstance(spread, (int, float)):
            ctx.checks.append((move < spread,
                               "props, the move is smaller than the seed spread it is compared to",
                               f"{spread}", f"{move:.4f}", "results/multiseed_headline.json"))
