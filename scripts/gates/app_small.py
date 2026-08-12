"""Checks for the four short appendices: budget, external, design, arch.

These four print about eighteen math-mode numerals that no check reached, and rather more figures
in running prose that the surrounding argument reasons from. The order below follows how
load-bearing a figure is rather than where it sits on the page: the matched-budget table and the
frontier the budget appendix argues from first, then the external ceiling and its splits, then the
design appendix's two measured margins, then the architecture's filter comparison and the census of
the rule bank.

Two conventions this module keeps.

*Every figure names one leaf.* ``cross_method_decomposition.json`` records GRAIL's ``recall@15`` and
its ``pool_coverage_recall_inf`` as the same number, and the appendix prints ``0.318`` for both; the
sentence about what GRAIL *reports* is bound to ``recall@15`` and the sentence about the pool to
``pool_coverage_recall_inf``, with a separate check that the artifact really does make them equal --
which is the claim the prose makes when it says the cut is not binding.

*The manuscript-side regex runs against ``ctx.flat``.* An anchored pattern over the whole flattened
manuscript is what ``check_coverage.py`` can see, so a figure matched that way stops being unread.
Counting how many times a figure is printed is done against ``ctx.tex[...]`` instead, since that
question is about one file and not about the manuscript.
"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

HERE = "scripts/gates/app_small.py"

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "thirteen": 13, "fifteen": 15,
          "twenty-four": 24}


def _need(ctx, name, pat):
    """Find the sentence a set of checks hangs off, and fail loudly when it has been rewritten.

    A regex that matches nothing hands ``None`` to every check below it, and ``None`` never
    compares equal -- but it reports as a value mismatch rather than as a missing sentence, which
    is the harder failure to read. So the anchor is its own check.
    """
    m = re.search(pat, ctx.flat)
    if m is None:
        ctx.checks.append((False, name, "the passage is present", "no match", HERE))
    return m


def _cmp(ctx, name, printed, computed, note):
    """Compare a figure read from the manuscript against the artifact, at the printed precision.

    Every value this module reads out of the manuscript arrives as a string, and for those the
    precision it is printed at is the whole yardstick: the manuscript is right when it prints the
    artifact rounded, and wrong otherwise. Rounding is half-up, the way a typesetter rounds, so a
    value sitting exactly on the boundary has one right answer and its neighbour is a defect --
    ``8.65`` mean output prints as ``8.7`` and ``8.6`` is a stale figure, not a rounding taste.
    Accepting both would make the check unable to fail on the likeliest perturbation there is, an
    adjacent digit.

    Anything that is not a manuscript string is an artifact-against-artifact comparison, and is
    handed to ``ctx.check`` so it keeps the tolerance that kind of check is written with.
    """
    if not isinstance(printed, str):
        ctx.check(name, printed, computed, note)
        return
    ok = False
    if computed is not None:
        p = printed.strip().lstrip("+").replace("$", "").replace("{,}", "").replace(",", "")
        places = len(p.split(".")[1]) if "." in p else 0
        q = Decimal(1).scaleb(-places)
        try:
            ok = Decimal(str(computed)).quantize(q, rounding=ROUND_HALF_UP) == Decimal(p)
        except Exception:
            ok = False
    ctx.checks.append((ok, name, printed, computed, note))


def _src(ctx, name, relpath, pat):
    """One capture out of a source file, so a constant the appendix describes cannot drift silently.

    Some of what the architecture appendix states is a property of the code that produced the bank
    and the checkpoints rather than of any result file; the file that defines the constant is the
    artifact for those.
    """
    p = ctx.root / relpath
    if not p.exists():
        ctx.checks.append((False, name, relpath, "file missing", HERE))
        return None
    m = re.search(pat, p.read_text(), re.M)
    if m is None:
        ctx.checks.append((False, name, f"{relpath} defines it", "pattern not found", HERE))
        return None
    return m


def register(ctx) -> None:
    _budget(ctx)
    _external(ctx)
    _design(ctx)
    _arch(ctx)


# --------------------------------------------------------------------------------------------
# paper/app/budget.tex
# --------------------------------------------------------------------------------------------
def _budget(ctx) -> None:
    bml = ctx.art("budget_matched_leaderboard.json")
    bmf = ctx.art("budget_matched_frontier.json")
    cmd = ctx.art("cross_method_decomposition.json")
    bws = ctx.art("bank_without_selection.json")
    sa = ctx.art("selection_ablation.json")
    if not (bml and bmf and cmd and bws and sa):
        ctx.checks.append((False, "budget, its five artifacts are present", "5 artifacts",
                          "one or more missing", HERE))
        return
    by = bml["by_method"]
    grail, sygma = cmd["methods"][0], cmd["methods"][1]

    # ---- the printed table: every row, every column, in order ----------------------------------
    # One pattern for the whole body binds the twenty-five cells together, so a row that goes stale
    # on its own, a row that is dropped, and a reordering of the five methods all fail here. A
    # per-row pattern would survive all three.
    order = ["GRAIL", "SyGMa", "BioTransformer", "MetaPredictor", "MetaTrans"]
    cell = r" & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+) \\\\ "
    tail = r" & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+) & (\d+\.\d+)"
    body = r"\\midrule " + cell.join(order) + tail + r" \\\\ \\bottomrule"
    m = _need(ctx, "budget, the matched-cut-off table body", body)
    if m:
        cols = ["recall@5", "recall@10", "recall@15", "precision", "mean_output"]
        for i, meth in enumerate(order):
            for j, col in enumerate(cols):
                _cmp(ctx, f"budget, table {meth} {col}", m.group(i * 5 + j + 1), by[meth][col],
                          "results/budget_matched_leaderboard.json")

    # The three recall columns are one measurement at growing k, so each row must be non-decreasing
    # across them. A row copied from an older run typically breaks this before it breaks anything a
    # reader would notice.
    bad = [meth for meth in order
           if not by[meth]["recall@5"] <= by[meth]["recall@10"] <= by[meth]["recall@15"]]
    ctx.checks.append((not bad, "budget, recall is non-decreasing in k on every row",
                      "all five rows", ", ".join(bad) or "all five rows",
                      "results/budget_matched_leaderboard.json"))

    # ---- the paragraph the table is introduced by ----------------------------------------------
    m = _need(ctx, "budget, the deployed-configuration sentence",
              r"SyGMa reaches (\d+\.\d+) by emitting about (\d+) predictions per substrate; GRAIL "
              r"reaches (\d+\.\d+) at about (\d+\.\d+)\.")
    if m:
        _cmp(ctx, "budget, prose SyGMa recall@15", m.group(1), by["SyGMa"]["recall@15"],
                  "results/budget_matched_leaderboard.json")
        _cmp(ctx, "budget, prose SyGMa output size", m.group(2),
                  round(by["SyGMa"]["mean_output"]), "rounded from mean_output 74.15")
        _cmp(ctx, "budget, prose GRAIL recall@15", m.group(3), by["GRAIL"]["recall@15"],
                  "results/budget_matched_leaderboard.json")
        _cmp(ctx, "budget, prose GRAIL output size", m.group(4), by["GRAIL"]["mean_output"],
                  "results/budget_matched_leaderboard.json")

    m = _need(ctx, "budget, the precision reordering sentence",
              r"SyGMa's precision is (\d+\.\d+), roughly four times below the low-output methods: "
              r"GRAIL at (\d+\.\d+), BioTransformer at (\d+\.\d+), MetaTrans at (\d+\.\d+) and "
              r"MetaPredictor at (\d+\.\d+)")
    if m:
        for i, meth in enumerate(["SyGMa", "GRAIL", "BioTransformer", "MetaTrans",
                                  "MetaPredictor"]):
            _cmp(ctx, f"budget, prose precision {meth}", m.group(i + 1), by[meth]["precision"],
                      "results/budget_matched_leaderboard.json")

    m = _need(ctx, "budget, the small-budget sentence",
              r"at \$k=5\$, SyGMa's (\d+\.\d+) is close to MetaPredictor's (\d+\.\d+), with "
              r"GRAIL at (\d+\.\d+)")
    if m:
        for i, meth in enumerate(["SyGMa", "MetaPredictor", "GRAIL"]):
            _cmp(ctx, f"budget, prose recall@5 {meth}", m.group(i + 1), by[meth]["recall@5"],
                      "results/budget_matched_leaderboard.json")

    # ---- the caption, which states the macro/micro pair the two artifacts disagree on ----------
    m = _need(ctx, "budget, the caption's macro-against-micro pair",
              r"which is why GRAIL reads \$(\d+\.\d+)\$ in this table and \$(\d+\.\d+)\$ there\.\}")
    if m:
        _cmp(ctx, "budget, caption macro recall@15", m.group(1), by["GRAIL"]["recall@15"],
                  "results/budget_matched_leaderboard.json")
        _cmp(ctx, "budget, caption micro recall@15", m.group(2), grail["recall@15"],
                  "results/cross_method_decomposition.json methods[GRAIL].recall@15")

    # ---- the frontier -------------------------------------------------------------------------
    m = _need(ctx, "budget, the frontier's cut-off grid",
              r"measured at (\w+) cut-offs between (\d+) and (\d+)")
    if m:
        _cmp(ctx, "budget, number of frontier cut-offs", _WORDS.get(m.group(1)), len(bmf["ks"]),
                  "results/budget_matched_frontier.json ks")
        _cmp(ctx, "budget, smallest frontier cut-off", m.group(2), min(bmf["ks"]),
                  "results/budget_matched_frontier.json ks")
        _cmp(ctx, "budget, largest frontier cut-off", m.group(3), max(bmf["ks"]),
                  "results/budget_matched_frontier.json ks")

    m = _need(ctx, "budget, the frontier's three quoted cut-offs",
              r"with no crossover: (\d+\.\d+) against (\d+\.\d+) at \$k=1\$, (\d+\.\d+) against "
              r"(\d+\.\d+) at \$k=15\$, and (\d+\.\d+) against (\d+\.\d+) at \$k=64\$")
    if m:
        # SyGMa at k=15 is 0.4925, exactly on the third-decimal boundary, and 0.492 is a stale
        # figure rather than a rounding taste -- which is why _cmp rounds one way and not both.
        _cmp(ctx, "budget, frontier SyGMa at k=1", m.group(1), bmf["sygma_recall_at_k"]["1"],
                  "results/budget_matched_frontier.json")
        _cmp(ctx, "budget, frontier GRAIL at k=1", m.group(2), bmf["grail_recall_at_k"]["1"],
                  "results/budget_matched_frontier.json")
        _cmp(ctx, "budget, frontier SyGMa at k=15", m.group(3), bmf["sygma_recall_at_k"]["15"],
                  "results/budget_matched_frontier.json")
        _cmp(ctx, "budget, frontier GRAIL at k=15", m.group(4), bmf["grail_recall_at_k"]["15"],
                  "results/budget_matched_frontier.json")
        _cmp(ctx, "budget, frontier SyGMa at k=64", m.group(5), bmf["sygma_recall_at_k"]["64"],
                  "results/budget_matched_frontier.json")
        _cmp(ctx, "budget, frontier GRAIL at k=64", m.group(6), bmf["grail_recall_at_k"]["64"],
                  "results/budget_matched_frontier.json")
    # "SyGMa leads GRAIL at every one" is a claim about all seven, not about the three quoted.
    leads = all(bmf["sygma_recall_at_k"][str(k)] > bmf["grail_recall_at_k"][str(k)]
                for k in bmf["ks"])
    ctx.checks.append((leads, "budget, SyGMa leads at every frontier cut-off", "no crossover",
                      "leads everywhere" if leads else "GRAIL leads somewhere",
                      "results/budget_matched_frontier.json"))

    # ---- the decomposition --------------------------------------------------------------------
    m = _need(ctx, "budget, the micro-against-macro sentence",
              r"macro, so GRAIL reads \$(\d+\.\d+)\$ here against \$(\d+\.\d+)\$ there on the same "
              r"substrates")
    if m:
        _cmp(ctx, "budget, decomposition micro recall@15", m.group(1), grail["recall@15"],
                  "results/cross_method_decomposition.json")
        _cmp(ctx, "budget, decomposition against the table's macro", m.group(2),
                  by["GRAIL"]["recall@15"], "results/budget_matched_leaderboard.json")

    m = _need(ctx, "budget, the deployed-against-frontier sentence",
              r"which reports \$(\d+\.\d+)\$ as deployed against \$(\d+\.\d+)\$ at \$k=15\$ on the "
              r"frontier")
    if m:
        _cmp(ctx, "budget, deployed recall@15", m.group(1), grail["recall@15"],
                  "results/cross_method_decomposition.json")
        _cmp(ctx, "budget, frontier recall at the same cut", m.group(2),
                  bmf["grail_recall_at_k"]["15"], "results/budget_matched_frontier.json")

    m = _need(ctx, "budget, SyGMa's output size on the shared subset",
              r"at about (\d+) outputs per substrate on this subset")
    if m:
        _cmp(ctx, "budget, SyGMa mean output on the shared subset", m.group(1),
                  round(sygma["mean_output"]), "rounded from mean_output 63.97")

    m = _need(ctx, "budget, SyGMa loses almost nothing at the cut",
              r"what it reports, (\d+\.\d+) against a recall@15 of (\d+\.\d+)")
    if m:
        _cmp(ctx, "budget, SyGMa pool coverage", m.group(1), sygma["pool_coverage_recall_inf"],
                  "results/cross_method_decomposition.json")
        _cmp(ctx, "budget, SyGMa recall@15 in the decomposition", m.group(2),
                  sygma["recall@15"], "results/cross_method_decomposition.json")

    m = _need(ctx, "budget, GRAIL's bank ceiling on the shared subset",
              r"its ceiling on this subset is (\d+\.\d+), against a full-test ceiling of "
              r"\$\\ceiling\$ and SyGMa's realised (\d+\.\d+)")
    if m:
        _cmp(ctx, "budget, GRAIL bank ceiling on the shared subset", m.group(1),
                  cmd["grail_bank_ceiling_on_shared"], "results/cross_method_decomposition.json")
        _cmp(ctx, "budget, SyGMa's realised coverage", m.group(2),
                  sygma["pool_coverage_recall_inf"], "results/cross_method_decomposition.json")

    m = _need(ctx, "budget, what selection and ranking retain",
              r"GRAIL nevertheless reports (\d+\.\d+), because selection and ranking together "
              r"retain (\d+\.\d+) of what the bank could reach")
    if m:
        _cmp(ctx, "budget, GRAIL's realised recall", m.group(1), grail["recall@15"],
                  "results/cross_method_decomposition.json")
        _cmp(ctx, "budget, retention against the bank ceiling", m.group(2),
                  cmd["grail_selection_retention_vs_bank_ceiling"],
                  "results/cross_method_decomposition.json")

    # "its recall@15 equals that pool's coverage exactly" is the claim that lets the same 0.318
    # stand for two leaves; if the artifact ever separates them the sentence is wrong.
    same = grail["recall@15"] == grail["pool_coverage_recall_inf"]
    ctx.checks.append((same and grail["selection_retention"] == 1.0,
                      "budget, GRAIL's cut is not binding on the shared subset",
                      "recall@15 = pool coverage",
                      f"{grail['recall@15']} vs {grail['pool_coverage_recall_inf']}",
                      "results/cross_method_decomposition.json"))

    # ---- the counterfactual that rules out the stronger reading -------------------------------
    m = _need(ctx, "budget, the no-selector counterfactual",
              r"at the same budget gives \$(\d+\.\d+)\$ against the deployed selector's "
              r"\$(\d+\.\d+)\$ at \$k=15\$ on \$(\d+)\$ substrates")
    if m:
        _cmp(ctx, "budget, the bank without a selector at k=15", m.group(1),
                  bws["arms"]["7581"]["by_budget"]["15"]["grail"],
                  "results/bank_without_selection.json arms[7581].by_budget[15].grail")
        _cmp(ctx, "budget, the deployed selector at k=15", m.group(2),
                  sa["by_top_k"]["30"]["recall@15"],
                  "results/selection_ablation.json by_top_k[30].recall@15")
        _cmp(ctx, "budget, the counterfactual's substrates", m.group(3), bws["n"],
                  "results/bank_without_selection.json n")
        _cmp(ctx, "budget, both arms ran on the same substrates", bws["n"], sa["n"],
                  "bank_without_selection.n against selection_ablation.n")

    # The two arms have to be the same measurement for the difference to mean anything.
    ctx.checks.append((bws["match"] == sa["match"] == cmd["match"],
                      "budget, one matching criterion across the three artifacts",
                      "inchikey_tautomer", f"{bws['match']}/{sa['match']}/{cmd['match']}",
                      "results/{bank_without_selection,selection_ablation,"
                      "cross_method_decomposition}.json"))

    # ---- the shared-subset macro, which four of these sentences are conditioned on -------------
    m = _need(ctx, "budget, the \\nshared macro", r"\\newcommand\{\\nshared\}\{(\d+)\}")
    if m:
        _cmp(ctx, "budget, \\nshared", m.group(1), cmd["n_shared"],
                  "results/cross_method_decomposition.json n_shared")
        _cmp(ctx, "budget, the frontier ran on the same subset", bmf["n"], cmd["n_shared"],
                  "budget_matched_frontier.n against cross_method_decomposition.n_shared")

    # ---- how often each repeated figure is printed in this appendix ---------------------------
    tex = ctx.tex.get("budget.tex", "")
    for pat, n, what in ((r"0\.318", 4, "GRAIL's micro recall@15"),
                         (r"0\.365", 4, "GRAIL's macro recall@15"),
                         (r"0\.350", 2, "GRAIL on the frontier at k=15"),
                         (r"0\.520", 3, "SyGMa's pool coverage")):
        found = len(re.findall(pat, tex))
        ctx.checks.append((found == n, f"budget, printings of {what}", f"{n} printings",
                          f"{found} found", "paper/app/budget.tex"))


# --------------------------------------------------------------------------------------------
# paper/app/external.tex
# --------------------------------------------------------------------------------------------
def _external(ctx) -> None:
    cev = ctx.art("ceiling_external_validity.json")
    spl = ctx.art("external_ceiling_split.json")
    rf = ctx.art("recall_factorization.json")
    if not (cev and spl and rf):
        ctx.checks.append((False, "external, its three artifacts are present", "3 artifacts",
                          "one or more missing", HERE))
        return
    ext = cev["external_ceiling_uncapped"]
    reg = cev["regression"]

    m = _need(ctx, "external, the internal ceiling's interval",
              r"cluster-bootstrap 95\\% confidence interval of \$\[(\d+\.\d+),(\d+\.\d+)\]\$ over "
              r"([\d{},]+) resamples of substrates")
    if m:
        _cmp(ctx, "external, internal ceiling interval, lower", m.group(1),
                  rf["factors"]["coverage_bank"]["lo"],
                  "results/recall_factorization.json factors.coverage_bank.lo")
        _cmp(ctx, "external, internal ceiling interval, upper", m.group(2),
                  rf["factors"]["coverage_bank"]["hi"],
                  "results/recall_factorization.json factors.coverage_bank.hi")
        _cmp(ctx, "external, resamples behind that interval", m.group(3).replace("{,}", ""),
                  rf["provenance"]["n_boot"],
                  "results/recall_factorization.json provenance.n_boot")

    m = _need(ctx, "external, the external population",
              r"the (\d+) GLORYx parent substrates, one uncapped depth-1 pass")
    if m:
        _cmp(ctx, "external, GLORYx parents", m.group(1), ext["n"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.n")

    m = _need(ctx, "external, the uncapped external ceiling",
              r"the external ceiling is (\d+\.\d+), with a 95\\% interval of \$\[(\d+\.\d+),"
              r"(\d+\.\d+)\]\$")
    if m:
        _cmp(ctx, "external, uncapped ceiling", m.group(1), ext["point"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.point")
        _cmp(ctx, "external, uncapped ceiling interval, lower", m.group(2), ext["lo"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.lo")
        _cmp(ctx, "external, uncapped ceiling interval, upper", m.group(3), ext["hi"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.hi")

    m = _need(ctx, "external, the capped figure that is not this one",
              r"A figure of (\d+\.\d+) appears elsewhere for the external ceiling")
    if m:
        _cmp(ctx, "external, the pool-capped figure", m.group(1),
                  cev["external_ceiling_capped_committed"],
                  "results/ceiling_external_validity.json external_ceiling_capped_committed")

    # ---- the seen/unseen split, all eight figures of it in one sentence ------------------------
    m = _need(ctx, "external, the seen-against-unseen split",
              r"gives \$(\d+\.\d+)\$ \$\[(\d+\.\d+),(\d+\.\d+)\]\$ on the twenty-four seen parents "
              r"against \$(\d+\.\d+)\$ \$\[(\d+\.\d+),(\d+\.\d+)\]\$ on the thirteen unseen, a "
              r"difference of \$\+(\d+\.\d+)\$ \$\[-(\d+\.\d+),\+(\d+\.\d+)\]\$")
    if m:
        seen, unseen = spl["seen_in_training"], spl["unseen"]
        _cmp(ctx, "external, coverage on seen parents", m.group(1), seen["coverage"],
                  "results/external_ceiling_split.json seen_in_training.coverage")
        _cmp(ctx, "external, seen interval, lower", m.group(2), seen["ci95"][0],
                  "results/external_ceiling_split.json")
        _cmp(ctx, "external, seen interval, upper", m.group(3), seen["ci95"][1],
                  "results/external_ceiling_split.json")
        _cmp(ctx, "external, coverage on unseen parents", m.group(4), unseen["coverage"],
                  "results/external_ceiling_split.json unseen.coverage")
        _cmp(ctx, "external, unseen interval, lower", m.group(5), unseen["ci95"][0],
                  "results/external_ceiling_split.json")
        _cmp(ctx, "external, unseen interval, upper", m.group(6), unseen["ci95"][1],
                  "results/external_ceiling_split.json")
        _cmp(ctx, "external, the seen-minus-unseen difference", m.group(7),
                  spl["seen_minus_unseen"]["delta"],
                  "results/external_ceiling_split.json seen_minus_unseen.delta")
        _cmp(ctx, "external, difference interval, lower", "-" + m.group(8),
                  spl["seen_minus_unseen"]["ci95"][0], "results/external_ceiling_split.json")
        _cmp(ctx, "external, difference interval, upper", m.group(9),
                  spl["seen_minus_unseen"]["ci95"][1], "results/external_ceiling_split.json")
    # The split is a partition of the population the ceiling is reported on, and the difference is
    # its two arms subtracted; both went stale independently in this paper before.
    _cmp(ctx, "external, the two arms partition the references",
              spl["seen_in_training"]["references"] + spl["unseen"]["references"],
              spl["all"]["references"], "results/external_ceiling_split.json")
    _cmp(ctx, "external, the two arms partition the recoveries",
              spl["seen_in_training"]["recovered"] + spl["unseen"]["recovered"],
              spl["all"]["recovered"], "results/external_ceiling_split.json")
    _cmp(ctx, "external, the difference is the two arms subtracted",
              spl["seen_minus_unseen"]["delta"],
              spl["seen_in_training"]["coverage"] - spl["unseen"]["coverage"], "seen - unseen")

    m = _need(ctx, "external, what the interval still admits",
              r"admits an inflation of nearly \$(\d+\.\d+)\$ in either direction")
    if m:
        _cmp(ctx, "external, the inflation the interval admits, upper",
                  m.group(1), spl["seen_minus_unseen"]["ci95"][1],
                  "results/external_ceiling_split.json seen_minus_unseen.ci95[1]")
        _cmp(ctx, "external, the inflation the interval admits, lower",
                  m.group(1), -spl["seen_minus_unseen"]["ci95"][0],
                  "results/external_ceiling_split.json seen_minus_unseen.ci95[0], negated")

    # ---- the tautomer enumerator's budget ------------------------------------------------------
    # 191 and the 5 it leaves have no artifact; the denominator does, and the sentence's own
    # arithmetic ties the other two to it, so a lone stale numeral in it fails here.
    m = _need(ctx, "external, the enumerator's transform limit",
              r"reaches a canonical form for (\d+) of (\d+) unique metabolites and exhausts its "
              r"transform limit on the remaining (\d+), or (\d+\.\d+)\\%")
    if m:
        reached, total, left, pct = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                     m.group(4))
        _cmp(ctx, "external, the external reference metabolites", m.group(2),
                  spl["all"]["references"],
                  "results/external_ceiling_split.json all.references")
        _cmp(ctx, "external, the metabolites left at a non-canonical form", m.group(3),
                  total - reached, "the sentence's own subtraction")
        _cmp(ctx, "external, the share left at a non-canonical form", pct,
                  100.0 * left / total, "the sentence's own division")

    # ---- composition -------------------------------------------------------------------------
    m = _need(ctx, "external, the regression's predictors",
              r"regression takes (\w+) molecular descriptors")
    if m:
        _cmp(ctx, "external, molecular descriptors in the fit", _WORDS.get(m.group(1)),
                  len(reg["descriptor_names"]) - 1,
                  "results/ceiling_external_validity.json regression.descriptor_names, "
                  "less the annotation count")

    m = _need(ctx, "external, the in-sample macro predictions",
              r"predicts macro coverages of about (\d+\.\d+) internally and (\d+\.\d+) "
              r"externally in sample")
    if m:
        _cmp(ctx, "external, in-sample internal macro", m.group(1),
                  reg["predicted_internal_mean"],
                  "results/ceiling_external_validity.json regression.predicted_internal_mean")
        _cmp(ctx, "external, in-sample external macro", m.group(2),
                  reg["predicted_external_mean"],
                  "results/ceiling_external_validity.json regression.predicted_external_mean")

    m = _need(ctx, "external, the in-sample fit overstates transferability",
              r"in-sample external prediction of (\d+\.\d+) sits above the measured external macro "
              r"coverage of (\d+\.\d+)")
    if m:
        _cmp(ctx, "external, the in-sample external prediction again", m.group(1),
                  reg["predicted_external_mean"],
                  "results/ceiling_external_validity.json regression.predicted_external_mean")
        _cmp(ctx, "external, the measured external macro coverage", m.group(2),
                  reg["external_macro_coverage"],
                  "results/ceiling_external_validity.json regression.external_macro_coverage")
    ctx.checks.append((reg["predicted_external_mean"] > reg["external_macro_coverage"],
                      "external, the in-sample prediction does sit above the measurement",
                      "prediction > measurement",
                      f"{reg['predicted_external_mean']:.4f} vs "
                      f"{reg['external_macro_coverage']:.4f}",
                      "results/ceiling_external_validity.json"))

    m = _need(ctx, "external, the out-of-sample prediction",
              r"gives (\d+\.\d+) out of sample and recovers (\d+)\\% of the gap")
    if m:
        _cmp(ctx, "external, the out-of-sample external prediction", m.group(1),
                  reg["predicted_external_mean_oos"],
                  "results/ceiling_external_validity.json regression.predicted_external_mean_oos")
        _cmp(ctx, "external, the share of the gap recovered", m.group(2),
                  100.0 * reg["gap_recovery_frac"],
                  "results/ceiling_external_validity.json regression.gap_recovery_frac")
    # The recovered share is defined off the two macro coverages and the out-of-sample prediction;
    # recomputing it catches a fit that moved while the ratio beside it did not.
    _cmp(ctx, "external, the recovered share follows from the three macro quantities",
              reg["gap_recovery_frac"],
              (reg["predicted_external_mean_oos"] - reg["internal_macro_coverage"])
              / (reg["external_macro_coverage"] - reg["internal_macro_coverage"]),
              "(oos - internal) / (external - internal)")

    m = _need(ctx, "external, the model behind the out-of-sample figure",
              r"point estimate from a (\w+)-predictor model fit on the internal substrates and "
              r"averaged over the \$(\d+)\$ external ones")
    if m:
        _cmp(ctx, "external, predictors in the fitted model", _WORDS.get(m.group(1)),
                  len(reg["descriptor_names"]),
                  "results/ceiling_external_validity.json regression.descriptor_names")
        _cmp(ctx, "external, external substrates averaged over", m.group(2), ext["n"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.n")
        _cmp(ctx, "external, one coefficient per predictor plus an intercept",
                  len(reg["coefficients"]), len(reg["descriptor_names"]) + 1,
                  "results/ceiling_external_validity.json regression")

    m = _need(ctx, "external, the sample size the interpretation is conditioned on",
              r"At \$n=(\d+)\$ this is a suggestive partial composition effect")
    if m:
        _cmp(ctx, "external, sample size in the interpretation", m.group(1), ext["n"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.n")

    m = _need(ctx, "external, the twenty-four seen parents stated as a count",
              r"Twenty-four of the \$(\d+)\$ GLORYx parents are substrates")
    if m:
        _cmp(ctx, "external, GLORYx parents in the contamination section", m.group(1), ext["n"],
                  "results/ceiling_external_validity.json external_ceiling_uncapped.n")

    tex = ctx.tex.get("external.tex", "")
    for pat, n, what in ((r"\$37\$", 2, "the GLORYx parent count in math mode"),
                         (r"0\.808", 2, "the in-sample external prediction")):
        found = len(re.findall(pat, tex))
        ctx.checks.append((found == n, f"external, printings of {what}", f"{n} printings",
                          f"{found} found", "paper/app/external.tex"))


# --------------------------------------------------------------------------------------------
# paper/app/design.tex
# --------------------------------------------------------------------------------------------
def _design(ctx) -> None:
    sh = ctx.art("setsize_headroom.json")
    if not sh:
        ctx.checks.append((False, "design, its artifact is present", "setsize_headroom.json",
                          "missing", HERE))
        return
    rule = sh["arms"]["gap rule a=0.5"]
    oracle = sh["arms"]["oracle count"]

    m = _need(ctx, "design, the measured margin over the best constant",
              r"beats the best global constant on frozen scores by \$\+(\d+\.\d+)\$ "
              r"\$\[\+(\d+\.\d+),\+(\d+\.\d+)\]\$ macro F1")
    if m:
        _cmp(ctx, "design, margin over the best constant", m.group(1),
                  rule["f1_gain_over_best_constant"],
                  "results/setsize_headroom.json arms[gap rule a=0.5].f1_gain_over_best_constant")
        _cmp(ctx, "design, that margin's interval, lower", m.group(2),
                  rule["ci95_vs_best_constant"][0], "results/setsize_headroom.json")
        _cmp(ctx, "design, that margin's interval, upper", m.group(3),
                  rule["ci95_vs_best_constant"][1], "results/setsize_headroom.json")

    m = _need(ctx, "design, the margin against the deployed budget",
              r"Against the deployed budget the margin is \$\+(\d+\.\d+)\$, but that is the arm")
    if m:
        _cmp(ctx, "design, margin over the deployed budget", m.group(1),
                  rule["f1_gain_over_k15"],
                  "results/setsize_headroom.json arms[gap rule a=0.5].f1_gain_over_k15")

    m = _need(ctx, "design, the margin restated as the price of the deployed filter",
              r"and the \$\+(\d+\.\d+)\$ over the best constant it cannot express is the price")
    if m:
        _cmp(ctx, "design, the price the deployed scorer pays", m.group(1),
                  rule["f1_gain_over_best_constant"],
                  "results/setsize_headroom.json arms[gap rule a=0.5].f1_gain_over_best_constant")

    # The design rests on the emission rule beating the constant while the oracle bounds it, and on
    # the deployed budget being the discredited arm rather than the comparator.
    ctx.checks.append((rule["separated_vs_best_constant"] is True,
                      "design, the emission rule separates from the best constant",
                      "interval excludes zero",
                      str(rule["ci95_vs_best_constant"]), "results/setsize_headroom.json"))
    ctx.checks.append((0 < rule["f1_gain_over_best_constant"]
                      < oracle["f1_gain_over_best_constant"],
                      "design, the rule sits between the constant and the oracle",
                      "0 < rule < oracle",
                      f"{rule['f1_gain_over_best_constant']} vs "
                      f"{oracle['f1_gain_over_best_constant']}",
                      "results/setsize_headroom.json"))
    _cmp(ctx, "design, the deployed budget is the fixed k=15 arm",
              rule["f1_gain_over_k15"] - rule["f1_gain_over_best_constant"],
              sh["arms"]["fixed k=1"]["f1"] - sh["arms"]["fixed k=15"]["f1"],
              "the two baselines differ by exactly the constant's lead over k=15")

    tex = ctx.tex.get("design.tex", "")
    for pat, n, what in ((r"\+0\.0147", 2, "the margin over the best constant"),
                         (r"\+0\.074", 1, "the margin over the deployed budget")):
        found = len(re.findall(pat, tex))
        ctx.checks.append((found == n, f"design, printings of {what}", f"{n} printings",
                          f"{found} found", "paper/app/design.tex"))


# --------------------------------------------------------------------------------------------
# paper/app/arch.tex
# --------------------------------------------------------------------------------------------
def _arch(ctx) -> None:
    fc = ctx.art("filter_compare_matched_sub800.json")
    cat = ctx.art("mined_rule_catalog_v2.json")
    census = ctx.art("convention_census.json")
    if not (fc and cat and census):
        ctx.checks.append((False, "arch, its three artifacts are present", "3 artifacts",
                          "one or more missing", HERE))
        return

    # ---- the bank the whole appendix is about --------------------------------------------------
    m = _need(ctx, "arch, the \\nbank macro", r"\\newcommand\{\\nbank\}\{([\d,{}]+)\}")
    if m:
        _cmp(ctx, "arch, \\nbank", m.group(1).replace("{,}", ""),
                  census["banks"]["this work, full bank"]["templates"],
                  "results/convention_census.json banks[this work, full bank].templates")

    # ---- the featurisation both stages share ---------------------------------------------------
    m = _need(ctx, "arch, the shared featurisation",
              r"molecular graphs with (\d+)-dimensional nodes and (\d+)-dimensional edges, and a "
              r"(\d+)-bit Morgan fingerprint per molecule")
    if m:
        for label, group, const in (("node", 1, "SINGLE_NODE_DIM"), ("edge", 2, "EDGE_DIM"),
                                    ("fingerprint", 3, "FINGERPRINT_DIM")):
            s = _src(ctx, f"arch, {const} is defined", "grail_metabolism/utils/transform.py",
                     rf"^{const}\s*=\s*(\d+)")
            if s:
                _cmp(ctx, f"arch, shared {label} dimension", m.group(group), int(s.group(1)),
                          f"grail_metabolism/utils/transform.py {const}")

    # ---- the generator's encoder, as the paper preset configures it ----------------------------
    pres = "grail_metabolism/experiments/presets.py"
    m = _need(ctx, "arch, the generator's substrate encoder",
              r"node dimensions \$(\d+)\\to(\d+)\\to(\d+)\\to(\d+)\$")
    if m:
        # Every width in these two sentences comes out of one preset block, so it is read with one
        # pattern: a literal repeated here would be the frozen constant this whole script exists to
        # stop, and reading them together also fails if the block is reordered.
        s = _src(ctx, "arch, the paper preset's generator block", pres,
                 r"generator=GeneratorConfig\(\s*in_channels=(\d+),\s*edge_dim=\d+,\s*"
                 r"hidden_dims=\[(\d+), (\d+)\],\s*rule_hidden_dims=\[(\d+), (\d+), (\d+)\],\s*"
                 r"projection_dim=(\d+),")
        if s:
            for label, group, src_group in (("input", 1, 1), ("first hidden", 2, 2),
                                            ("second hidden", 3, 3), ("projection", 4, 7)):
                _cmp(ctx, f"arch, generator encoder, {label}", m.group(group),
                          int(s.group(src_group)), pres)
            m2 = _need(ctx, "arch, the rule-graph encoder",
                       r"hidden dimensions \$(\d+)\\to(\d+)\\to(\d+)\$")
            if m2:
                for i in range(3):
                    _cmp(ctx, f"arch, rule encoder layer {i + 1}", m2.group(i + 1),
                              int(s.group(4 + i)), pres)

    m = _need(ctx, "arch, the prior term and the applicability penalty",
              r"frequency-prior term at weight (\d+\.\d+), the log-odds .*? then subtracts a "
              r"penalty of (\d+\.\d+) for rules whose SMARTS does not match")
    if m:
        s = _src(ctx, "arch, the preset's prior strength", pres, r"prior_strength=(\d+\.\d+),")
        if s:
            _cmp(ctx, "arch, the frequency-prior weight", m.group(1), float(s.group(1)), pres)
        s = _src(ctx, "arch, the applicability penalty", "grail_metabolism/config.py",
                 r"applicability_penalty: float = (\d+\.\d+)")
        if s:
            _cmp(ctx, "arch, the applicability penalty", m.group(2), float(s.group(1)),
                      "grail_metabolism/config.py GeneratorConfig.applicability_penalty")

    m = _need(ctx, "arch, the margin-ranking auxiliary",
              r"margin-ranking auxiliary at weight (\d+\.\d+) and margin (\d+\.\d+)")
    if m:
        s = _src(ctx, "arch, the preset's ranking auxiliary", pres,
                 r"rank_weight=(\d+\.\d+),\s*\n\s*ranking_margin=(\d+\.\d+),")
        if s:
            _cmp(ctx, "arch, the ranking auxiliary's weight", m.group(1), float(s.group(1)), pres)
            _cmp(ctx, "arch, the ranking auxiliary's margin", m.group(2), float(s.group(2)), pres)

    # ---- the deployed filter's head -------------------------------------------------------------
    m = _need(ctx, "arch, the filter's concatenated vector",
              r"the resulting (\d+)-dimensional vector is mapped through a three-layer perceptron")
    if m:
        s = _src(ctx, "arch, FINGERPRINT_DIM for the filter head",
                 "grail_metabolism/utils/transform.py", r"^FINGERPRINT_DIM\s*=\s*(\d+)")
        if s:
            _cmp(ctx, "arch, the filter's concatenated width", m.group(1),
                      2 * 128 + 2 * int(s.group(1)),
                      "two 128-dim embeddings and two Morgan fingerprints")

    m = _need(ctx, "arch, the merged-graph variant's nodes",
              r"encodes one merged substrate--product graph, whose (\d+)-dimensional nodes")
    if m:
        s = _src(ctx, "arch, PAIR_NODE_DIM is defined", "grail_metabolism/utils/transform.py",
                 r"^PAIR_NODE_DIM\s*=\s*(\d+)")
        if s:
            _cmp(ctx, "arch, the merged graph's node dimension", m.group(1), int(s.group(1)),
                      "grail_metabolism/utils/transform.py PAIR_NODE_DIM")

    m = _need(ctx, "arch, the filter's early stopping",
              r"filter under the second\. Optimisation uses Adam at a learning rate of "
              r"\$10\^\{(-\d+)\}\$ for at most \d+ epochs with early stopping at a patience of "
              r"(\d+)")
    if m:
        s = _src(ctx, "arch, the optimiser defaults", "grail_metabolism/config.py",
                 r"class OptimConfig:\s*\n\s*lr: float = 1e-0*(\d+)[\s\S]*?patience: int = (\d+)")
        if s:
            _cmp(ctx, "arch, the learning-rate exponent", m.group(1), -int(s.group(1)),
                      "grail_metabolism/config.py OptimConfig.lr")
            _cmp(ctx, "arch, the early-stopping patience", m.group(2), int(s.group(2)),
                      "grail_metabolism/config.py OptimConfig.patience")

    # ---- choosing between the two filters -------------------------------------------------------
    m = _need(ctx, "arch, the matched filter-training subset",
              r"trained on an identical (\d+)-substrate subset")
    if m:
        s = _src(ctx, "arch, the subset size the trainer defaults to",
                 "scripts/train_filter_subset.py", r'"--max-train", type=int, default=(\d+)')
        if s:
            _cmp(ctx, "arch, the filter comparison's training subset", m.group(1),
                      int(s.group(1)), "scripts/train_filter_subset.py --max-train")

    m = _need(ctx, "arch, the filter comparison",
              r"Recall@15 is (\d+\.\d+) for the independent variant against (\d+\.\d+) for the "
              r"merged-graph variant, a paired difference of \$(-\d+\.\d+)\$ with a 95\\% "
              r"paired-bootstrap interval of \$\[(-\d+\.\d+),(-\d+\.\d+)\]\$ over \$n=\\ntest\$ "
              r"substrates; at recall@5 the margin widens, (\d+\.\d+) against (\d+\.\d+)")
    if m:
        d = fc["paired_delta_recall15_pair_minus_single"]
        # 0.3655 and 0.3565 sit exactly on the third-decimal boundary and are printed rounded up.
        _cmp(ctx, "arch, independent encoder recall@15", m.group(1),
                  fc["rankings"]["single"]["recall@15"],
                  "results/filter_compare_matched_sub800.json rankings.single.recall@15")
        _cmp(ctx, "arch, merged-graph recall@15", m.group(2),
                  fc["rankings"]["pair"]["recall@15"],
                  "results/filter_compare_matched_sub800.json rankings.pair.recall@15")
        _cmp(ctx, "arch, the paired difference", m.group(3), d["point"],
                  "results/filter_compare_matched_sub800.json "
                  "paired_delta_recall15_pair_minus_single.point")
        _cmp(ctx, "arch, that difference's interval, lower", m.group(4), d["lo"],
                  "results/filter_compare_matched_sub800.json")
        _cmp(ctx, "arch, that difference's interval, upper", m.group(5), d["hi"],
                  "results/filter_compare_matched_sub800.json")
        _cmp(ctx, "arch, independent encoder recall@5", m.group(6),
                  fc["rankings"]["single"]["recall@5"],
                  "results/filter_compare_matched_sub800.json rankings.single.recall@5")
        _cmp(ctx, "arch, merged-graph recall@5", m.group(7), fc["rankings"]["pair"]["recall@5"],
                  "results/filter_compare_matched_sub800.json rankings.pair.recall@5")
        # The difference is the two rankings subtracted, and it is evaluated on the clean test
        # split the appendix names by macro rather than by number.
        _cmp(ctx, "arch, the difference is the two variants subtracted", d["point"],
                  fc["rankings"]["pair"]["recall@15"] - fc["rankings"]["single"]["recall@15"],
                  "pair - single")
    m = _need(ctx, "arch, the \\ntest macro", r"\\newcommand\{\\ntest\}\{([\d,{}]+)\}")
    if m:
        _cmp(ctx, "arch, the filter comparison ran on the full clean test split",
                  m.group(1).replace("{,}", ""), fc["n"],
                  "results/filter_compare_matched_sub800.json n")

    # ---- how the bank was mined ------------------------------------------------------------------
    mine = "scripts/mine_rules.py"
    m = _need(ctx, "arch, the MCS coverage gate",
              r"the match must cover at least (\d+)\\% of the smaller molecule's atoms")
    if m:
        s = _src(ctx, "arch, the miner's MCS coverage gate", mine,
                 r"if mcs\.numAtoms < (\d+\.\d+) \* min_atoms")
        if s:
            _cmp(ctx, "arch, the MCS coverage gate", float(m.group(1)) / 100, float(s.group(1)),
                      f"{mine} rejects an MCS below this share of the smaller molecule")

    m = _need(ctx, "arch, the selectivity filter",
              r"rejects templates producing more than (\d+) distinct products on average across a "
              r"(\d+)-substrate sample, and templates that are both very general and very "
              r"prolific, applicable to more than (\d+)\\% of sampled substrates while still "
              r"averaging more than (\d+)\s*products")
    if m:
        s = _src(ctx, "arch, the miner's unselective cut", mine, r"if mean_products > (\d+):")
        if s:
            _cmp(ctx, "arch, the mean-products cut", m.group(1), int(s.group(1)), mine)
        s = _src(ctx, "arch, the miner's sample size", mine, r"FILTER_SAMPLE_SIZE = (\d+)")
        if s:
            _cmp(ctx, "arch, the selectivity sample", m.group(2), int(s.group(1)),
                      f"{mine} FILTER_SAMPLE_SIZE")
        s = _src(ctx, "arch, the miner's general-and-prolific cut", mine,
                 r"if applicability_fraction > (\d+\.\d+) and mean_products > (\d+):")
        if s:
            _cmp(ctx, "arch, the applicability cut", float(m.group(3)) / 100, float(s.group(1)),
                      mine)
            _cmp(ctx, "arch, the products cut applied with it", m.group(4), int(s.group(2)), mine)

    # ---- what the bank contains ----------------------------------------------------------------
    m = _need(ctx, "arch, the four curated banks the union was built from",
              r"deduplicated union of four earlier curated banks, of (\d+), (\d+), (\d+) and "
              r"([\d{},]+) rules")
    if m:
        nested = census["nested_not_counted"]
        _cmp(ctx, "arch, the middle curated bank", m.group(2),
                  nested["hand-written collection, middle"]["templates"],
                  "results/convention_census.json nested_not_counted[middle]")
        _cmp(ctx, "arch, the largest curated bank", m.group(4).replace("{,}", ""),
                  nested["hand-written collection, largest"]["templates"],
                  "results/convention_census.json nested_not_counted[largest]")
        comp = ctx.root / "grail_metabolism" / "compressed_rules.smarts"
        n_comp = len([ln for ln in comp.read_text().splitlines() if ln.strip()]) \
            if comp.exists() else None
        _cmp(ctx, "arch, the compressed curated bank", m.group(3), n_comp,
                  "grail_metabolism/compressed_rules.smarts, non-blank lines")
    # The union is a partition of the deployed bank: the curated half and the mined half have to
    # add back to it, which is the totals row this sentence does not print.
    banks = census["banks"]
    _cmp(ctx, "arch, the curated and mined halves sum to the bank",
              banks["this work, full bank"]["templates"],
              banks["this work, curated half"]["templates"]
              + banks["this work, mined half"]["templates"],
              "results/convention_census.json banks")

    counts = [v["count"] for v in cat.values()]
    m = _need(ctx, "arch, the positional miner's yield",
              r"positional miner alone extracts ([\d{},]+) unique self-tested templates")
    if m:
        _cmp(ctx, "arch, templates the positional miner extracts",
                  m.group(1).replace("{,}", ""), len(cat),
                  "results/mined_rule_catalog_v2.json, one entry per template")

    m = _need(ctx, "arch, how the miner's support is distributed",
              r"Of these, ([\d{},]+) \((\d+)\\%\) rest on a single training pair, and only (\d+) "
              r"on five or more; the single best-supported template covers ([\d{},]+) pairs")
    if m:
        one = sum(1 for c in counts if c == 1)
        _cmp(ctx, "arch, templates resting on one pair", m.group(1).replace("{,}", ""), one,
                  "results/mined_rule_catalog_v2.json, entries with count == 1")
        _cmp(ctx, "arch, their share of the mined templates", float(m.group(2)) / 100,
                  one / len(cat), "results/mined_rule_catalog_v2.json")
        _cmp(ctx, "arch, templates resting on five or more pairs", m.group(3),
                  sum(1 for c in counts if c >= 5), "results/mined_rule_catalog_v2.json")
        _cmp(ctx, "arch, the best-supported template's pairs",
                  m.group(4).replace("{,}", ""), max(counts),
                  "results/mined_rule_catalog_v2.json, the largest count")

    m = _need(ctx, "arch, the miner re-derives nothing new",
              r"re-derives templates that are all already in the bank: ([\d{},]+) of ([\d{},]+), "
              r"none new")
    if m:
        _cmp(ctx, "arch, re-derived templates already in the bank",
                  m.group(1).replace("{,}", ""), len(cat),
                  "results/mined_rule_catalog_v2.json")
        _cmp(ctx, "arch, templates the re-run produced", m.group(2).replace("{,}", ""), len(cat),
                  "results/mined_rule_catalog_v2.json")
        bank = ctx.root / "grail_metabolism" / "resources" / "extended_smirks.txt"
        if bank.exists():
            shipped = {ln.strip() for ln in bank.read_text().splitlines() if ln.strip()}
            new = sum(1 for r in cat if r not in shipped)
            _cmp(ctx, "arch, none of the re-derived templates is new", 0, new,
                      "mined_rule_catalog_v2.json keys against resources/extended_smirks.txt")
        else:
            ctx.checks.append((False, "arch, the shipped bank is readable",
                              "resources/extended_smirks.txt", "missing", HERE))
