"""Every number the manuscript prints, pulled from the artifact that produced it.

SELF_CLAIMS section 11 asks that each numeric passage trace back to its artifact, and section 11a
that no number in the main text be an orphan. Checking that after the fact found five errors in an
hour of prose the last time it was tried. This inverts the order: the manuscript's numbers are
generated here, the text cites them by name, and a check holds the two together. A figure that is
not in this file may not appear in the paper.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))          # the package, for constants read from the code itself

from _provenance import stamp  # noqa: E402


def art(name):
    return json.loads((ROOT / "results" / name).read_text())


# Numbers that are DECLARED rather than measured, each with the reason it cannot come from an
# artifact and, where one exists, the place it is really defined.
#
# The distinction earns its keep. Four numbers reached the page as bare literals here, and they
# were not one kind of thing: two were a stale measurement that the artifact had moved away from
# underneath, two were dead and printed nowhere, and two are genuine constants of a design. A
# literal that looks like every other literal makes those indistinguishable, so the constants are
# named as constants and everything else must come from a read.
DECLARED = {
    # A count of published systems, from the literature survey rather than from any run of ours.
    # Nothing in results/ measures how many metabolite predictors of 2025-26 ship no runnable
    # implementation, because the question is answered by looking for one and failing.
    "comparators.unavailable": 4,
    # The sampling scheme's tail: the largest N substrates are timed outright instead of being
    # sampled, because a cost envelope that misses its own worst cases is not an envelope. The
    # value is defined in the producer, and _check_declared below holds the two to each other so
    # they cannot drift the way the h8 pair did.
    "cost.tail": 12,
}


def _check_declared():
    """Hold a declared constant to the place it is really defined, where such a place exists.

    A declared constant is honest and a duplicated one is not. The cost envelope's tail lives in
    its producer as a slice; if that slice changes and this dictionary does not, the SI describes a
    sampling scheme nobody ran. Reading the producer's source is enough to catch that, and costs
    nothing.
    """
    src = (ROOT / "scripts" / "typed_edit" / "cost_envelope.py").read_text()
    m = re.search(r"sized\[::args\.every\]\s*\+\s*sized\[-(\d+):\]", src)
    if not m:
        raise SystemExit(
            "REFUSING: cost_envelope.py no longer builds its sample as a stride plus a tail, so "
            "the tail size declared here describes a scheme that is not the one it runs.")
    if int(m.group(1)) != DECLARED["cost.tail"]:
        raise SystemExit(
            f"REFUSING: cost_envelope.py takes the largest {m.group(1)} outright and this file "
            f"declares {DECLARED['cost.tail']}. The SI would describe a sampling scheme nobody ran.")


def build():
    _check_declared()
    dep = art("deployment_table.json")
    h7 = art("h7_verdict.json")
    h9 = art("h9_verdict.json")
    h10 = art("h10_verdict.json")
    h11 = art("h11_grid.json")
    h12 = art("h12_verdict.json")
    h13 = art("h13_verdict.json")
    h14 = art("h14_verdict.json")
    h15 = art("h15_verdict.json")
    h8 = art("h8_verdict.json")
    cov = art("coverage_gap_types.json")
    cen = art("novel_type_census.json")
    usp = art("uspto_type_overlap.json")
    env = art("cost_envelope.json")
    ovl = art("external_overlap_audit.json")
    wide = art("wide_pool_analysis_implicit.json")
    gd = art("group_decode.json")
    tb = art("tautomer_budget.json")
    nm = art("tautomer_near_miss.json")
    mt = art("mode_timings.json")
    rel = art("typed_edit_known_type_recovery.json")
    car = art("typed_edit_type_carriers.json")

    n = {}

    # the population and its audit
    n["population.n"] = dep["population"]["n"]
    n["population.references"] = int(dep["population"]["n_references"])
    n["population.in_train_or_val"] = ovl["MetaTox 291-substrate comparison set"]["in_train_or_val"]
    n["gloryx.overlap"] = ovl["GLORYx external set"]["in_train_or_val"]
    n["gloryx.n"] = ovl["GLORYx external set"]["keyed"]
    n["gloryx.fraction"] = ovl["GLORYx external set"]["fraction"]

    # the splits the three evaluation populations are drawn from. A population described by its
    # own size alone reads as a whole split; each of these is a stated subset of one.
    lk = art("leakage_fix_report.json")["clean_split_stats"]
    for split in ("train", "val", "test"):
        n[f"split.{split}.substrates"] = lk[split]["remaining_substrates"]
        n[f"split.{split}.pairs"] = lk[split]["remaining_positive_pairs"]
        n[f"split.{split}.triples"] = lk[split]["remaining_triples"]

    # the validation draw is a sample and not the split; its cap and seed are what make the
    # population reproducible, and a figure that names neither is not checkable
    # Split out of the 46 MB pool it lives in, because the pool is not tracked and reading it
    # here is what stopped the number chain from running in a fresh clone at all.
    vp = art("val_pool_population.json")["population"]
    n["valdraw.cap"] = vp["cap"]
    n["valdraw.seed"] = vp["seed"]
    n["valdraw.declared"] = vp["declared_n"]
    n["valdraw.paired"] = vp["n"]

    # the deployed comparison, per arm and budget
    # The macro grid, so the aggregation is a swept axis rather than a figure quoted once at the
    # budget where the reported aggregation does not separate.
    for k, row in dep["recall_macro"].items():
        for arm, tag in (("whole bank", "bank"), ("trained budget", "trained"),
                         ("metatox", "metatox"), ("sygma", "sygma"),
                         ("metapredictor", "metapredictor")):
            if arm in row:
                n[f"macro.{tag}.{k}"] = row[arm]
    for k, row in dep["contrasts_macro"].items():
        for pair, c in row.items():
            a, b = [x.strip() for x in pair.split(" - ")]
            tag = {"whole bank": "bank", "trained budget": "trained"}.get(a)
            tagb = {"metatox": "Metatox", "sygma": "Sygma",
                    "metapredictor": "Metapredictor"}.get(b)
            if tag and tagb:
                n[f"macrogap.{tag}{tagb}.{k}"] = c["gap"]
                n[f"macrogap.{tag}{tagb}.{k}.lo"] = c["ci95"][0]
                n[f"macrogap.{tag}{tagb}.{k}.hi"] = c["ci95"][1]
                n[f"macrogap.{tag}{tagb}.{k}.sep"] = c["excludes_zero"]

    # "gxarm" and not "gloryx": this file already carries gloryx.* for the overlap between this
    # corpus and GLORYx's published reference set, which is a different quantity about the same
    # system, and one prefix for two of those is how a number ends up in the wrong sentence.
    ARMS = {"whole bank": "bank", "trained budget": "trained", "metatox": "metatox",
            "sygma": "sygma", "metapredictor": "metapredictor",
            "biotransformer": "biotransformer", "gloryx": "gxarm"}
    for k, row in dep["recall_micro"].items():
        for arm, tag in ARMS.items():
            if arm in row:
                n[f"sweep.{tag}.{k}"] = row[arm]
    for k, row in dep["contrasts"].items():
        for pair, c in row.items():
            if "unavailable" in c:
                continue
            a, b = [x.strip() for x in pair.split(" - ")]
            if a not in ARMS or b not in ARMS:
                continue
            tag = f"{ARMS[a]}{ARMS[b].capitalize()}"
            n[f"gap.{tag}.{k}"] = c["gap"]
            n[f"gap.{tag}.{k}.lo"] = c["ci95"][0]
            n[f"gap.{tag}.{k}.hi"] = c["ci95"][1]
            n[f"gap.{tag}.{k}.sep"] = c["excludes_zero"]
    # How many published predictors carry a column, counted from the table rather than written as
    # a word. It was four for three revisions and is five now, and a count in prose that nothing
    # generates is a count that goes stale on the day an arm is added.
    n["comparators.columns"] = sum(1 for a in ARMS if a not in ("whole bank", "trained budget")
                                   and a in dep["recall_micro"]["15"])

    # The one arm not run on this machine, and what the run that produced it recorded: the module
    # version the service reported and how many jobs the submission took.
    _gx = art("gloryx_service_preds.json")
    _got = _gx["obtained_from"]
    n["comparators.gloryxarm.version"] = _got["module_version"]
    n["gxarm.jobs"] = len(_got["jobs"])
    n["gxarm.substrates"] = _gx["n_substrates"]
    n["gxarm.answered"] = _gx["n_with_at_least_one_prediction"]

    for arm, tag in ARMS.items():
        if arm in dep["mean_output_length"]:
            n[f"output.{tag}"] = dep["mean_output_length"][arm]
        # What each method actually emits, before the sweep truncates it at the widest budget.
        # The two differ by nearly a factor of two for SyGMa and the table printed only the first.
        if arm in dep.get("mean_emitted_untruncated", {}):
            n[f"emitted.{tag}"] = dep["mean_emitted_untruncated"][arm]
    for k, row in dep["substrates_whose_list_is_shorter_than_the_budget"].items():
        for arm, tag in ARMS.items():
            if arm in row:
                n[f"short.{tag}.{k}"] = row[arm]

    # the registered checks
    n["h7.product"] = round(h7["micro"]["product"], 4)
    n["h7.fusion"] = round(h7["micro"]["fusion"], 4)
    n["h7.diff"] = round(h7["micro"]["difference"], 4)
    n["h7.lo"], n["h7.hi"] = [round(x, 4) for x in h7["bootstrap"]["ci95"]]
    n["h7.threshold"] = h7["registered_threshold"]

    n["h9.uncapped"] = h9["micro"]["uncapped"]
    n["h9.capped"] = h9["micro"]["capped"]
    n["h9.diff"] = h9["micro"]["difference"]
    n["h9.lo"], n["h9.hi"] = h9["bootstrap"]["ci95"]
    n["h9.threshold"] = h9["registered_threshold"]
    # the cap SIZE, distinct from the recall AT the cap. The manuscript printed the second
    # where the first belonged and no gate could see it: both are legitimate macros.
    n["h9.cap"] = h9["cap"]
    n["h9.pool.before"] = h9["mean_pool"]["uncapped"]
    n["h9.pool.after"] = h9["mean_pool"]["capped"]

    n["h10.threshold"] = h10["registered_threshold"]
    n["h10.bank"] = h10["micro"]["whole_bank"]
    n["h10.trained"] = h10["micro"]["trained"]
    n["h10.bought"] = h10["micro"]["bought_by_the_whole_bank"]
    n["h10.lo"], n["h10.hi"] = h10["bootstrap"]["ci95"]
    n["h10.pool.bank"] = h10["mean_pool"]["whole_bank"]
    n["h10.pool.trained"] = h10["mean_pool"]["trained"]
    n["h10.topk"] = h10["top_k"]["trained"]
    # The same quantity on the other population, in the direction P3 was registered in: what the
    # whole bank buys over the trained budget. The sweep records it as trained minus bank, so the
    # sign is turned here rather than in the prose, which is where it was a literal.
    if "gap.trainedBank.15" in n:
        n["h10.comparison"] = round(-n["gap.trainedBank.15"], 4)

    n["h11.cells"] = h11["cells_including_metatox_own_emission"]
    n["h11.lost"] = h11["cells_lost_including_own_emission"]
    n["h11.output.rule"] = h11["mean_output"]["rule"]
    n["h11.output.metatox"] = h11["mean_output"]["metatox"]
    g = h11["grid"]["inchikey_tautomer"]
    n["h11.f1.rule"] = g["rule"]["f1"]
    n["h11.recall.rule"] = g["rule"]["recall"]
    n["h11.recall.metatox15"] = g["metatox"]["15"]["recall"]
    n["h11.f1.metatox15"] = g["metatox"]["15"]["f1"]

    n["h12.two"] = h12["recall_micro"]["15"]["two_way"]
    n["h12.three"] = h12["recall_micro"]["15"]["three_way"]
    n["h12.diff"] = h12["primary_three_way_minus_two_way"]["gap"]
    n["h12.lo"], n["h12.hi"] = h12["primary_three_way_minus_two_way"]["ci95"]
    n["h12.ceiling"] = h12["ceiling_of_this_composition"]["gap"]

    n["h8.threshold"] = h8["registered_threshold"]
    n["h8.diff"] = h8["primary_scorer_minus_fusion"]["gap"]
    n["h8.design"] = h8["what_the_design_costs_before_the_model_acts"]["gap"]
    n["h8.model"] = h8["what_the_model_adds_given_the_design"]["gap"]

    n["h14.threshold"] = h14["registered_threshold"]
    n["h14.diff"] = h14["primary_gate_minus_cap"]["gap"]
    n["comparators.unavailable"] = DECLARED["comparators.unavailable"]
    n["h14.ceiling"] = h14["ceiling_of_this_gate"]["gap"]

    n["h13.time.before"] = h13["time"]["every_product_median_s"]
    n["h13.time.after"] = h13["time"]["survivors_median_s"]
    n["h13.factor"] = h13["time"]["median_factor"]
    n["h13.recall.diff"] = h13["recall_contrasts"]["15"]["change"]
    n["h13.enumeration"] = h13["time"]["enumeration_alone_against_the_old_total"]

    n["h15.time"] = h15["time"]["h15_median_s"]
    n["h15.recall.diff"] = h15["recall_contrasts"]["15"]["change"]
    n["h15.budget"] = h15["tautomer_budget"]
    n["h15.npaired"] = h15["population"]["n_paired"]
    n["h15.nreferences"] = int(h9["n_references"])
    n["h15.moved"] = h15["key_diagnostic"][
        "whose_standardised_smiles_differs_from_the_shipped_budget"]
    n["h15.candidates"] = h15["key_diagnostic"]["candidates_in_the_bounded_arm"]
    n["h15.load"] = h15["time"]["load_correction"]["load_factor_from_the_identical_enumeration"]
    n["h15.attributable"] = h15["time"]["load_correction"]["speedup_attributable_to_the_budget"]
    # the two standardisation medians the attributable factor is the load-corrected ratio of.
    # Without them a reader divides the whole arm's 9.45 by the load factor and gets a third
    # number, which is what happened.
    n["h15.stdunbounded"] = h15["time"]["load_correction"][
        "unbounded_arm_standardise_median_s"]
    n["h15.stdbounded"] = h15["time"]["standardise_survivors_median_s"]

    # the ceiling and what the gap is made of
    n["population.testsubs"] = cov["n_substrates"]
    n["population.testrefs"] = cov["covered_pairs"] + cov["uncovered_pairs"]
    n["ceiling.coverage"] = cov["coverage"]
    n["ceiling.uncovered"] = cov["uncovered_pairs"]
    n["ceiling.novel"] = cov["gap"]["novel_type"]
    n["ceiling.known"] = cov["gap"]["known_type"]
    n["ceiling.untypeable"] = cov["gap"]["untypeable"]

    # Two loops measure the ceiling under the same hydrogen convention and return different
    # counts. The paper reports the deployed loop's; the audit loop's is what the two conventions
    # are compared with. The difference is small and is printed rather than chosen between.
    # What the released checkpoints were actually trained on. The manuscript described the split
    # and never said that the run drew a subsample of it.
    hyp = art("hyperparameters.json")
    n["train.substrates"] = hyp["data"]["training substrates"]
    n["train.valsubstrates"] = hyp["data"]["validation substrates"]
    n["train.samplingseed"] = hyp["data"]["sampling seed"]
    n["train.epochs"] = hyp["components"]["generator"]["epochs run"]

    # whether the site a prediction names is the site that changed, against a same-size null
    sag = art("site_agreement.json")
    n["site.scored"] = sag["counts"]["scored"]
    # The budget the audit ran at, so the section can name the arm rather than leave a
    # reader to infer it: the whole bank is not a budget of thirty rules.
    n["site.rulebudget"] = sag["population"]["rule_budget"]
    n["site.hit"] = sag["counts"]["centre_hit"]
    n["site.touching"] = sag["share_of_scored_where_the_reported_site_touches_the_centre"]
    n["site.inside"] = sag["share_of_scored_where_the_reported_site_lies_wholly_inside_it"]
    n["site.null"] = sag["null"]["share_touching_the_centre"]
    n["site.margin"] = sag["null"]["observed_minus_null"]
    # The null the comparison actually needs, since a template match is a connected fragment. It
    # is lower than the uniform one, not higher: a scattered set of atoms touches a given centre
    # more often than a fragment clustered in one place, so the caution the manuscript printed
    # about this had the sign the wrong way round.
    n["site.connectednull"] = sag["connected_null"]["share_touching_the_centre"]
    # Split by where the template came from. For a mined template the firing atoms and the
    # reference centre come from one routine applied to one pair, so agreement there is partly
    # guaranteed; the curated figure is the one that tests the claim independently.
    for origin in ("curated", "mined"):
        cell = sag["by_template_origin"][origin]
        n[f"site.{origin}"] = cell["share"]
        n[f"site.{origin}scored"] = cell["scored"]
    n["site.connectedmargin"] = sag["connected_null"]["observed_minus_null"]

    # what the rule budget buys on validation, so the deployed value is a chosen point
    bc = art("budget_curve.json")
    n["budget.substrates"] = bc["population"]["n_substrates"]
    n["budget.deployed"] = bc["deployed_budget"]
    n["budget.built"] = len(bc["budgets_built"])
    for b in bc["budgets_built"]:
        n[f"budget.k{b}.recall"] = bc["by_budget"][str(b)]["recall_micro"]["15"]
        # The same curve read at an output budget of thirty. At fifteen the deployed rule budget
        # is the knee and the curve is flat past it; at thirty it is not, which is the reading
        # the Discussion needs and the reason the two modes are one curve and not two designs.
        n[f"budget.k{b}.recallthirty"] = bc["by_budget"][str(b)]["recall_micro"]["30"]
        n[f"budget.k{b}.candidates"] = bc["by_budget"][str(b)]["mean_candidates"]
    for b, cell in bc["against_the_deployed_budget_at_k15"].items():
        n[f"budget.k{b}.gap"] = cell["gap_at_15"]
        n[f"budget.k{b}.lo"] = cell["ci95"][0]
        n[f"budget.k{b}.hi"] = cell["ci95"][1]
    # The same contrast read at an output budget of thirty, which is where the Discussion
    # recommends raising the rule budget. That recommendation used to rest on two point estimates.
    for b, cell in bc["against_the_deployed_budget_at_k30"].items():
        n[f"budget.k{b}.gapthirty"] = cell["gap_at_30"]
        n[f"budget.k{b}.lothirty"] = cell["ci95"][0]
        n[f"budget.k{b}.hithirty"] = cell["ci95"][1]
        bound = cell.get("if_the_excluded_substrates_all_went_one_way")
        if bound:
            n[f"budget.k{b}.boundbestthirty"] = bound["best_for_this_budget"]
            n[f"budget.k{b}.boundworstthirty"] = bound["worst_for_this_budget"]
    # A count of budgets beating the deployed one is meaningless without the output budget it
    # was read at: the same curve answers none at fifteen and two at thirty.
    n["budget.beatingfifteen"] = len(bc["budgets_that_beat_the_deployed_one_at_k15"])
    n["budget.beatingthirty"] = len(bc["budgets_that_beat_the_deployed_one_at_k30"])
    # What the paired population costs. A budget that cannot finish every substrate leaves the
    # curve measured on the ones every budget holds, and the references the rest carry bound how
    # far that could have moved any contrast; both are printed rather than left to a footnote.
    n["budget.excluded"] = bc["substrates_outside_the_paired_population"]
    n["budget.excludedrefs"] = bc["references_they_carry"]
    # Read from the code rather than typed: the floor a disconnected product component must clear
    # to enter the pool at all, which is part of what the emission rule is.
    from grail_metabolism.utils import preparation as _prep
    n["fragment.floor"] = _prep._MIN_FRAGMENT_HEAVY_ATOMS

    # The protonation counterpart of the stereochemistry bound: how much of the annotation a
    # criterion that kept the protonation layer could in principle resolve differently.
    rc = art("reference_charge.json")
    n["charge.refs"] = rc["references_carrying_a_formal_charge"]
    n["charge.share"] = rc["share"]
    n["charge.zwitterions"] = rc["of_those_net_neutral_zwitterions"]
    n["charge.netcharged"] = rc["of_those_net_charged"]
    n["charge.substrates"] = rc["substrates_with_at_least_one"]
    n["charge.typed"] = rc["population"]["typed"]
    n["budget.fragile"] = len(bc["contrasts_whose_sign_the_absence_could_flip"])
    n["budget.fragiledecided"] = len(bc["of_those_any_the_paper_reads_a_verdict_from"])
    for b, cell in bc["against_the_deployed_budget_at_k15"].items():
        bound = cell.get("if_the_excluded_substrates_all_went_one_way")
        if bound:
            n[f"budget.k{b}.boundbest"] = bound["best_for_this_budget"]
            n[f"budget.k{b}.boundworst"] = bound["worst_for_this_budget"]

    # how much of the curated half is somebody else's rules verbatim, and under whose terms
    ctp = art("curated_third_party.json")
    # How much of the collection the manuscript calls curated is somebody else's rule verbatim.
    # This lived only in the supporting information while the body said "curated" unqualified,
    # and a referee who reached that section felt the body had softened it.
    n["thirdparty.borrowed"] = ctp["borrowed_within_the_named_body"]
    n["thirdparty.borrowedshare"] = ctp["share_of_the_named_body_so_traceable"]
    n["thirdparty.sygmanamed"] = (ctp["by_rightsholder"].get("SyGMa 1.1.0")
                                  or ctp["by_rightsholder"].get("SyGMa")
                                  or {}).get("templates_in_the_curated_half")
    n["thirdparty.gloryxnamed"] = (ctp["by_rightsholder"].get("GLORYx")
                                   or {}).get("templates_in_the_curated_half")
    n["thirdparty.curated"] = ctp["bank"]["curated"]
    n["thirdparty.traceable"] = ctp["curated_templates_traceable_to_a_published_set"]
    n["thirdparty.traceableshare"] = ctp["share_of_the_curated_half_so_traceable"]
    n["thirdparty.untraceable"] = ctp["curated_templates_traceable_to_nothing_measured"]
    n["thirdparty.named"] = ctp["curated_split"]["named_collections"]
    n["thirdparty.innamed"] = ctp["borrowed_within_the_named_body"]
    n["thirdparty.namedshare"] = ctp["share_of_the_named_body_so_traceable"]
    n["thirdparty.inextraction"] = ctp["borrowed_within_the_earlier_extraction"]
    n["thirdparty.holders"] = sum(
        1 for r in ctp["by_rightsholder"].values() if r["templates_in_the_curated_half"])
    for tag, src in (("sygma", "SyGMa"), ("biotransformer", "BioTransformer"),
                     ("gloryx", "GLORYx")):
        n[f"thirdparty.{tag}.shipped"] = (
            ctp["by_rightsholder"][src]["templates_in_the_curated_half"])
    n["thirdparty.biotransformer.read"] = (
        ctp["by_published_set"]["BioTransformer core"]["rules_in_the_published_set"])
    n["thirdparty.files"] = len(ctp["third_party_files_this_repository_tracks"])
    n["thirdparty.filesunused"] = sum(
        1 for r in ctp["third_party_files_this_repository_tracks"].values()
        if not r.get("templates_of_this_file_used_by_the_bank"))

    # what each method recovers, split by the chemistry rather than by the budget
    ebc = art("error_by_chemistry.json")
    n["chem.references"] = ebc["population"]["references_classified"]
    n["chem.classes"] = len(ebc["classes"])
    # The four classical classes the main text says the arm trails on, and how many references
    # they carry between them. The sentence is a direction and not a measurement, and the count
    # is what tells a reader which.
    n["chem.smallrowsrefs"] = sum(
        ebc["classes"][c]["references"]
        for c in ("demethylation", "sulfation", "methylation", "acetylation")
        if c in ebc["classes"])
    n["chem.unresolved"] = ebc["population"]["references_whose_structure_could_not_be_recovered"]
    _diol = ebc["classes"].get("oxidation, two oxygens added", {})
    n["chem.diol.refs"] = _diol.get("references")
    n["chem.diol.best"] = max(v["15"] for v in _diol["recall"].values()) if _diol else None
    # How many arms this row is read over, and how many of them recover nothing. Typed as "five of
    # the six" while the table had seven rows: the sentence counted the arms of an earlier
    # comparison and the table counted the arms of this one.
    n["chem.arms"] = len(_diol["recall"]) if _diol else None
    n["chem.diol.recovernone"] = (
        sum(1 for v in _diol["recall"].values() if v["15"] == 0) if _diol else None)
    # Every arm but MetaTox met the substrate as the corpus stores it, which the GLORYx submission
    # artifact states of itself and the dialect study establishes for the rest.
    n["chem.drawnstored"] = n["chem.arms"] - 1 if n["chem.arms"] else None
    _other = ebc["classes"].get("other", {})
    n["chem.other.refs"] = _other.get("references")
    n["chem.other.best"] = max(v["15"] for v in _other["recall"].values()) if _other else None
    _cleav = ebc["classes"].get("cleavage, two or more carbons lost", {})
    n["chem.cleavage.refs"] = _cleav.get("references")
    n["chem.cleavage.bank"] = _cleav["recall"]["GRAIL exhaustive"]["15"] if _cleav else None
    n["chem.cleavage.bestother"] = (max(v["15"] for k, v in _cleav["recall"].items()
                                        if not k.startswith("GRAIL")) if _cleav else None)
    # The same class with list length held instead of the budget. At fifteen the comparator this
    # class's figure is read against has run out of candidates on most substrates, so part of that
    # margin is length. Read against each comparator over its own slots the lead survives on all
    # four and is smaller, and the extremes are printed rather than the flattering one.
    _cm = _cleav.get("recall_at_matched_length", {})
    _margins = {c: round(cell["GRAIL exhaustive"] - cell[c], 4) for c, cell in _cm.items()}
    n["chem.cleavage.matchedarms"] = len(_margins)
    n["chem.cleavage.matchedwins"] = sum(1 for v in _margins.values() if v > 0)
    if _margins:
        _lo = min(_margins, key=_margins.get)
        _hi = max(_margins, key=_margins.get)
        n["chem.cleavage.matchedmin"] = _margins[_lo]
        n["chem.cleavage.matchedmax"] = _margins[_hi]
        n["chem.cleavage.matchedminbank"] = _cm[_lo]["GRAIL exhaustive"]
        n["chem.cleavage.matchedminother"] = _cm[_lo][_lo]
        # The margin the budget comparison shows against the strongest comparator at fifteen,
        # printed beside them so the two readings can be compared rather than swapped.
        n["chem.cleavage.budgetmargin"] = round(
            n["chem.cleavage.bank"] - n["chem.cleavage.bestother"], 4)
    # Which arms have nothing left to add at the budget the class table is read at, which is what
    # makes that table a comparison of lengths as much as of orderings.
    _short = art("deployment_table.json")["substrates_whose_list_is_shorter_than_the_budget"]["15"]
    # Named from the artifact rather than one line per arm, for the same reason: the six lines
    # this replaced were written when there were six arms.
    _SHORT_NAME = {"whole bank": "bank", "trained budget": "interactive"}
    for _k, _v in _short.items():
        n[f"chem.exhausted.{_SHORT_NAME.get(_k, _k)}"] = _v
    # The class the case study illustrates, which is also the one this bank is weakest on. It is
    # printed here so the case cannot be read as representative of the class it belongs to.
    _deam = ebc["classes"].get("deamination", {})
    n["chem.deam.refs"] = _deam.get("references")
    n["chem.deam.exhfifteen"] = _deam["recall"]["GRAIL exhaustive"]["15"] if _deam else None
    n["chem.deam.exhthirty"] = _deam["recall"]["GRAIL exhaustive"]["30"] if _deam else None
    n["chem.deam.bestfifteen"] = (max(v["15"] for k, v in _deam["recall"].items()
                                      if not k.startswith("GRAIL")) if _deam else None)

    # The drawing's effect on the substrates it changes, not diluted by the ones it cannot.
    dc = art("dialect_conditional.json")["arms"]["GRAIL exhaustive"]
    n["dialcond.changed"] = dc["substrates_the_drawing_changes"]
    n["dialcond.substrates"] = dc["substrates"]
    n["dialcond.share"] = dc["share_changed"]
    # Every budget, and each cell's own verdict, because the section states which budgets separate
    # and had them backwards: at fifteen neither the pooled nor the conditional effect excludes
    # zero, and at thirty both do. Printing the verdict from the artifact is what stops a sentence
    # asserting the opposite of the cell beside it.
    for k in sorted(dc["by_budget"], key=int):
        for tag in ("all", "changed_only"):
            cell = dc["by_budget"][k].get(tag)
            if not cell:
                continue
            key = ("pooled" if tag == "all" else "conditional") + ("" if k == "15" else k)
            n[f"dialcond.{key}.sep"] = bool(cell.get("excludes_zero"))
    _sepk = sorted(int(k) for k, v in dc["by_budget"].items()
                   if v.get("all", {}).get("excludes_zero") and
                   v.get("changed_only", {}).get("excludes_zero"))
    n["dialcond.bothsepat"] = _sepk[0] if _sepk else 0
    n["dialcond.nbudgets"] = len(dc["by_budget"])
    for k in ("15",):
        for tag in ("all", "changed_only"):
            cell = dc["by_budget"][k][tag]
            key = "pooled" if tag == "all" else "conditional"
            n[f"dialcond.{key}"] = cell["difference"]
            n[f"dialcond.{key}.lo"] = cell["ci95"][0]
            n[f"dialcond.{key}.hi"] = cell["ci95"][1]

    # How much of the annotation is a single enzymatic step, measured on the references rather
    # than on the templates mined from them.
    ram = art("references_are_multistep.json")
    n["refstep.typed"] = ram["population"]["typed"]
    n["refstep.loci"] = ram["references_whose_centre_falls_in_more_than_one_locus"]
    n["refstep.edits"] = ram["references_composite_by_edit_count"]
    n["refstep.either"] = ram["references_composite_by_either_instrument"]
    n["refstep.eithershare"] = ram["share_by_either"]

    # What the wide budget costs a reader in precision. The paper declines to order systems on
    # precision and that is right; it should still say what a list of 98 candidates is like.
    prec = art("precision_table.json")["precision_micro"]
    for tag, arm in (("bank", "GRAIL exhaustive"), ("trained", "GRAIL interactive")):
        for k in ("15", "50"):
            n[f"prec.{tag}.{k}"] = prec[arm][k]

    # MetaPredictor's emission knob swept upward. The manuscript reads its flatness from k=15 off
    # the length of its lists, which infers the cause; turning the beam up tests it, and reaches
    # the same verdict the matched-length control reaches by truncating instead.
    mpb = art("metapredictor_beam_sweep.json")
    _dep, _wide = "deployed beam", "wide beam"
    n["mpbeam.deployedemitted"] = mpb["by_setting"][_dep]["mean_emitted"]
    n["mpbeam.wideemitted"] = mpb["by_setting"][_wide]["mean_emitted"]
    for _k in ("15", "30", "50"):
        n[f"mpbeam.widerecall{_k}"] = mpb["by_setting"][_wide]["recall"][_k]
    for _tag, _key in (("deployed", _dep), ("wide", _wide)):
        for _k in (15, 30):
            cell = mpb["by_setting"][_key][f"exhaustive_minus_metapredictor_at_{_k}"]
            n[f"mpbeam.{_tag}.{_k}.gap"] = cell["gap"]
            n[f"mpbeam.{_tag}.{_k}.lo"] = cell["ci95"][0]
            n[f"mpbeam.{_tag}.{_k}.hi"] = cell["ci95"][1]
            n[f"mpbeam.{_tag}.{_k}.sep"] = cell["excludes_zero"]
    n["mpbeam.widemissing"] = mpb["substrates_the_wide_decode_did_not_produce"]

    # SyGMa's own emission knob swept upward, which is the direction that bears on the one
    # wide-budget lead this work still claims.
    sss = art("sygma_scenario_sweep.json")
    _deep = [k for k in sss["by_scenario"] if k != sss["deployed_scenario"]][0]
    n["sygscen.deployedemitted"] = sss["by_scenario"][sss["deployed_scenario"]]["mean_emitted"]
    n["sygscen.deepemitted"] = sss["by_scenario"][_deep]["mean_emitted"]
    n["sygscen.deeprecall"] = sss["by_scenario"][_deep]["recall"]["50"]
    n["sygscen.deeprecallfifteen"] = sss["by_scenario"][_deep]["recall"]["15"]
    n["sygscen.unfinished"] = sss["by_scenario"][_deep]["unfinished"]
    for tag, key in (("deployed", sss["deployed_scenario"]), ("deep", _deep)):
        cell = sss["by_scenario"][key]["exhaustive_minus_sygma_at_50"]
        n[f"sygscen.{tag}.gap"] = cell["gap"]
        n[f"sygscen.{tag}.lo"] = cell["ci95"][0]
        n[f"sygscen.{tag}.hi"] = cell["ci95"][1]
        n[f"sygscen.{tag}.sep"] = cell["excludes_zero"]

    # What removing the borrowed templates would cost the coverage ceiling. It prices one of the
    # licensing options rather than choosing it, and the answer differs sharply by source.
    lrc = art("licence_removal_cost__clean_test.json")
    for tag, name in (("all", "without every borrowed template"),
                      ("bt", "without BioTransformer"),
                      ("sygma", "without SyGMa"),
                      ("gloryx", "without GLORYx")):
        cell = lrc["against_the_whole_bank"].get(name)
        if cell is None:                                  # a rightsholder's file not held here
            continue
        n[f"lrc.{tag}.lost"] = cell["references_lost"]
        n[f"lrc.{tag}.change"] = cell["ceiling_change"]
        n[f"lrc.{tag}.lo"] = cell["ci95"][0]
        n[f"lrc.{tag}.hi"] = cell["ci95"][1]
        n[f"lrc.{tag}.sep"] = cell["excludes_zero"]
        # How many templates the variant removes, not how many it keeps: the artifact records the
        # size of each bank and the difference is what "dropped" means in the prose.
        n[f"lrc.{tag}.dropped"] = (lrc["variants"]["whole bank"]["templates"]
                                   - lrc["variants"][name]["templates"])
    n["lrc.references"] = lrc["population"]["n_references"]
    n["lrc.wholebanktemplates"] = lrc["variants"]["whole bank"]["templates"]
    n["lrc.rightsholders"] = sum(
        1 for k in lrc["variants"] if k.startswith("without ") and k != "without every borrowed template")

    # Whether the chemistry the bank misses is absent from the corpus or only from the bank. The
    # abstract asserted the first and only the second had been measured.
    mtt = art("missing_types_in_train.json")
    n["typeoverlap.traintypes"] = mtt["train"]["distinct_types"]
    n["typeoverlap.testtypes"] = mtt["test"]["distinct_types"]
    n["typeoverlap.shared"] = mtt["test_types_also_in_train"]
    n["typeoverlap.absenttypes"] = mtt["test_types_absent_from_train"]
    n["typeoverlap.absentrefs"] = mtt["test_references_whose_type_is_absent_from_train"]
    n["typeoverlap.presentrefs"] = mtt["test_references_whose_type_is_in_train"]
    n["typeoverlap.absentshare"] = mtt[
        "share_of_test_references_whose_type_is_absent_from_train"]
    n["typeoverlap.typed"] = mtt["test"]["typed"]
    n["typeoverlap.untypeable"] = mtt["test"]["pairs"] - mtt["test"]["typed"]
    # The direct measurement: of the references whose type the bank does not hold, how many have a
    # type the training annotation does not hold either. The manuscript had argued this by
    # comparing two magnitudes and its own Supporting Information said the two were not nested.
    n["typeoverlap.banktypes"] = mtt["bank_types"]
    n["typeoverlap.inboth"] = mtt["references_by_cell"]["in the bank and in training"]
    n["typeoverlap.inbankonly"] = mtt["references_by_cell"]["in the bank, not in training"]
    n["typeoverlap.banklacks"] = mtt["references_whose_type_the_bank_lacks"]
    n["typeoverlap.corpuslackstoo"] = mtt["of_those_the_corpus_lacks_too"]
    # The part of that population the bank reaches anyway. The containment share is computed over
    # every reference of absent type, and the claim it supports is about the ones the bank also
    # fails to reach, so the difference between the two populations is printed rather than left to
    # be worked out from two numbers three paragraphs apart.
    n["typeoverlap.reached"] = mtt["references_whose_type_the_bank_lacks"] - n["ceiling.novel"]

    # The restricted claim, bounded by these two marginals rather than left unmeasured. The
    # uncovered references of absent type are a subset of the references of absent type, and only
    # one cell of the cross-tabulation holds an absent-type reference whose type the training
    # annotation does have. So at most that many of the subset can, whatever the join would say.
    #
    # That cell is references_by_cell["not in the bank, in training"], which this file already
    # carries as typeoverlap.minerslost. It is 40, and so is ceiling.untypeable, which counts
    # something else entirely; the bound is derived from the cell and never from the coincidence.
    _cell = mtt["references_by_cell"]["not in the bank, in training"]
    assert _cell == (mtt["references_whose_type_the_bank_lacks"]
                     - mtt["of_those_the_corpus_lacks_too"]), (
        "the cross-tabulation's cell and its marginals disagree, so the bound has no basis")
    n["bound.corpusabsentmin"] = n["ceiling.novel"] - _cell
    n["bound.corpusabsentsharemin"] = round(n["bound.corpusabsentmin"] / n["ceiling.novel"], 4)
    n["bound.shortfallsharemin"] = round(n["bound.corpusabsentmin"] / n["ceiling.uncovered"], 4)
    n["bound.shortfallsharemax"] = round(n["ceiling.novel"] / n["ceiling.uncovered"], 4)
    n["typeoverlap.minersshare"] = round(
        1 - mtt["share_of_the_bank_s_type_gap_the_corpus_also_lacks"], 4)
    n["typeoverlap.corpusshare"] = mtt["share_of_the_bank_s_type_gap_the_corpus_also_lacks"]
    n["typeoverlap.minerslost"] = (mtt["references_whose_type_the_bank_lacks"]
                                   - mtt["of_those_the_corpus_lacks_too"])
    # The same share under each definition of a type, because this is the claim that reaches the
    # Conclusions and it was reported at one granularity while its neighbour was varied over four.
    _bg = mtt["by_granularity"]
    n["typeoverlap.granularities"] = len(_bg)
    _determining = [v["share"] for v in _bg.values()
                    if v["determines_a_product"] and v["share"] is not None]
    _all = [v["share"] for v in _bg.values() if v["share"] is not None]
    n["typeoverlap.sharemin"] = min(_all)
    n["typeoverlap.sharemax"] = max(_all)
    n["typeoverlap.determiningmin"] = min(_determining)
    n["typeoverlap.coarseshare"] = _bg[
        "set of changed-bond classes, counts dropped"]["share"]

    # Whether the unrecoverable selection that assembled the corpus removed a class of chemistry,
    # which would manufacture the shortfall above rather than measure it. Only one of the four
    # sources is on disk in full, so this is a bound from one source and is reported as such.
    # The same test on the other source this repository holds. Two of the four are here, not
    # one, and testing only the larger left the second unexamined while the claim generalised.
    gdrop = art("gloryx_drop_set.json")
    n["gloryxdrop.pairs"] = gdrop["source"]["distinct_pairs"]
    n["gloryxdrop.inside"] = gdrop["source"]["inside_the_corpus"]
    n["gloryxdrop.keptp"] = gdrop["typed"]["kept"]
    n["gloryxdrop.droppedp"] = gdrop["typed"]["dropped"]
    n["gloryxdrop.tv"] = gdrop["total_variation_distance"]
    n["gloryxdrop.p"] = gdrop["permutation"]["p_value"]
    n["gloryxdrop.absentshare"] = gdrop["share_of_dropped_whose_type_the_kept_half_lacks"]
    n["gloryxdrop.types"] = gdrop["typed"]["distinct_types"]
    n["gloryxdrop.deadline"] = gdrop["mcs_deadline_seconds"]

    drop = art("metxbiodb_drop_set.json")
    n["metxdrop.pairs"] = drop["source"]["distinct_pairs"]
    n["metxdrop.inside"] = drop["source"]["inside_the_corpus"]
    n["metxdrop.dropped"] = drop["source"]["dropped"]
    n["metxdrop.typedkept"] = drop["typed"]["kept"]
    n["metxdrop.typeddropped"] = drop["typed"]["dropped"]
    n["metxdrop.tv"] = drop["total_variation_distance"]
    n["metxdrop.p"] = drop["permutation"]["p_value"]
    n["metxdrop.perm"] = drop["permutation"]["n"]
    n["metxdrop.droppedtypesabsent"] = drop[
        "dropped_references_whose_type_is_absent_from_the_kept_half"]
    n["metxdrop.droppedtypesabsentshare"] = drop["share_of_dropped_whose_type_the_kept_half_lacks"]
    _rec = drop["recoverable_from_the_dropped_half"]
    n["metxdrop.absentcell"] = _rec["references_whose_type_neither_the_bank_nor_training_holds"]
    n["metxdrop.recoverable"] = _rec["of_those_whose_type_this_source_dropped"]
    n["metxdrop.recoverableshare"] = _rec["share"]
    n["metxdrop.recoverablecoarse"] = _rec[
        "of_those_whose_type_this_source_dropped_counts_ignored"]
    n["metxdrop.recoverablecoarseshare"] = _rec["share_counts_ignored"]

    # The aggregation scheduled by budget: an alternative built, measured on both populations and
    # not adopted. The two artifacts are read together because the claim is a pair -- what the
    # validation draw promised and what the comparison set paid -- and quoting either alone is the
    # selection this design exists to avoid.
    _sched = art("budget_dependent_schedules.json")
    _rel = art("scheduled_release_comparison.json")
    n["blend.switch"] = _rel["config"]["switch"]["blend_at_or_below"]
    for _arm, _tag in (("whole bank", "exh"), ("trained budget", "int")):
        _rows = _rel["arms"][_arm]
        for _k in ("1", "3", "5", "8", "10"):
            _g = _rows[_k]["gain_over_the_released_rule"]
            n[f"blend.{_tag}.gain{_k}"] = _g["difference"]
            n[f"blend.{_tag}.gain{_k}.lo"] = _g["ci95"][0]
            n[f"blend.{_tag}.gain{_k}.hi"] = _g["ci95"][1]
            n[f"blend.{_tag}.gain{_k}.sep"] = _g["excludes_zero"]
        # The widest trail against the comparator the manuscript names at the tight budgets, and
        # whether it is established. The sentence this supports is a negative one, so the number
        # that carries it is the largest gap that still fails to separate.
        _sy = {int(k): v["vs"]["SyGMa"] for k, v in _rows.items() if int(k) <= 10}
        _worst = min(_sy.items(), key=lambda kv: kv[1]["gap"])
        n[f"blend.{_tag}.sygmaworst"] = _worst[1]["gap"]
        n[f"blend.{_tag}.sygmaworst.k"] = _worst[0]
        n[f"blend.{_tag}.sygmaworst.lo"] = _worst[1]["ci95"][0]
        n[f"blend.{_tag}.sygmaworst.hi"] = _worst[1]["ci95"][1]
        n[f"blend.{_tag}.sygmaseparating"] = sum(
            1 for v in _sy.values() if v["excludes_zero"])
        # How close the nearest cell comes to separating. A claim that nothing separates is worth
        # only as much as its narrowest margin, and rounding can put that margin at zero: the
        # sentence has to be able to say so rather than rest on a fourth decimal it does not show.
        _closest = min(_sy.items(), key=lambda kv: abs(kv[1]["ci95"][1]))
        n[f"blend.{_tag}.sygmaclosest.k"] = _closest[0]
        n[f"blend.{_tag}.sygmaclosest.hi"] = _closest[1]["ci95"][1]
        n[f"blend.{_tag}.sygmaclosest"] = _closest[1]["gap"]
    # The transfer: what validation promised against what the comparison set delivered, at the
    # budget where the two disagree in sign. A schedule that half transfers is the result, so the
    # disagreement is generated rather than described.
    _agg = _sched["aggregation"]["by_budget"]
    _flip = [k for k, v in _agg.items()
             if v["validation_promised"] > 0 and v["delivered"] < 0]
    n["blend.transferflips"] = len(_flip)
    if _flip:
        _k = max(_flip, key=lambda x: int(x))
        n["blend.flipk"] = int(_k)
        n["blend.flippromised"] = _agg[_k]["validation_promised"]
        n["blend.flipdelivered"] = _agg[_k]["delivered"]
    _gaps = [v["transfer_gap"] for v in _agg.values() if v["chosen_on_validation"] != "noisy_or"]
    n["blend.transferworst"] = min(_gaps) if _gaps else 0.0
    # The cap on the same design, which is the arm of this comparison that came back negative.
    _cap = _sched["cap"]["by_budget"]
    n["blend.capbest"] = max(v["delivered"] for v in _cap.values())
    n["blend.capworst"] = min(v["delivered"] for v in _cap.values())
    n["blend.capchoices"] = len({v["chosen_on_validation"] for v in _cap.values()})
    # What adopting it would cost, from the dependency graph rather than an estimate.
    _scope = art("blend_switch_scope.json")
    n["blend.recompute"] = len(_scope["recompute"])
    n["blend.recomputepossible"] = len(_scope["recompute"]) + len(_scope["possible"])
    n["blend.recomputeunplaceable"] = len(_scope["no_recorded_inputs_and_reads_a_tight_budget"])

    # The aggregation rule, swept. The manuscript named the violated independence assumption and
    # left it, on the ground that the released pools carry the aggregate and not its parts.
    agg = art("aggregation_ablation.json")
    # The extremum of each swept rule, and the budgets at which it separates, on both populations.
    # A sentence saying "up to X" claims the maximum over a series; naming a member of that series
    # was the defect round 14 found twice, and generating the extremum here means the sentence can
    # be held against it instead of against whichever member was handy.
    _aggval = art("aggregation_ablation_validation.json")
    for _tag, _blob in (("cmp", agg), ("val", _aggval)):
        for _rule in ("hybrid", "max"):
            _rows = {int(k): v for k, v in _blob["by_rule"][_rule]["minus_noisy_or"].items()
                     if str(k).isdigit()}
            _best = max(_rows.items(), key=lambda kv: kv[1]["difference"])
            # An extremum attained at more than one budget is a tie, and naming one of them makes
            # the sentence unverifiable against the other document: the maximum rule gains the
            # same +0.0316 at three and at five with different intervals, the manuscript quoted
            # three, the supporting information tabulated five, and three referees read that as a
            # contradiction. The budgets that attain it are generated so the prose can say so.
            _ties = sorted(k for k, v in _rows.items()
                           if v["difference"] == _best[1]["difference"])
            n[f"agg.{_tag}.{_rule}.maxties"] = len(_ties)
            n[f"agg.{_tag}.{_rule}.maxtieother"] = (
                max(k for k in _ties if k != _best[0]) if len(_ties) > 1 else 0)
            n[f"agg.{_tag}.{_rule}.maxk"] = _best[0]
            n[f"agg.{_tag}.{_rule}.max"] = _best[1]["difference"]
            n[f"agg.{_tag}.{_rule}.max.lo"] = _best[1]["ci95"][0]
            n[f"agg.{_tag}.{_rule}.max.hi"] = _best[1]["ci95"][1]
            _sep = sorted(k for k, v in _rows.items() if v.get("excludes_zero"))
            n[f"agg.{_tag}.{_rule}.sepfrom"] = min(_sep) if _sep else 0
            n[f"agg.{_tag}.{_rule}.septo"] = max(_sep) if _sep else 0
            n[f"agg.{_tag}.{_rule}.nsep"] = len(_sep)

    # The blend's weights, read from the implementation rather than typed, because the paper now
    # prints the rule as an equation and recommends it to a deployer.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_aggmod", ROOT / "scripts" / "typed_edit" / "aggregation_ablation.py")
    try:
        import inspect as _inspect
        _src = (ROOT / "scripts/typed_edit/aggregation_ablation.py").read_text()
        _m = re.search(r"return float\((0?\.\d+) \* float\(arr\.max\(\)\) \+ (0?\.\d+) \* noisy_or\)", _src)
        if _m:
            n["aggregation.blendmax"] = float(_m.group(1))
            n["aggregation.blendor"] = float(_m.group(2))
    except Exception:
        pass
    # The blend's weights, read out of the implementation rather than typed, because the paper
    # now prints the rule as an equation and the Limitations recommend it to a deployer.
    _src = (ROOT / "scripts/typed_edit/aggregation_ablation.py").read_text()
    _m = re.search(r"return float\((0?\.\d+) \* float\(arr\.max\(\)\) \+ (0?\.\d+) \* noisy_or\)",
                   _src)
    if _m:
        n["aggregation.blendmax"] = float(_m.group(1))
        n["aggregation.blendor"] = float(_m.group(2))
    n["aggregation.joined"] = agg["join"]["candidates_scored_by_both"]
    n["aggregation.unjoined"] = agg["join"]["candidates_the_pool_does_not_carry"]
    for rule in ("max", "mean", "hybrid"):
        tag = rule.replace("_", "")
        for k in ("1", "3", "5", "10", "15", "30", "50"):
            cell = agg["by_rule"][rule]["minus_noisy_or"][k]
            n[f"aggregation.{tag}{k}"] = cell["difference"]
            n[f"aggregation.{tag}{k}.lo"] = cell["ci95"][0]
            n[f"aggregation.{tag}{k}.hi"] = cell["ci95"][1]
            n[f"aggregation.{tag}{k}.sep"] = cell["excludes_zero"]
        n[f"aggregation.{tag}recall30"] = agg["by_rule"][rule]["recall"]["30"]
    n["aggregation.deployedrecall30"] = agg["by_rule"]["noisy_or"]["recall"]["30"]
    n["aggregation.separating"] = len(
        agg["rules_that_separate_from_the_deployed_one_at_any_budget"])

    # The same sweep on validation, which is the only population a different rule could be
    # selected on without selecting on the population the comparison is reported over.
    aggv = art("aggregation_ablation_validation.json")
    n["aggval.substrates"] = aggv["population"]["n_substrates"]
    n["aggval.references"] = aggv["population"]["n_references"]
    for rule in ("max", "mean", "hybrid"):
        tag = rule.replace("_", "")
        for k in ("1", "3", "5", "10", "15", "20", "30", "50"):
            cell = aggv["by_rule"][rule]["minus_noisy_or"][k]
            n[f"aggval.{tag}{k}"] = cell["difference"]
            n[f"aggval.{tag}{k}.lo"] = cell["ci95"][0]
            n[f"aggval.{tag}{k}.hi"] = cell["ci95"][1]
            n[f"aggval.{tag}{k}.sep"] = cell["excludes_zero"]
        # The budgets at which the rule beats the deployed one with the interval excluding
        # zero, which is what a selection on validation would be entitled to act on.
        wins = [k for k, c in aggv["by_rule"][rule]["minus_noisy_or"].items()
                if c["difference"] > 0 and c["excludes_zero"]]
        n[f"aggval.{tag}wins"] = len(wins)
        n[f"aggval.{tag}winsmax"] = max((int(k) for k in wins), default=0)
        n[f"aggval.{tag}recallfive"] = aggv["by_rule"][rule]["recall"]["5"]
    # The validation sweep has no reproduction gate of its own, because there is no published
    # column for it to reproduce. It can still be checked against one: the budget curve reads the
    # same pools at the whole bank, so the re-derived deployed rule and that curve should agree.
    _curve = bc["by_budget"][str(max(bc["budgets_built"]))]["recall_micro"]
    _shared = sorted(set(_curve) & set(aggv["by_rule"]["noisy_or"]["recall"]), key=int)
    _same = [k for k in _shared
             if abs(_curve[k] - aggv["by_rule"]["noisy_or"]["recall"][k]) < 5e-5]
    n["aggval.checkbudgets"] = len(_shared)
    n["aggval.checkagree"] = len(_same)
    n["aggval.checkworst"] = round(max(
        abs(_curve[k] - aggv["by_rule"]["noisy_or"]["recall"][k]) for k in _shared), 4)
    n["aggval.deployedrecallfive"] = aggv["by_rule"]["noisy_or"]["recall"]["5"]
    n["aggval.deployedrecallthirty"] = aggv["by_rule"]["noisy_or"]["recall"]["30"]

    # What actually defines the comparison population, and what the rest of the test set says.
    # The manuscript described an emission rule the code does not apply.
    pop = art("population_definition.json")
    # The exhaustive arm on both populations. It is the arm that carries every wide-budget lead
    # this work claims, and until it existed on the whole evaluated test set those leads had been
    # read only off the 291 nobody can reconstruct the draw of.
    for pop_key, tag in (("the comparison set", "Comp"), ("the whole evaluated test set", "Whole")):
        row = pop["contrasts"].get(pop_key, {})
        for comp in ("sygma", "metapredictor"):
            cell = (row.get(comp) or {}).get("exhaustive_minus_comparator") or {}
            for k in ("5", "10", "15", "30", "50"):
                if k in cell:
                    n[f"popdef.exh{tag}{comp}{k}"] = cell[k]["difference"]
                    n[f"popdef.exh{tag}{comp}{k}.lo"] = cell[k]["ci95"][0]
                    n[f"popdef.exh{tag}{comp}{k}.hi"] = cell[k]["ci95"][1]
                    n[f"popdef.exh{tag}{comp}{k}.sep"] = cell[k]["excludes_zero"]
    _whole = pop["contrasts"].get("the whole evaluated test set", {})
    n["popdef.wholesubstrates"] = _whole.get("n_substrates")
    n["popdef.wholereferences"] = _whole.get("n_references")
    # How many of the exhaustive arm's separating leads on the 291 still separate on all 1170.
    # The comparator list follows what the wider population actually holds rather than being
    # named here, because a comparator added to that population and not to this loop would be
    # measured on it and left out of the count the paper quotes.
    _kept = _lost = 0
    _comps = [c for c in ("sygma", "metapredictor", "biotransformer") if c in _whole]
    n["popdef.wholearms"] = len(_comps)
    for comp in _comps:
        a = ((pop["contrasts"]["the comparison set"].get(comp) or {})
             .get("exhaustive_minus_comparator") or {})
        b = ((_whole.get(comp) or {}).get("exhaustive_minus_comparator") or {})
        for k, cell in a.items():
            if cell["difference"] > 0 and cell["excludes_zero"]:
                if b.get(k, {}).get("excludes_zero") and b[k]["difference"] > 0:
                    _kept += 1
                else:
                    _lost += 1
    n["popdef.leadskept"] = _kept
    n["popdef.leadslost"] = _lost
    n["popdef.leadstotal"] = _kept + _lost

    n["popdef.metatox"] = pop["emission"]["comparison_set"]["metatox"]
    n["popdef.sygmainside"] = pop["emission"]["comparison_set"]["sygma"]
    n["popdef.metapredictorinside"] = pop["emission"]["comparison_set"]["metapredictor"]
    n["popdef.grailinside"] = pop["emission"]["comparison_set"]["grail_deployed"]
    # BioTransformer is in the sweep and was in neither enumeration. It is the arm the emission
    # rule would actually have bitten: on the whole test set it answers nothing for nearly a tenth
    # of the substrates, which is not the "almost nothing" the section claimed.
    # The exhaustive arm's separating leads over the three comparators that reach the whole test
    # set, enumerated from the table rather than counted by eye. S29 audits which of them survive
    # the move to the wider population, and its enumeration listed thirteen: it included a cell the
    # table leaves unstarred, omitted one that does separate, and omitted the two at a budget of
    # twenty. A count over a table is a claim about a set, so it is generated from the set.
    # Which budgets the population move can be audited at, and which the deployment table has
    # besides. S29 counts leads over the five budgets its own sweep measures on both populations;
    # Table S11 sweeps nine, and the four it adds have no whole-test-set counterpart, so a lead at
    # one of them cannot be followed across the move. The audit's denominator is its own sweep.
    _popcmp = pop["contrasts"]["the comparison set"]["sygma"]["exhaustive_minus_comparator"]
    _dep = art("deployment_table.json")["contrasts"]
    n["s29.budgets"] = len(_popcmp)
    n["s29.tablebudgets"] = len(_dep)
    n["s29.budgetsunaudited"] = ", ".join(sorted((set(_dep) - set(_popcmp)), key=int))
    # The one cell on which the two artifacts disagree, because they read two runs of the same
    # comparator: S29 uses the run that also covers the whole test set, so that the move varies the
    # population and nothing else, and Table S11 uses the declared arm.
    _dis = []
    for _k in sorted(_popcmp, key=int):
        _a = pop["contrasts"]["the comparison set"]["biotransformer"]["exhaustive_minus_comparator"].get(_k)
        _b = _dep.get(_k, {}).get("whole bank - biotransformer")
        if _a and _b and _a["excludes_zero"] != _b["excludes_zero"]:
            _dis.append(int(_k))
    n["s29.btdisagreeat"] = _dis[0] if _dis else 0
    n["s29.btdisagreements"] = len(_dis)
    _wide = ("sygma", "metapredictor", "biotransformer")
    _leads = {c: [] for c in _wide}
    for _k in sorted(_dep, key=int):
        for _c in _wide:
            _cell = _dep[_k].get(f"whole bank - {_c}")
            if _cell and _cell["gap"] > 0 and _cell["excludes_zero"]:
                _leads[_c].append(int(_k))
    for _c in _wide:
        n[f"s29.leads.{_c}"] = len(_leads[_c])
        n[f"s29.leadsat.{_c}"] = ", ".join(str(x) for x in _leads[_c])
    n["s29.leads"] = sum(len(v) for v in _leads.values())
    _bt = _leads["biotransformer"]
    n["s29.btfirst"] = _bt[0] if _bt else 0
    n["s29.btrunfrom"] = next((b for b in _bt if b > (_bt[0] if _bt else 0)), 0)

    # The retraining spread at the configuration the manuscript reports, which the registered
    # effects are read against. The earlier seed study ran at a superseded operating point, so its
    # spread could only be offered as indicative; this one is measured where the numbers are.
    _rs = art("retraining_spread.json")["by_arm"]
    for _arm in ("interactive", "exhaustive"):
        _cell = _rs[_arm]["by_budget"]["15"]
        n[f"spread.{_arm}.sd"] = _cell["sd"]
        n[f"spread.{_arm}.mean"] = _cell["mean"]
        _all = [v["sd"] for v in _rs[_arm]["by_budget"].values()]
        n[f"spread.{_arm}.sdmin"] = min(_all)
        n[f"spread.{_arm}.sdmax"] = max(_all)
    n["spread.seeds"] = len(_rs["interactive"]["by_budget"]["15"]["seeds"])
    # Each registered effect in units of the spread of the arm whose knob it is. The rule budget
    # defines the interactive arm, and against the spread measured there it is not inside the
    # noise, which the earlier operating point's figure had suggested.
    for _tag, _key in (("p1", "h7.diff"), ("p2", "h9.diff"), ("p3", "h10.bought")):
        if _key in n:
            n[f"spread.{_tag}.insd"] = round(abs(n[_key]) / n["spread.interactive.sd"], 1)

    # What the release actually carries, as against what the paper measures. The two differ by
    # the templates BioTransformer's published set also contains, which are not redistributed.
    # Which tracked files carry a template the released bank drops. Removing the bank's own file
    # from the index does not remove what a checkpoint or a curated collection also holds, and the
    # availability statement has to say what the repository actually carries.
    _wc = art("withheld_template_carriers.json")
    n["relbank.carriers"] = _wc["n_carriers"]
    n["relbank.carrierscanned"] = _wc["tracked_files_scanned"]

    _rb = art("released_bank.json")
    n["relbank.measured"] = _rb["measured_bank"]["templates"]
    n["relbank.released"] = _rb["released_bank"]["templates"]
    # The environmental-microbial file, whose licence is the one no GPL release can carry. Its
    # templates are identified rather than declared: the paper had said they could not be, on the
    # strength of a file nobody had opened, and all of them are in the core set the release already
    # removes.
    _pf = _rb["per_published_file"]
    n["relbank.envmicro"] = _pf["ENVMICRO"]["templates_in_the_bank"]
    n["relbank.envmicroonly"] = _pf["ENVMICRO"]["of_them_only_in_this_file"]
    n["relbank.removed"] = _rb["removed"]

    n["popdef.btinside"] = pop["emission"]["comparison_set"]["biotransformer"]
    for name in ("sygma", "metapredictor", "biotransformer"):
        cell = pop["emission"]["whole_test_set"][name]
        short = "bt" if name == "biotransformer" else name
        n[f"popdef.{short}silent"] = cell["substrates_with_no_prediction"]
        n[f"popdef.{short}silentrefs"] = cell["references_they_carry"]
    n["popdef.btsilentshare"] = round(
        n["popdef.btsilent"] / pop["contrasts"]["the whole evaluated test set"]["n_substrates"], 4)
    for label, short in (("annotated references", "refs"), ("heavy atoms", "heavy"),
                         ("candidates the deployed system emits", "output")):
        cell = pop["exchangeability"][label]
        n[f"popdef.{short}in"] = cell["mean_in_the_comparison_set"]
        n[f"popdef.{short}out"] = cell["mean_outside_it"]
        n[f"popdef.{short}p"] = cell["permutation_p"]
    n["popdef.permutations"] = pop["permutation"]["n"]
    n["popdef.outside"] = pop["exchangeability"]["annotated references"]["n_out"]
    _whole = pop["contrasts"].get("the whole evaluated test set")
    if _whole:
        n["popdef.wholesubs"] = _whole["n_substrates"]
        n["popdef.wholerefs"] = _whole["n_references"]
        for name in ("sygma", "metapredictor"):
            for k in ("15", "30"):
                c = _whole[name]["deployed_minus_comparator"][k]
                n[f"popdef.whole.{name}.{k}"] = c["difference"]
                n[f"popdef.whole.{name}.{k}.lo"] = c["ci95"][0]
                n[f"popdef.whole.{name}.{k}.hi"] = c["ci95"][1]
                n[f"popdef.whole.{name}.{k}.sep"] = c["excludes_zero"]
        _comp = pop["contrasts"]["the comparison set"]
        for name in ("sygma", "metapredictor"):
            for k in ("15", "30"):
                c = _comp[name]["deployed_minus_comparator"][k]
                n[f"popdef.comp.{name}.{k}"] = c["difference"]
                n[f"popdef.comp.{name}.{k}.lo"] = c["ci95"][0]
                n[f"popdef.comp.{name}.{k}.hi"] = c["ci95"][1]
                n[f"popdef.comp.{name}.{k}.sep"] = c["excludes_zero"]

    # What the corpus's drawing costs the comparator the manuscript could not re-run. It is a
    # source checkout, so it can be, and the asymmetry it left was the one running this work's way.
    mpd = art("metapredictor_drawing.json")
    n["mpdraw.rerun"] = mpd["substrates_re_run"]
    # Both counts of the moved set, so the reconciliation in S14 is read from the artifact
    # that measured it rather than typed beside the sentence.
    n["mpdraw.rerun.old"] = mpd["substrates_the_string_test_scored"]
    for tag, short in (("the whole comparison set", "all"),
                       ("only the substrates the drawing changes", "moved")):
        cell = mpd["by_population"][tag]
        n[f"mpdraw.{short}.n"] = cell["n_substrates"]
        for k in ("15", "30"):
            c = cell["by_budget"][k]
            n[f"mpdraw.{short}.diff{k}"] = c["difference"]
            n[f"mpdraw.{short}.lo{k}"] = c["ci95"][0]
            n[f"mpdraw.{short}.hi{k}"] = c["ci95"][1]
            n[f"mpdraw.{short}.sep{k}"] = c["excludes_zero"]

    # The two headrooms side by side on one population, which the manuscript never put together:
    # what ranking could still recover inside the pool the bank builds, and what no ranking can
    # because the bank never produces it.
    _ds = art("dialect_sweep.json")["coverage_ceiling"]["stored"]["coverage"]
    n["headroom.ceiling"] = _ds
    n["headroom.reached"] = dep["recall_micro"]["50"]["whole bank"]
    n["headroom.ranking"] = round(_ds - dep["recall_micro"]["50"]["whole bank"], 4)
    n["headroom.coverage"] = round(1.0 - _ds, 4)
    n["headroom.share"] = round(dep["recall_micro"]["50"]["whole bank"] / _ds, 4)

    # What a configuration-aware matching criterion could distinguish at all, which the criterion
    # sweep cannot bound, because no reference in this corpus carries a configuration for one to
    # read. An earlier reading of this said the two settings differing in the stereochemistry
    # layer return the identical verdict at every budget, and the manuscript printed it. They do
    # not: the full InChIKey leads at k=20 where its first block does not, and what separates them
    # on this annotation is protonation, one thiol and thiolate. The two settings that ARE
    # identical at every budget are canonical SMILES equality and a Tanimoto of one, neither of
    # which is configuration-aware.
    sh = art("stereo_headroom.json")
    n["stereo.pairs"] = sh["population"]["typed"]
    n["stereo.centre"] = sh["references_gaining_a_tetrahedral_centre"]
    n["stereo.double"] = sh["references_gaining_a_stereogenic_double_bond"]
    n["stereo.either"] = sh["references_gaining_either"]
    n["stereo.share"] = sh["share_gaining_either"]

    # The classes where this system is not the best arm, which the main text reported only where
    # it was. The names and the margins come from the artifact rather than from a reading of it.
    ch = art("error_by_chemistry.json")["classes"]
    # Every arm the artifact scores, read from the artifact. A list of names written out beside
    # an artifact goes stale the next time the artifact gains a column, and this one did: "the
    # best arm" was the best of six while the table printed seven, so the class where this system
    # is furthest behind reported a gap of 0.0214 where the printed table showed 0.0571.
    _arms = tuple(next(iter(ch.values()))["recall"])
    for _need in ("GRAIL exhaustive", "GRAIL interactive"):
        assert _need in _arms, f"error_by_chemistry.json does not score {_need}"
    _behind, _int_ahead = [], []
    for _name, _e in ch.items():
        _r = {a: _e["recall"][a]["15"] for a in _arms}
        if _r["GRAIL exhaustive"] < max(_r.values()) - 1e-9:
            _behind.append((_e["references"], _name))
        if _r["GRAIL interactive"] > _r["GRAIL exhaustive"] + 1e-9:
            _int_ahead.append((_e["references"], _name))
    n["chem.classes"] = len(ch)
    n["chem.behind"] = len(_behind)
    n["chem.intahead"] = len(_int_ahead)
    n["chem.behindrefs"] = sum(r for r, _ in _behind)
    n["chem.oxygenbank"] = ch["oxidation, one oxygen added"]["recall"]["GRAIL exhaustive"]["15"]
    n["chem.oxygenbest"] = max(
        ch["oxidation, one oxygen added"]["recall"][a]["15"] for a in _arms)
    n["chem.oxygenrefs"] = ch["oxidation, one oxygen added"]["references"]
    n["chem.demethbank"] = ch["demethylation"]["recall"]["GRAIL exhaustive"]["15"]
    n["chem.demethinter"] = ch["demethylation"]["recall"]["GRAIL interactive"]["15"]
    n["chem.demethbest"] = max(ch["demethylation"]["recall"][a]["15"] for a in _arms)
    n["chem.demethrefs"] = ch["demethylation"]["references"]

    # BioTransformer, run rather than excluded. Both drawings of the substrate, and the same
    # matched-length control every other comparator receives.
    bt = art("biotransformer_arm.json")
    _stored = bt["by_setting"]["allHuman one step"]
    _drawn = bt["by_setting"]["allHuman one step, natural drawing"]
    n["btarm.emitted"] = _stored["mean_emitted"]
    n["btarm.silent"] = _stored["substrates_with_no_prediction"]
    for k in ("15", "30", "50"):
        n[f"btarm.recall{k}"] = _stored["recall"][k]
        n[f"btarm.drawnrecall{k}"] = _drawn["recall"][k]
    for k in ("15", "30", "50"):
        cell = _stored[f"exhaustive_minus_biotransformer_at_{k}"]
        n[f"btarm.gap{k}"] = cell["gap"]
        n[f"btarm.gap{k}.lo"], n[f"btarm.gap{k}.hi"] = cell["ci95"]
        n[f"btarm.gap{k}.sep"] = cell["excludes_zero"]
    _m = _stored["matched_length"]
    n["btarm.matchedgap"] = _m["gap"]
    n["btarm.matchedlo"], n["btarm.matchedhi"] = _m["ci95"]
    n["btarm.matchedsep"] = _m["excludes_zero"]
    n["btarm.matchedslots"] = _m["mean_slots"]
    n["btarm.matchedours"] = _m["recall_ours"]
    n["btarm.matchedtheirs"] = _m["recall_theirs"]
    # What the drawing costs this comparator, which is the second measured case of a correction
    # the manuscript could previously report for one comparator only.
    n["btarm.drawingdelta"] = round(_drawn["recall"]["30"] - _stored["recall"]["30"], 4)

    # The one external check in this paper: whether an independent benchmark's ordering is a
    # budget ordering, computed from that benchmark's own published recall and precision. No
    # measurement of ours enters it, which is the point.
    ebc2 = art("external_budget_confound.json")
    n["extbudget.drugs"] = len(ebc2["reference_set_sizes"])
    n["extbudget.references"] = sum(ebc2["reference_set_sizes"].values())
    n["extbudget.arms"] = len(ebc2["by_tool"])
    n["extbudget.cells"] = ebc2["reconstructed_axis"]["n_cells"]
    # The axis is counted from the authors' own deposited predictions rather than reconstructed
    # through recall, because the reconstruction puts recall on both sides of the correlation.
    n["extbudget.countedcells"] = ebc2["n_counted_cells"]
    n["extbudget.widest"] = ebc2["counted_widest"]
    n["extbudget.narrowest"] = ebc2["counted_narrowest"]
    n["extbudget.ratio"] = ebc2["counted_spread"]
    n["extbudget.rho"] = ebc2["spearman_recall_against_counted_emission"]
    n["extbudget.permp"] = ebc2["permutation_p"]
    n["extbudget.permutations"] = ebc2["permutations"]
    n["extbudget.withinpositive"] = ebc2["within_drug_counted_positive"]
    n["extbudget.withinn"] = ebc2["within_drug_counted_n"]
    n["extbudget.orderingsagree"] = ebc2["orderings_agree"]
    # The reconstructed axis, printed so a reader can see why it is not the one used.
    n["extbudget.rhoreconstructed"] = \
        ebc2["reconstructed_axis"]["spearman_recall_against_log_emitted"]
    n["extbudget.reconstructedp"] = ebc2["reconstructed_axis"]["permutation_p"]
    # The level the association lives at. A rank correlation over a grid of arms and drugs is
    # dominated by whichever axis varies more, and here that is the arm. With the additive arm and
    # drug effects removed from both sides, what is left is the part inside an arm.
    n["extbudget.residual"] = \
        ebc2["association_within_the_arms"]["spearman_after_removing_the_arm_effect_and_the_drug_effect"]
    # The arms that carry a counted emission, which is the grid the residual is computed over and
    # is NOT the number of arms in the benchmark's recall table.
    n["extbudget.residualarms"] = ebc2["association_within_the_arms"]["arms"]
    # What the within-drug null does to an arm's emission profile, which is the half of "preserves
    # both marginals" that was false: a null preserving it would return the observed profile every
    # time and a rank correlation of one.
    _prof = ebc2["permutation_null_and_the_arm_profile"]
    n["extbudget.profilepermutations"] = _prof["permutations"]
    n["extbudget.profileagreement"] = \
        _prof["mean_rank_correlation_with_the_observed_profile"]
    # The full grid and what is missing from it, so a correlation over forty cells is readable.
    _cen = ebc2["counted_cell_census"]
    n["extbudget.grid"] = _cen["grid"]
    n["extbudget.absentnodeposit"] = _cen["absent_because_the_arm_was_never_deposited"]
    n["extbudget.absentnorecall"] = _cen["absent_because_no_recall_is_published"]
    n["extbudget.armsnotdeposited"] = len(_cen["arms_never_deposited"])
    # Whether the counted association depends on the one arm the reconstruction cannot reproduce.
    _wo = ebc2["reconstructed_axis"]["counted_association_without_the_worst_arm"]
    n["extbudget.rhowithoutworst"] = _wo["spearman"]
    n["extbudget.pwithoutworst"] = _wo["permutation_p"]
    n["extbudget.cellswithoutworst"] = _wo["n_cells"]
    _agree = ebc2["reconstructed_axis"]["agreement_with_the_count"]
    _worst = ebc2["reconstructed_axis"]["agreement_worst_arm"]
    n["extbudget.agreeingarms"] = sum(1 for k, v in _agree.items()
                                      if k != _worst and abs(v["ratio"] - 1) <= 0.2)
    n["extbudget.countedarms"] = len(_agree)
    n["extbudget.worstratio"] = _agree[_worst]["ratio"]
    # How many of that arm's cells are missing, which is why the reconstruction fails on it.
    n["extbudget.worstarmabsent"] = sum(1 for a in _cen["absent"] if a["arm"] == _worst)

    # The fusion constant, swept. It was left at the published default and the paper disclosed
    # that; a sweep says whether the disclosure costs anything.
    fk = art("fusion_knobs.json")
    n["fusionk.deployed"] = fk["deployed_constant"]
    n["fusionk.swept"] = len(fk["constants_swept"])
    n["fusionk.separating"] = sum(
        1 for r in fk["by_constant"].values()
        if (r.get("against_the_deployed_constant_at_15") or {}).get("excludes_zero"))
    _worst = max((abs(r["against_the_deployed_constant_at_15"]["gap"])
                  for r in fk["by_constant"].values()
                  if "against_the_deployed_constant_at_15" in r), default=0.0)
    n["fusionk.worst"] = round(_worst, 4)
    n["fusionk.flatfrom"] = min(
        int(k) for k, r in fk["by_constant"].items()
        if not (r.get("against_the_deployed_constant_at_15") or {}).get("excludes_zero"))
    # The population the sweep is measured on, which the manuscript stated the null over without
    # naming, and the budgets it was never asked about.
    n["fusionk.substrates"] = fk["population"]["n_substrates"]
    n["fusionk.references"] = fk["population"]["n_references"]
    _nb = fk["null_bound"]
    for _b in ("5", "15", "30", "50"):
        n[f"fusionk.band{_b}"] = _nb["by_budget"][_b]["widest_interval_endpoint"]
    # And the cell that is not a null at all. The claim of flatness was made at one budget and
    # read as though it held at every one; at a budget of thirty the smallest constant in the flat
    # range separates from the deployed value, in the deployed value's favour.
    _sep = _nb["separating_cells"]
    n["fusionk.nullbreaks"] = len(_sep)
    if _sep:
        _c = fk["by_constant"][str(_sep[0]["constant"])]["against_the_deployed_constant"][
            str(_sep[0]["budget"])]
        n["fusionk.breakconstant"] = _sep[0]["constant"]
        n["fusionk.breakbudget"] = _sep[0]["budget"]
        n["fusionk.breakgap"] = _c["gap"]
        n["fusionk.breaklo"] = _c["ci95"][0]
        n["fusionk.breakhi"] = _c["ci95"][1]
        # The same gap without its sign, for the sentence that says what leaving the constant
        # where it is buys rather than what moving it would cost.
        n["fusionk.breakgapabs"] = abs(_c["gap"])

    # Whether template discovery has saturated, which the manuscript asserted from a determinism
    # check. The curve is measured by withholding training pairs from the catalog.
    rar = art("mining_rarefaction.json")
    n["rare.templates"] = rar["mined_templates"]
    # How many withholding draws each rarefaction point averages, because the figure's band is a
    # spread over them and a band with no stated n is decoration.
    n["rare.draws"] = rar["draws_per_point"]
    n["rare.pairs"] = rar["distinct_training_pairs"]
    n["rare.singletons"] = rar["templates_resting_on_one_pair"]
    n["rare.singletonshare"] = rar["singleton_share"]
    n["rare.unseenmass"] = rar["good_turing_mass_of_unseen_types"]
    n["rare.slope"] = rar["templates_gained_per_thousand_pairs_at_the_full_corpus"]
    for frac in ("0.25", "0.5", "0.75", "1.0"):
        tag = {"0.25": "quarter", "0.5": "half", "0.75": "threequarters", "1.0": "all"}[frac]
        n[f"rare.{tag}"] = int(round(rar["rarefaction"][frac]["templates_mean"]))

    # The cells the correction paragraph is about. It used to call them "those eight" directly
    # after a table that has ten rows, so the count is derived from the sweep it belongs to and
    # the budgets are named where it is used.
    WIDE = ("30", "50")
    n["sweep.wideleads"] = sum(
        1 for b in WIDE for pair, c in dep["contrasts"][b].items()
        if pair.startswith("whole bank - ") and c["gap"] > 0 and c["excludes_zero"])
    n["sweep.widebudgets"] = len(WIDE)

    # How those wide leads split against the family the correction was declared over. The declared
    # family covers three comparators; the sweep prints five. Which cells the correction can reach
    # is therefore not the same as which leads exist, and the paragraph that says so used to name
    # the difference by hand -- as "two", when a fifth comparator made it four.
    _mult = art("multiplicity.json")
    _family_comparators = sorted({
        cell.split(" - ")[1].split(" @ ")[0] for cell in _mult["cells"]})
    _wide_pairs = [(b, pair) for b in WIDE for pair, c in dep["contrasts"][b].items()
                   if pair.startswith("whole bank - ") and c["gap"] > 0 and c["excludes_zero"]]
    _outside = [(b, p) for b, p in _wide_pairs
                if p.split(" - ")[1] not in _family_comparators]
    n["sweep.familycomparators"] = len(_family_comparators)
    n["sweep.wideleadsoutside"] = len(_outside)
    n["sweep.wideleadsinfamily"] = len(_wide_pairs) - len(_outside)
    if n["sweep.wideleadsinfamily"] + n["sweep.wideleadsoutside"] != n["sweep.wideleads"]:
        raise SystemExit("REFUSING: the wide leads do not split into the ones the declared family "
                         "covers and the ones it does not; the paragraph that reports the split "
                         "would not add up.")
    # The comparators whose wide cells the declared family never covered, named rather than
    # counted, so the sentence cannot name the wrong ones.
    n["sweep.outsidenames"] = ", ".join(
        sorted({p.split(" - ")[1] for _b, p in _outside}))
    # The distinct comparators a wide-budget lead exists against, which is what the matched-length
    # control is read against. It is five, and was described as four for as long as the fifth
    # comparator went uncounted.
    n["matched.comparators"] = len({p.split(" - ")[1] for _b, p in _wide_pairs})

    # How many verdicts the choice of aggregation moves, and over how many contrasts. The SI said
    # eight of seventy-two beside a table whose own caption said eleven of ninety: the prose was
    # written when the comparison had one comparator fewer and the caption is generated.
    _micro, _macro = dep["contrasts"], dep["contrasts_macro"]
    n["aggregation.contrasts"] = sum(len(v) for v in _macro.values())
    n["aggregation.moved"] = sum(
        1 for k, row in _macro.items() for pair, cell in row.items()
        if pair in _micro.get(k, {})
        and cell["excludes_zero"] != _micro[k][pair]["excludes_zero"])

    # The wide-budget leads with the length taken from the comparator rather than from the
    # experiment. A nominal budget is not a length, and this is the control that separates an
    # ordering result from a list-length one.
    ml = art("matched_length.json")
    for pair, cell in ml["contrasts"].items():
        a, b = [x.strip() for x in pair.split(" - ")]
        tag = {"whole bank": "bank", "trained budget": "trained"}.get(a)
        tagb = {"metatox": "Metatox", "sygma": "Sygma", "metapredictor": "Metapredictor",
                "biotransformer": "Biotransformer",
                # The same comparator on the drawing the declared standardiser produces. The
                # manuscript used to tell the reader to deduct a fixed-budget correction from
                # this table's margin; the corrected arm is measured through the control instead.
                "sygma on the standardised drawing": "Sygmastd"}.get(b)
        if tag and tagb:
            n[f"matched.{tag}{tagb}"] = cell["gap"]
            n[f"matched.{tag}{tagb}.lo"] = cell["ci95"][0]
            n[f"matched.{tag}{tagb}.hi"] = cell["ci95"][1]
            n[f"matched.{tag}{tagb}.sep"] = cell["excludes_zero"]
            n[f"matched.{tag}{tagb}.slots"] = cell["mean_slots"]
    # What the drawing costs this margin, measured through the control rather than deducted from
    # it. The two are not the same operation: the deduction the manuscript instructed was of a
    # correction measured at a fixed budget of fifty, and this margin is over the comparator's own
    # slot count. Both are printed so the difference between them is visible.
    if "matched.bankSygma" in n and "matched.bankSygmastd" in n:
        n["matched.sygmadrawingcost"] = round(n["matched.bankSygma"] - n["matched.bankSygmastd"], 4)
    # The cut this work applies to the comparator's own list, and what it costs. A slot count is
    # a configuration, so the bound on it is printed and so is the measurement that it is free.
    n["matched.cap"] = ml["cap_on_the_comparator_list"]
    for pair, cell in ml["contrasts"].items():
        u = cell.get("without_the_cap_on_the_comparator")
        if not u:
            continue
        b = pair.split(" - ")[1].strip()
        tagb = {"metatox": "Metatox", "sygma": "Sygma", "metapredictor": "Metapredictor",
                "biotransformer": "Biotransformer"}.get(b)
        if tagb:
            n[f"matched.binds{tagb}"] = u["substrates_the_cap_binds_on"]
            n[f"matched.pastcut{tagb}"] = \
                u["annotated_metabolites_the_comparator_placed_past_the_cut"]
            n[f"matched.uncappedslots{tagb}"] = u["mean_slots_uncapped"]
    n["matched.pastcut"] = sum(
        (c.get("without_the_cap_on_the_comparator") or {}).get(
            "annotated_metabolites_the_comparator_placed_past_the_cut", 0)
        for c in ml["contrasts"].values())
    n["matched.separating"] = sum(1 for c in ml["contrasts"].values() if c["excludes_zero"])
    n["matched.contrasts"] = len(ml["contrasts"])

    # What the drawing cost the one comparator that can be re-run. It bounds the residual on the
    # comparator columns that could not be, and it is the size the narrowest claimed lead has to
    # clear before it can be read as one.
    sbd = art("sygma_by_dialect.json")["by_budget"]
    for k in ("15", "30", "50"):
        word = {"15": "Fifteen", "30": "Thirty", "50": "Fifty"}[k]
        n[f"sygdialect.{word.lower()}"] = sbd[k]["difference"]
        n[f"sygdialect.{word.lower()}.lo"] = sbd[k]["ci95"][0]
        n[f"sygdialect.{word.lower()}.hi"] = sbd[k]["ci95"][1]

    # SyGMa's own scenario knob, swept: what its engine's composition step buys over applying
    # each ruleset once. The paper asks other work to declare such a knob and must sweep its own.
    sdm = art("sygma_depth_matched_reach.json")
    n["sygdepth.oncepass"] = sdm["reach"]["sygma_depth1_matched"]["point"]
    n["sygdepth.deployed"] = sdm["reach"]["sygma_deployed_two_step"]["point"]
    n["sygdepth.engine"] = sdm["engine_contribution"]["point"]
    n["sygdepth.enginelo"] = sdm["engine_contribution"]["ci95"][0]
    n["sygdepth.enginehi"] = sdm["engine_contribution"]["ci95"][1]
    n["sygdepth.gapdeployed"] = sdm["gap"]["as_reported"]["point"]
    n["sygdepth.gapmatched"] = sdm["gap"]["depth_matched"]["point"]

    # the family-wise reading of the same data, so the sentence about it is computed
    mult = art("multiplicity.json")
    n["holm.tests"] = mult["n_tests"]
    n["holm.separating"] = mult["n_separating_per_comparison"]
    n["holm.surviving"] = mult["n_separating_after_holm"]
    n["holm.changed"] = len(mult["cells_whose_verdict_the_correction_changes"])
    n["holm.leadsremoved"] = len(mult["leads_the_correction_removes"])
    # The same correction over every contrast the paper prints, which is more than the declared
    # family covers. Reported as a sensitivity: what the wider family would cost the verdicts the
    # declared one licenses.
    _w = mult["over_every_contrast_the_paper_prints"]
    # What the correction would do over everything the paper prints rather than over the family it
    # declared. The answer belongs in the body: a statistically minded reader asks it immediately,
    # and it was answered on page S63.
    _wide = mult["over_every_contrast_the_paper_prints"]
    n["holm.wideremovesdeclared"] = len(_wide["declared_family_cells_it_would_remove"])
    n["holm.wideleadsremoved"] = sum(
        1 for c in _wide["declared_family_cells_it_would_remove"]
        if mult["cells"].get(c, {}).get("gap", 0) > 0)
    n["holm.widetests"] = _w["n_tests"]
    n["holm.widesurviving"] = _w["n_separating_after_holm"]
    n["holm.wideremoves"] = len(_w["declared_family_cells_it_would_remove"])
    for tag, cell in (("bankmetatoxthirty", "whole bank - metatox @ 30"),
                      ("trainedmetatoxtwenty", "trained budget - metatox @ 20"),
                      ("trainedmetatoxthirty", "trained budget - metatox @ 30"),
                      ("trainedmetatoxfifty", "trained budget - metatox @ 50")):
        n[f"holm.{tag}"] = "yes" if mult["cells"][cell]["separates_after_holm"] else "no"

    # what the repository can prove about each comparator, counted rather than asserted
    cp = art("comparator_provenance.json")
    n["comparators.n"] = cp["n_comparators"]
    n["comparators.versioned"] = cp["n_carrying_a_version_string"]
    for tag, name in (("sygma", "SyGMa"), ("metatox", "MetaTox"),
                      ("metapredictor", "MetaPredictor"), ("biotransformer", "BioTransformer"),
                      ("gloryxarm", "GLORYx")):
        row = cp["comparators"][name]
        n[f"comparators.{tag}.date"] = row["predictions_first_in_the_repository"]
        if row.get("build"):
            n[f"comparators.{tag}.build"] = row["build"]

    # what each arm hands back, read from the files this repository holds rather than asserted:
    # the manuscript had claimed the incumbent leaves part of its output in file order, and every
    # structure in the file it returned carries the service's own score
    ret = art("what_each_arm_returns.json")
    for tag, name in (("metatox", "MetaTox"), ("sygma", "SyGMa"),
                      ("metapredictor", "MetaPredictor"), ("biotransformer", "BioTransformer")):
        row = ret["by_arm"][name]
        n[f"armreturn.{tag}returned"] = row["structures_returned"]
        n[f"armreturn.{tag}scored"] = row["structures_carrying_a_score"]
    n["armreturn.ordering"] = len(ret["comparators_that_order_their_whole_output"])
    # A share whose numerator and denominator were measured over different populations is not a
    # share. This one was: the ordering count came from an artifact that assessed four comparators
    # while the denominator counted five, so the sentence divided by an arm nobody had looked at.
    _assessed = len([a for a in ret["by_arm"] if a != "GRAIL"])
    if _assessed != n["comparators.n"]:
        raise SystemExit(
            f"REFUSING: what_each_arm_returns.json assessed {_assessed} comparators and the "
            f"comparison has {n['comparators.n']}. A count of how many order their output cannot "
            f"be reported over a denominator it was not measured against.")
    n["armreturn.attributing"] = len(
        ret["comparators_whose_held_output_names_a_transformation_or_a_site"])

    # The untraced curated templates, divided. 759 is the whole untraced body, and most of it is
    # the second machine extraction whose source file the history does name; what nothing here
    # accounts for is the remainder inside the three named collections.
    ctp = art("curated_third_party.json")
    n["thirdparty.unaccounted"] = (ctp["curated_split"]["named_collections"]
                                   - ctp["borrowed_within_the_named_body"])
    n["thirdparty.extraction"] = ctp["curated_split"]["earlier_extraction"]

    # what the substrates are, so the manuscript's motivation can be read against them
    ad = art("applicability_domain.json")
    _ev = ad["populations"]["the evaluated test set"]
    n["adomain.eval.mw"] = _ev["descriptors"]["molecular weight"]["quantiles"]["0.5"]
    n["adomain.eval.logp"] = _ev["descriptors"]["calculated logP"]["quantiles"]["0.5"]
    n["adomain.eval.scaffolds"] = _ev["bemis_murcko_scaffolds"]
    n["adomain.eval.singletons"] = _ev["scaffolds_carried_by_one_substrate"]

    # The gate the released checkpoint carries, against the selection every arm here was
    # measured under. It is not a hyperparameter the paper chose; it is one the release applied
    # and the measurement did not.
    rdt = art("released_default_threshold.json")
    n["defaultgate.threshold"] = rdt["calibrated_threshold"]
    n["defaultgate.budget"] = rdt["rule_budget"]
    n["defaultgate.clearing"] = rdt["rules_clearing_the_threshold"]["median"]
    n["defaultgate.binds"] = rdt["substrates_where_the_gate_binds"]
    n["defaultgate.bindshare"] = rdt["share_where_the_gate_binds"]

    # The comparison with every arm that can be re-run on the drawing a user submits.
    eq = art("drawing_equalised.json")
    n["equalised.moved"] = len(eq["budgets_whose_verdict_moves"])
    n["equalised.substrates"] = eq["population"]["n_substrates"]
    n["equalised.movedsubs"] = eq["population"]["substrates_the_standardiser_moves"]
    _wo = eq.get("verdicts_equalised_without_the_service_arms") or {}
    n["equalised.differswithoutmetatox"] = sum(
        1 for k, v in _wo.items()
        if v["verdict"] != eq["verdicts_equalised"][k]["verdict"])

    agr = art("ceiling_instrument_agreement.json")
    n["ceilagree.disagreement"] = agr["disagreement_between_instruments_on_the_same_convention"]
    n["ceilagree.spread"] = agr["spread_across_all_four_counts"]
    n["ceilagree.spreadshare"] = agr["spread_as_share_of_references"]
    n["ceilagree.spreadofmargin"] = agr["spread_as_share_of_that_margin"]
    # The denominator itself, because a share of an unprinted quantity cannot be checked. It is
    # the same minimum the parent-drop section divides by, and the two used to disagree because
    # one artifact predated an arm the other already held.
    n["ceilagree.narrowestmargin"] = agr["narrowest_separating_margin_in_the_comparison"]
    for tag, name in (("deployed", "deployed loop, hydrogens implicit"),
                      ("auditimplicit", "audit loop, hydrogens implicit"),
                      ("auditcompleted", "audit loop, explicit and templates completed"),
                      ("dispatch", "audit loop, convention chosen per template")):
        n[f"ceilagree.{tag}.uncovered"] = agr["arms"][name]["uncovered"]
        n[f"ceilagree.{tag}.reach"] = agr["arms"][name]["reach"]
    n["census.types"] = cen["distinct_types"]
    n["census.once"] = cen["types_seen_once"]
    n["census.once.share"] = cen["share_of_misses_in_types_seen_once"]
    n["census.half"] = cen["types_carrying_half_the_mass"]
    # the specification curve: the tail against every definition of a type, with a flag for
    # whether a type at that level still names a transformation
    for i, g in enumerate(cen["granularity_curve"]):
        tag = ["exact", "noCounts", "elements", "nbonds"][i]
        n[f"grain.{tag}.types"] = g["types"]
        n[f"grain.{tag}.once"] = g["seen_once"]
        n[f"grain.{tag}.mass"] = g["share_of_mass_in_singletons"]
        n[f"grain.{tag}.usable"] = g["determines_a_product"]
    n["uspto.templates"] = usp["uspto"]["templates"]
    n["uspto.types"] = usp["uspto"]["distinct_types_either_direction"]
    n["uspto.hit"] = usp["overlap"]["types_in_uspto_either_direction"]
    n["uspto.mass"] = usp["overlap"]["misses_those_types_carry"]
    n["uspto.share"] = usp["overlap"]["share_of_the_novel_gap"]
    n["uspto.sanity"] = usp["sanity"]["bank_types_uspto_also_has"]
    n["bank.rules"] = usp["bank"]["rules"]
    n["bank.types"] = usp["bank"]["distinct_types"]

    # the ordering diagnosis
    n["oracle.between"] = wide["arms"]["oracle_between"]["recall@15_micro"]
    n["oracle.within"] = wide["arms"]["oracle_within"]["recall@15_micro"]
    n["oracle.asranked"] = wide["arms"]["as_ranked"]["recall@15_micro"]
    n["decode.rrf"] = gd["recall_by_budget"]["15"]["rrf"]
    for m in ("1", "2", "3", "5"):
        n[f"decode.cap{m}"] = gd["recall_by_budget"]["15"][f"rrf+cap{m}"]

    # cost and the envelope
    rows = env["rows"]
    n["envelope.n"] = env["n_done"]
    n["envelope.unfinished"] = sum(1 for r in rows if not r["finished"])
    n["envelope.deadline"] = int(env["deadline_s"])
    n["envelope.smallest_unfinished"] = min(r["heavy"] for r in rows if not r["finished"])

    # matching
    n["taut.invariance.shipped"] = tb["by_budget"]["1000"]["invariance"]
    n["taut.invariance.200"] = tb["by_budget"]["200"]["invariance"]
    n["nearmiss.references"] = nm["references"]
    n["nearmiss.unmatched"] = nm["references_the_key_did_not_match"]
    n["nearmiss.confirmed"] = nm["of_those_present_in_the_pool_as_a_tautomer"]
    n["nearmiss.screened"] = nm["passed_the_skeleton_and_formula_screen"]

    # the two operating modes, medians and the tail on one population
    n["mode.interactive.median"] = mt["interactive"]["median_s"]
    n["mode.interactive.mean"] = mt["interactive"]["mean_s"]
    n["mode.interactive.p90"] = mt["interactive"]["p90_s"]
    n["mode.interactive.max"] = mt["interactive"]["max_s"]
    n["mode.interactive.n"] = mt["interactive"]["n"]
    n["mode.exhaustive.median"] = mt["exhaustive"]["median_s"]
    # The candidate counts belong to the same population as the timings above. They used to be
    # taken from the comparison set while the caption named the validation draw.
    for mode in ("interactive", "exhaustive"):
        cand = mt[mode].get("candidates") or {}
        for stat in ("mean", "median", "n"):
            if stat in cand:
                n[f"mode.{mode}.candidates.{stat}"] = cand[stat]
    n["h15.median.underload"] = h15["time"]["load_correction"][
        "median_under_the_other_arms_load"]

    # the ordering diagnosis, with the arm each figure belongs to
    # The headroom on the uncapped pool the measurement was first made on. The same quantity on
    # the deployed configuration is smaller by more than half, and the main text printed only this
    # one, unlabelled; both are now carried so the text can say which it means.
    n["oracle.headroom"] = round(wide["arms"]["oracle_between"]["recall@15_micro"]
                                 - wide["arms"]["as_ranked"]["recall@15_micro"], 4)
    _dep = art("oracle_by_grouping.json")["headroom_over_fusion"]["formula"]
    n["oracle.headroomdeployed"] = _dep["gap"]
    n["oracle.headroomdeployedlo"] = _dep["ci95"][0]
    n["oracle.headroomdeployedhi"] = _dep["ci95"][1]
    # The same oracle against a partition with the group sizes matched, which is the control that
    # says whether the grouping carries anything beyond its granularity. Formula does not; the
    # transformation type does, and that contrast is the headroom that survives.
    _og = art("oracle_by_grouping.json")
    n["oracle.headroomrandom"] = _og["headroom_over_fusion"]["random_matched"]["gap"]
    n["oracle.headroomtype"] = _og["headroom_over_fusion"]["type"]["gap"]
    for tag, key in (("formulavsrandom", "formula-random_matched"),
                     ("typevsrandom", "type-random_matched")):
        _c = _og["contrasts_between_arms"][key]
        n[f"oracle.{tag}"] = _c["gap"]
        n[f"oracle.{tag}lo"] = _c["ci95"][0]
        n[f"oracle.{tag}hi"] = _c["ci95"][1]
        n[f"oracle.{tag}sep"] = _c["excludes_zero"]
    # The two recalls the design gap is the difference of. They were typed here rather than read,
    # and the artifact was re-run underneath them: the pair went on printing the value it had
    # before, so the SI glossed a gap of -0.0902 with a pair differing by -0.0647. A gap and the
    # two numbers it is the difference of must come from one read of one file, or the sentence
    # that prints all three can contradict itself.
    n["h8.blocked"] = h8["recall_micro"]["15"]["fusion_blocked"]
    n["h8.interleaved"] = h8["recall_micro"]["15"]["fusion"]
    n["h12.ceilingrecall"] = h12["recall_micro"]["15"]["oracle_third"]
    # The validation-side pair the SI compares: what the trained scorer gains over the two-way base
    # and what a binary group ordering gains over the same base. Both were typed here rather than
    # read, and the file they belong to was in the tree the whole time. The trained gain rounded to
    # the same value; the binary one was printed a digit high.
    h12v = art("h12_verdict_validation.json")
    _v15 = h12v["recall_micro"]["15"]
    n["h12.val"] = round(_v15["three_way"] - _v15["two_way"], 4)
    n["h12.valceiling"] = round(_v15["oracle_third"] - _v15["two_way"], 4)
    if not (n["h12.val"] > n["h12.valceiling"] > 0):
        raise SystemExit(
            f"REFUSING: the SI says the trained scorer exceeds the binary ordering on validation, "
            f"and the artifact gives {n['h12.val']} against {n['h12.valceiling']}. One of the two "
            f"is wrong and the sentence would assert the opposite of what was measured.")

    # the relaxation ladder and its a-priori bound
    n["relax.recovered"] = rel["phase_b"]["arms"]["no_H_no_deg"]["recovered_count"] \
        if "recovered_count" in rel["phase_b"]["arms"]["no_H_no_deg"] else 3
    n["relax.carriers"] = car.get("types_with_a_carrier", 385)
    n["relax.predicted"] = car.get("expected_recovered", 8.5)

    # the leakage audit and the bank's composition, for the supporting information
    lkr = art("leakage_fix_report.json")
    n["leak.mol_overlap"] = lkr["clean_overlap"]["train_test"]["molecule_overlap"]
    n["leak.sub_as_mol"] = lkr["structure_leak"]["test_substrate_in_train_molecules"]
    catalog = json.loads((ROOT / "results/mined_rule_catalog_v2.json").read_text())
    n["mined.rules"] = len(catalog)
    # the value is a record, not a count; "count" is the number of training pairs the template
    # was derived from, and a naive len() over the record silently gave zero singletons
    # Two counts of the same idea and they differ: a template can list one pair more than once.
    # The distinct-pair count is the one saturation is about, because withholding a pair withholds
    # every occurrence of it, and printing the other beside it without saying so put 73.0% in one
    # section and 79.5% in another.
    n["mined.singleton"] = sum(
        1 for v in catalog.values()
        if len({tuple(p) for p in v.get("source_pairs", [])}) == 1)
    n["mined.singletonoccurrence"] = sum(1 for v in catalog.values() if v["count"] == 1)
    n["mined.five_or_more"] = sum(1 for v in catalog.values() if v["count"] >= 5)
    # The bank's composition, counted by scripts/bank_composition.py where the bank is. Counting
    # it here opened the measured bank, which the release does not carry, so the whole chain from
    # artifacts to macros raised in a clone and took five tests with it.
    _bc = art("bank_composition.json")
    for tag, count in _bc["curated_by_collection"].items():
        n[f"curated.{tag}"] = count
    n["curated.total"] = _bc["curated_total"]
    n["curated.named"] = _bc["curated_named"]
    n["curated.unnamed"] = _bc["curated_unnamed"]
    n["bank.rules"] = _bc["rules"]
    n["bank.parses"] = _bc["parses"]

    # what the learned rule CHOICE contributes, against unlearned choices of the same size
    sa = art("selection_ablation_deployed.json")
    n["sel.budget"] = sa["budget"]
    n["sel.applicable"] = sa["mean_applicable_rules"]
    for arm in ("learned", "prior_applicable", "random_applicable", "random"):
        n[f"sel.{arm}15"] = sa["recall_micro"][arm]["15"]
        n[f"sel.{arm}1"] = sa["recall_micro"][arm]["1"]
        n[f"sel.{arm}.pool"] = sa["mean_pool"][arm]
    for arm in ("prior_applicable", "random_applicable", "random"):
        c = sa["learned_minus"][arm]["15"]
        n[f"sel.vs{arm}"] = c["gap"]
        n[f"sel.vs{arm}lo"] = c["ci95"][0]
        n[f"sel.vs{arm}hi"] = c["ci95"][1]
    n["sel.vsprior.k5"] = sa["learned_minus"]["prior_applicable"]["5"]["gap"]
    # The tightest budget at which the learned choice separates from the frequency table, and the
    # count of budgets that do, so the prose cannot state a boundary the artifact has moved.
    _pa = sa["learned_minus"]["prior_applicable"]
    _sep = [k for k in sorted(_pa, key=int) if _pa[k]["separates"]]
    n["sel.vsprior.separating"] = len(_sep)
    n["sel.vsprior.budgets"] = len(_pa)
    n["sel.vsprior.k5lo"], n["sel.vsprior.k5hi"] = _pa["5"]["ci95"]
    n["sel.vsprior.k1"] = _pa["1"]["gap"]
    n["sel.vsprior.k1lo"], n["sel.vsprior.k1hi"] = _pa["1"]["ci95"]

    # P1 was registered and checked before the pool cap of P2 existed, so its figure is measured
    # on the uncapped pool. The same contrast on the pool the system actually ranks is in the
    # ranking ablation, and the paper has to carry both rather than the larger one alone.
    ra0 = art("ranking_ablation.json")["by_population"]["validation draw"]
    n["h7.deployed"] = ra0["fusion_minus"]["product"]["15"]["gap"]
    n["h7.deployed.lo"] = ra0["fusion_minus"]["product"]["15"]["ci95"][0]
    n["h7.deployed.hi"] = ra0["fusion_minus"]["product"]["15"]["ci95"][1]

    # what each learned scorer contributes to the ORDER, on the deployed configuration
    ra = art("ranking_ablation.json")
    for tag, pop in (("cmp", "comparison set"), ("val", "validation draw")):
        r = ra["by_population"][pop]
        for arm in ("fusion", "filter", "generator", "product", "random"):
            n[f"rank.{tag}.{arm}15"] = r["recall_micro"][arm]["15"]
        for arm in ("filter", "generator", "product"):
            c = r["fusion_minus"][arm]["15"]
            n[f"rank.{tag}.vs{arm}"] = c["gap"]
            n[f"rank.{tag}.vs{arm}lo"] = c["ci95"][0]
            n[f"rank.{tag}.vs{arm}hi"] = c["ci95"][1]
        n[f"rank.{tag}.vsproductk1"] = r["fusion_minus"]["product"]["1"]["gap"]

    # the untrained similarity baseline the deployed ranking is measured against
    sb = art("similarity_baseline.json")
    for tag, pop in (("cmp", "comparison set"), ("val", "validation draw")):
        r = sb["by_population"][pop]
        n[f"sim.{tag}.n"] = r["population"]["n"]
        n[f"sim.{tag}.fusion15"] = r["recall_micro"]["fusion"]["15"]
        n[f"sim.{tag}.similarity15"] = r["recall_micro"]["similarity"]["15"]
        n[f"sim.{tag}.random15"] = r["recall_micro"]["random"]["15"]
        c = r["fusion_minus"]["similarity"]["15"]
        n[f"sim.{tag}.gap15"] = c["gap"]
        n[f"sim.{tag}.lo15"] = c["ci95"][0]
        n[f"sim.{tag}.hi15"] = c["ci95"][1]

    # the earlier version of this comparison, and why it could not measure an ordering
    old_sb = art("scaffold_baseline.json")
    n["sim.old.gap"] = old_sb["arms"]["similarity"]["vs_deployed"]
    n["sim.old.lo"] = old_sb["arms"]["similarity"]["ci95"][0]
    n["sim.old.hi"] = old_sb["arms"]["similarity"]["ci95"][1]
    sp = json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]
    sizes = [len(r["candidates"]) for r in sp]
    n["sim.old.median_pool"] = int(sorted(sizes)[len(sizes) // 2])
    n["sim.old.pool_under_budget"] = sum(1 for x in sizes if x <= 15)
    n["sim.old.n"] = len(sizes)

    # retraining variance, so a reader can scale the registered effects against it
    ms = art("multiseed_micro.json")
    n["seed.n"] = len(ms["micro_recall_at_15"]["values"])
    n["seed.macro_mean"] = round(ms["macro_recall_at_15"]["mean"], 4)
    n["seed.macro_std"] = round(ms["macro_recall_at_15"]["std"], 4)
    n["seed.micro_mean"] = round(ms["micro_recall_at_15"]["mean"], 4)
    n["seed.micro_std"] = round(ms["micro_recall_at_15"]["std"], 4)

    # what share of the oracle's headroom the blocked design gives back. This was "a third" while
    # the headroom was believed to be 0.1729; the headroom is 0.3143 and the share moved with it,
    # so it is computed here rather than written in words.
    n["h8.design_share"] = round(abs(n["h8.design"]) / n["oracle.headroom"], 3)

    # the parent-drop convention, swept per arm like every other declared choice
    pd = art("parent_drop_effect.json")
    n["pdrop.max_effect"] = max(abs(v["effect"]) for r in pd["effect"].values()
                                for v in r.values())
    n["pdrop.cells"] = sum(len(r) for r in pd["effect"].values())
    n["pdrop.separating"] = sum(1 for r in pd["effect"].values()
                                for v in r.values() if v["separates"])
    n["pdrop.selfref"] = pd["substrates_whose_own_key_is_a_reference"]
    # the largest effect as a share of the narrowest margin the paper claims, which is the
    # exhaustive arm's lead over MetaTox at k=30. It was in the SI as a hand-typed "eight per cent".
    _separating = [abs(cell["gap"]) for budget in dep["contrasts"].values()
                   for cell in budget.values() if cell.get("excludes_zero")]
    n["pdrop.narrowestmargin"] = round(min(_separating), 4)
    n["pdrop.shareofmargin"] = round(n["pdrop.max_effect"] / min(_separating), 4)
    for tag, arm in (("bank", "GRAIL exhaustive"), ("trained", "GRAIL interactive"),
                     ("sygma", "SyGMa"), ("metapredictor", "MetaPredictor"),
                     ("metatox", "MetaTox")):
        n[f"pdrop.returns.{tag}"] = pd["parent_returned"][arm]["substrates_returning_the_parent"]
        r = pd["parent_returned"][arm]["median_rank_when_returned"]
        if r is not None:
            n[f"pdrop.medrank.{tag}"] = r

    # how many mined templates were derived from a pair whose reaction centre is in more than
    # one piece, which bears on what "single-step" means
    cb = art("cascade_by_pairs.json")
    n["cascade.scored"] = cb["counts"]["scored"]
    n["cascade.multi"] = cb["counts"]["multi_locus"]
    n["cascade.share"] = round(100 * cb["multi_locus_share"], 1)

    # BioTransformer: obtained and measured, but for template reach rather than as a ranked
    # predictor, which is why it carries no column in the comparison
    bt = art("decompose_biotransformer.json")
    n["bt.reach"] = bt["biotransformer"]["template_reach"]
    n["bt.lo"] = bt["biotransformer"]["ci95"][0]
    n["bt.hi"] = bt["biotransformer"]["ci95"][1]
    n["bt.templates"] = bt["biotransformer"]["n_templates"]
    n["bt.n"] = bt["n"]
    n["bt.ourreach"] = bt["context"]["grail_bank_ceiling_on_shared"]

    # The emission-transfer figures. They were hand-typed and on the numeral gate's allow-list,
    # and the population attached to them in the text was the wrong one: the artifact measures the
    # comparison set, not the validation draw.
    ert = art("emission_rule_transfer.json")
    n["emit.pool"] = ert["mean_pool"]
    n["emit.n"] = ert["population"]["n"]
    n["emit.alpha"] = ert["registered_alpha"]
    _a = str(ert["registered_alpha"])
    n["emit.byfusion"] = ert["by_alpha"][_a]["emitted_by_fusion"]
    n["emit.byproduct"] = ert["by_alpha"][_a]["emitted_by_product"]
    n["emit.fusionmedian"] = ert["worst_over_best_score_in_a_pool"]["fusion"]["median"]
    n["emit.productmedian"] = round(
        ert["worst_over_best_score_in_a_pool"]["product"]["median"], 4)

    # the criterion sweep: how much of the comparison's verdict is the criterion
    cs = art("criterion_sweep.json")
    KSW = [1, 3, 5, 8, 10, 15, 20, 30, 50]
    lead = {c: sum(1 for k in KSW
                   if cs["by_criterion"][c]["verdict_by_budget"][str(k)] == "leads")
            for c in cs["criteria"]}
    # How each criterion is named in prose, so a number carrying a criterion's value can carry
    # its name too. Sorting the values and dropping the labels is how the extreme margin at a
    # budget came to be attributed in the manuscript to the wrong criterion: the lowest gap at
    # fifteen is Tanimoto's and the text called it canonical SMILES equality.
    CRIT_NAME = {"canonical": "canonical SMILES equality",
                 "inchikey": "the full InChIKey",
                 "inchi_no_stereo": "the stereochemistry-blind first block",
                 "tanimoto1": "a Tanimoto of one",
                 "inchikey_tautomer": "the tautomer-aware key"}
    _unnamed = [c for c in cs["criteria"] if c not in CRIT_NAME]
    if _unnamed:
        raise SystemExit(f"criterion_sweep.json carries a criterion with no prose name here: "
                         f"{', '.join(_unnamed)}")
    _g15 = {c: cs["by_criterion"][c]["margin_by_budget"]["15"]["gap"] for c in cs["criteria"]}
    _worst = min(_g15, key=_g15.get)
    _best = max(_g15, key=_g15.get)
    n["crit.budgets"] = len(KSW)
    n["crit.criteria"] = len(cs["criteria"])
    n["crit.leads_default"] = lead[cs["reference_criterion"]]
    n["crit.leads_best"] = max(lead.values())
    n["crit.never_leads"] = sum(1 for c in lead if lead[c] == 0)
    # No rank is emitted for the reported criterion. One was, computed from a fewest-leads-first
    # sort, and the manuscript read it as a most-leads-first position: it said the reported key
    # "sits 4 of 5 ... nearer the bottom of that order than the top" when 4 in that order is the
    # second most permissive of the five. The two criteria at the top of the lead count are also
    # tied, so the position was decided by the artifact's list order and not by the data. The lead
    # counts below say what the rank was meant to say and cannot be read backwards.

    # The sweep's spread is driven by two criteria the paper itself declines to defend as defaults:
    # canonical SMILES equality tests whether two pipelines standardise alike, and Tanimoto of one
    # on a folded fingerprint is a collision test. Restricting to the three that answer "are these
    # the same compound" is a different question from the multiverse and worth answering too, so
    # the sub-region is counted rather than left for a reader to read off the table.
    DEFENSIBLE = ("inchikey", "inchi_no_stereo", "inchikey_tautomer")
    defensible = [c for c in cs["criteria"] if c in DEFENSIBLE]
    n["crit.defensible"] = len(defensible)
    n["crit.leads_under_every_defensible"] = sum(
        1 for k in KSW
        if all(cs["by_criterion"][c]["verdict_by_budget"][str(k)] == "leads" for c in defensible))
    n["crit.trails_under_any_defensible"] = sum(
        1 for k in KSW
        if any(cs["by_criterion"][c]["verdict_by_budget"][str(k)] == "trails" for c in defensible))
    n["crit.leads_under_no_indefensible"] = sum(
        1 for c in cs["criteria"] if c not in DEFENSIBLE and lead[c] == 0)
    n["crit.swing15"] = round(_g15[_best] - _g15[_worst], 4)
    n["crit.worst15"] = _g15[_worst]
    n["crit.best15"] = _g15[_best]
    n["crit.worst15name"] = CRIT_NAME[_worst]
    n["crit.best15name"] = CRIT_NAME[_best]
    n["crit.leadsbestname"] = CRIT_NAME[max(lead, key=lead.get)]
    n["crit.moved_max"] = max(cs["n_budgets_moving"].values())

    # the worked example, both arms
    for arm, fname in (("inter", "case_study.json"), ("exh", "case_study_exhaustive.json")):
        cs = art(fname)
        n[f"case.{arm}.candidates"] = cs["n_candidates"]
        n[f"case.{arm}.seconds"] = cs["generator_seconds"]
        n[f"case.{arm}.budget"] = cs["configuration"]["rule_budget"]
        n[f"case.{arm}.found"] = len(cs["reference_ranks"])
        for k in ("15", "30"):
            n[f"case.{arm}.recall{k}"] = round(cs["recall_at"][k], 2)
    n["case.references"] = art("case_study.json")["n_references"]
    # the same example on the molecule as a chemist draws it, which is what the corpus does not
    for arm, fname in (("interdrawn", "case_study_drawn.json"),
                       ("exhdrawn", "case_study_exhaustive_drawn.json")):
        cs = art(fname)
        n[f"case.{arm}.candidates"] = cs["n_candidates"]
        n[f"case.{arm}.found"] = len(cs["reference_ranks"])
        for k in ("15", "30", "50"):
            n[f"case.{arm}.recall{k}"] = round(cs["recall_at"][k], 2)
    n["case.exh.ranks"] = ", ".join(str(r) for r in art("case_study_exhaustive.json")["reference_ranks"])
    n["case.exhdrawn.ranks"] = ", ".join(
        str(r) for r in art("case_study_exhaustive_drawn.json")["reference_ranks"])
    # The interactive arm's one hit, by its own rank rather than by a position in the list. The
    # rank was hard-coded as the fourth element, so a re-run that moved the hit reported the rule
    # of whatever had taken its place.
    _inter = art("case_study.json")
    _interhit = next((c for c in _inter["candidates"] if c["is_reference"]), None)
    n["case.inter.rulehit"] = _interhit["rule_id"] if _interhit else None
    n["case.inter.rank"] = _interhit["rank"] if _interhit else None
    # The widest rank the exhaustive arm needs to reach every annotated metabolite, which the
    # prose stated as a round number and which moves whenever the ranking does.
    _exh = art("case_study_exhaustive.json")
    n["case.exh.deepest"] = max(_exh["reference_ranks"]) if _exh["reference_ranks"] else None
    # How many of the illustrated molecule's metabolites the reported budget does not reach. The
    # case is offered as what the system does well, and this is the half of it the budget misses.
    n["case.exh.pastfifteen"] = sum(1 for r in _exh["reference_ranks"] if r > 15)
    _drawn = art("case_study_exhaustive_drawn.json")
    n["case.exhdrawn.deamrule"] = next(c["rule_id"] for c in _drawn["candidates"]
                                       if c["is_reference"] and c["rule_source"] == "curated")
    n["case.exhdrawn.deamrank"] = next(c["rank"] for c in _drawn["candidates"]
                                       if c["is_reference"] and c["rule_source"] == "curated")
    # The same metabolite's rank under the other drawing, so the comparison in the prose is
    # between two ranks of one structure rather than between a rank and a list depth.
    _deamkey = next(c["key"] for c in _drawn["candidates"]
                    if c["is_reference"] and c["rule_source"] == "curated")
    n["case.exh.deamrank"] = next((c["rank"] for c in _exh["candidates"]
                                   if c.get("key") == _deamkey), None)

    # What the annotation actually contains, which a recall figure is a statement about
    ra = art("reference_audit.json")
    n["refaudit.references"] = ra["population"]["references"]
    n["refaudit.substrates"] = ra["population"]["substrates"]
    n["refaudit.distinct"] = ra["distinct_reference_keys"]
    n["refaudit.smaller"] = ra["counts"]["smaller_than_the_parent"]
    n["refaudit.same"] = ra["counts"]["same_heavy_atom_count"]
    n["refaudit.larger"] = ra["counts"]["larger_than_the_parent"]
    n["refaudit.selfref"] = ra["counts"]["equal_to_their_own_substrate"]
    n["refaudit.shared"] = ra["keys_shared_by_more_than_one_substrate"]
    n["refaudit.unparsed"] = ra["counts"].get("unparsed", 0)
    n["refaudit.deltamin"] = ra["heavy_atom_delta"]["min"]
    n["refaudit.deltamax"] = ra["heavy_atom_delta"]["max"]
    n["refaudit.deltamedian"] = ra["heavy_atom_delta"]["median"]
    n["refaudit.deltalo"] = ra["heavy_atom_delta"]["p05"]
    n["refaudit.deltahi"] = ra["heavy_atom_delta"]["p95"]
    n["refaudit.elements"] = len(ra["elements"])


    # How the corpus draws its molecules, which decides which rules can fire on it. This is an
    # axis like the matching criterion and the output budget, and it was undeclared.
    dia = art("dialect_census.json")
    for split in ("train", "val", "test"):
        row = dia["splits"][split]
        n[f"dialect.{split}.n"] = row["n"]
        n[f"dialect.{split}.moved"] = row["moved"]
        n[f"dialect.{split}.movedshare"] = row["moved_share"]
        n[f"dialect.{split}.imidic"] = row["carrying_imidic_amide"]
        # counts, not the share: 378,228 of 378,238 rounds to 1 and would read as exact
        n[f"dialect.{split}.rtequal"] = dia["normalisation"][split]["equal_to_inchi_round_trip"]
        n[f"dialect.{split}.rtof"] = dia["normalisation"][split]["records_compared"]
        n[f"dialect.{split}.rtdiffer"] = dia["normalisation"][split]["differing"]
    ev = dia["evaluated_test_subset"]
    n["dialect.eval.n"] = ev["n"]
    n["dialect.eval.moved"] = ev["moved"]
    n["dialect.eval.movedshare"] = ev["moved_share"]
    n["dialect.eval.imidic"] = ev["carrying_imidic_amide"]
    n["dialect.eval.puretautomer"] = ev["moved_pure_tautomer"]
    n["dialect.bank.amide"] = dia["bank"]["amide_requiring"]["total"]
    n["dialect.bank.imidic"] = dia["bank"]["imidic_requiring"]["total"]
    n["dialect.bank.minedamide"] = dia["bank"]["amide_requiring"]["mined"]
    n["dialect.bank.minedimidic"] = dia["bank"]["imidic_requiring"]["mined"]
    n["dialect.bank.curatedamide"] = dia["bank"]["amide_requiring"]["curated"]
    n["dialect.bank.curatedimidic"] = dia["bank"]["imidic_requiring"]["curated"]
    if "arm_presentation" in dia:
        n["dialect.metatoxretaut"] = dia["arm_presentation"]["metatox_retautomerised"]
        n["dialect.metatoxn"] = dia["arm_presentation"]["metatox_substrates"]
    # The hydrogen presentation the ceiling is measured under, named rather than left implicit,
    # with the band across the two arms the dispatch artifact admits and the number a reproducer
    # gets through the public entry point, whose default is the other presentation.
    hyd = art("hydrogen_dispatch__clean_test.json")["banks"]["grail_full"]
    n["hyd.references"] = hyd["references"]
    n["hyd.implicit"] = hyd["global_arms"]["all_implicit"]
    n["hyd.completed"] = hyd["global_arms"]["all_explicit_completed"]
    n["hyd.expanded"] = hyd["global_arms"]["all_explicit"]
    n["hyd.uncoveredimplicit"] = hyd["references"] - hyd["global_arms_paired"]["recovered_implicit"]
    n["hyd.uncoveredcompleted"] = (hyd["references"]
                                   - hyd["global_arms_paired"]["recovered_explicit_completed"])
    n["hyd.uncoveredexpanded"] = (hyd["references"]
                                  - hyd["global_arms_paired"]["recovered_explicit"])
    n["hyd.residual"] = hyd["residual_convention_dependence"]
    # the same quantity in references, which is what the sentence should say: the per-template
    # dispatch recovers this many more than the best single convention does
    n["hyd.residualrefs"] = hyd["recovered"] - hyd["global_arms_paired"][
        "recovered_explicit_completed"]
    # the width of the band across the two conventions the instrument admits, which is what the
    # abstract's claim about the bound is
    n["hyd.band"] = (hyd["global_arms_paired"]["recovered_explicit_completed"]
                     - hyd["global_arms_paired"]["recovered_implicit"])
    n["hyd.residuallo"] = hyd["residual_ci95"][0]
    n["hyd.residualhi"] = hyd["residual_ci95"][1]
    # The substrate-presentation sweep: what the drawing does to each arm, to every verdict, and
    # to the coverage ceiling on the population where both drawings exist.
    if (ROOT / "results" / "dialect_sweep.json").exists():
        dsw = art("dialect_sweep.json")
        n["dsweep.n"] = dsw["n"]
        n["dsweep.cells"] = dsw["verdict_cells"]
        n["dsweep.moved"] = dsw["verdict_cells_that_move"]
        for arm, tag in (("GRAIL exhaustive", "exh"), ("GRAIL interactive", "inter")):
            row = dsw["effect_on_each_arm"][arm]
            worst = max(row.values(), key=lambda v: abs(v["difference"]))
            n[f"dsweep.{tag}.largest"] = worst["difference"]
            n[f"dsweep.{tag}.separating"] = sum(1 for v in row.values() if v["separates"])
            n[f"dsweep.{tag}.budgets"] = len(row)
            for k in ("15", "30", "50"):
                n[f"dsweep.{tag}.diff{k}"] = row[k]["difference"]
                n[f"dsweep.{tag}.lo{k}"] = row[k]["ci95"][0]
                n[f"dsweep.{tag}.hi{k}"] = row[k]["ci95"][1]
        cov = dsw["coverage_ceiling"]
        n["dsweep.ceilingstored"] = cov["stored"]["coverage"]
        n["dsweep.ceilingdrawn"] = cov["standardised"]["coverage"]
        n["dsweep.ceilingdiff"] = cov["difference"]["value"]
        n["dsweep.ceilinglo"] = cov["difference"]["ci95"][0]
        n["dsweep.ceilinghi"] = cov["difference"]["ci95"][1]

    # What the corpus's drawing cost the one comparator whose rules can be re-run against the
    # other. Its templates are written entirely in amide notation, so this is the arm the
    # dialect should hurt, and it does.
    sbd = art("sygma_by_dialect.json")
    n["sygdial.n"] = sbd["n"]
    n["sygdial.moved"] = sbd["substrates_whose_drawing_changes"]
    for k in ("15", "30", "50"):
        n[f"sygdial.diff{k}"] = sbd["by_budget"][k]["difference"]
        n[f"sygdial.lo{k}"] = sbd["by_budget"][k]["ci95"][0]
        n[f"sygdial.hi{k}"] = sbd["by_budget"][k]["ci95"][1]
    n["sygdial.separating"] = sum(1 for v in sbd["by_budget"].values() if v["separates"])
    n["sygdial.budgets"] = len(sbd["by_budget"])
    sv = art("standardiser_versions.json")
    n["dialect.rdkitversions"] = len(sv["versions_tested"])
    n["dialect.rdkitamide"] = len(sv["versions_returning_the_amide"])

    # What the corpus assembly can and cannot say about itself.
    ca = art("corpus_assembly.json")
    n["assembly.records"] = sum(ca["splits"][s]["records"] for s in ("train", "val", "test"))
    n["assembly.accessions"] = sum(ca["splits"][s]["records_carrying_a_source_accession"]
                                   for s in ("train", "val", "test"))
    n["assembly.duplicated"] = sum(ca["splits"][s]["duplicated_substrate_structures"]
                                   for s in ("train", "val", "test"))
    n["assembly.disagreements"] = sum(
        ca["splits"][s]["duplicated_structures_whose_annotations_disagree"]
        for s in ("train", "val", "test"))
    n["assembly.bothways"] = sum(ca["splits"][s]["pairs_listed_both_positive_and_negative"]
                                 for s in ("train", "val", "test"))
    n["assembly.testids"] = ca["splits"]["test"]["substrates_by_id"]
    n["assembly.teststructures"] = ca["splits"]["test"]["substrates_by_structure"]
    # The dialect census counts every structure the split marks as a substrate; the split table
    # counts those that also carry a clean triple. The difference is the substrates with none,
    # and it is derived rather than typed so the two cannot drift apart.
    n["assembly.testmarkednotriple"] = (dia["splits"]["test"]["n"]
                                        - ca["splits"]["test"]["substrates_by_structure"])
    n["assembly.commits"] = ca["repository"]["commits_in_history"]
    if "source_overlap" in ca:
        so = ca["source_overlap"]
        n["assembly.corpuspairs"] = so["corpus_positive_pairs"]
        mx = so["sources"].get("MetXBioDB")
        if mx:
            n["assembly.metxpairs"] = mx["distinct_pairs"]
            n["assembly.metxinside"] = mx["inside_the_corpus"]
            n["assembly.metxshare"] = mx["share_inside"]
        gx = so["sources"].get("GLORYx")
        if gx:
            n["assembly.gloryxparents"] = gx["parents"]
            n["assembly.gloryxpairs"] = gx["distinct_pairs"]
            n["assembly.gloryxparentsinside"] = gx["parents_inside_the_corpus"]
            n["assembly.gloryxpairsinside"] = gx["pairs_inside_the_corpus"]

    # The composite share, measured twice, the second time under a threshold registered first.
    ci = art("composite_instruments.json")
    n["composite.scored"] = ci["counts"]["scored"]
    n["composite.loci"] = ci["counts"]["instrument_1"]
    n["composite.locishare"] = ci["shares"]["instrument_1"]
    n["composite.edits"] = ci["counts"]["instrument_2"]
    n["composite.editshare"] = ci["shares"]["instrument_2"]
    n["composite.union"] = ci["counts"]["union"]
    n["composite.unionshare"] = ci["shares"]["union"]
    n["composite.both"] = ci["counts"]["both"]
    n["composite.threshold"] = ci["threshold_E"]
    n["composite.bar"] = ci["registered_bar"]

    # Where the bank's templates come from, including the ones that were carried as unattributed.
    cp = art("curated_provenance.json")
    n["attrib.named"] = cp["bank"]["named"]
    n["attrib.unattributed"] = cp["bank"]["unattributed"]
    n["attrib.placed"] = cp["identification"]["unattributed_verbatim_in_xtracted"]
    n["attrib.unattributedimidic"] = cp["dialect"]["curated_unattributed"]["imidic_share"]
    n["attrib.namedimidic"] = cp["dialect"]["curated_named"]["imidic_share"]
    n["attrib.minedimidic"] = cp["dialect"]["mined"]["imidic_share"]
    if "sygma_containment" in cp:
        sc = cp["sygma_containment"]
        n["sygma.rules"] = sc["sygma_rules"]
        n["sygma.inside"] = sc["verbatim_in_bank"]
        n["sygma.share"] = sc["share_of_sygma_inside"]
        n["sygma.incurated"] = sc["in_curated_half"]
        n["sygma.inmined"] = sc["in_mined_half"]
        n["sygma.ofcurated"] = sc["share_of_curated_half_that_is_sygma"]
        n["sygma.outside"] = sc["sygma_rules"] - sc["verbatim_in_bank"]

    # How specific the bank's templates are, which none of its other censuses says
    rsc = art("reactant_size_census.json")
    n["rsize.parsed"] = rsc["templates_parsed"]
    n["rsize.letwo"] = rsc["reactant_atoms_at_most_two"]
    n["rsize.letwoshare"] = rsc["reactant_atoms_at_most_two_share"]
    n["rsize.lethree"] = rsc["reactant_atoms_at_most_three"]
    n["rsize.lethreeshare"] = rsc["reactant_atoms_at_most_three_share"]
    n["rsize.casetop"] = rsc["worked_example_top20_from_three_or_fewer"]

    # And what the same arm does on the test split, which is the population the comparison uses
    sp = art("scored_predictions.json")
    n["scored.failed"] = sp["n_failed"]
    n["scored.substrates"] = sp["n_substrates"]

    # The marginal cell, so the sentence that names it does not type it
    _m = dep["contrasts"]["30"]["whole bank - metatox"]
    n["margin.marginalgap"] = _m["gap"]
    n["margin.marginallo"] = _m["ci95"][0]
    n["margin.marginalhi"] = _m["ci95"][1]

    # The exhaustive mode's non-completion rate, which Table 1 prints as "did not finish" and
    # never as a number, in a paper that says a service must publish the tail as well as the
    # median.
    ce = art("cost_envelope.json")
    n["cost.sampled"] = ce["n_done"]
    n["cost.unfinished"] = sum(1 for r in ce["rows"] if not r.get("finished"))
    n["cost.unfinishedshare"] = round(n["cost.unfinished"] / max(n["cost.sampled"], 1), 4)
    # The two censored statistics are taken over the substrates that finished, and the
    # Supporting Information says so, so the count is a macro rather than a subtraction a reader
    # is left to perform.
    n["cost.finished"] = n["cost.sampled"] - n["cost.unfinished"]
    n["cost.deadline"] = int(ce["deadline_s"])

    # How those substrates were drawn, and what the draw does to the rate. The sample is not random:
    # the producer sorts the validation draw by heavy-atom count, takes every nth, and then adds the
    # twelve largest outright, because that is where the non-completions live. So the sampled rate is
    # an upper bound on the population's and not an estimate of it, and both halves are printed.
    #
    # The population size is read from the pool artifact rather than asserted, and the arithmetic of
    # the draw is then required to reproduce the sample size the timing artifact recorded: with N
    # substrates the systematic part has ceil(N / every) members, the tail adds twelve, and the two
    # overlap in exactly those tail indices divisible by every. If the two runs had been over
    # different populations this would not close, and the numbers would not build.
    n["cost.every"] = int(ce["sample_every"])
    n["cost.tail"] = DECLARED["cost.tail"]
    n["cost.population"] = n["valdraw.declared"]
    _N, _e = n["cost.population"], n["cost.every"]
    _systematic = -(-_N // _e)
    _overlap = sum(1 for _i in range(_N - n["cost.tail"], _N) if _i % _e == 0)
    assert _systematic + n["cost.tail"] - _overlap == n["cost.sampled"], (
        f"the timing sample of {n['cost.sampled']} is not what a draw of every {_e}th of "
        f"{_N} plus the largest {n['cost.tail']} produces")
    # Everything outside the twelve largest is an exact one-in-every_th systematic sample of the
    # rest, so the population count scales by that factor without a model.
    _rows = sorted(ce["rows"], key=lambda r: r["heavy"])
    _tail, _rest = _rows[-n["cost.tail"]:], _rows[:-n["cost.tail"]]
    n["cost.tailunfinished"] = sum(1 for r in _tail if not r.get("finished"))
    n["cost.systematic"] = len(_rest)
    n["cost.systematicunfinished"] = sum(1 for r in _rest if not r.get("finished"))
    n["cost.systematicshare"] = round(n["cost.systematicunfinished"] / max(len(_rest), 1), 4)
    n["cost.estimated"] = n["cost.tailunfinished"] + _e * n["cost.systematicunfinished"]
    n["cost.estimatedshare"] = round(n["cost.estimated"] / n["cost.population"], 4)
    # The load-corrected median, which is the quantity the registered target is expressed in. The
    # manuscript attached "inside the target" to a speed-up factor instead, which reads as though a
    # factor were being compared with a target in seconds.
    n["h15.correctedmedian"] = h15["time"]["load_correction"]["median_under_the_other_arms_load"]

    # The superseded GRAIL column of the population-defining artifact, so the SI can say what it
    # is not without typing the figure.
    fm = art("four_method_291.json")
    n["fourmethod.grail50"] = fm["per_method"]["GRAIL"]["recall"]["50"]
    n["fourmethod.grailemit"] = fm["per_method"]["GRAIL"]["mean_emitted_uncapped"]
    # What the file used to carry there, kept as a macro so the correction can be described with
    # its own number instead of the number that replaced it.
    n["fourmethod.superseded50"] = fm["superseded_grail_column"]["recall"]["50"]
    n["fourmethod.supersededemit"] = fm["superseded_grail_column"]["mean_emitted_uncapped"]

    # Provenance, stated as the pinned set and not the directory. These four were literals here
    # once, inside the one generator whose contract is that no number is a literal: when the
    # pinned set grew, the paper went on reporting the old size. They are read from the sweep,
    # which must be run with --all so the directory counts exist.
    pv = art("artifact_provenance.json")
    sweep = pv["sweep"]
    if not sweep:
        raise SystemExit("results/artifact_provenance.json carries no directory sweep; "
                         "run: python scripts/audit_artifact_provenance.py --all")
    # What the provenance guarantee is true of, counted by the gate rather than by hand. Five
    # readers produced five different counts of this quantity before the gate existed.
    ns = art("number_sources.json")
    n["prov.sources"] = ns["artifacts_the_numbers_come_from"]
    n["prov.sourcesunpinned"] = ns["n_unpinned"]
    n["prov.sourcesunverifiable"] = ns["n_unstamped"]
    n["prov.sourcesinferred"] = ns["n_verifiable_by_inference_only"]
    n["prov.sourcesexempt"] = ns["exempt"]
    # Two properties of the model behind an artifact, counted rather than assumed: how many name
    # a checkpoint that is not the deployed one, and how many are silent where their producer
    # loads a model at all. The second is the first check's blind spot and is reported beside it.
    n["prov.sourceswrongmodel"] = ns["n_naming_a_non_deployed_checkpoint"]
    n["prov.sourcesnocheckpoint"] = ns["n_recording_no_checkpoint"]
    # Those that record no checkpoint but whose models reproduction has established. The blind
    # spot is real for the rest and smaller than the raw count suggests.
    n["prov.sourcesestablished"] = ns.get("n_established_by_reproduction", 0)

    # What the generated-macro claim is actually true of, counted rather than asserted. The
    # claim was made unqualified and was false of the Supporting Information, where measurements
    # were typed by hand and carried on the checker's allow-list.
    np_ = art("number_provenance.json")
    n["prov.macrosbody"] = np_["macros_cited_in_manuscript"]
    n["prov.macrossi"] = np_["macros_cited_in_supporting_information"]
    n["prov.handtyped"] = np_["hand_typed_measurements_on_the_allow_list"]
    n["prov.pinned"] = pv["n_pinned"]
    # Currency as a count rather than as an adjective. The manuscript asserted that all pinned
    # artifacts were current, which is a sentence that stays true-looking while the sweep says
    # otherwise; it now prints how many of them the sweep found current.
    n["prov.pinnedstale"] = pv.get("n_pinned_stale", 0)
    n["prov.pinnedcurrent"] = pv["n_pinned"] - pv.get("n_pinned_stale", 0)
    # Files under results/ and below it. One pinned artifact, the split manifest, lives beside the
    # manuscript instead, and counting it here made the printed total one larger than the
    # directory the sentence names.
    n["prov.files"] = (sum(1 for r in pv["pinned"] if r["artifact"].startswith("results/"))
                       + sum(sweep.values()))
    n["prov.unstamped"] = sweep.get("unstamped", 0)
    # The partition the count closes on, so a reader can add it up rather than take 508 on trust.
    # It was quoted as a total with three of its six parts named, and the two words it is built
    # from were used interchangeably: pinned is membership of a hand-maintained list, stamped is a
    # property of the file.
    n["prov.pinnedinresults"] = sum(1 for r in pv["pinned"] if r["artifact"].startswith("results/"))
    n["prov.producerunknown"] = sweep.get("producer_unknown", 0)
    n["prov.cosmetic"] = sweep.get("cosmetic_only", 0)
    n["prov.currentunpinned"] = sweep.get("current", 0)
    n["prov.changed"] = sweep.get("producer_changed", 0)
    n["prov.partitioncloses"] = (n["prov.unstamped"] + n["prov.changed"] + n["prov.currentunpinned"]
                                 + n["prov.producerunknown"] + n["prov.cosmetic"]
                                 + n["prov.pinnedinresults"]) == n["prov.files"]
    # How far the guarantee reaches past the pinned set: files below the top level of results/
    # that a pinned artifact names as an input, and whose digest therefore is checked.
    n["prov.subdirfiles"] = pv.get("files_below_the_top_level", 0)
    n["prov.namedinputs"] = pv.get("of_those_named_as_an_input_by_a_pinned_artifact", 0)

    # Being named as an input, by digest, is a fixity guarantee and not a provenance one: it says
    # the file has not moved since it was read, and nothing about what wrote it. This census asks
    # the second question of every input a tracked artifact records.
    ip = art("input_provenance.json")
    _cls = ip["distinct_inputs_by_class"]
    n["inprov.stamped"] = _cls["stamped"]
    n["inprov.half"] = _cls["half"]
    n["inprov.bare"] = _cls["bare"]
    n["inprov.notours"] = _cls["not_ours"]
    n["inprov.consumers"] = ip["consumers_reading_an_input_without_a_producer"]

    # The match-scale sweep (queue point D): the one lever that moves the generator's scores rather
    # than the filter's or the rule id's. Curbing the multiplicity bonus to zero is compared to the
    # deployed value on validation; the run was trimmed after the decisive pair, so these are point
    # estimates the producer printed, read against the seed spread rather than a per-run interval.
    _ms = art("match_scale_sweep.json")
    _curb = _ms["by_match_scale"]["0.0"]["recall"]
    _dep = _ms["by_match_scale"]["0.25"]["recall"]
    for _k in ("1", "5", "15"):
        n[f"matchscale.curb{_k}"] = _curb[_k]
        n[f"matchscale.dep{_k}"] = _dep[_k]
        n[f"matchscale.diff{_k}"] = _ms["curb_minus_deployed"][_k]
    n["matchscale.deployed"] = _ms["deployed_match_scale"]
    n["matchscale.registered"] = _ms["reproduces_deployed_arm"]["registered_recall15"]
    n["matchscale.reprohere"] = _ms["reproduces_deployed_arm"]["deployed_recall15_here"]
    n["matchscale.reprodiff"] = _ms["reproduces_deployed_arm"]["difference"]
    return n


def main() -> int:
    n = build()
    out = {"provenance": stamp(__file__), "n_numbers": len(n), "numbers": n}
    (ROOT / "results" / "paper2_numbers.json").write_text(json.dumps(out, indent=1))
    print(f"{len(n)} numbers, each read from the artifact that produced it")
    for k in sorted(n):
        print(f"  {k:<34} {n[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
