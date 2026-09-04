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


def build():
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
    vp = art("val_pools.json")["population"]
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

    ARMS = {"whole bank": "bank", "trained budget": "trained", "metatox": "metatox",
            "sygma": "sygma", "metapredictor": "metapredictor",
            "biotransformer": "biotransformer"}
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
    n["comparators.unavailable"] = 4
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
    n["chem.exhausted.metapredictor"] = _short["metapredictor"]
    n["chem.exhausted.biotransformer"] = _short["biotransformer"]
    n["chem.exhausted.metatox"] = _short["metatox"]
    n["chem.exhausted.sygma"] = _short["sygma"]
    n["chem.exhausted.bank"] = _short["whole bank"]
    n["chem.exhausted.interactive"] = _short["trained budget"]
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
    _rb = art("released_bank.json")
    n["relbank.measured"] = _rb["measured_bank"]["templates"]
    n["relbank.released"] = _rb["released_bank"]["templates"]
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
    # sweep cannot bound: two of its five settings differ only in the stereochemistry layer and
    # return the identical verdict at every budget, because the annotation carries none.
    sh = art("stereo_headroom.json")
    n["stereo.pairs"] = sh["population"]["typed"]
    n["stereo.centre"] = sh["references_gaining_a_tetrahedral_centre"]
    n["stereo.double"] = sh["references_gaining_a_stereogenic_double_bond"]
    n["stereo.either"] = sh["references_gaining_either"]
    n["stereo.share"] = sh["share_gaining_either"]

    # The classes where this system is not the best arm, which the main text reported only where
    # it was. The names and the margins come from the artifact rather than from a reading of it.
    ch = art("error_by_chemistry.json")["classes"]
    _arms = ("GRAIL exhaustive", "GRAIL interactive", "metatox", "sygma", "metapredictor",
             "biotransformer")
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
    n["rare.pairs"] = rar["distinct_training_pairs"]
    n["rare.singletons"] = rar["templates_resting_on_one_pair"]
    n["rare.singletonshare"] = rar["singleton_share"]
    n["rare.unseenmass"] = rar["good_turing_mass_of_unseen_types"]
    n["rare.slope"] = rar["templates_gained_per_thousand_pairs_at_the_full_corpus"]
    for frac in ("0.25", "0.5", "0.75", "1.0"):
        tag = {"0.25": "quarter", "0.5": "half", "0.75": "threequarters", "1.0": "all"}[frac]
        n[f"rare.{tag}"] = int(round(rar["rarefaction"][frac]["templates_mean"]))

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
                      ("metapredictor", "MetaPredictor"), ("biotransformer", "BioTransformer")):
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
    _wo = eq.get("verdicts_equalised_without_metatox") or {}
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
    n["h8.blocked"] = 0.4376
    n["h8.interleaved"] = 0.5023
    n["h12.ceilingrecall"] = h12["recall_micro"]["15"]["oracle_third"]
    n["h12.val"] = 0.0412
    n["h12.valceiling"] = 0.0260

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
    bank = [ln for ln in (ROOT / "grail_metabolism/resources/extended_smirks.txt"
                          ).read_text().splitlines() if ln.strip()]

    # The bank's composition, counted from the files rather than described. mined_only_v2.txt is
    # the mined half that matches the catalog and the deployed bank; mined_only.txt is a
    # superseded earlier cut of 5,866. Three curated collections ship with the code; the rest of
    # the curated half comes from a fourth whose file is not in the repository.
    def _rules(rel):
        return {ln.strip() for ln in (ROOT / rel).read_text().splitlines() if ln.strip()}
    bankset = set(bank)
    minedset = _rules("grail_metabolism/resources/mined_only_v2.txt")
    curated_files = {
        "hydroxylation": "grail_metabolism/data/smirks.txt",
        "merged": "grail_metabolism/data/merged_smirks.txt",
        "notebooks": "grail_metabolism/resources/notebooks_rules.txt",
    }
    named = set()
    for tag, rel in curated_files.items():
        r = _rules(rel) & bankset
        n[f"curated.{tag}"] = len(r)
        named |= r
    n["curated.total"] = len(bankset - minedset)
    n["curated.named"] = len(named)
    n["curated.unnamed"] = len(bankset - minedset - named)
    n["bank.rules"] = len(bank)
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    def _parses(smirks):
        # ReactionFromSmarts raises on a malformed template rather than returning None, so the
        # count has to catch as well as test; one template in the bank does exactly this
        try:
            return AllChem.ReactionFromSmarts(smirks.strip()) is not None
        except Exception:
            return False
    n["bank.parses"] = sum(1 for r in bank if _parses(r))

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
    order = sorted(cs["criteria"], key=lambda c: lead[c])
    gaps = sorted(cs["by_criterion"][c]["margin_by_budget"]["15"]["gap"] for c in cs["criteria"])
    n["crit.budgets"] = len(KSW)
    n["crit.criteria"] = len(cs["criteria"])
    n["crit.leads_default"] = lead[cs["reference_criterion"]]
    n["crit.leads_best"] = max(lead.values())
    n["crit.never_leads"] = sum(1 for c in lead if lead[c] == 0)
    n["crit.default_rank"] = order.index(cs["reference_criterion"]) + 1

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
    n["crit.swing15"] = round(gaps[-1] - gaps[0], 4)
    n["crit.worst15"] = gaps[0]
    n["crit.best15"] = gaps[-1]
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
    n["cost.tail"] = 12
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
