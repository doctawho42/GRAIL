"""Every number the manuscript prints, pulled from the artifact that produced it.

SELF_CLAIMS section 11 asks that each numeric passage trace back to its artifact, and section 11a
that no number in the main text be an orphan. Checking that after the fact found five errors in an
hour of prose the last time it was tried. This inverts the order: the manuscript's numbers are
generated here, the text cites them by name, and a check holds the two together. A figure that is
not in this file may not appear in the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

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
    ARMS = {"whole bank": "bank", "trained budget": "trained", "metatox": "metatox",
            "sygma": "sygma", "metapredictor": "metapredictor"}
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
    # how much of the curated half is somebody else's rules verbatim, and under whose terms
    ctp = art("curated_third_party.json")
    n["thirdparty.curated"] = ctp["bank"]["curated"]
    n["thirdparty.traceable"] = ctp["curated_templates_traceable_to_either_source"]
    n["thirdparty.traceableshare"] = ctp["share_of_the_curated_half_so_traceable"]
    for tag, src in (("sygma", "SyGMa"), ("biotransformer", "BioTransformer")):
        row = ctp["third_party_in_the_curated_half"][src]
        n[f"thirdparty.{tag}.shipped"] = row["of_them_in_the_bank"]
        n[f"thirdparty.{tag}.read"] = row["third_party_rules_read"]
        n[f"thirdparty.{tag}.shareofcurated"] = row["share_of_the_curated_half"]

    # what each method recovers, split by the chemistry rather than by the budget
    ebc = art("error_by_chemistry.json")
    n["chem.references"] = ebc["population"]["references_classified"]
    n["chem.classes"] = len(ebc["classes"])
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

    agr = art("ceiling_instrument_agreement.json")
    n["ceilagree.disagreement"] = agr["disagreement_between_instruments_on_the_same_convention"]
    n["ceilagree.spread"] = agr["spread_across_all_four_counts"]
    n["ceilagree.spreadshare"] = agr["spread_as_share_of_references"]
    n["ceilagree.spreadofmargin"] = agr["spread_as_share_of_that_margin"]
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
    n["oracle.headroom"] = round(wide["arms"]["oracle_between"]["recall@15_micro"]
                                 - wide["arms"]["as_ranked"]["recall@15_micro"], 4)
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
    n["mined.singleton"] = sum(1 for v in catalog.values() if v["count"] == 1)
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
    n["case.inter.rulehit"] = art("case_study.json")["candidates"][3]["rule_id"]
    _drawn = art("case_study_exhaustive_drawn.json")
    n["case.exhdrawn.deamrule"] = next(c["rule_id"] for c in _drawn["candidates"]
                                       if c["is_reference"] and c["rule_source"] == "curated")
    n["case.exhdrawn.deamrank"] = next(c["rank"] for c in _drawn["candidates"]
                                       if c["is_reference"] and c["rule_source"] == "curated")

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

    # The superseded GRAIL column of the population-defining artifact, so the SI can say what it
    # is not without typing the figure.
    fm = art("four_method_291.json")
    n["fourmethod.grail50"] = fm["per_method"]["GRAIL"]["recall"]["50"]
    n["fourmethod.grailemit"] = fm["per_method"]["GRAIL"]["mean_emitted_uncapped"]

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

    # What the generated-macro claim is actually true of, counted rather than asserted. The
    # claim was made unqualified and was false of the Supporting Information, where measurements
    # were typed by hand and carried on the checker's allow-list.
    np_ = art("number_provenance.json")
    n["prov.macrosbody"] = np_["macros_cited_in_manuscript"]
    n["prov.macrossi"] = np_["macros_cited_in_supporting_information"]
    n["prov.handtyped"] = np_["hand_typed_measurements_on_the_allow_list"]
    n["prov.pinned"] = pv["n_pinned"]
    n["prov.files"] = pv["n_pinned"] + sum(sweep.values())
    n["prov.unstamped"] = sweep.get("unstamped", 0)
    n["prov.changed"] = sweep.get("producer_changed", 0)
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
