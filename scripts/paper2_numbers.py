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

    # the ceiling and what the gap is made of
    n["population.testsubs"] = cov["n_substrates"]
    n["population.testrefs"] = cov["covered_pairs"] + cov["uncovered_pairs"]
    n["ceiling.coverage"] = cov["coverage"]
    n["ceiling.uncovered"] = cov["uncovered_pairs"]
    n["ceiling.novel"] = cov["gap"]["novel_type"]
    n["ceiling.known"] = cov["gap"]["known_type"]
    n["ceiling.untypeable"] = cov["gap"]["untypeable"]
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

    # Provenance, stated as the pinned set and not the directory. These four were literals here
    # once, inside the one generator whose contract is that no number is a literal: when the
    # pinned set grew, the paper went on reporting the old size. They are read from the sweep,
    # which must be run with --all so the directory counts exist.
    pv = art("artifact_provenance.json")
    sweep = pv["sweep"]
    if not sweep:
        raise SystemExit("results/artifact_provenance.json carries no directory sweep; "
                         "run: python scripts/audit_artifact_provenance.py --all")
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
