#!/usr/bin/env python3
"""Every number the registration leans on, checked against the code that wrote it.

Artifacts here are gated on write and read blind, and this project has three of its own
defects from that asymmetry. The sharpest is `bank_without_selection.json`, which carried
`ceiling_on_this_subset: 0.7284` through a correction that took the same quantity to 0.8007:
the value went on looking committed because nothing that read it asked which version of the
code had produced it. That artifact is the first thing this sweep flags.

Two strengths of evidence, and they are labelled apart.

  recorded    the artifact carries the digest of its producer's source, written at the same
              moment as the numbers. Nothing is assumed.
  inferred    the artifact predates stamping. The commit that ADDED it is recovered from the
              log and the producer as it stood there is hashed. That is the producer at write
              time only if the script did not change between the run and the commit -- an
              assumption about how the work was done, not a fact the artifact records.

PINNED lists the artifacts the preregistration and the freeze depend on. Those must be current;
everything else is counted and reported. A changed producer does not prove a number wrong, it
proves nobody has checked, which is the state all three of this project's defects shared.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import COSMETIC, CURRENT, check_inputs, infer, verify  # noqa: E402

# not_checkable_here is not a pass: it says the tree cannot answer, and the reason is that
# the release withholds the file. It must not fail a clone and must not be mistaken for a
# verification, so it is counted and named separately wherever the totals are printed.
NOT_CHECKABLE = "not_checkable_here"
OK = (CURRENT, COSMETIC, NOT_CHECKABLE)

TE = "scripts/typed_edit"
# Three artifacts were removed rather than re-run, and what replaced each is named here so the
# removal is a record and not a gap. results/vs_metatox.json and results/vs_metatox_pools.json
# were an earlier route to the comparison the paper reports; every figure now comes from
# results/widepools_implicit/ and results/deployment_table.json, and the one audit that still read
# the older pair reads the live pools instead. results/bank_without_selection_full.json was scored
# with a filter checkpoint the system does not deploy and no figure in either manuscript is drawn
# from it. Keeping a stale pin would have been a gate that reports currency it cannot check.
# One pin fails permanently, and the reason is recorded here rather than repaired, because
# repairing it would mean reporting a currency nothing can check.
#
#   results/uspto_type_overlap.json  producer_changed (was 81e44a0d3d8a, is now 23314d652a09)
#
# The artifact cannot be regenerated at all: its input, grail_metabolism/uspto_templates.csv.gz,
# is a zero-byte placeholder in every checkout, was never tracked, and no producer in this
# repository writes it. So the usual closure -- re-run and re-stamp -- does not exist.
#
# What the change was, reviewed against the log rather than asserted: the producer has exactly two
# commits, the one that created it and 5a4aadf, which inserts one line, `"recovered_types":
# sorted(hit_b)`, adding a set to the report beside the count and examples it already wrote. None
# of the six values the manuscript reads from this artifact (uspto.templates, uspto.types,
# uspto.hit, uspto.mass, uspto.share, uspto.sanity) is computed from that line.
#
# That argument is not promoted to a pass. semantic_equal on the recorded and current sources
# returns False -- the delta is a statement, not a comment -- so COSMETIC correctly does not
# apply, and reading the code is weaker evidence than re-running it. The status stays
# producer_changed: it says nobody has checked, which is exactly true and cannot be made
# untrue while the input is missing.
PINNED = {
    # the type vocabulary the H1 stratum and the appendix's counts are keyed to
    "results/typed_edit_type_curve.json": f"{TE}/type_curve.py",
    "results/typed_edit_type_carriers.json": f"{TE}/type_carrier_provenance.py",
    # the strata H1 and H6 are registered on
    "results/h1_stratum.json": f"{TE}/build_h1_stratum.py",
    "results/h1_join_sensitivity.json": f"{TE}/h1_join_sensitivity.py",
    "results/h6_stratum.json": f"{TE}/build_h6_stratum.py",
    # the measurements H5 and the step-0 verdict rest on
    "results/typed_edit_known_pairs.json": f"{TE}/known_type_recovery.py",
    "results/typed_edit_known_type_recovery.json": f"{TE}/known_type_recovery.py",
    "results/typed_edit_step0.json": f"{TE}/run_bank.py",
    "results/typed_edit_relaxation.json": f"{TE}/relaxation_ladder.py",
    # the emission comparison, its grid, and the freeze itself
    "results/emission_leaderboard.json": f"{TE}/emission_leaderboard.py",
    "results/emission_grid.json": f"{TE}/emission_grid.py",
    # the comparison that decides the release, and the pools it wrote
    "results/aggregation_ablation.json": f"{TE}/aggregation_ablation.py",
    "results/biotransformer_arm.json": f"{TE}/biotransformer_arm.py",
    "results/metapredictor_drawing.json": f"{TE}/metapredictor_drawing.py",
    "results/metxbiodb_drop_set.json": f"{TE}/metxbiodb_drop_set.py",
    "results/stereo_headroom.json": f"{TE}/stereo_headroom.py",
    "results/reference_source_coverage.json": f"{TE}/reference_source_coverage.py",
    "results/test_reference_descriptors.json": f"{TE}/reference_descriptors.py",
    "results/reproduce_from_descriptors.json": f"{TE}/reproduce_from_descriptors.py",
    "results/population_definition.json": f"{TE}/population_definition.py",
    # the second MetaTox submission: what came back, what it covers, and the measured reason the
    # arm still does not join the population above. Its inputs are two SDF files far too large to
    # track, so the artifact carries their sha256 and the producer refuses rather than guesses when
    # they are not on disk; what is pinned here is the code, which is the part that can move.
    "results/metatox_outside_submission.json": "scripts/metatox_outside_ingest.py",
    # MetaTox as one arm over the evaluated population: the second submission ingested on the ids
    # of its own submission set, and the merge of the two runs that the axis reads.
    "results/metatox_smirks_preds_1170.json": "scripts/metatox_smirks_ingest.py",
    "results/metatox_smirks_preds_evaluated1170.json": "revision/phase2_metatox_merge.py",
    # how much of that column's recall at each budget is the method and how much is the order its
    # supplier's file happened to have, and which contrast cells the difference could have decided
    "results/metatox_tie_break_sensitivity.json": "scripts/metatox_tie_break_sensitivity.py",
    # the negative result that closes the cheap route to the between-group headroom
    "results/group_decode.json": f"{TE}/group_decode.py",
    # the three matched training variants the supplement reports, and the frontier they share
    "results/rulegate_survivors_summary.json": f"{TE}/rulegate_survivors_summary.py",
    # the generator's multiplicity bonus, swept at inference on the deployed model
    "results/match_scale_sweep.json": f"{TE}/match_scale_sweep.py",
    # what the matching's tautomer canonicalisation costs, measured on the data
    "results/tautomer_near_miss.json": f"{TE}/tautomer_near_miss.py",
    "results/tautomer_budget.json": f"{TE}/tautomer_budget.py",
    # the group scorer: its selection on validation and its verdict on the 291
    "results/group_scorer_selection.json": f"{TE}/train_group_scorer.py",
    "results/h8_verdict.json": f"{TE}/h8_verdict.py",
    "results/train_pools.json": f"{TE}/build_train_pools.py",
    # the same scorer without the blocking, and its selection
    "results/group_scorer_selection_h12.json": f"{TE}/train_group_scorer.py",
    "results/h12_verdict.json": f"{TE}/h12_verdict.py",
    "results/h12_verdict_validation.json": f"{TE}/h12_verdict.py",
    # what the novel-type gap is made of, and whether a library on disk already holds it
    "results/novel_type_census.json": f"{TE}/novel_type_census.py",
    "results/uspto_type_overlap.json": f"{TE}/uspto_type_overlap.py",
    # the group signal as a gate, and its selection
    "results/group_scorer_selection_h14.json": f"{TE}/train_group_scorer.py",
    "results/h14_verdict.json": f"{TE}/h14_verdict.py",
    # the survivors arm with the tautomer budget bounded
    "results/h15_verdict.json": f"{TE}/h15_verdict.py",
    # standardisation off the hot loop
    "results/h13_verdict.json": f"{TE}/h13_verdict.py",
    # the emission rule's grid
    "results/h11_grid.json": f"{TE}/h11_grid.py",
    # why the registered emission threshold does not survive the change of ranking
    "results/emission_rule_transfer.json": f"{TE}/emission_rule_transfer.py",
    # what each operating mode costs, medians and the tail on one population
    "results/mode_timings.json": f"{TE}/mode_timings.py",
    # the comparison as it would ship, both budgets against MetaTox
    "results/deployment_table.json": f"{TE}/deployment_table.py",
    "results/oracle_by_grouping.json": f"{TE}/oracle_by_grouping.py",
    "results/criterion_sweep.json": f"{TE}/criterion_sweep.py",
    "results/similarity_baseline.json": f"{TE}/similarity_baseline.py",
    "results/ranking_ablation.json": f"{TE}/ranking_ablation.py",
    "results/precision_table.json": f"{TE}/precision_table.py",
    "results/cascade_by_pairs.json": f"{TE}/cascade_by_pairs.py",
    "results/parent_drop_effect.json": f"{TE}/parent_drop_effect.py",
    "results/selection_ablation_deployed.json": f"{TE}/selection_verdict.py",
    "results/selection_pools_deployed.json": f"{TE}/selection_ablation_deployed.py",
    "results/case_study.json": f"{TE}/case_study.py",
    "results/case_study_exhaustive.json": f"{TE}/case_study.py",
    # the same example on the molecule as a chemist draws it, which is the other half of the
    # substrate-presentation axis and is cited beside the first two
    "results/case_study_drawn.json": f"{TE}/case_study.py",
    "results/case_study_exhaustive_drawn.json": f"{TE}/case_study.py",
    # the substrate-presentation sweep on both GRAIL arms
    "results/dialect_sweep.json": f"{TE}/dialect_sweep.py",
    # what the corpus's drawing cost the comparator whose rules can be re-run against it
    "results/sygma_by_dialect.json": f"{TE}/sygma_by_dialect.py",
    # the axis itself, and what it did to the bank
    "results/dialect_census.json": f"{TE}/dialect_census.py",
    "results/standardiser_versions.json": f"{TE}/standardiser_versions.py",
    # what the corpus assembly can and cannot say about itself
    "results/corpus_assembly.json": f"{TE}/corpus_assembly.py",
    # the composite share, both instruments, the second one registered as H16
    "results/composite_instruments.json": f"{TE}/composite_instruments.py",
    # where the bank's templates come from, including the 492 carried as unattributed
    "results/curated_provenance.json": f"{TE}/curated_provenance.py",
    # what the annotation contains, which every recall figure is a statement about
    "results/reference_audit.json": f"{TE}/reference_audit.py",
    # what the generated-macro claim is true of, so the claim's own figures are generated
    "results/number_provenance.json": "scripts/check_paper2_numbers.py",
    # the rule budget: the curve's shape and the check on validation
    "results/h10_verdict.json": f"{TE}/h10_verdict.py",
    # the cap, its upper-bound table and the check that fixed it
    "results/pool_cap_cost.json": f"{TE}/pool_cap_cost.py",
    "results/h9_verdict.json": f"{TE}/h9_verdict.py",
    # the H7 check and the validation pool it reads
    "results/val_pools.json": f"{TE}/build_val_pools.py",
    "results/h7_verdict.json": f"{TE}/h7_verdict.py",
    "paper2/split_manifest.json": f"{TE}/freeze_split.py",
    # Everything else the number generator reads. These were outside the pinned set while the
    # paper asserted that every number came from a pinned artifact, which made the assertion
    # false of the coverage bound, the split counts, the leakage audit, the contamination claim
    # and the multiseed spread. Several predate stamping and are verified by inference from the
    # commit that added them, which the sweep reports as inferred rather than as current.
    # The aggregation taken from the budget: an alternative measured on both populations and not
    # adopted. Its numbers are printed, so it is pinned like any other source the paper reads.
    "results/budget_dependent_schedules.json": "scripts/budget_dependent_schedules.py",
    "results/scheduled_release_comparison.json": "scripts/scheduled_release_comparison.py",
    "results/blend_switch_scope.json": "scripts/blend_switch_scope.py",
    "results/coverage_gap_types.json": "scripts/coverage_gap_types.py",
    "results/ceiling_instrument_agreement.json": "scripts/typed_edit/ceiling_instrument_agreement.py",
    "results/comparator_provenance.json": "scripts/typed_edit/comparator_provenance.py",
    "results/hyperparameters.json": "scripts/typed_edit/hyperparameters.py",
    "results/multiplicity.json": "scripts/typed_edit/multiplicity.py",
    "results/sygma_depth_matched_reach.json": "scripts/sygma_depth_matched_reach.py",
    "results/error_by_chemistry.json": "scripts/typed_edit/error_by_chemistry.py",
    "results/curated_third_party.json": "scripts/typed_edit/curated_third_party.py",
    "results/site_agreement.json": "scripts/typed_edit/site_agreement.py",
    # what retraining moves, at the configuration the manuscript reports
    "results/retraining_spread.json": f"{TE}/retraining_spread.py",
    # How the pools that spread is computed from were made, which for a while nothing recorded.
    "results/seedpool_recipe.json": f"{TE}/seedpool_recipe.py",
    # The census of which inputs can say what produced them, which is the question this sweep
    # does not ask of the level below itself.
    "results/input_provenance.json": "scripts/audit_input_provenance.py",
    # Which tracked files carry a template the released bank drops, which is the difference
    # between what the bank ships and what the repository holds.
    "results/withheld_template_carriers.json": "scripts/check_no_withheld_templates.py",
    # The validation draw's four scalars, split out of a 46 MB pool that is not tracked so that
    # the number chain runs in a clone.
    "results/val_pool_population.json": "scripts/val_pool_population.py",
    # The measured bank's composition, counted where the bank is because the bank is not shipped.
    "results/bank_composition.json": "scripts/bank_composition.py",
    # Every sentence in which the work concedes something, recorded so a cut for length cannot
    # take one out without the decision being made deliberately.
    "results/disclosure_inventory.json": "scripts/disclosure_inventory.py",
    # the bank the repository ships, which is the measured bank minus what it may not carry
    "results/released_bank.json": "scripts/build_released_bank.py",
    # the one comparator column obtained from a service rather than run here
    "results/gloryx_service_preds.json": f"{TE}/gloryx_via_service.py",
    # this work's thesis read off an independent benchmark's own published tables
    "results/external_budget_confound.json": f"{TE}/external_budget_confound.py",
    # the compounds the manuscript names, as structures, for the submission checklist
    "results/worked_example_structures.json": f"{TE}/worked_example_structures.py",
    # what a retraining under an unpinnable RDKit would standardise differently
    "results/rdkit_version_drift.json": f"{TE}/rdkit_version_drift.py",
    # the comparator's own emission knob turned up, which the manuscript prints both rows of
    "results/metapredictor_beam_sweep.json": f"{TE}/metapredictor_beam_sweep.py",
    # the comparison with every arm that can be re-run on one drawing
    "results/drawing_equalised.json": f"{TE}/drawing_equalised.py",
    # the same kept-against-dropped test on the second source this repository holds
    "results/gloryx_drop_set.json": f"{TE}/metxbiodb_drop_set.py",
    # the comparison population written out under the key everything is scored by
    "results/comparison_set_members.json": f"{TE}/comparison_set_members.py",
    # what kind of molecules the numbers were measured on
    "results/applicability_domain.json": f"{TE}/applicability_domain.py",
    # the rule gate the released checkpoint carries, against the one the paper measured
    "results/released_default_threshold.json": f"{TE}/released_default_threshold.py",
    # which arms return an ordering and which return an attribution, from the held files
    "results/what_each_arm_returns.json": "scripts/typed_edit/what_each_arm_returns.py",
    "results/budget_curve.json": "scripts/typed_edit/budget_curve.py",
    "results/matched_length.json": "scripts/typed_edit/matched_length.py",
    "results/mining_rarefaction.json": "scripts/typed_edit/mining_rarefaction.py",
    "results/fusion_knobs.json": "scripts/typed_edit/fusion_knobs.py",
    "results/missing_types_in_train.json": "scripts/typed_edit/missing_types_in_train.py",
    "results/licence_removal_cost__clean_test.json": "scripts/typed_edit/licence_removal_cost.py",
    "results/sygma_scenario_sweep.json": "scripts/typed_edit/sygma_scenario_sweep.py",
    "results/references_are_multistep.json": "scripts/typed_edit/references_are_multistep.py",
    "results/dialect_conditional.json": "scripts/typed_edit/dialect_conditional.py",
    "results/decompose_biotransformer.json": "scripts/decompose_biotransformer.py",
    "results/external_overlap_audit.json": "scripts/external_overlap_audit.py",
    "results/hydrogen_dispatch__clean_test.json": "scripts/hydrogen_dispatch.py",
    "results/leakage_fix_report.json": "scripts/audit_leakage.py",
    "results/multiseed_micro.json": "scripts/multiseed_micro.py",
    "results/scaffold_baseline.json": "scripts/scaffold_baseline.py",
    "results/cost_envelope.json": f"{TE}/cost_envelope.py",
    "results/reactant_size_census.json": f"{TE}/reactant_size_census.py",
    "results/four_method_291.json": "scripts/four_method_291.py",
    "results/scored_predictions.json": "scripts/dump_scored_predictions.py",
    "results/aggregation_ablation_validation.json": f"{TE}/aggregation_ablation.py",
    "results/reference_charge.json": f"{TE}/reference_charge.py",
    "results/pool_checkpoints.json": f"{TE}/pool_checkpoints.py",
    "results/wide_pool_analysis_implicit.json": f"{TE}/wide_pool_analysis.py",
}


def _withheld_by_design(rel: str) -> bool:
    """Whether a path is absent because the release does not carry it.

    A file the ignore rules cover and the index does not hold is one the release deliberately
    withholds -- the measured bank, a candidate pool, the corpus. A file that IS tracked and still
    missing is a broken checkout and stays a failure. Asking git rather than keeping a list here
    means a newly withheld artifact is classified correctly without this file being edited.
    """
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                             capture_output=True).returncode == 0
    if tracked:
        return False
    return subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT,
                          capture_output=True).returncode == 0


def check(rel: str, producer: str | None) -> dict:
    path = ROOT / rel
    if not path.exists():
        # Absent because the release withholds it is not the same as absent because something
        # broke. A clone reaches the first for every untracked pool, and reporting the release as
        # stale there told a reader the software was wrong when it was working as documented.
        if _withheld_by_design(rel):
            return {"artifact": rel, "status": "not_checkable_here",
                    "detail": "not redistributed; absent from a clone by design"}
        return {"artifact": rel, "status": "absent", "detail": "not in this checkout"}
    v = verify(path)
    if v["status"] not in OK and producer is not None and v["status"] == "unstamped":
        v = infer(path, producer)
    # A stamp says which script wrote the file; it cannot say what that script was pointed at.
    # An artifact that names its inputs is checked against them, and one written from an input
    # that has since vanished or moved is not current however clean its producer is. This is the
    # case a perturbation run against the real results directory produces, and the one the
    # producer check is blind to by construction.
    if v["status"] in OK:
        try:
            gone = check_inputs(json.loads(path.read_text()))
        except Exception:
            gone = []
        if gone:
            # An input that is gone BECAUSE it is not redistributed is the same case one level
            # down: the digest cannot be recomputed in a clone and its absence is not evidence
            # that anything moved. An input that is present and differs still fails.
            withheld = [g for g in gone
                        if g.startswith("input gone: ")
                        and _withheld_by_design(g[len("input gone: "):])]
            if withheld and len(withheld) == len(gone):
                return {**v, "status": "not_checkable_here",
                        "detail": "input not redistributed: " + "; ".join(withheld)}
            return {**v, "status": "input_changed", "detail": "; ".join(gone)}
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also sweep every other artifact")
    ap.add_argument("--diff", action="store_true",
                    help="print the producer diff for every artifact whose code moved")
    ap.add_argument("--out", default=str(ROOT / "results" / "artifact_provenance.json"))
    args = ap.parse_args()

    pinned = [check(rel, prod) for rel, prod in sorted(PINNED.items())]
    bad = [r for r in pinned if r["status"] not in OK]

    print(f"{'artifact':<46}{'status':<18}evidence")
    for r in pinned:
        how = "recorded" if r.get("how") == "recorded source digest" else (
            "inferred" if r.get("inferred") else r.get("detail", "")[:34])
        print(f"  {r['artifact']:<44}{r['status']:<18}{how}")
    if args.diff:
        moved = [r for r in pinned if r.get("diff")]
        for r in moved:
            print(f"\n--- {r['artifact']}: {r['status']} "
                  f"({r.get('diff_recovered_by', '')}) ---")
            print(r["diff"])
        if not moved:
            print("\nno pinned producer has moved, so there is nothing to diff")

    # The sweep is always computed and always written. It used to run only under --all, so an
    # ordinary invocation replaced the directory counts in the artifact with an empty dict, and
    # the manuscript's numbers path -- which reads those counts -- could not be regenerated after
    # anyone ran the audit the plain way. --all now controls the printing and nothing else.
    sweep = Counter()
    others = []
    # Recursive. It used to glob the top level only, which left every sharded pool outside the
    # sweep -- including the validation pools a published curve is read from, three of which then
    # turned out to have been scored by a checkpoint nobody deploys. A directory the sweep does
    # not enter is a directory whose contents nothing checks.
    for p in sorted(glob.glob(str(ROOT / "results" / "**" / "*.json"), recursive=True)):
        rel = str(Path(p).relative_to(ROOT))
        if rel in PINNED:
            continue
        v = verify(p)
        sweep[v["status"]] += 1
        if v["status"] not in OK:
            others.append(v)

    # An unstamped file in a subdirectory is not necessarily unaccounted for: a pinned artifact
    # that records what it read names those files by digest, so the guarantee reaches one hop
    # past the pinned set. How far it reaches is counted rather than asserted.
    named = set()
    for rel in PINNED:
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        for row in (blob.get("inputs") or []):
            named.add(row.get("path"))
    in_subdirs = [str(Path(p).relative_to(ROOT))
                  for p in glob.glob(str(ROOT / "results" / "**" / "*.json"), recursive=True)
                  if Path(p).parent != ROOT / "results"]
    reached = sorted(f for f in in_subdirs if f in named)
    if args.all:
        print(f"\nthe other {sum(sweep.values())} artifacts: " +
              ", ".join(f"{k} {v}" for k, v in sweep.most_common()))
        changed = [o for o in others if o["status"] == "producer_changed"]
        if changed:
            print(f"  {len(changed)} name a producer that has changed since they were written:")
            for o in changed[:10]:
                print(f"    {o['artifact']:<46}{o.get('detail','')[:44]}")

    rep = {"pinned": pinned, "n_pinned": len(pinned), "n_pinned_stale": len(bad),
           "sweep": dict(sweep), "sweep_not_current": others,
           "files_below_the_top_level": len(in_subdirs),
           "of_those_named_as_an_input_by_a_pinned_artifact": len(reached),
           "named_inputs": reached}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    # Say where the output went. This script's default --out is a tracked artifact, so a caller
    # that forgets to redirect it rewrites the file other gates verify against -- which happened,
    # silently, from the paper-gate suite, and took an hour to attribute.
    print(f"  wrote {args.out}")

    if bad:
        print(f"\nFAIL: {len(bad)} pinned artifacts are not current")
        for r in bad:
            print(f"  {r['artifact']}: {r['status']} -- {r.get('detail','')}")
        return 1
    print(f"{len(in_subdirs)} of the swept files sit below the top level of results/; "
          f"{len(reached)} of those are named as an input by a pinned artifact, so their digest "
          f"is checked even where the file carries no stamp of its own")
    cos = [r for r in pinned if r["status"] == COSMETIC]
    nck = [r for r in pinned if r["status"] == NOT_CHECKABLE]
    # A tree that cannot answer must not print that it did. The count of artifacts this checkout
    # could actually verify leads, and the ones it could not are named with the reason.
    print(f"\n{len(pinned) - len(nck)} of {len(pinned)} pinned artifacts trace to the code that "
          f"wrote them"
          + (f"; {len(cos)} of them through a change proved cosmetic" if cos else ""))
    if nck:
        print(f"  {len(nck)} could not be checked in this checkout, because the release does not "
              f"carry the file or an input it names:")
        for r in nck:
            print(f"    {r['artifact']}: {r['detail']}")
        print("  That is the release working as documented. In a checkout holding the withheld "
              "files every one of them is verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
