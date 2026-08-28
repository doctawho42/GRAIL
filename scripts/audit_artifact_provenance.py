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
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import COSMETIC, CURRENT, infer, verify  # noqa: E402

OK = (CURRENT, COSMETIC)

TE = "scripts/typed_edit"
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
    # the pool the target function is derived from, measured on the split
    "results/bank_without_selection_full.json": "scripts/bank_without_selection.py",
    # the comparison that decides the release, and the pools it wrote
    "results/vs_metatox.json": f"{TE}/vs_metatox.py",
    "results/vs_metatox_pools.json": f"{TE}/vs_metatox.py",
    # the negative result that closes the cheap route to the between-group headroom
    "results/group_decode.json": f"{TE}/group_decode.py",
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
    "results/selection_ablation_deployed.json": f"{TE}/selection_verdict.py",
    "results/selection_pools_deployed.json": f"{TE}/selection_ablation_deployed.py",
    "results/case_study.json": f"{TE}/case_study.py",
    "results/case_study_exhaustive.json": f"{TE}/case_study.py",
    # the rule budget: the curve's shape and the check on validation
    "results/h10_verdict.json": f"{TE}/h10_verdict.py",
    # the cap, its upper-bound table and the check that fixed it
    "results/pool_cap_cost.json": f"{TE}/pool_cap_cost.py",
    "results/h9_verdict.json": f"{TE}/h9_verdict.py",
    # the H7 check and the validation pool it reads
    "results/val_pools.json": f"{TE}/build_val_pools.py",
    "results/h7_verdict.json": f"{TE}/h7_verdict.py",
    "paper2/split_manifest.json": f"{TE}/freeze_split.py",
}


def check(rel: str, producer: str | None) -> dict:
    path = ROOT / rel
    if not path.exists():
        return {"artifact": rel, "status": "absent", "detail": "not in this checkout"}
    v = verify(path)
    if v["status"] in OK or producer is None:
        return v
    if v["status"] == "unstamped":
        return infer(path, producer)
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
    for p in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
        rel = str(Path(p).relative_to(ROOT))
        if rel in PINNED:
            continue
        v = verify(p)
        sweep[v["status"]] += 1
        if v["status"] not in OK:
            others.append(v)
    if args.all:
        print(f"\nthe other {sum(sweep.values())} artifacts: " +
              ", ".join(f"{k} {v}" for k, v in sweep.most_common()))
        changed = [o for o in others if o["status"] == "producer_changed"]
        if changed:
            print(f"  {len(changed)} name a producer that has changed since they were written:")
            for o in changed[:10]:
                print(f"    {o['artifact']:<46}{o.get('detail','')[:44]}")

    rep = {"pinned": pinned, "n_pinned": len(pinned), "n_pinned_stale": len(bad),
           "sweep": dict(sweep), "sweep_not_current": others}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    if bad:
        print(f"\nFAIL: {len(bad)} pinned artifacts are not current")
        for r in bad:
            print(f"  {r['artifact']}: {r['status']} -- {r.get('detail','')}")
        return 1
    cos = [r for r in pinned if r["status"] == COSMETIC]
    print(f"\nall {len(pinned)} pinned artifacts trace to the code that wrote them"
          + (f"; {len(cos)} of them through a change proved cosmetic" if cos else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
