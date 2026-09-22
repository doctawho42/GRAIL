"""BioTransformer's step count, swept, and what sweeping it found about the arm already reported.

The manuscript conceded that this comparator's one declared knob was not turned. Turning it needed
a steps=1 arm from the same harness, and producing one exposed three things about the arms the
paper already prints, none of which were visible before the tool was run twice under a runner that
distinguishes a crash from an answer.

1. A CRASH WAS NOT AN ANSWER. Both frozen prediction files were written by a runner that caught
   every exception and stored []. A substrate whose run crashed and a substrate for which
   BioTransformer genuinely predicts nothing were the same record, so the comparator was scored
   zero recall for its own crashes and nothing could count them.

2. THE TWO FROZEN FILES WERE NOT TWO PURPOSES. The Supporting Information records that they
   disagree and attributes it to each being right for its own population. Measured, the
   whole-test-set file reproduces exactly under a per-parent re-run and the comparison-set file
   does not, because the latter was assembled by joining an external CSV on a precursor
   identifier and silently dropping rows that did not join.

3. THE KNOB IS THE BUDGET AXIS. A second generation's candidate set CONTAINS the first's on every
   substrate where both complete, so it cannot lose a structure; what it does is emit an order of
   magnitude more of them. That is the free parameter this paper is about, appearing inside one
   tool rather than between two.

Every figure here is computed from the artifacts named in `inputs`. Nothing is carried in from a
conversation, and the two failure-rate strata are reported with the ceiling a stratum of their
size could reach, because a factor near its ceiling says only that the stratum is large.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)

FRESH_291   = "results/bt_steps/preds_s1_n291.json"
FRESH_S2    = "results/bt_steps/preds_s2_n291.json"
FRESH_1170  = "results/bt_steps/preds_s1_n1170.json"
# The arm AS PUBLISHED, kept at its own path because the working path now holds the per-parent
# re-run that replaced it. Comparing the two is the whole of section `reproduction`, and
# after the replacement both names pointed at one file, which would have reported perfect
# agreement and deleted the finding.
FROZEN_291  = "results/biotransformer_arm_as_published.json"
FROZEN_1170 = "results/biotransformer_fulltest_preds.json"
PARENTS_291 = "results/bt_steps/parents_n291.txt"

# Strata tested against the failures. Each is a structural feature a reader can check by eye, and
# each was chosen BEFORE the full arm finished -- phosphorus and sulfur because the two exceptions
# recovered from stderr name them, the imidic acid because this corpus stores amides in that form
# and the hypothesis was that the dialect was the cause. It was not.
STRATA = {
    "phosphorus": re.compile(r"P"),
    "sulfur": re.compile(r"S|s"),
    "imidic_acid": re.compile(r"N=C\(O\)|C\(O\)=N"),
    "charged_ring_nitrogen": re.compile(r"\[[nN]\+\]"),
}


# Every input is read ONCE, through here, and its digest is kept. This producer was first run
# while other processes were working in the same checkout and it returned a comparison-set
# reproduction of 82 of 262 where a hand measurement of the same two files gave 89 of 276 -- it
# had read a state that is not on disk now. Nothing in it could have noticed: a producer that
# re-opens a file per question cannot tell a changed input from a consistent one, and the numbers
# it wrote were plausible. The digests are re-checked before the report is written, and a change
# refuses rather than reports.
_READ: dict = {}


def _art(rel):
    if rel not in _READ:
        raw = (ROOT / rel).read_bytes()
        _READ[rel] = (hashlib.sha256(raw).hexdigest(), json.loads(raw.decode()))
    return _READ[rel][1]


def _verify_inputs_unchanged() -> None:
    moved = []
    for rel, (digest, _) in _READ.items():
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if now != digest:
            moved.append(f"{rel}: read at {digest[:12]}, now {now[:12]}")
    if moved:
        sys.exit("REFUSING: an input changed while this producer was running, so its figures "
                 "describe no single state of the repository:\n  " + "\n  ".join(moved))


def _compare(fresh_rel, frozen_rel):
    """How a fresh per-parent run relates to a frozen file, structure by structure."""
    fresh, frozen = _art(fresh_rel), _art(frozen_rel)
    both = [k for k in fresh if k in frozen]
    eq_list = sum(1 for k in both if fresh[k] == frozen[k])
    eq_set = sum(1 for k in both if set(fresh[k]) == set(frozen[k]))
    diff = [k for k in both if set(fresh[k]) != set(frozen[k])]
    return {
        "fresh_parents": len(fresh), "frozen_parents": len(frozen), "shared": len(both),
        "list_identical": eq_list, "set_identical": eq_set,
        "differ": len(diff),
        "frozen_strict_subset_of_fresh": sum(1 for k in diff if set(frozen[k]) < set(fresh[k])),
        "frozen_strict_superset_of_fresh": sum(1 for k in diff if set(frozen[k]) > set(fresh[k])),
        "neither_contains_the_other": sum(
            1 for k in diff if not (set(frozen[k]) < set(fresh[k]))
            and not (set(frozen[k]) > set(fresh[k]))),
        "metabolites_frozen": sum(len(frozen[k]) for k in both),
        "metabolites_fresh": sum(len(fresh[k]) for k in both),
        # Structures the frozen file carries that the re-run never produces anywhere. A join that
        # merely dropped rows could not do this: it says the frozen file came from a DIFFERENT
        # invocation, so the arm cannot be repaired by re-joining and can only be replaced.
        "frozen_structures_the_rerun_never_produces": len(
            {s for v in frozen.values() for s in v} - {s for v in fresh.values() for s in v}),
        "distinct_structures_frozen": len({s for v in frozen.values() for s in v}),
        "distinct_structures_fresh": len({s for v in fresh.values() for s in v}),
    }


def _failures(parents_rel, fresh_rel):
    """A parent absent from a completed run's output is one the tool could not answer for.

    Read by ABSENCE rather than from the sidecar: the sidecar records one invocation's failures and
    a resumed run rewrites it, so the file that survives a resume is not the whole tally.
    """
    parents = [ln.strip() for ln in (ROOT / parents_rel).read_text().splitlines() if ln.strip()]
    ok = _art(fresh_rel)
    return parents, [p for p in parents if p not in ok]


def _enrichment(parents, fails):
    n, K = len(parents), len(fails)
    out = {}
    for name, rx in STRATA.items():
        G = sum(1 for p in parents if rx.search(p))
        g = sum(1 for p in fails if rx.search(p))
        if not G:
            continue
        base = K / n
        # The largest factor a stratum of this size could show, reached when every failure is in
        # it. A stratum covering most of the population cannot be enriched much whatever it does.
        out[name] = {
            "in_population": G, "in_failures": g,
            "rate_in_stratum": round(g / G, 4),
            "enrichment": round((g / G) / base, 2) if base else None,
            "ceiling": round((min(K, G) / G) / base, 2) if base else None,
        }
    return {"n": n, "failures": K, "baseline_rate": round(K / n, 4), "strata": out}


def _load_pools():
    pools, refs = {}, {}
    for spec in ("results/widepools_implicit/w*.json", "results/widepools_k30/all.json"):
        for f in sorted(glob.glob(str(ROOT / spec))) or [str(ROOT / spec)]:
            d = json.loads(Path(f).read_text())
            pools.setdefault(Path(f).parent.name, {}).update(d["pools"])
            refs.update(d["references"])
    keys = list(pools)
    subs = sorted(set.intersection(*[set(pools[k]) for k in keys]) if keys else set())
    return [s for s in subs if refs.get(s)], refs


def _recall(arm_preds, subs, refs, pool):
    from bank_without_selection import _dedup, _key as tautkey
    parent = {s: tautkey(s) for s in subs}
    N = float(sum(len(refs[s]) for s in subs))
    ranked = {s: [k for k in _dedup(arm_preds.get(s, []), max(KS) + 5, pool)
                  if k and k != parent[s]][:max(KS)] for s in subs}
    return {str(k): round(sum(len(set(ranked[s][:k]) & set(refs[s])) for s in subs) / N, 4)
            for k in KS}


def main() -> int:
    subs, refs = _load_pools()
    parents291, fails291 = _failures(PARENTS_291, FRESH_291)
    s1, s2 = _art(FRESH_291), _art(FRESH_S2)
    paired = [s for s in subs if s in s1 and s in s2]

    report = {
        "provenance": stamp(__file__),
        "question": "what BioTransformer's one declared knob does, and what running it twice "
                    "revealed about the arms already reported",
        "reproduction": {
            "whole_test_set": _compare(FRESH_1170, FROZEN_1170),
            "comparison_set": _compare(FRESH_291, FROZEN_291),
        },
        "failures": {
            "comparison_set": _enrichment(parents291, fails291),
        },
        "steps_two": {},
        "inputs": record_inputs([ROOT / r for r in
                                 (FRESH_291, FRESH_S2, FRESH_1170, FROZEN_291, FROZEN_1170)]),
    }

    # how the whole-test-set file's empties divide, now that a crash can be told from an answer
    frozen1170, fresh1170 = _art(FROZEN_1170), _art(FRESH_1170)
    empty = {k for k, v in frozen1170.items() if not v}
    crashed = {k for k in frozen1170 if k not in fresh1170}
    report["failures"]["whole_test_set"] = {
        "n": len(frozen1170), "frozen_empties": len(empty),
        "of_those_a_crash": len(empty & crashed),
        "of_those_a_genuine_empty": len(empty - crashed),
        "crashes_not_empty_in_frozen": len(crashed - empty),
        "crash_share_of_population": round(len(crashed) / len(frozen1170), 4),
        "references_the_crashes_carry": int(sum(len(refs.get(s, [])) for s in crashed)),
    }

    # the knob itself
    contained = sum(1 for s in paired if set(s1[s]) <= set(s2[s]))
    report["steps_two"] = {
        "attempted": len(parents291),
        "completed": len(s2),
        "crashed_or_timed_out": len(parents291) - len(s2),
        "paired_with_steps_one": len(paired),
        "steps_one_set_contained_in_steps_two": contained,
        "generation_one_structures_lost": sum(len(set(s1[s]) - set(s2[s])) for s in paired),
        "mean_emitted_steps_one": round(st.mean([len(s1[s]) for s in paired]), 1),
        "mean_emitted_steps_two": round(st.mean([len(s2[s]) for s in paired]), 1),
    }
    report["steps_two"]["emission_ratio"] = round(
        report["steps_two"]["mean_emitted_steps_two"]
        / report["steps_two"]["mean_emitted_steps_one"], 1)

    with Pool() as pool:
        report["recall_micro"] = {
            "population": {"comparison_set": len(subs),
                           "paired_for_the_steps_contrast": len(paired)},
            "biotransformer_as_printed": _recall(_art(FROZEN_291), subs, refs, pool),
            "biotransformer_per_parent": _recall(_art(FRESH_291), subs, refs, pool),
            "steps_one_paired": _recall(s1, paired, refs, pool),
            "steps_two_paired": _recall(s2, paired, refs, pool),
        }
    r = report["recall_micro"]
    r["correction_delta"] = {k: round(r["biotransformer_per_parent"][k]
                                      - r["biotransformer_as_printed"][k], 4) for k in
                             r["biotransformer_as_printed"]}
    r["steps_delta"] = {k: round(r["steps_two_paired"][k] - r["steps_one_paired"][k], 4)
                        for k in r["steps_one_paired"]}
    r["largest_correction"] = max(abs(v) for v in r["correction_delta"].values())
    r["largest_steps_gain"] = max(r["steps_delta"].values())

    # What the sweep cost, and what the arm that was NOT run would have cost. Both read from the
    # runs' own sidecars rather than from a stopwatch, and the projection is the completed
    # steps=2 rate applied to the wider population -- a partial run's rate understated it.
    cost = {}
    for tag, rel in (("steps_one_1170", "results/bt_steps/preds_s1_n1170.json.run.json"),
                     ("steps_two_291", "results/bt_steps/preds_s2_n291.json.run.json")):
        s = _art(rel)
        cost[tag] = {"substrates": s["done"], "hours": round(s["elapsed_s"] / 3600, 2),
                     "per_minute": round(s["done"] / s["elapsed_s"] * 60, 2),
                     "workers": s["workers"], "timeout_s": s["timeout_s"]}
    _r2 = cost["steps_two_291"]["per_minute"]
    cost["steps_two_1170_projected_hours"] = round(1170 / _r2 / 60)
    cost["not_run"] = ("steps=2 over the whole evaluated test set")
    report["cost"] = cost

    _verify_inputs_unchanged()
    out = ROOT / "results/biotransformer_steps.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    rp = report["reproduction"]
    print(f"  whole test set reproduces list-identically: "
          f"{rp['whole_test_set']['list_identical']} of {rp['whole_test_set']['shared']}")
    print(f"  comparison set reproduces set-identically : "
          f"{rp['comparison_set']['set_identical']} of {rp['comparison_set']['shared']}"
          f"  (metabolite deficit "
          f"{rp['comparison_set']['metabolites_fresh'] - rp['comparison_set']['metabolites_frozen']})")
    print(f"  steps=2 completes {report['steps_two']['completed']} of "
          f"{report['steps_two']['attempted']}, contains steps=1 on "
          f"{report['steps_two']['steps_one_set_contained_in_steps_two']} of "
          f"{report['steps_two']['paired_with_steps_one']}, emits "
          f"{report['steps_two']['emission_ratio']}x")
    print(f"  largest correction to the printed column: {r['largest_correction']:+.4f}; "
          f"largest steps=2 gain: {r['largest_steps_gain']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
