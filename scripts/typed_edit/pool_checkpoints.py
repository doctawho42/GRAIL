#!/usr/bin/env python3
"""Which checkpoints scored each candidate pool, established by reproduction rather than recorded.

A pool is a set of candidates with a generator score and a filter score on each. Which models
produced those scores is the pool's defining configuration and most of the pools here record it
nowhere: the field was added to the builder after they were written. A curve assembled from pools
scored by different models measures the model as much as the parameter it varies, and that is not
hypothetical -- three points of the rule-budget curve were scored by a filter checkpoint nobody
deploys, while the other two were scored by the deployed one, and nothing in the artifacts said so.

Both scores are deterministic functions of the substrate and the candidate, so the model can be
recovered: score a pool's own candidates with each checkpoint in the tree and count exact matches.
The model that wrote them matches every score and every other matches almost none, which makes the
answer a measurement with a margin rather than a recollection.

    python scripts/typed_edit/pool_checkpoints.py
    python scripts/typed_edit/pool_checkpoints.py --stage filter     # the cheap half alone
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

def deployed_runs() -> dict:
    """{stage: run} for what this repository actually releases, read from git rather than named.

    "Deployed" is not a fact about a directory on this machine, it is a fact about what ships, and
    naming it in a constant is how the two came apart: a constant said both stages were one run
    while the tree tracked a generator from one and a filter from another, and the artifact the
    comparison is read from matched the tracked pair rather than the named one. Reading the answer
    out of `git ls-files` makes the gate follow the release instead of a recollection of it, so
    changing what is released changes what every check here demands, in one place.
    """
    import subprocess

    out = {}
    try:
        tracked = subprocess.run(["git", "ls-files", "artifacts/"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30).stdout.splitlines()
    except Exception:
        tracked = []
    for rel in tracked:
        parts = rel.split("/")
        if len(parts) >= 4 and parts[-2] == "checkpoints" and parts[-1].endswith(".pt"):
            stage = parts[-1][:-3]
            if stage in ("generator", "filter"):
                out.setdefault(stage, parts[1])
    return out


DEPLOYED_BY_STAGE = deployed_runs()
# Kept for the printed header only; where the two stages differ this is not a single answer.
DEPLOYED = DEPLOYED_BY_STAGE.get("generator") or "full5000_implicit"
# Every trained run whose checkpoints could plausibly have scored a pool. A candidate that is not
# in this list cannot be identified, so an unidentified pool is reported as such and never as the
# deployed one by default.
CANDIDATES = ("full5000_implicit", "full5000_priors", "full5000_single",
              "full5000_expanded_control", "full2500_single",
              "multiseed_full5000_implicit_seed0", "multiseed_full5000_seed0")
# How many substrates and candidates to test. The signal is all-or-nothing, so a handful settles
# it; the counts are reported so a reader can see the margin rather than trust the verdict.
N_SUBS, N_CANDS = 3, 40


# Sharded pool directories a published number is read from. The comparison table itself comes
# from one of these, so leaving them out of the check left the paper's central artifact
# unexamined while the validation pools beside it were being verified.
SHARDED = ("widepools_implicit", "widepools_fulltest", "widepools_k30", "widepools_k30_fulltest")


def pools_to_check():
    """{name: (path, rule budget or None)} for every pool a published number is read from."""
    out = {}
    for path in sorted(glob.glob(str(ROOT / "results" / "valpools_k*" / "all.json"))):
        blob = json.loads(Path(path).read_text())
        out[Path(path).parent.name] = (Path(path), blob.get("top_k"))
    merged = ROOT / "results" / "val_pools.json"
    if merged.exists():
        out["val_pools"] = (merged, json.loads(merged.read_text()).get("top_k"))
    for name in SHARDED:
        shards = sorted(glob.glob(str(ROOT / "results" / name / "w*.json")))
        if shards:
            first = json.loads(Path(shards[0]).read_text())
            out[name] = (Path(shards[0]), first.get("top_k"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("filter", "generator", "both"), default="both")
    ap.add_argument("--out", default=str(ROOT / "results" / "pool_checkpoints.json"))
    args = ap.parse_args()

    from bank_without_selection import _load
    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.workflows.factory import build_filter, build_generator

    pools = pools_to_check()
    if not pools:
        raise SystemExit("no pools found")

    def digest(path):
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:16]

    def load(kind, builder):
        """{run: model} and {run: digest}. Two runs can hold the same file.

        Seed 0 of the multi-seed sweep and the deployed run ship a byte-identical generator, so
        both reproduce a pool's scores exactly and neither can be told from the other by any
        measurement over scores. That is not an ambiguity about which model wrote the pool; it is
        two names for one file, and the identification groups by digest so it reads as such
        instead of reporting the pool as unidentified.
        """
        models, digests = {}, {}
        for name in CANDIDATES:
            path = ROOT / f"artifacts/{name}/checkpoints/{kind}.pt"
            if not path.exists():
                continue
            try:
                models[name] = _load(path, builder)
                digests[name] = digest(path)
            except Exception as exc:                       # a run whose arch no longer builds
                print(f"  ({name}/{kind}: {exc})", file=sys.stderr)
        return models, digests

    filters, filter_digests = (
        load("filter", lambda a, r: build_filter(FilterConfig(**a)))
        if args.stage in ("filter", "both") else ({}, {}))
    generators, generator_digests = (
        load("generator", lambda a, r: build_generator(GeneratorConfig(**a), r))
        if args.stage in ("generator", "both") else ({}, {}))
    digests_by_stage = {"filter": filter_digests, "generator": generator_digests}

    rows = {}
    for name, (path, budget) in pools.items():
        blob = json.loads(path.read_text())
        subs = sorted(blob["pools"])[:N_SUBS]
        entry = {"path": str(path.relative_to(ROOT)), "rule_budget": budget,
                 "records_its_own_checkpoints": bool(blob.get("checkpoints")),
                 "recorded": blob.get("checkpoints")}

        for stage, models in (("filter", filters), ("generator", generators)):
            if not models:
                continue
            counts, total = {}, 0
            for model_name, model in models.items():
                hit = shared = 0
                for s in subs:
                    stored = blob["pools"][s][:N_CANDS]
                    if not stored:
                        continue
                    if stage == "filter":
                        got = dict(zip((c["smiles"] for c in stored),
                                       model.score_batch(s, [c["smiles"] for c in stored])))
                    else:
                        det = model.generate_scored_with_details(
                            s, top_k=budget or 7581, threshold=None, compute_sites=False)
                        got = {d[0]: float(d[1]) for d in det}
                    for c in stored:
                        if c["smiles"] in got:
                            shared += 1
                            if abs(float(got[c["smiles"]]) - float(c[stage])) < 1e-6:
                                hit += 1
                counts[model_name] = {"exact": hit, "comparable": shared}
                total = max(total, shared)
            # A checkpoint is identified when it reproduces every comparable score and no file
            # with a different digest does. Runs sharing one file are one candidate, and a rival
            # that happens to agree on a few scores is not a rival: an untrained-on-this-substrate
            # checkpoint matched 4 of 120 by coincidence and, under a rule that demanded no
            # agreement at all from the runner-up, made a pool with a perfect unique match read
            # as unidentified. What matters is that the perfect match is unique.
            digests = digests_by_stage[stage]
            best = max(counts, key=lambda m: counts[m]["exact"])
            best_n, best_of = counts[best]["exact"], counts[best]["comparable"]
            tied = sorted(m for m in counts if digests.get(m) == digests.get(best))
            others = [m for m in counts if m not in tied]
            runner = max(others, key=lambda m: counts[m]["exact"], default=None)
            runner_n = counts[runner]["exact"] if runner else 0
            identified = bool(
                best_of and best_n == best_of
                and all(counts[m]["exact"] < counts[m]["comparable"] for m in others))
            entry[stage] = {
                "identified_as": best if identified else None,
                "matches": f"{best_n} of {best_of}",
                "sha256_16": digests.get(best),
                "runs_sharing_that_file": tied,
                "next_best_with_a_different_file": f"{runner} {runner_n}" if runner else None,
                "margin": (f"{best_n}/{best_of} against {runner_n}/{counts[runner]['comparable']}"
                           if runner else None),
                "is_the_deployed_run": bool(
                    identified and DEPLOYED_BY_STAGE.get(stage) in tied),
                "the_released_run_for_this_stage": DEPLOYED_BY_STAGE.get(stage),
                "counts": counts}
        rows[name] = entry

    # `s in (args.stage, "both")` reads naturally and is wrong: with --stage both the tuple is
    # ("both", "both") and neither stage is in it, so every check below was skipped in exactly the
    # mode that runs them all. The artifact then recorded an empty disagreement list while holding
    # per-pool evidence of a disagreement, and budget_curve.py's gate, which reads that list,
    # passed on it.
    stages = [s for s in ("generator", "filter") if args.stage in (s, "both")]
    disagreeing = {}
    for stage in stages:
        seen = {r[stage]["identified_as"] for r in rows.values()
                if stage in r and r[stage]["identified_as"]}
        if len(seen) > 1:
            disagreeing[stage] = sorted(seen)
    off_deployed = sorted(n for n, r in rows.items()
                          if any(stage in r and r[stage]["identified_as"]
                                 and not r[stage]["is_the_deployed_run"] for stage in stages))
    unidentified = sorted(n for n, r in rows.items()
                          if any(stage in r and not r[stage]["identified_as"] for stage in stages))

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs(p for p, _ in pools.values()),
        "question": ("which generator and which filter scored each candidate pool, where the pool "
                     "itself records nothing"),
        "method": (f"score each pool's own candidates on {N_SUBS} substrates with every trained "
                   f"checkpoint in the tree and count exact agreements; the model that wrote them "
                   f"matches all of them and every other matches none"),
        "deployed_run": DEPLOYED,
        "released_runs_by_stage": DEPLOYED_BY_STAGE,
        "how_the_release_is_determined": (
            "read from `git ls-files artifacts/`: the checkpoint a stage ships is the one this "
            "repository tracks for it, so the gate follows what is released rather than a name "
            "written beside it"),
        "candidates_considered": list(CANDIDATES),
        "pools": rows,
        "stages_whose_checkpoint_is_not_the_same_across_pools": disagreeing,
        "pools_scored_by_something_other_than_the_deployed_run": off_deployed,
        "pools_whose_checkpoint_could_not_be_identified": unidentified,
        "reading": (
            "A curve across pools is a comparison of the parameter the pools vary only if every "
            "other thing they vary is held. The model is one of those things and it was the one "
            "no artifact recorded, so it is measured here and the measurement is what the curve's "
            "producer refuses on."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{'pool':16s} {'budget':>7s}  {'generator':>34s}  {'filter':>34s}")
    for name, r in rows.items():
        def cell(stage):
            if stage not in r:
                return "not tested"
            c = r[stage]
            mark = "" if c["is_the_deployed_run"] else "  <-- not the deployed run"
            return f"{c['identified_as'] or 'unidentified'} ({c['matches']}){mark}"
        print(f"{name:16s} {str(r['rule_budget']):>7s}  {cell('generator'):>34s}  "
              f"{cell('filter'):>34s}")
    if disagreeing:
        print(f"\nthe pools do not share one {', '.join(disagreeing)}: "
              + "; ".join(f"{k}: {v}" for k, v in disagreeing.items()))
    else:
        print("\nevery identified pool shares one generator and one filter")
    if off_deployed:
        print(f"scored by something other than the deployed run: {off_deployed}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
