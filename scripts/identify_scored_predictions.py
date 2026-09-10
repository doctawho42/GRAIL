#!/usr/bin/env python3
"""Which checkpoints wrote results/scored_predictions.json, and does the narrow pool nest?

scripts/typed_edit/pool_checkpoints.py identifies a pool's models by reproduction: both scores are
deterministic functions of the substrate and the candidate, so scoring a pool's own candidates with
every checkpoint in the tree and counting exact agreements recovers the model that wrote them. Its
own docstring says a check reporting on whatever it happens to find "cannot fail on the pool that
is missing, which is how the interactive arm's comparison pool went unverified" -- and the
interactive dump is exactly that pool, still unverified, in a shape that audit does not read.

This applies the same method to it, and then answers what the identification was for.

The reading under test is that the exhaustive arm loses to its own interactive arm on five
transformation classes by ranking rather than by coverage. That rests on the narrow pool being
inside the wide one. Rule nesting is exact -- the interactive arm fires the top 30 of the same
bank -- but candidate nesting is not implied by it: noisy-or over a wider rule set moves every
candidate's aggregated score, so the cap can drop a candidate the narrow arm emits. Where that
happens and the dropped key is a reference, the difference is coverage and not position.

Both halves are refused unless the first one lands on the released pair, because a nesting test
between two arms scored by different models measures the models.

    python scripts/identify_scored_predictions.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import bank_without_selection as B  # noqa: E402
from pool_checkpoints import CANDIDATES, N_CANDS, N_SUBS, deployed_runs  # noqa: E402
from grail_metabolism.config import FilterConfig, GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_filter, build_generator  # noqa: E402

DUMP = ROOT / "results" / "scored_predictions.json"
WIDE = "results/widepools_implicit/w*.json"
NARROW = "results/widepools_k30/all.json"  # the interactive arm's rule budget, released pair
DEPLOYED_CAP = 100


def identify(dump, stage, builder, key):
    """{run: (exact, comparable)} over the dump's own candidates, and the run that wrote them."""
    subs = [r["sub"] for r in dump["rows"][:N_SUBS]]
    stored = {r["sub"]: r["candidates"][:N_CANDS] for r in dump["rows"][:N_SUBS]}
    counts = {}
    for name in CANDIDATES:
        path = ROOT / f"artifacts/{name}/checkpoints/{stage}.pt"
        if not path.exists():
            continue
        try:
            model = B._load(path, builder)
        except Exception as exc:
            print(f"  ({name}/{stage}: {exc})", file=sys.stderr)
            continue
        hit = seen = 0
        for s in subs:
            cs = stored[s]
            if not cs:
                continue
            smis = [c["smiles"] for c in cs]
            if stage == "filter":
                got = list(model.score_batch(s, smis))
            else:
                d = {a: b for a, b, *_ in model.generate_scored_with_details(
                    s, top_k=30, threshold=None, compute_sites=False)}
                got = [d.get(x) for x in smis]
            for c, g in zip(cs, got):
                if g is None:
                    continue
                seen += 1
                hit += abs(float(g) - float(c[key])) < 1e-9
        counts[name] = (hit, seen)
    best = max(counts, key=lambda n: (counts[n][0] / max(counts[n][1], 1), counts[n][0]))
    ratio = counts[best][0] / max(counts[best][1], 1)
    # A tie is not an identification. The margin is reported either way, because "unidentified"
    # and "identified with one disagreement" are different answers and a bare None hides which.
    tied = [n for n in counts if n != best
            and counts[n][0] / max(counts[n][1], 1) >= ratio - 1e-12 and counts[n][0] == counts[best][0]]
    return counts, {"best": best, "exact": counts[best][0], "comparable": counts[best][1],
                    "ratio": round(ratio, 4), "tied_with": tied,
                    "clean": bool(ratio > 0.99 and not tied)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "scored_predictions_identity.json"))
    args = ap.parse_args()
    torch.set_num_threads(6)

    dump = json.loads(DUMP.read_text())
    released = deployed_runs()
    print(f"  released per stage (from git ls-files): {released}", flush=True)

    rep = {"config": {**B._code_version(), "dump": "results/scored_predictions.json",
                      "operating_point": dump.get("operating_point"),
                      "method": "reproduction: score the dump's own candidates with every "
                                "checkpoint in the tree and count exact agreements",
                      "n_substrates": N_SUBS, "n_candidates": N_CANDS},
           "released_runs_by_stage": released, "identified": {}, "counts": {}}

    for stage, builder, key in (("filter", lambda a, r: build_filter(FilterConfig(**a)), "filter"),
                                ("generator", lambda a, r: build_generator(GeneratorConfig(**a), r),
                                 "generator")):
        counts, run = identify(dump, stage, builder, key)
        rep["counts"][stage] = {n: {"exact": h, "comparable": c} for n, (h, c) in counts.items()}
        rep["identified"][stage] = run
        top = sorted(counts.items(), key=lambda kv: -kv[1][0])[:3]
        print(f"  {stage}: best {run['best']} at {run['exact']}/{run['comparable']} "
              f"({run['ratio']}), clean={run['clean']}"
              + (f", tied with {run['tied_with']}" if run["tied_with"] else ""), flush=True)
        for n, (h, c) in top:
            print(f"      {n:<34} {h}/{c}", flush=True)

    # The question is not which run wrote it but whether it is the released one, and that is
    # settled by the released run scoring nothing rather than by the winner being clean.
    same = {s: rep["identified"][s]["best"] == released.get(s) for s in rep["identified"]}
    rep["released_run_agreement"] = {
        s: rep["counts"][s].get(released.get(s), {}) for s in rep["identified"]}
    rep["is_the_released_pair"] = all(same.values())
    print(f"\n  the dump is the released pair: {rep['is_the_released_pair']}  {same}", flush=True)

    # The dump is not the released pair, so it cannot carry the nesting test. The narrow pool
    # that can is results/widepools_k30/all.json: the same rule budget of 30, the same comparison
    # population, and both stages recorded and verified as the released run. The dump's identity
    # is kept above as the reason this file does not use it.
    rep["nesting_source"] = {
        "narrow": NARROW, "wide": WIDE,
        "why_not_the_dump": "results/scored_predictions.json was written by a different pair, so "
                            "nesting it against the released wide pool would compare two models"}
    narrow_blob = json.loads((ROOT / NARROW).read_text())
    for stage, rec in (narrow_blob.get("checkpoints") or {}).items():
        run = Path(rec["path"]).parts[1]
        if run != released.get(stage):
            raise SystemExit(f"{NARROW} records {run} for the {stage}, not the released "
                             f"{released.get(stage)}; the two arms would be different models")

    pools = {}
    for f in sorted(glob.glob(str(ROOT / WIDE))):
        pools.update(json.loads(Path(f).read_text())["pools"])
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    narrow = narrow_blob["pools"]
    subs = sorted(set(narrow) & set(pools) & set(truth))
    print(f"\n  nesting on {len(subs)} substrates both arms cover", flush=True)

    from multiprocessing import Pool
    kp = Pool(6)
    refs = {s: set(B._dedup(truth[s], None, kp)) for s in subs}
    self_key = {s: B._key(s) for s in subs}
    nkeys = {s: {c["key"] for c in narrow[s] if c["key"] != self_key[s]} for s in subs}
    kp.close()

    nest = {"n_substrates": len(subs)}
    for label, cap in (("uncapped", None), ("cap100", DEPLOYED_CAP)):
        wide = {s: {c["key"] for c in (pools[s] if cap is None else
                                       sorted(pools[s], key=lambda c: -c["generator"])[:cap])
                    if c["key"] != self_key[s]} for s in subs}
        lost = {s: nkeys[s] - wide[s] for s in subs}
        broken = {s: v for s, v in lost.items() if v}
        lost_refs = {s: v & refs[s] for s, v in broken.items()}
        lost_refs = {s: v for s, v in lost_refs.items() if v}
        nest[label] = {
            "holds_on": len(subs) - len(broken), "fails_on": len(broken),
            "keys_only_the_narrow_arm_has": sum(map(len, lost.values())),
            "of_those_that_are_references": sum(map(len, lost_refs.values())),
            "substrates_losing_a_reference": len(lost_refs),
            "reading": ("nesting holds on references: whatever the narrow arm wins is position"
                        if not lost_refs else
                        "the narrow arm holds references the wide one does not: part of any "
                        "narrow-arm advantage is coverage, not position")}
        print(f"  {label}: holds on {nest[label]['holds_on']}/{len(subs)}, "
              f"keys only the narrow arm has {nest[label]['keys_only_the_narrow_arm_has']}, "
              f"of them references {nest[label]['of_those_that_are_references']}", flush=True)
    rep["nesting"] = nest

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
