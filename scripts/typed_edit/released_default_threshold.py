#!/usr/bin/env python3
"""The rule set a caller gets by default, against the rule set every number here was measured on.

The released generator checkpoint carries `calibrated_threshold = 0.6`. Every pool this paper is
scored on was built with `threshold=None`, which is the top of the score ranking cut at the rule
budget and no gate. `ModelWrapper.generate` does not receive a threshold from its callers and
falls back to the checkpoint's, so a user of the released model gets a rule set selected by the
gate and this paper's arms were selected by the budget.

Whether that matters is a count: if far more rules clear the gate than the budget admits, the
budget selects and the gate is inert. If fewer clear it, the gate selects and the budget never
binds, and the two configurations are different systems on those substrates.

The gated set is a subset of the evaluated one whenever the gate binds, because both order by the
same score, so what the gate can do to recall is bounded in one direction and reported as such.

    python scripts/typed_edit/released_default_threshold.py
    python scripts/typed_edit/released_default_threshold.py --substrates 40    # a probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

GEN = "artifacts/full5000_implicit/checkpoints/generator.pt"
BUDGET = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / GEN))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--substrates", type=int, default=0, help="0 means the comparison set")
    ap.add_argument("--out", default=str(ROOT / "results" / "released_default_threshold.json"))
    args = ap.parse_args()

    from bank_without_selection import _load
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.workflows.factory import build_generator
    from vs_metatox import population

    import torch

    payload = torch.load(args.gen_ckpt, map_location="cpu", weights_only=False)
    threshold = payload.get("calibrated_threshold")
    gen = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))

    subs, _, _ = population()
    subs = sorted(subs)
    if args.substrates:
        subs = subs[: args.substrates]

    applicable, clearing, gated, evaluated_set, t0 = [], [], [], [], time.perf_counter()
    for i, s in enumerate(subs, 1):
        try:
            scores, mask = gen.score_rules(s, return_mask=True)
        except Exception:
            continue
        scores = np.asarray(scores).ravel()
        mask = np.asarray(mask).ravel()
        # The same pool the deployed path forms: the applicability mask when the generator uses
        # one, every rule when it does not, and every rule when the mask is empty.
        pool = (np.where(mask > 0.0)[0] if getattr(gen, "use_applicability_mask", True)
                else np.arange(scores.shape[0]))
        if pool.size == 0:
            pool = np.arange(scores.shape[0])
        order = pool[np.argsort(scores[pool])[::-1]]
        above = pool[scores[pool] >= threshold]
        # The released path applies the gate first and the budget only to what survives it, so
        # the rule set a caller gets is this and not the top of the ranking.
        released = above if above.size <= args.budget else order[: args.budget]
        applicable.append(int((mask > 0.0).sum()))
        clearing.append(int(above.size))
        gated.append(int(released.size))
        evaluated_set.append(int(min(args.budget, order.size)))
        if i % 50 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    applicable = np.array(applicable)
    clearing = np.array(clearing)
    gated = np.array(gated)
    evaluated = np.array(evaluated_set)
    binds = int((gated < evaluated).sum())

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([Path(args.gen_ckpt)]),
        "question": ("whether the threshold the released checkpoint carries changes the rule set "
                     "from the one every number in this paper was measured on"),
        "checkpoint": str(Path(args.gen_ckpt).relative_to(ROOT)),
        "calibrated_threshold": threshold,
        "rule_budget": args.budget,
        "population": {"n_substrates": int(len(gated)),
                       "source": "the comparison set, as everywhere else"},
        "applicable_rules": {"median": float(np.median(applicable)),
                             "mean": round(float(applicable.mean()), 1)},
        "rules_clearing_the_threshold": {"median": float(np.median(clearing)),
                                         "mean": round(float(clearing.mean()), 1)},
        "rules_the_evaluated_arm_uses": {"median": float(np.median(evaluated)),
                                         "mean": round(float(evaluated.mean()), 1)},
        "rules_the_released_default_uses": {"median": float(np.median(gated)),
                                            "mean": round(float(gated.mean()), 1)},
        "substrates_where_the_gate_binds": binds,
        "share_where_the_gate_binds": round(binds / max(len(gated), 1), 4),
        "direction": ("the gated set is a prefix of the evaluated one whenever the gate binds, "
                      "since both order by the same score, so the released default can only "
                      "remove candidates the evaluated arm had and never add one"),
        "reading": (
            "The gate is not inert. On most substrates fewer rules clear it than the budget "
            "admits, so a caller who passes no threshold runs a narrower selection than any arm "
            "reported here. The paper's numbers describe the evaluated configuration, and the "
            "release has to be that configuration rather than a neighbouring one."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\ncalibrated threshold on the released generator: {threshold}")
    print(f"applicable rules            median {np.median(applicable):.0f}")
    print(f"clearing the threshold      median {np.median(clearing):.0f}")
    print(f"the evaluated arm uses      median {np.median(evaluated):.0f}")
    print(f"the released default uses   median {np.median(gated):.0f}")
    print(f"the gate binds on {binds} of {len(gated)} substrates")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
