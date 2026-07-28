#!/usr/bin/env python3
"""Dump the deployed ranking with its per-candidate scores, for calibration work.

`cardinality_crossfit.py` closed the substrate-conditioned direction: a head reading only the
substrate recovers 3-4% of the headroom an oracle cut reaches, because most of that headroom is
knowledge of where the hits fell in the ranking, which the substrate does not determine. The lever
that remains is a stopping rule on the model's own scores, and testing it needs the scores, which no
existing artifact retains.

This runs the deployed operating point unchanged -- the same top_k, threshold, candidate cap and
rank-only policy `factorize_recall.py` uses, so the emitted candidates reproduce
`recall_factorization.json`'s `deployed_top15` -- and writes the ranked candidates with the filter
score, the generator score and the product the pipeline ranks by. `max_output` is lifted so the
stopping rule can be swept past the deployed cut; the first 15 entries of each row are the deployed
output.

The scores come out of `ModelWrapper.generate(return_scores=True)`, which reads the same ranked list
the default path returns, so the dump cannot drift from the deployed behaviour.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from grail_metabolism.config import DatasetConfig
from grail_metabolism.workflows.data import load_dataset_bundle

# Reuse the deployed model construction rather than restating it. Three attempts at restating it
# diverged from the artifact in three different ways -- output normalisation, the generator
# threshold, and the calibrated thresholds the checkpoint payload carries -- each silent. The
# operating-point constants come from the same module for the same reason.
sys.path.insert(0, str(ROOT / "scripts"))
from factorize_recall import (  # noqa: E402
    CANDIDATE_TOP_K,
    FILTER_CANDIDATE_CAP,
    build_deployed_model,
)

DEPLOYED_K = 15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-substrates", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--dump-k", type=int, default=64, help="how deep to record the ranking")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "scored_predictions.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model = build_deployed_model(copy_prior=True)
    gen_threshold = getattr(model.generator, "calibrated_threshold", None)
    print(f"generator threshold {gen_threshold}, {len(model.generator.rule_names)} rules", flush=True)

    # Same dataset configuration factorize_recall.py uses, so the split and its ordering match the
    # artifact this dump has to reproduce. Train and val are trimmed to nothing: only test is read.
    cfg = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8,
        max_test_substrates=args.max_substrates or 0,
        sampling_seed=42,
    )
    bundle = load_dataset_bundle(cfg)
    frame = bundle.test
    subs = list(frame.map.keys())
    print(f"{len(subs)} substrates, recording {args.dump_k} deep", flush=True)

    rows, started = [], time.perf_counter()
    for i, sub in enumerate(subs, 1):
        try:
            scored = model.generate(
                sub,
                top_k=CANDIDATE_TOP_K,
                threshold=gen_threshold,
                max_output=args.dump_k,
                gate_by_filter=False,
                filter_candidate_cap=FILTER_CANDIDATE_CAP,
                return_scores=True,
            )
        except Exception as exc:  # a substrate that fails to enumerate is recorded, not dropped
            rows.append({"sub": sub, "error": str(exc)[:200], "candidates": []})
            continue
        rows.append({
            "sub": sub,
            "candidates": [{"smiles": s, "combined": c, "filter": f, "generator": g}
                           for s, c, f, g in scored],
        })
        if i % 50 == 0:
            rate = (time.perf_counter() - started) / i
            print(f"  {i}/{len(subs)}  {rate:.2f}s/substrate  eta {rate * (len(subs) - i) / 60:.1f}min",
                  flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({
        "operating_point": {"candidate_top_k": CANDIDATE_TOP_K, "filter_candidate_cap": FILTER_CANDIDATE_CAP,
                            "gate_by_filter": False, "deployed_max_output": DEPLOYED_K,
                            "dump_k": args.dump_k},
        "n_substrates": len(rows),
        "n_failed": sum(1 for r in rows if r.get("error")),
        "rows": rows,
    }, indent=1))
    print(f"\nwrote {out}  ({sum(1 for r in rows if r.get('error'))} failed)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
