"""M0 census: how many annotated metabolites are depth-2-reachable-in-the-rule-env
but NOT depth-1? Go/no-go for the Stage-2b multi-step RECALL claim (the diversity
method claim does not depend on this). Cheap, CPU, no training."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey


def _children_ik(sub, generator, top_k, max_pool):
    # NOTE: the real Generator.generate_scored_with_details(sub, top_k=None,
    # threshold=None, compute_sites=True) has no `max_pool` kwarg -- only the
    # synthetic test stub does. `max_pool` is accepted here (and by
    # census_depth2) to match the task brief's interface, but is intentionally
    # NOT forwarded to the real generator call, since it has no equivalent
    # there; top_k alone controls the candidate-rule pool size.
    out = {}
    for smiles, gen_score, rule_id, *_ in generator.generate_scored_with_details(
        sub, top_k=top_k, compute_sites=False
    ):
        ik = _tautomer_inchikey(smiles)
        if ik is not None and ik not in out:
            out[ik] = smiles
    return out  # ik -> smiles


def census_depth2(sub, annotated_ik, generator, top_k=200, max_pool=150):
    d1 = _children_ik(sub, generator, top_k, max_pool)          # ik -> smiles at depth 1
    depth1_hits = set(d1) & annotated_ik
    d2 = set()
    for child_smiles in d1.values():                            # expand each depth-1 child once
        d2 |= set(_children_ik(child_smiles, generator, top_k, max_pool))
    depth2_only = (d2 & annotated_ik) - depth1_hits
    reached = depth1_hits | (d2 & annotated_ik)
    return {
        "n_annot": len(annotated_ik),
        "depth1": len(depth1_hits),
        "depth2_only": len(depth2_only),
        "unreach": len(annotated_ik - reached),
    }


def main() -> None:
    from grail_metabolism.config import DatasetConfig, GeneratorConfig
    from grail_metabolism.model.grail import _read_checkpoint
    from grail_metabolism.workflows.data import load_dataset_bundle
    from grail_metabolism.workflows.factory import build_generator

    ap = argparse.ArgumentParser(description="M0 depth-2 reachability census")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--substrates", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-pool", type=int, default=150)
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/census_multistep.json"))
    args = ap.parse_args()

    state = _read_checkpoint(args.gen_ckpt)
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False); generator.eval()

    cfg = DatasetConfig(use_clean_splits=True, standardize=False, cache_preprocessed=False,
                        max_test_substrates=args.substrates + 60, sampling_seed=0)
    bundle = load_dataset_bundle(cfg)
    frame = getattr(bundle, args.split)

    agg = {"n_annot": 0, "depth1": 0, "depth2_only": 0, "unreach": 0, "n_subs": 0}
    for sub, prods in list(frame.map.items())[: args.substrates]:
        annotated_ik = {_tautomer_inchikey(p) for p in prods} - {None}
        if not annotated_ik:
            continue
        c = census_depth2(
            sub, annotated_ik, generator, top_k=args.top_k, max_pool=args.max_pool
        )
        for k in ("n_annot", "depth1", "depth2_only", "unreach"):
            agg[k] += c[k]
        agg["n_subs"] += 1
    agg["depth2_only_frac"] = agg["depth2_only"] / max(agg["n_annot"], 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
