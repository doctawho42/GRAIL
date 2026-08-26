"""Train the H8 group scorer on the training split and select it on validation.

Every hyperparameter is chosen by micro recall@15 on validation and the 291 are not read here.
The selection table is written whole rather than only its winner, so a reader can see how flat
or peaked the choice was, and the fusion baseline is computed on the same examples so the
comparison the verdict will report is not against a number from a different population.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from bank_without_selection import _load  # noqa: E402
from grail_metabolism.config import GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402
from group_scorer import (GROUP_FEATURES, Featuriser, GroupScorer, build_examples,  # noqa: E402
                          fusion_recall, reorder_recall)

GRID = [{"hidden": h, "lr": lr, "dropout": dr}
        for h in (32, 64, 128) for lr in (1e-3, 3e-4) for dr in (0.0, 0.2)]
EPOCHS, PATIENCE, SEED = 60, 12, 0


def load_pools(spec):
    pools, refs = {}, {}
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    return pools, refs


def train_one(cfg, tr, va, in_dim, device):
    torch.manual_seed(SEED)
    model = GroupScorer(in_dim, cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    Xt = [torch.from_numpy(e["X"]).to(device) for e in tr]
    Yt = [torch.from_numpy(e["y"]).to(device) for e in tr]
    best, best_state, since = -1.0, None, 0
    order = np.arange(len(tr))
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        model.train()
        rng.shuffle(order)
        for i in order:
            opt.zero_grad()
            logits = model(Xt[i])
            loss = -(Yt[i] * torch.log_softmax(logits, dim=0)).sum()
            loss.backward()
            opt.step()
        r = reorder_recall(model, va, 15, device)
        if r > best:
            best, since = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pools", default="results/trainpools/s*.json")
    ap.add_argument("--val-pools", default=str(ROOT / "results/val_pools.json"))
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/group_scorer_selection.json"))
    ap.add_argument("--model-out", default=str(ROOT / "artifacts/group_scorer.pt"))
    args = ap.parse_args()

    device = "cpu"
    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.eval()
    feat = Featuriser(generator)
    in_dim = feat.enc_dim + GROUP_FEATURES

    tp, tr_refs = load_pools(args.train_pools)
    vp, va_refs = load_pools(args.val_pools)
    print(f"train pools {len(tp)}  val pools {len(vp)}", file=sys.stderr, flush=True)
    tr = build_examples(tp, tr_refs, feat)
    va = build_examples(vp, va_refs, feat)
    print(f"usable examples: train {len(tr)}  val {len(va)}", file=sys.stderr, flush=True)

    base = fusion_recall(va, 15)
    rows = []
    best = None
    for n, cfg in enumerate(GRID, 1):
        model, r, eps = train_one(cfg, tr, va, in_dim, device)
        rows.append({**cfg, "val_recall@15": round(r, 4), "epochs_run": eps})
        print(f"  [{n}/{len(GRID)}] hidden={cfg['hidden']} lr={cfg['lr']} "
              f"dropout={cfg['dropout']}  val r@15 {r:.4f}  ({eps} epochs)",
              file=sys.stderr, flush=True)
        if best is None or r > best[1]:
            best = (cfg, r, model)
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__),
             "selected_on": "validation micro recall@15", "grid": rows,
             "fusion_baseline_val_recall@15": round(base, 4),
             "n_train_examples": len(tr), "n_val_examples": len(va),
             "in_dim": in_dim, "epochs_max": EPOCHS, "patience": PATIENCE, "seed": SEED,
             "best": None if best is None else {**best[0], "val_recall@15": round(best[1], 4)}},
            indent=1))

    cfg, r, model = best
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "in_dim": in_dim,
                "elements_and_group_features": GROUP_FEATURES,
                "val_recall@15": r, "fusion_baseline": base}, args.model_out)
    print(f"\nfusion baseline on validation: {base:.4f}")
    print(f"best: hidden={cfg['hidden']} lr={cfg['lr']} dropout={cfg['dropout']}  "
          f"val r@15 {r:.4f}  ({r - base:+.4f} over fusion)")
    print(f"wrote {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
