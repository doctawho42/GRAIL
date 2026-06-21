#!/usr/bin/env python3
"""Train the standalone site-of-metabolism (SoM) prior on MCS-derived node labels.

Node-level BCE over substrate atoms; labels = reacting atoms (+1-hop) of the annotated
(substrate, metabolite) pairs (see grail_metabolism/model/som.py). Cheap (one forward per
substrate per epoch). Saves som.pt with arch+state_dict for reeval_ranking.py --som-ckpt.

See docs/superpowers/specs/2026-06-21-som-prior-design.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from grail_metabolism.config import DatasetConfig, SoMConfig
from grail_metabolism.model.som import build_som_dataset
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_som


def _auc(model, samples) -> float:
    if not samples:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for data in samples:
            logits.append(model.node_logits(data).cpu().numpy().reshape(-1))
            labels.append(data.y.cpu().numpy().reshape(-1))
    y = np.concatenate(labels)
    s = np.concatenate(logits)
    if y.min() == y.max():  # need both classes
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-substrates", type=int, default=0, help="0 = all available")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "subset_train_v" / "checkpoints" / "som.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=(args.train_substrates or None), sampling_seed=args.seed,
    )
    print("loading dataset bundle...", flush=True)
    bundle = load_dataset_bundle(dataset)
    print("building SoM node-label datasets (MCS-derived)...", flush=True)
    train_samples = build_som_dataset(bundle.train)
    val_samples = build_som_dataset(bundle.val)
    n_pos = int(sum(float(d.y.sum()) for d in train_samples))
    n_tot = int(sum(int(d.y.numel()) for d in train_samples))
    print(f"train graphs={len(train_samples)} val graphs={len(val_samples)} "
          f"atoms={n_tot} positives={n_pos} ({100.0 * n_pos / max(n_tot, 1):.1f}%)", flush=True)
    if not train_samples or n_pos == 0:
        print("ERROR: no SoM training signal (empty dataset or no positive atoms)", flush=True)
        return 1

    model = build_som(SoMConfig())
    # Sparse positives -> weight them up so BCE doesn't collapse to all-negative.
    pos_weight = torch.tensor([max(1.0, (n_tot - n_pos) / max(n_pos, 1))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True)

    t = time.perf_counter()
    best_auc = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model.node_logits(batch), batch.y.float())
            loss.backward()
            opt.step()
            total += float(loss) * int(batch.num_graphs)
        auc = _auc(model, val_samples)
        if not np.isnan(auc) and (np.isnan(best_auc) or auc > best_auc):
            best_auc = auc
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:>3}/{args.epochs}  loss={total / max(len(train_samples), 1):.4f}  "
                  f"val_auc={auc:.4f}  ({time.perf_counter() - t:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "best_val_auc": best_auc}, out)
    print(f"best val AUC={best_auc:.4f}; wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
