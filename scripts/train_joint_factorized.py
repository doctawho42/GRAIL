#!/usr/bin/env python3
"""D3 joint-training arm: fine-tune the factorized type/site heads with a ranking loss.

The bolt-on re-ranker (D2) multiplies INDEPENDENTLY-trained type/site probabilities into the rank.
Here we instead fine-tune those heads to *rank* true metabolites given the FIXED deployed generator
and filter. For each training substrate we forward the trainable heads once and score every pooled
candidate in the log domain,

    s_i = log(filter_i) + log(gen_i) + log(type_prob_i) + log(site_prob_i),

where `log(filter_i)` and `log(gen_i)` are frozen constants from the pool (`build_joint_pools.py`),
`type_prob_i = sigmoid(type_logits)[tid_i]` (mean type prob when the rule maps to no known type),
and `site_prob_i = max sigmoid(site_logits)[reacting_i]` (neutral 1.0 when the site is un-localizable
-- exactly the deployed `product_som_score` aggregation). The loss is a multi-positive listwise
softmax cross-entropy, `-mean_{i in positives} log_softmax(s)_i`; gradients flow only through the two
trainable heads. Warm-started from the bolt-on `factorized_v1`, early-stopped on a val top-15
hit-rate proxy. Saves a drop-in `FactorizedGenerator` checkpoint the reranker can load.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import Chem

from grail_metabolism.model.factorized import FactorizedGenerator
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import from_rdmol

WARM = ROOT / "artifacts" / "factorized_v1" / "checkpoints" / "factorized.pt"
_EPS = 1e-6


def load_rows(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            rows.extend(json.loads(Path(p).read_text())["rows"])
    return rows


def _prep(rows):
    """Attach the featurized substrate graph to each row (cache by SMILES)."""
    cache, out = {}, []
    for r in rows:
        sub = r["sub"]
        if sub not in cache:
            mol = Chem.MolFromSmiles(sub)
            cache[sub] = from_rdmol(mol) if mol is not None else None
        data = cache[sub]
        if data is None or data.x.size(0) != r["n_atoms"]:
            continue  # atom-count mismatch -> reacting indices would misalign; skip
        out.append((data, r["cands"]))
    return out


def _cand_scores(model, data, cands):
    """Differentiable per-candidate combined log-score s_i for one substrate."""
    type_p = torch.sigmoid(model.type_logits(data))[0]           # [num_types]
    site_p = torch.sigmoid(model.site_logits(data))              # [n_atoms]
    type_floor = type_p.mean()
    n_types = type_p.size(0)
    n_atoms = site_p.size(0)
    scores, ys = [], []
    for c in cands:
        tid = c["tid"]
        tp = type_p[tid] if 0 <= tid < n_types else type_floor
        reacting = [a for a in c["reacting"] if 0 <= a < n_atoms]
        sp = torch.max(site_p[reacting]) if reacting else site_p.new_tensor(1.0)
        scores.append(c["lf"] + c["lg"] + torch.log(tp + _EPS) + torch.log(sp + _EPS))
        ys.append(c["y"])
    return torch.stack(scores), ys


def _val_hitrate(model, val, k=15):
    """Fraction of val substrates with >=1 positive in the top-k by combined score (recall proxy)."""
    model.eval()
    hits = 0
    with torch.no_grad():
        for data, cands in val:
            if not any(c["y"] for c in cands):
                continue
            scores, ys = _cand_scores(model, data, cands)
            order = torch.argsort(scores, descending=True)[:k].tolist()
            if any(ys[i] for i in order):
                hits += 1
    return hits / max(len(val), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="glob(s) of train pool shards")
    ap.add_argument("--val", required=True, help="glob of val pool file(s)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "factorized_joint" / "checkpoints" / "factorized.pt"))
    args = ap.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(6)

    train = _prep(load_rows(args.train))
    val = _prep(load_rows([args.val]))
    print(f"train {len(train)} substrates, val {len(val)} substrates", flush=True)

    model = FactorizedGenerator.load(WARM)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_hit, best_state, base_hit = -1.0, None, _val_hitrate(model, val)
    print(f"warm-start val top-15 hit-rate {base_hit:.4f}", flush=True)

    order = list(range(len(train)))
    for epoch in range(args.epochs):
        model.train()
        seed_everything(args.seed + epoch + 1)
        import random
        random.shuffle(order)
        total, nb = 0.0, 0
        opt.zero_grad()
        for step, idx in enumerate(order, 1):
            data, cands = train[idx]
            scores, ys = _cand_scores(model, data, cands)
            pos = [i for i, y in enumerate(ys) if y]
            if not pos or scores.numel() < 2:
                continue
            logp = torch.log_softmax(scores, dim=0)
            loss = -logp[pos].mean()
            loss.backward()
            total += float(loss); nb += 1
            if step % 16 == 0:  # mini-batch of 16 substrates
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        hit = _val_hitrate(model, val)
        print(f"epoch {epoch+1}: train_loss {total/max(nb,1):.4f}  val_hit@15 {hit:.4f}", flush=True)
        if hit > best_hit:
            best_hit = hit
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nbest val top-15 hit-rate {best_hit:.4f} (warm-start {base_hit:.4f}) -> {args.out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
