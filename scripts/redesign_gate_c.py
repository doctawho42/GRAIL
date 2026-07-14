#!/usr/bin/env python3
"""Redesign Gate C — is the already-trained SoM site head sharp enough to carry selection?

The factorized redesign ranks by P(site|type,s); its whole recall story rests on the site head
localizing the true reacting atom. Gate C runs the EXISTING trained SoMPredictor
(artifacts/subset_train_v/checkpoints/som.pt) on the test substrates and asks: is the true
reacting atom (derive_som_labels, MCS-derived) among the top-k predicted atoms?

GO only if true site in top-3 for >= ~70% of localizable test substrates (the synthesis's Gate C
bar). If it is much lower, Gate D (which needs the real site) is doomed and the direction dies here
cheaply.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from grail_metabolism.model.som import SoMPredictor, derive_som_labels
from scripts.run_benchmark import load_test_map

CKPT = ROOT / "artifacts" / "subset_train_v" / "checkpoints" / "som.pt"


def main() -> int:
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    arch = state.get("arch") if isinstance(state, dict) else None
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    som = SoMPredictor(**arch) if arch else SoMPredictor()
    som.load_state_dict(sd, strict=False)
    som.eval()

    test_map = load_test_map(None, 42)
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    localizable = 0
    n_true_atoms = []
    n_atoms_total = []
    for i, (sub, mets) in enumerate(test_map.items(), 1):
        if i % 200 == 0:
            print(f"  {i}/{len(test_map)}", flush=True)
        true_sites = derive_som_labels(sub, mets)
        if not true_sites:
            continue
        scores = som.score_atoms(sub)
        if scores.size == 0:
            continue
        localizable += 1
        n_true_atoms.append(len(true_sites))
        n_atoms_total.append(int(scores.size))
        ranked = list(np.argsort(-scores))  # atom indices, best first
        for k in ks:
            topk = set(int(a) for a in ranked[:k])
            if topk & set(int(a) for a in true_sites if 0 <= a < scores.size):
                hits[k] += 1

    report = {
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "n_test": len(test_map),
        "localizable": localizable,
        "mean_true_sites": round(float(np.mean(n_true_atoms)), 2) if n_true_atoms else None,
        "mean_atoms_per_mol": round(float(np.mean(n_atoms_total)), 1) if n_atoms_total else None,
        "site_hit@1": round(hits[1] / localizable, 3) if localizable else None,
        "site_hit@3": round(hits[3] / localizable, 3) if localizable else None,
        "site_hit@5": round(hits[5] / localizable, 3) if localizable else None,
        # random baseline for @3: expected top-3 hit if sites were random ~ 1-(1-|true|/|atoms|)^3
    }
    if n_true_atoms:
        import statistics
        rr = statistics.mean(1 - (1 - t / max(a, 1)) ** 3 for t, a in zip(n_true_atoms, n_atoms_total))
        report["random_baseline_hit@3"] = round(rr, 3)
    out = ROOT / "results" / "redesign_gate_c.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    go = (report["site_hit@3"] or 0) >= 0.70
    print(f"\nGATE C: SoM true-site hit@3 = {report['site_hit@3']} (bar 0.70; random ~{report.get('random_baseline_hit@3')}) "
          f"=> {'GO (site head can carry selection)' if go else 'WEAK — site head under-sharp; Gate D at risk'}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
