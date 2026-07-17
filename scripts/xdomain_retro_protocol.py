#!/usr/bin/env python3
"""Cross-domain protocol-sensitivity probe (B): retrosynthesis top-k accuracy under match conventions.

Deferred to future-work: a full cross-domain RANK-FLIP needs >=2 comparable models with differing
match-sensitivity, whose raw ranked predictions are not publicly available (a decisive-test
feasibility wall we assessed rather than forced). This is the modest, honest probe that IS feasible:
run ONE strong published retrosynthesis model (sagawa/ReactionT5v2-retrosynthesis-USPTO_50k, top-1
71% on this split) on the USPTO-50k test set and score the SAME predictions under the conventions the
field uses inconsistently -- canonical-with-stereo, stereo-stripped, InChIKey, tautomer-InChIKey.

A materially different top-k number across conventions demonstrates that "the reported recall depends
on the match convention" is NOT metabolism-specific -- it recurs in retrosynthesis. That supports
TAME's premise cross-domain without claiming a leaderboard reorder (which we cannot show with one
model). NOT a rank-flip.

Prediction is correct at k if any of the top-k predicted reactant SETS equals the true reactant SET
under the convention (order-independent; ReactionT5 emits dot-joined reactants).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
MODEL = "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k"
MODES = ["canonical", "nostereo", "inchikey", "tautomer"]
KS = [1, 3, 5]


def reactant_set_key(dotjoined: str, mode: str):
    parts = [p for p in dotjoined.split(".") if p]
    if not parts:
        return None
    keys = set()
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return None
        try:
            if mode == "canonical":
                keys.add(Chem.MolToSmiles(m))
            elif mode == "nostereo":
                Chem.RemoveStereochemistry(m)
                keys.add(Chem.MolToSmiles(m))
            elif mode == "inchikey":
                keys.add(Chem.MolToInchiKey(m))
            elif mode == "tautomer":
                keys.add(_tautomer_inchikey(p))
        except Exception:
            return None
    return frozenset(keys)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "xdomain_retro_protocol.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("loading ReactionT5v2 (downloads on first run) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()

    rows = list(csv.DictReader(open(args.test_csv)))[: args.n]
    print(f"test reactions: {len(rows)}  beams={args.beams}", flush=True)

    # correct[mode][k] = # test reactions with a top-k hit under that convention
    correct = {m: {k: 0 for k in KS} for m in MODES}
    n = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        product, true_r = row["PRODUCT"], row["REACTANT"]
        inp = tok(product, return_tensors="pt", truncation=True, max_length=200)
        with torch.no_grad():
            out = model.generate(**inp, num_beams=args.beams, num_return_sequences=args.beams, max_length=200)
        preds = [tok.decode(o, skip_special_tokens=True).replace(" ", "").rstrip(".") for o in out]
        n += 1
        for mode in MODES:
            true_key = reactant_set_key(true_r, mode)
            if true_key is None:
                continue
            pred_keys = [reactant_set_key(p, mode) for p in preds]
            for k in KS:
                if any(pk is not None and pk == true_key for pk in pred_keys[:k]):
                    correct[mode][k] += 1

    acc = {m: {f"top{k}": round(correct[m][k] / n, 4) for k in KS} for m in MODES}
    # spread across conventions at each k (the "protocol changes the number" gap)
    spread = {f"top{k}": round(max(acc[m][f"top{k}"] for m in MODES) - min(acc[m][f"top{k}"] for m in MODES), 4) for k in KS}
    report = {"model": MODEL, "n": n, "beams": args.beams, "accuracy_by_mode": acc, "spread_across_modes": spread}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n=== retrosynthesis top-k accuracy under match conventions (ReactionT5v2, n={n}) ===", flush=True)
    print(f"{'mode':10s} | " + " | ".join(f"top{k}".rjust(7) for k in KS), flush=True)
    for m in MODES:
        print(f"{m:10s} | " + " | ".join(str(acc[m][f'top{k}']).rjust(7) for k in KS), flush=True)
    print(f"{'SPREAD':10s} | " + " | ".join(str(spread[f'top{k}']).rjust(7) for k in KS), flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
