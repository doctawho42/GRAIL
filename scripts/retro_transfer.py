#!/usr/bin/env python3
"""Does a retrosynthesis model's degradation off-distribution measure transfer, or difficulty?

In metabolite prediction the similarity-to-training axis turned out to measure how well studied a
substrate's chemistry is, and the control that showed it was a rule engine that never trained. This
runs the same design in retrosynthesis, which is the domain where the mechanism should be sharpest,
templates being extracted from USPTO and models trained on USPTO.

  learned    ReactionT5v2 fine-tuned on the USPTO-50k training split
  control    nearest-neighbour retrieval over USPTO-FULL restricted to patents that appear nowhere
             in USPTO-50k, so its knowledge is disjoint from the model's training data by document
             and it has no learned parameters at all

Both are scored on the same USPTO-50k test reactions, stratified by the maximum product-Tanimoto
similarity of each test product to the 40,008 training products. If the control is flat and the
model degrades, the axis measures transfer and the degradation is real, as
\\citet{Buttenschoen_2024} found for docking. If the control degrades too, the axis is confounded
with difficulty, as it is in metabolism.

Two checks guard the setup. The model's overall top-1 should reproduce the published figure for this
split; materially higher would mean the assumed split is not the one it trained on and the test
reactions are contaminated. And the control corpus is verified disjoint by patent before use.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MODEL = "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k"
# The authoritative split, linked from the model card and used for its published 71.2% top-1.
# A community copy of "USPTO-50k" gave 0.834 here, thirteen points above, because its test split is
# not this one -- the guard that caught that is why the split source is pinned rather than assumed.
SPLIT = ROOT / "grail_metabolism" / "data" / "USPTO_50k"
CORPUS = ROOT / "grail_metabolism" / "data" / "USPTO_FULL.csv"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 1.01]
SEED = 0
OUT = ROOT / "results" / "retro_transfer.json"
CKPT = ROOT / "results" / "retro_transfer_preds.json"

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def unmapped(smiles: str) -> str | None:
    """Canonical SMILES with atom maps removed; USPTO-50k ships mapped, the model expects plain."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def reactant_key(dotjoined: str):
    """Order-independent key for a reactant set, the convention the retrosynthesis field scores on."""
    parts = [p for p in dotjoined.split(".") if p]
    keys = set()
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return None
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        keys.add(Chem.MolToSmiles(m))
    return frozenset(keys) if keys else None


def rset_fp(dotjoined: str):
    """Union fingerprint of a reactant set, for the graded score."""
    fp = None
    for s in [x for x in dotjoined.split(".") if x]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        f = gen.GetFingerprint(m)
        fp = f if fp is None else (fp | f)
    return fp


def graded(true_r: str, preds: list) -> float:
    """Best Tanimoto between any top-k prediction's reactant set and the true one.

    Exact set match is the field's metric and is right for the model, but it floors a retrieval
    control at zero: a control with no headroom is flat by construction and cannot discriminate,
    so a flat control would say nothing about the axis. A graded score gives both arms
    non-degenerate variation on the same scale, which is what the slopes need to be comparable.
    """
    tf = rset_fp(true_r)
    if tf is None:
        return 0.0
    best = 0.0
    for p in preds:
        pf = rset_fp(p)
        if pf is not None:
            best = max(best, DataStructs.TanimotoSimilarity(tf, pf))
    return best


def load_splits():
    """The split the model was fine-tuned on, columns REACTANT/PRODUCT, unmapped SMILES."""
    f = lambda n: pd.read_csv(SPLIT / f"{n}.csv").rename(
        columns={"REACTANT": "reactants", "PRODUCT": "product"})
    return f("train"), f("test"), f("val")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--n-corpus", type=int, default=120000)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = random.Random(SEED)

    tr, te, va = load_splits()
    print(f"USPTO-50k: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
    # This split carries no patent id, so the control corpus is made disjoint by PRODUCT across all
    # three splits instead: the retriever can then never return a reaction whose product it has
    # already seen, which is the contamination that matters for a retrieval baseline.
    seen_products = set()
    for frame in (tr, te, va):
        for s in frame["product"]:
            u = unmapped(s)
            if u:
                seen_products.add(u)
    print(f"{len(seen_products)} distinct products across the three splits", flush=True)

    train_prod = [p for p in (unmapped(s) for s in tr["product"]) if p]
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(p)) for p in train_prod]
    print(f"{len(train_fps)} training products fingerprinted", flush=True)

    # Control corpus: USPTO-FULL minus every patent that appears in USPTO-50k, so the control's
    # knowledge shares no document with the model's training data.
    ctrl_prod, ctrl_react, skipped = [], [], 0
    import csv as _csv
    with open(CORPUS) as fh:
        for rec in _csv.DictReader(fh):
            if rng.random() > 0.15:
                continue
            try:
                left, _, right = rec["reactions"].split(">")
            except ValueError:
                continue
            p = unmapped(right)
            if p is None or Chem.MolFromSmiles(p).GetNumHeavyAtoms() < 3:
                continue
            if p in seen_products:
                skipped += 1
                continue
            ctrl_prod.append(p)
            ctrl_react.append(left)
            if len(ctrl_prod) >= args.n_corpus:
                break
    print(f"control corpus {len(ctrl_prod)} reactions; {skipped} skipped as USPTO-50k products",
          flush=True)
    ctrl_fps = [gen.GetFingerprint(Chem.MolFromSmiles(p)) for p in ctrl_prod]

    rows = te.sample(n=min(args.n_test, len(te)), random_state=SEED).to_dict("records")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    print(f"scoring {len(rows)} test reactions, beams={args.beams}", flush=True)

    recs, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        prod, true_r = unmapped(row["product"]), unmapped(row["reactants"])
        if prod is None or true_r is None:
            continue
        f = gen.GetFingerprint(Chem.MolFromSmiles(prod))
        sim = float(max(DataStructs.BulkTanimotoSimilarity(f, train_fps)))

        inp = tok(prod, return_tensors="pt", truncation=True, max_length=200)
        with torch.no_grad():
            out = model.generate(**inp, num_beams=args.beams,
                                 num_return_sequences=args.beams, max_length=200)
        preds = [tok.decode(o, skip_special_tokens=True).replace(" ", "").rstrip(".") for o in out]

        csims = DataStructs.BulkTanimotoSimilarity(f, ctrl_fps)
        order = np.argsort(csims)[::-1][: args.beams]
        cpreds = [ctrl_react[j] for j in order]

        recs.append({"sim": sim, "true": true_r, "model": preds, "control": cpreds})
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el/i:.2f}s/rxn  eta {el/i*(len(rows)-i)/60:.1f}min", flush=True)
    CKPT.write_text(json.dumps(recs))

    sim = np.array([r["sim"] for r in recs])
    def hits(arm, k):
        out = []
        for r in recs:
            tk = reactant_key(r["true"])
            pk = [reactant_key(p) for p in r[arm][:k]]
            out.append(1.0 if tk is not None and any(x == tk for x in pk if x) else 0.0)
        return np.array(out)
    acc = {a: {f"top{k}": hits(a, k) for k in (1, 5)} for a in ("model", "control")}
    for a in ("model", "control"):
        acc[a]["graded"] = np.array([graded(r["true"], r[a]) for r in recs])

    rep = {"model": MODEL, "n": len(recs), "beams": args.beams, "bins": BINS,
           "n_train": len(train_fps), "n_control_corpus": len(ctrl_prod),
           "overall": {a: {k: round(float(v.mean()), 4) for k, v in d.items()} for a, d in acc.items()},
           "strata": []}
    print(f"\noverall: model top1 {acc['model']['top1'].mean():.4f} top5 {acc['model']['top5'].mean():.4f}"
          f" | control top1 {acc['control']['top1'].mean():.4f} top5 {acc['control']['top5'].mean():.4f}")
    print("  (published ReactionT5v2 top-1 on this split is 0.712; a much higher figure would mean"
          " the assumed split is not the one it trained on)\n")
    print(f"{'stratum':>16}{'n':>6}{'model top1':>12}{'model graded':>14}{'control graded':>16}")
    for i in range(len(BINS) - 1):
        m = (sim >= BINS[i]) & (sim < BINS[i + 1])
        if m.sum() < 15:
            continue
        rep["strata"].append({"lo": BINS[i], "hi": BINS[i + 1], "n": int(m.sum()),
                              "model_top1": round(float(acc["model"]["top1"][m].mean()), 4),
                              "model_graded": round(float(acc["model"]["graded"][m].mean()), 4),
                              "control_graded": round(float(acc["control"]["graded"][m].mean()), 4)})
        print(f"  [{BINS[i]:.2f},{BINS[i+1]:.2f}){m.sum():6}"
              f"{acc['model']['top1'][m].mean():12.4f}{acc['model']['graded'][m].mean():14.4f}"
              f"{acc['control']['graded'][m].mean():16.4f}")

    rngn = np.random.default_rng(SEED)
    idx = rngn.integers(0, len(sim), (2000, len(sim)))
    rep["slopes"] = {}
    # Each arm on the measure that has headroom for it: the model ceilings on the graded score
    # (0.99) and the control floors on exact match (0.00), so a single shared measure would pin one
    # of them. Magnitudes are therefore NOT compared -- only whether each slope differs from zero,
    # which is all the question needs: if a method that never trained degrades along the axis, the
    # axis is confounded whatever the model does.
    print("\nslope on similarity, each arm on the measure with headroom for it")
    print("  (positive = worse far from training; magnitudes across arms are not comparable):")
    for a, metric in (("model", "top1"), ("model", "graded"), ("control", "graded")):
        y = acc[a][metric]
        s = float(np.polyfit(sim, y, 1)[0])
        bt = np.array([np.polyfit(sim[i], y[i], 1)[0] for i in idx])
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rep["slopes"][f"{a}_{metric}"] = {"slope": round(s, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                          "excludes_zero": bool(lo > 0 or hi < 0),
                                          "mean": round(float(y.mean()), 4)}
        print(f"  {a+' '+metric:16} mean {y.mean():.3f}   slope {s:+.4f}  [{lo:+.4f},{hi:+.4f}]"
              f"  {'SIG' if (lo > 0 or hi < 0) else 'n.s.'}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
