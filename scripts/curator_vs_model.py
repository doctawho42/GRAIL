#!/usr/bin/env python3
"""Do models reach the level at which two human curations agree, on the same drugs?

Two independent curations of the same 25 drugs recover 0.647 and 0.539 of each other's metabolite
sets under the paper's default criterion, against 0.585 for the best published method on the full
test split. Those two numbers are not comparable: the agreement is measured on 25 well-studied drugs
carrying 5.7 to 6.5 references each, the 0.585 on 1,170 substrates averaging 2.2. Putting them side
by side is the defect this manuscript has had to repair five times.

This removes the mismatch by scoring the models where the curators were scored: the same 25 drugs,
the same references, the same matcher. A curation is treated exactly as a predictor would be -- its
metabolite set is scored against the other source's -- so the comparison asks one question. Does a
model recover as much of the corpus's annotation as an independent expert curation does?

Both directions are reported, because they answer different things. Recall of the CORPUS references
is what a curator scores 0.539 on and is the like-for-like number. Recall of the GLORYx references
is what a curator scores 0.647 on. Neither model is contaminated here: SyGMa does not train at all
and MetaPredictor was trained by its authors on their own data, so no restriction is needed.
"""
from __future__ import annotations
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey
from scripts.gloryx_rank_flip_ci import DATA, PRED_FILES, sygma_predict

RDLogger.DisableLog("rdApp.*")
K, N_BOOT, SEED = 15, 10000, 0
OUT = ROOT / "results" / "curator_vs_model.json"


def tk(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def load_gloryx() -> dict:
    raw = (DATA / "gloryx_test.json").read_text()
    data = json.loads(re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", raw))

    def flat(ns):
        o = []
        for n in ns or []:
            if n.get("smiles"):
                o.append(n["smiles"])
            o.extend(flat(n.get("metabolites")))
        return o

    return {p["smiles"]: flat(p.get("metabolites")) for p in data if p.get("smiles")}


def load_corpus() -> dict:
    out: dict = {}
    for split in ("train", "val", "test"):
        sdf, tri = ROOT / f"grail_metabolism/data/{split}.sdf", ROOT / f"grail_metabolism/data/{split}_triples_clean.txt"
        if not sdf.exists() or not tri.exists():
            continue
        ids = {}
        for mol in Chem.SDMolSupplier(str(sdf)):
            if mol is None:
                continue
            p = mol.GetPropsAsDict()
            i = str(p.get("Index", ""))
            s = p.get("SMILES") or Chem.MolToSmiles(mol)
            if i and s:
                ids[i] = s
        with open(tri) as fh:
            for line in fh:
                a = line.split()
                if len(a) == 3 and a[2] == "1":
                    s, m = ids.get(a[0]), ids.get(a[1])
                    if s and m:
                        out.setdefault(s, set()).add(m)
    return out


def recall(pred_smiles, ref_smiles, cap=None) -> float:
    ref = {k for k in (tk(x) for x in ref_smiles) if k}
    if not ref:
        return float("nan")
    seen, keys = set(), []
    for s in pred_smiles:
        k = tk(s)
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    if cap:
        keys = keys[:cap]
    return len(set(keys) & ref) / len(ref)


def main() -> int:
    gl, corp = load_gloryx(), load_corpus()
    ckey = {}
    for s in corp:
        k = tk(s)
        if k:
            ckey.setdefault(k, s)
    shared = [(g, ckey[tk(g)]) for g in gl if tk(g) and tk(g) in ckey and gl[g] and corp[ckey[tk(g)]]]
    subs = [g for g, _ in shared]
    print(f"{len(shared)} drugs curated by both sources\n", flush=True)

    preds = {n: {s: json.loads(p.read_text()).get(s, []) for s in subs} for n, p in PRED_FILES.items()}
    preds["SyGMa"] = sygma_predict(subs, 256)
    # Each curation, scored exactly as a predictor would be.
    preds["GLORYx curation"] = {g: gl[g] for g, _ in shared}
    preds["corpus curation"] = {g: sorted(corp[c]) for g, c in shared}

    rep = {"n_shared": len(shared), "k": K, "arms": {}}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(shared), (N_BOOT, len(shared)))

    for target, label in (("corpus", "recall of CORPUS references"), ("gloryx", "recall of GLORYx references")):
        print(f"=== {label} ===")
        print(f"{'arm':20}{'uncapped':>12}{'top-15':>12}{'mean output':>14}")
        for name, pr in preds.items():
            if name == f"{target} curation" or (target == "gloryx" and name == "GLORYx curation"):
                continue  # a source cannot be scored against itself
            unc, cap, size = [], [], []
            for g, c in shared:
                refs = sorted(corp[c]) if target == "corpus" else gl[g]
                p = pr.get(g, [])
                size.append(len(p))
                unc.append(recall(p, refs))
                cap.append(recall(p, refs, K))
            unc, cap = np.array(unc), np.array(cap)
            bt = unc[idx].mean(axis=1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            rep["arms"].setdefault(target, {})[name] = {
                "uncapped": round(float(unc.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "top_k": round(float(cap.mean()), 4), "mean_output": round(st.mean(size), 1)}
            print(f"{name:20}{unc.mean():12.4f}{cap.mean():12.4f}{st.mean(size):14.1f}"
                  f"   [{lo:.3f},{hi:.3f}]")
        print()

    OUT.write_text(json.dumps(rep, indent=1))
    cur = rep["arms"]["corpus"].get("GLORYx curation", {})
    best = max((v["uncapped"] for k, v in rep["arms"]["corpus"].items() if "curation" not in k),
               default=float("nan"))
    print(f"on the same {len(shared)} drugs, an independent expert curation recovers "
          f"{cur.get('uncapped')} of the corpus annotation")
    print(f"the best model recovers {best:.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
