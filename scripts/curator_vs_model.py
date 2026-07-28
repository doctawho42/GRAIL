#!/usr/bin/env python3
"""Do models reach the level at which two human curations agree, on the same drugs?

Two independent curations of the same 25 drugs recover 0.647 and 0.539 of each other's metabolite
sets under the paper's default criterion, against 0.585 for the best published method on the full
test split. Those two numbers are not comparable: the agreement is measured on 25 well-studied drugs
carrying 5.7 to 6.5 references each, the 0.585 on 1,170 substrates averaging 2.2. Putting them side
by side is the defect this manuscript has had to repair five times.

This removes the mismatch by scoring the models where the curators were scored: the same 25 drugs,
the same references, the same matcher. A curation is scored the way a predictor is -- its metabolite
set against the other source's -- but it is not one: its list is what that source's inclusion
criteria admitted, not an attempt to reconstruct the other source's list, and it is read here as the
level at which two independent readings of the same chemistry coincide.

Budget is a condition of the answer, not a footnote to it. The manuscript makes the output budget a
free parameter, so an uncapped model emitting 10.6 against a curation emitting 6.0 reproduces the
mismatch it warns about. Both matched budgets are therefore reported -- a fixed k=6, which truncates
the curation identically, and a per-substrate cap equal to that drug's own curated set size -- and
the gap against the curation is bootstrapped paired on the substrate.

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
        print(f"{'arm':20}{'uncapped':>12}{'at k=6':>10}{'budget-matched':>12}{'output':>12}")
        vecs = {}
        for name, pr in preds.items():
            if name == f"{target} curation" or (target == "gloryx" and name == "GLORYx curation"):
                continue  # a source cannot be scored against itself
            unc, cap, size, at6, matched = [], [], [], [], []
            for g, c in shared:
                refs = sorted(corp[c]) if target == "corpus" else gl[g]
                p = pr.get(g, [])
                size.append(len(p))
                unc.append(recall(p, refs))
                cap.append(recall(p, refs, K))
                # Budget-matched: the paper makes the output budget a parameter, so comparing a
                # model emitting 10.6 against a curation emitting 6.0 is exactly the mismatch it
                # warns about. k=6 is the curation's mean; the per-substrate match uses that
                # substrate's own curation size, which is the tighter of the two.
                at6.append(recall(p, refs, 6))
                other = gl[g] if target == "corpus" else sorted(corp[c])
                matched.append(recall(p, refs, max(1, len({k for k in (tk(x) for x in other) if k}))))
            unc, cap = np.array(unc), np.array(cap)
            at6, matched = np.array(at6), np.array(matched)
            vecs[name] = {"uncapped": unc, "at_k6": at6, "budget_matched": matched}
            bt = unc[idx].mean(axis=1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            b6 = at6[idx].mean(axis=1)
            bm = matched[idx].mean(axis=1)
            rep["arms"].setdefault(target, {})[name] = {
                "uncapped": round(float(unc.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "top_k": round(float(cap.mean()), 4), "mean_output": round(st.mean(size), 1),
                "at_k6": round(float(at6.mean()), 4),
                "at_k6_ci95": [round(float(np.quantile(b6, .025)), 4), round(float(np.quantile(b6, .975)), 4)],
                "budget_matched": round(float(matched.mean()), 4),
                "budget_matched_ci95": [round(float(np.quantile(bm, .025)), 4),
                                        round(float(np.quantile(bm, .975)), 4)]}
            print(f"{name:20}{unc.mean():12.4f}{at6.mean():10.4f}{matched.mean():12.4f}"
                  f"{st.mean(size):12.1f}")
        print()

        # The gap against the curation, paired on the substrate. A point estimate at n=25 is not a
        # claim; the reversal at matched budget is only reportable with an interval on the gap.
        ref = "GLORYx curation" if target == "corpus" else "corpus curation"
        if ref in vecs:
            print(f"gap: {ref} minus each model, paired")
            for col in ("uncapped", "at_k6", "budget_matched"):
                for name in vecs:
                    if name == ref:
                        continue
                    d = vecs[ref][col] - vecs[name][col]
                    b = d[idx].mean(axis=1)
                    lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
                    rep["arms"][target][name][f"gap_{col}"] = round(float(d.mean()), 4)
                    rep["arms"][target][name][f"gap_{col}_ci95"] = [round(lo, 4), round(hi, 4)]
                    print(f"  {col:16}{name:20}{d.mean():+8.4f} [{lo:+.4f},{hi:+.4f}]"
                          f" {'SIG' if (lo > 0 or hi < 0) else 'n.s.'}")
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
