#!/usr/bin/env python3
"""Does the GRAIL-minus-SyGMa coverage gap survive on substrates unlike the training corpus?

Appendix C.11 shows bank coverage rising with maximum Tanimoto similarity to the training
substrates, and attributes that to shared provenance: training corpus, rule bank and annotation
all come from the literature of well-studied metabolism. A reader can then ask whether the
0.735-vs-0.542 ceiling gap of section 4 is that confound rather than bank breadth.

The confound acts on the level of the ceiling. Whether it acts on the *gap* is a measurement:
SyGMa's bank never saw this training split, so if the gap is provenance it must shrink where the
provenance is weakest. Both banks are scored on the same substrates, under the same matching
criterion, and the difference is bootstrapped paired within stratum.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MATCH, N_BOOT, SEED = "inchikey_tautomer", 10000, 0
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 1.01]
OUT = ROOT / "results" / "ceiling_gap_by_similarity.json"


def keyed(items, table):
    out, seen = [], set()
    for it in items:
        k = table.get(it)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    fac = {r["sub"]: r for r in json.loads((ROOT / "results/recall_factorization.json").read_text())["per_substrate"]}
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    sygma = json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text())
    table = json.loads((ROOT / f"results/key_tables/{MATCH}.json").read_text())

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return gen.GetFingerprint(m) if m else None

    sup = Chem.SDMolSupplier(str(ROOT / "grail_metabolism/data/train.sdf"))
    tr = [fp(m.GetPropsAsDict().get("SMILES") or Chem.MolToSmiles(m)) for m in sup
          if m is not None and str(m.GetPropsAsDict().get("State", "")) == "Substrate"]
    tr = [f for f in tr if f is not None]
    print(f"train substrates: {len(tr)}", flush=True)

    sim, U, Cg, Cs = [], [], [], []
    for s, r in fac.items():
        if s not in truth or not truth[s] or not r["U"] or s not in sygma:
            continue
        f = fp(s)
        if f is None:
            continue
        refs = {table[x] for x in truth[s] if table.get(x)}
        sim.append(max(DataStructs.BulkTanimotoSimilarity(f, tr)))
        U.append(r["U"])
        Cg.append(r["Cfull"])
        Cs.append(len(refs & set(keyed(sygma[s], table))))
    sim, U, Cg, Cs = map(np.array, (sim, U, Cg, Cs))
    print(f"paired substrates: {len(U)}", flush=True)

    rng = np.random.default_rng(SEED)

    def stratum(mask):
        u, g, y = U[mask], Cg[mask], Cs[mask]
        cov_g, cov_s = g.sum() / u.sum(), y.sum() / u.sum()
        idx = np.arange(len(u))
        boot = np.array([
            (lambda b: (g[b].sum() - y[b].sum()) / u[b].sum())(rng.integers(0, len(u), len(u)))
            for _ in range(N_BOOT)
        ])
        return {
            "n": int(mask.sum()),
            "coverage_GRAIL": round(float(cov_g), 4),
            "coverage_SyGMa": round(float(cov_s), 4),
            "gap": round(float(cov_g - cov_s), 4),
            "gap_ci95": [round(float(np.quantile(boot, 0.025)), 4),
                         round(float(np.quantile(boot, 0.975)), 4)],
        }

    rep = {
        "match": MATCH, "n_boot": N_BOOT, "seed": SEED, "bins": BINS,
        "estimand": "micro coverage_bank(GRAIL) - micro coverage_bank(SyGMa), paired on substrate, within stratum",
        "all": stratum(np.ones(len(U), bool)),
        "strata": [],
    }
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sim >= lo) & (sim < hi)
        if m.sum() < 20:
            continue
        rep["strata"].append({"lo": lo, "hi": hi, **stratum(m)})

    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
