#!/usr/bin/env python3
"""Does a rule-grounded predictor degrade more slowly off the training distribution?

The reactions a rule bank encodes -- oxidation, hydrolysis, conjugation -- are chemically general,
while the patterns a sequence model learns are not, so a rule-grounded method should in principle
lose less on substrates far from anything it was trained on. That is a claim about the paradigm
rather than about a leaderboard position, and it is testable on predictions that already exist: the
test substrates are stratified by their maximum Morgan-Tanimoto similarity to any training
substrate, and each method's score is reported per stratum.

The design turns on a control that makes the axis interpretable. SyGMa is a pure rule engine and
never saw the training split at all, so its curve is the null: if similarity-to-training is really
measuring transfer, SyGMa should be flat across strata. If SyGMa degrades with the learned methods,
the axis is measuring substrate difficulty rather than distance from training, and no transfer claim
can be read off it. Reporting the learned methods without that control would make a difficulty
gradient look like a transfer result.

Scores are reported at a fixed budget rather than as emitted, since the budget sweep showed the
deployed sizes are not comparable, and at recall@15 as well so the stratification can be read
against the number the field reports.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
K_FIXED, K_RECALL = 5, 15
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 1.01]
N_BOOT, SEED = 10000, 0
OUT = ROOT / "results" / "transfer_stratified.json"


def train_substrates() -> list[str]:
    """Substrates of the training split, read by the convention preparation.py uses.

    The SDF keys records by `Index`, not by the empty `ID` field, and marks substrates with
    `State == "Substrate"`; the triples file's first column is that same Index. Both are used and
    reconciled, so a silent mismatch shows up as a count disagreement rather than as an empty set.
    """
    ids = set()
    with open(ROOT / "grail_metabolism/data/train_triples_clean.txt") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                ids.add(parts[0])
    smiles, by_state, by_index = [], 0, 0
    supplier = Chem.SDMolSupplier(str(ROOT / "grail_metabolism/data/train.sdf"))
    for mol in supplier:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        is_substrate = str(props.get("State", "")) == "Substrate"
        in_triples = str(props.get("Index", "")) in ids
        by_state += is_substrate
        by_index += in_triples
        if is_substrate:
            s = props.get("SMILES") or Chem.MolToSmiles(mol)
            if s:
                smiles.append(s)
    print(f"substrates by State={by_state}, ids referenced in triples={len(ids)}, "
          f"records whose Index is a triples substrate={by_index}", flush=True)
    if not smiles:
        raise SystemExit("no training substrates found -- the SDF convention has changed")
    return smiles


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())
    grail = {r["sub"]: r["deployed_top15"]
             for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
    }
    subs = [s for s in sorted(set.intersection(*(set(m) for m in methods.values())) & set(truth)) if truth[s]]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def fp(s):
        mol = Chem.MolFromSmiles(s)
        return gen.GetFingerprint(mol) if mol is not None else None

    train_fps = [f for f in (fp(s) for s in train_substrates()) if f is not None]
    print(f"{len(train_fps)} training substrates, {len(subs)} test substrates", flush=True)

    sim = []
    keep = []
    for s in subs:
        f = fp(s)
        if f is None:
            continue
        sim.append(max(DataStructs.BulkTanimotoSimilarity(f, train_fps)))
        keep.append(s)
    sim = np.array(sim)
    print(f"max-Tanimoto to train: mean {sim.mean():.3f}, median {np.median(sim):.3f}, "
          f"min {sim.min():.3f}, max {sim.max():.3f}\n", flush=True)

    def keyed(items):
        seen, out = set(), []
        for x in items:
            k = table.get(x)
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    scores = {}
    for name, preds in methods.items():
        f1k, rec = [], []
        for s in keep:
            pk = keyed(preds.get(s, []))
            real = {k for k in (table.get(x) for x in truth[s]) if k}
            tp = len(set(pk[:K_FIXED]) & real)
            nk = min(K_FIXED, len(pk))
            f1k.append(2 * tp / (nk + len(real)) if (nk + len(real)) else 0.0)
            rec.append(len(set(pk[:K_RECALL]) & real) / len(real))
        scores[name] = {"f1": np.array(f1k), "recall": np.array(rec)}

    idx_bins = np.digitize(sim, BINS[1:-1], right=False)
    rng = np.random.default_rng(SEED)
    rep = {"mode": MODE, "k_fixed": K_FIXED, "k_recall": K_RECALL, "bins": BINS,
           "n_train": len(train_fps), "n_test": len(keep), "strata": [], "slopes": {}}

    print(f"{'stratum':>14}{'n':>6}  " + "  ".join(f"{m:>22}" for m in methods))
    print(f"{'':>20}  " + "  ".join(f"{'F1@5':>10}{'rec@15':>12}" for _ in methods))
    for b in range(len(BINS) - 1):
        m = idx_bins == b
        if m.sum() < 20:
            continue
        row = {"lo": BINS[b], "hi": BINS[b + 1], "n": int(m.sum()), "methods": {}}
        cells = []
        for name in methods:
            f1 = float(scores[name]["f1"][m].mean())
            rc = float(scores[name]["recall"][m].mean())
            row["methods"][name] = {"f1_at_k": round(f1, 4), "recall_at_15": round(rc, 4)}
            cells.append(f"{f1:10.4f}{rc:12.4f}")
        rep["strata"].append(row)
        print(f"  [{BINS[b]:.2f},{BINS[b+1]:.2f}){m.sum():6}  " + "  ".join(cells))

    # Degradation slope: per-method regression of the score on similarity, with a paired bootstrap
    # on the difference of slopes against SyGMa, the method that never saw the training split.
    print(f"\nslope of score on max-Tanimoto (positive = worse when far from training):")
    boot = rng.integers(0, len(keep), (N_BOOT, len(keep)))
    slopes, draws = {}, {}
    for metric in ("f1", "recall"):
        slopes[metric], draws[metric] = {}, {}
        for name in methods:
            y = scores[name][metric]
            s_hat = float(np.polyfit(sim, y, 1)[0])
            bt = np.array([np.polyfit(sim[i], y[i], 1)[0] for i in boot[:1000]])
            draws[metric][name] = bt
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            slopes[metric][name] = {"slope": round(s_hat, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            print(f"  {metric:7} {name:16} {s_hat:+.4f}  [{lo:+.4f},{hi:+.4f}]")
    rep["slopes"] = slopes

    # The control's whole force is that SyGMa's slope is not distinguishable from the learned
    # methods', and a null asserted from two overlapping per-method intervals is not a null. The
    # difference is resampled on the same draws, so it is paired, and the interval states the width
    # of the equivalence the data actually support.
    print("\npaired slope difference against GRAIL (positive = the comparator degrades faster):")
    diffs = {}
    for metric in ("f1", "recall"):
        diffs[metric] = {}
        for name in methods:
            if name == "GRAIL":
                continue
            d = draws[metric][name] - draws[metric]["GRAIL"]
            pt = slopes[metric][name]["slope"] - slopes[metric]["GRAIL"]["slope"]
            lo, hi = float(np.quantile(d, .025)), float(np.quantile(d, .975))
            diffs[metric][f"{name}_minus_GRAIL"] = {
                "diff": round(float(pt), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "half_width": round(float(max(abs(lo), abs(hi))), 4)}
            print(f"  {metric:7} {name:16} {pt:+.4f}  [{lo:+.4f},{hi:+.4f}]")
    rep["slope_differences"] = {
        "note": "paired on the same bootstrap resamples as the per-method slopes; "
                "half_width is the largest slope difference the interval still admits",
        "by_metric": diffs}
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
