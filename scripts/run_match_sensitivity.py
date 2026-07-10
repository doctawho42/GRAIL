#!/usr/bin/env python3
"""Match-sensitivity ("rank-flip") experiment -- the D&B benchmark's headline.

Each metabolite-prediction paper matches predicted vs reference structures by a different
protocol (GLORYx: InChI without stereo; MetaTrans: Tanimoto=1; LAGOM: canonical SMILES;
strict: InChIKey; ours: tautomer-canonical InChIKey). We hold each method's RANKED RAW
predictions fixed, apply a UNIFORM canonical dedup, and re-score recall@k under every
protocol -- showing the absolute numbers swing and the leaderboard can reorder.

Methods are decoupled via prediction files (so external baselines drop in without re-running
anything): GRAIL from its exported test_predictions.csv; SyGMa generated here if installed;
any other method as a JSON `{substrate_smiles: [pred_smiles_ranked]}`. Ground-truth labels
come from the GRAIL CSV's `real` column (the annotated metabolites), shared across methods.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import aggregate_prediction_metrics

try:  # SyGMa rule application emits a flood of sanitization warnings
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
MODE_LABEL = {
    "canonical": "canon-SMILES (LAGOM)", "inchikey": "InChIKey (strict)",
    "inchi_no_stereo": "no-stereo (GLORYx)", "tanimoto1": "Tanimoto=1 (MetaTrans)",
    "inchikey_tautomer": "tautomer-InChIKey (ours)",
}


def _canon(smiles: str) -> str:
    from grail_metabolism.utils.preparation import _canonicalize_smiles_cached
    try:
        return _canonicalize_smiles_cached(smiles)
    except Exception:
        return smiles


def _dedup_canon(smiles_list: List[str]) -> List[str]:
    """Uniform, mode-agnostic dedup so the match protocol is the only varying factor."""
    out, seen = [], set()
    for s in smiles_list:
        c = _canon(s)
        if c in seen:
            continue
        seen.add(c)
        out.append(s)
    return out


def load_grail_csv(path: Path) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    preds, reals = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sub = row["substrate"]
            preds[sub] = [s for s in row["predicted"].split("|") if s]
            reals[sub] = [s for s in row["real"].split("|") if s]
    return preds, reals


def sygma_predictions(substrates: List[str]) -> Dict[str, List[str]]:
    import sygma
    from rdkit import Chem

    scenario = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    out: Dict[str, List[str]] = {}
    t = time.perf_counter()
    for i, sub in enumerate(substrates, 1):
        if i == 1 or i % 50 == 0 or i == len(substrates):
            print(f"  [sygma] {i}/{len(substrates)} ({time.perf_counter()-t:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            out[sub] = []
            continue
        try:
            tree = scenario.run(mol)
            tree.calc_scores()              # sorts to_smiles() by score
            ranked = [e[0] for e in tree.to_smiles()]
        except Exception:
            ranked = []
        parent = _canon(sub)
        out[sub] = [s for s in ranked if _canon(s) != parent]  # drop the parent
    return out


def score(method_preds: Dict[str, List[str]], reals: Dict[str, List[str]], ks: List[int]) -> Dict[str, Dict]:
    """recall@k + mean_output for one method under every match mode."""
    subs = [s for s in reals if reals[s]]
    by_mode = {}
    for mode in MODES:
        rows = [{"predicted": _dedup_canon(method_preds.get(s, [])), "real": reals[s]} for s in subs]
        m = aggregate_prediction_metrics(rows, ks, match=mode)
        by_mode[mode] = {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in ks} | {
            "precision": round(m.get("precision", 0.0), 3),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }
    return by_mode


def _score_cached(name: str, method_preds: Dict[str, List[str]], reals: Dict[str, List[str]],
                  ks: List[int], cache_dir: Path) -> Dict[str, Dict]:
    """Disk-cache score() per method (keyed by the method's predictions + reals + ks + modes).
    Re-scoring the leaderboard is dominated by tautomer-canonicalizing each method's output
    (SyGMa's ~12k molecules ~= 14 min); caching means ADDING a method re-scores ONLY that
    method, not every existing one."""
    payload = json.dumps({
        "preds": {s: method_preds.get(s, []) for s in sorted(reals)},
        "reals": {s: sorted(reals[s]) for s in sorted(reals)},
        "ks": list(ks), "modes": MODES,
    }, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    cf = cache_dir / f"score_{name}_{key}.json"
    if cf.exists():
        print(f"  [score-cache] {name}: HIT", flush=True)
        return json.loads(cf.read_text())
    t = time.perf_counter()
    bm = score(method_preds, reals, ks)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(bm))
    print(f"  [score-cache] {name}: scored in {time.perf_counter()-t:.0f}s -> cached", flush=True)
    return bm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grail-csv", type=str, default=str(ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"))
    ap.add_argument("--extra", type=str, nargs="*", default=[], help="method=path.json files {sub:[preds]}")
    ap.add_argument("--max-substrates", type=int, default=150)
    ap.add_argument("--ks", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--no-sygma", action="store_true")
    ap.add_argument("--cache-dir", type=str, default=str(ROOT / "results" / "match_sens_cache"))
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "match_sensitivity.json"))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    grail_preds, reals = load_grail_csv(Path(args.grail_csv))
    subs = [s for s in reals if reals[s]][: args.max_substrates]
    reals = {s: reals[s] for s in subs}
    print(f"substrates: {len(subs)}  (real labels present)", flush=True)

    methods: Dict[str, Dict[str, List[str]]] = {"GRAIL": {s: grail_preds.get(s, []) for s in subs}}
    if not args.no_sygma:
        subs_key = hashlib.sha256("|".join(sorted(subs)).encode()).hexdigest()[:16]
        sygma_cf = cache_dir / f"sygma_preds_{subs_key}.json"
        if sygma_cf.exists():
            print(f"loading cached SyGMa predictions ({sygma_cf.name})", flush=True)
            methods["SyGMa"] = json.loads(sygma_cf.read_text())
        else:
            print("generating SyGMa predictions...", flush=True)
            methods["SyGMa"] = sygma_predictions(subs)
            cache_dir.mkdir(parents=True, exist_ok=True)
            sygma_cf.write_text(json.dumps(methods["SyGMa"]))
    for spec in args.extra:
        name, path = spec.split("=", 1)
        methods[name] = json.loads(Path(path).read_text())

    report = {"n_substrates": len(subs), "ks": args.ks, "modes": MODES,
              "by_method": {name: _score_cached(name, p, reals, args.ks, cache_dir)
                            for name, p in methods.items()}}
    Path(args.out).write_text(json.dumps(report, indent=2))

    k = args.ks[-1]
    print(f"\n==== recall@{k} by method x match-protocol (the rank-flip table) ====", flush=True)
    head = f"{'method':<8}" + "".join(f" | {MODE_LABEL[m]:>22}" for m in MODES)
    print(head, flush=True)
    for name, bm in report["by_method"].items():
        print(f"{name:<8}" + "".join(f" | {bm[m][f'recall@{k}']:>22.3f}" for m in MODES), flush=True)
    print("\nper-protocol method ranking (best->worst by recall@%d):" % k, flush=True)
    for m in MODES:
        order = sorted(report["by_method"], key=lambda nm: -report["by_method"][nm][m][f"recall@{k}"])
        print(f"  {MODE_LABEL[m]:>22}: {' > '.join(order)}", flush=True)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
