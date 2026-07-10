#!/usr/bin/env python3
"""Paired-bootstrap CI on the match-sensitivity rank-flip.

The rank-flip table (``run_match_sensitivity.py``) shows the leaderboard reorders with the
match protocol, but a referee's fair question is whether a given pair's reversal is real at
this sample size or noise. This computes, for one method PAIR across two protocols, the
per-substrate recall@k difference and a paired bootstrap 95% CI (resampling substrates, the
correct unit -- predictions within a substrate are not independent).

The flip is statistically demonstrated when the two protocols' CIs for (A - B) exclude zero
with OPPOSITE signs: A leads under one protocol, B leads under the other. We also bootstrap
the interaction (how much more B gains from normalization than A) -- a single CImthat, if it
excludes zero, certifies the differential protocol-sensitivity directly.

Predictions and the uniform canonical dedup are taken through the SAME code path as the
table (``run_match_sensitivity._dedup_canon`` + ``metrics._match_keys``/``top_k_recall``) so
the point estimates reconcile exactly with ``match_sensitivity_*.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _match_keys, top_k_recall  # noqa: E402
from scripts.run_match_sensitivity import _dedup_canon, load_grail_csv  # noqa: E402

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


def per_substrate_recall(preds: Dict[str, List[str]], reals: Dict[str, List[str]],
                         subs: List[str], match: str, k: int) -> np.ndarray:
    """recall@k per substrate under one protocol, with the table's uniform canonical dedup."""
    out = np.empty(len(subs), dtype=float)
    for i, s in enumerate(subs):
        raw = _dedup_canon(preds.get(s, []))
        ranked = [next(iter(_match_keys([item], match))) for item in raw]
        real = _match_keys(reals[s], match)
        out[i] = top_k_recall(ranked, real, k)
    return out


def boot_ci(delta: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired bootstrap over substrates: resample the per-substrate delta vector."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = delta[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(delta.mean()), float(lo), float(hi)


def load_method(spec: str, grail_preds, subs) -> Dict[str, List[str]]:
    """`GRAIL` -> the CSV preds; otherwise `name=path.json`."""
    if spec == "GRAIL":
        return {s: grail_preds.get(s, []) for s in subs}
    _, path = spec.split("=", 1)
    d = json.loads(Path(path).read_text())
    return {s: d.get(s, []) for s in subs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grail-csv", default=str(ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"))
    ap.add_argument("--a", default="GRAIL", help="method A (`GRAIL` or `name=path.json`)")
    ap.add_argument("--b", default="BioTransformer=" + str(ROOT / "artifacts" / "tier2" / "biotransformer_preds.json"),
                    help="method B (`name=path.json`)")
    ap.add_argument("--extra-methods", nargs="*", default=[],
                    help="further `name=path.json` methods to include in the per-method "
                         "protocol-sensitivity table (e.g. SyGMa, MetaPredictor)")
    ap.add_argument("--strict", default="canonical", help="strict-structural protocol (A expected to lead)")
    ap.add_argument("--normalized", default="inchikey_tautomer", help="normalized protocol (B expected to lead)")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-substrates", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "rank_flip_ci.json"))
    args = ap.parse_args()

    grail_preds, reals = load_grail_csv(Path(args.grail_csv))
    subs = [s for s in reals if reals[s]][: args.max_substrates]
    reals = {s: reals[s] for s in subs}
    a_name = args.a.split("=", 1)[0]
    b_name = args.b.split("=", 1)[0]
    A = load_method(args.a, grail_preds, subs)
    B = load_method(args.b, grail_preds, subs)
    print(f"pair: {a_name} vs {b_name} | n={len(subs)} | k={args.k} | boot={args.n_boot} seed={args.seed}", flush=True)

    report = {"a": a_name, "b": b_name, "k": args.k, "n_substrates": len(subs),
              "n_boot": args.n_boot, "seed": args.seed, "protocols": {}}
    per = {}  # (method, protocol) -> per-substrate recall vector
    for proto in (args.strict, args.normalized):
        ra = per_substrate_recall(A, reals, subs, proto, args.k)
        rb = per_substrate_recall(B, reals, subs, proto, args.k)
        per[proto] = (ra, rb)
        # delta = A - B (positive => A leads)
        mean, lo, hi = boot_ci(ra - rb, args.n_boot, args.seed)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
        leader = a_name if mean > 0 else b_name
        report["protocols"][proto] = {
            f"recall_{a_name}": round(float(ra.mean()), 4),
            f"recall_{b_name}": round(float(rb.mean()), 4),
            "delta_A_minus_B": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "leader": leader, "verdict": sig,
        }
        print(f"  [{proto:>18}] {a_name} {ra.mean():.3f} vs {b_name} {rb.mean():.3f} | "
              f"A-B={mean:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] -> {leader} leads, {sig}", flush=True)

    # interaction: (B - A)|normalized  -  (B - A)|strict  == how much MORE B gains from
    # normalization than A. If CI > 0, B is more protocol-sensitive than A (the flip's cause).
    ra_s, rb_s = per[args.strict]
    ra_n, rb_n = per[args.normalized]
    inter = (rb_n - ra_n) - (rb_s - ra_s)
    mean, lo, hi = boot_ci(inter, args.n_boot, args.seed)
    inter_sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
    report["interaction_B_extra_gain_from_normalization"] = {
        "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": inter_sig}
    print(f"  [interaction] {b_name} gains {mean:+.4f} more than {a_name} from "
          f"{args.strict}->{args.normalized} | 95%CI[{lo:+.4f},{hi:+.4f}] -> {inter_sig}", flush=True)

    # per-method protocol-sensitivity: recall(normalized) - recall(strict), with a paired
    # bootstrap CI. Quantifies how much each method's measured recall depends on the match
    # protocol -- a property of the method's output style (a rule enumerator emitting many
    # tautomer/charge variants gains from normalized matching; a canonicalizing method does
    # not). This is the mechanism the interaction CI certifies, shown across all methods.
    print("\n  per-method protocol-sensitivity (recall_normalized - recall_strict):", flush=True)
    sens = {}
    method_specs = [(a_name, A), (b_name, B)]
    for spec in args.extra_methods:
        nm = spec.split("=", 1)[0]
        method_specs.append((nm, load_method(spec, grail_preds, subs)))
    for nm, preds in method_specs:
        rs = per_substrate_recall(preds, reals, subs, args.strict, args.k)
        rn = per_substrate_recall(preds, reals, subs, args.normalized, args.k)
        mean, lo, hi = boot_ci(rn - rs, args.n_boot, args.seed)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        sens[nm] = {"sensitivity": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": sig}
        print(f"    {nm:<15} {mean:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}]  {sig}", flush=True)
    report["protocol_sensitivity_per_method"] = sens

    flip = (report["protocols"][args.strict]["leader"] != report["protocols"][args.normalized]["leader"]
            and report["protocols"][args.strict]["verdict"].startswith("SIG")
            and report["protocols"][args.normalized]["verdict"].startswith("SIG"))
    report["flip_significant"] = bool(flip)
    print(f"\nFLIP {'DEMONSTRATED (both CIs exclude 0, opposite leaders)' if flip else 'NOT significant'}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
