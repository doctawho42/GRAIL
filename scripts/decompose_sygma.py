#!/usr/bin/env python3
"""Apply the recall decomposition to SyGMa, on the same split and in the same frame as GRAIL.

The decomposition has so far been self-diagnosis: one method, our own. It is a comparative
primitive only if it separates two methods that lose recall for different reasons. SyGMa is
rule-based with an enumerable bank, so it decomposes too, and its profile is the opposite of
GRAIL's: a lower ceiling that it converts almost completely, against a higher ceiling that
converts badly.

SyGMa has no selection stage -- it applies every applicable rule and emits the result -- so its
budgeted pool is its full pool and its selection factor is 1 by construction. That is a property
of the method, not an artefact of how we measure it, and it is what the comparison is about.

Micro (pooled ratio-of-sums) throughout, matching factorize_recall.py, with a cluster bootstrap
over substrates.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# No torch here on purpose: the key tables are precomputed by build_key_tables.py and verified
# against the paper's serial scorer by verify_key_tables.py.

MATCH, K, N_BOOT, SEED = "inchikey_tautomer", 15, 10000, 0
PRED = ROOT / "results" / "sygma_fulltest_predictions.json"
OUT = ROOT / "results" / "decompose_sygma.json"


def keyed(items, table):
    """Per-item keys in rank order, deduplicated -- the paper's matcher convention."""
    out, seen = [], set()
    for it in items:
        k = table.get(it)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    preds = json.loads(PRED.read_text())
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MATCH}.json").read_text())
    subs = sorted(s for s in truth if truth[s] and s in preds)
    print(f"substrates: {len(subs)}", flush=True)

    # Key every distinct structure once, in parallel; tautomer canonicalisation dominates and
    # SyGMa's pools barely repeat, so a per-process cache does not help.
    U, Cfull, H = [], [], []
    for s in subs:
        refs = {table[r] for r in truth[s] if table.get(r)}
        pool = keyed(preds[s], table)
        U.append(len(refs))
        Cfull.append(len(refs & set(pool)))
        H.append(len(refs & set(pool[:K])))
    U, Cfull, H = map(np.array, (U, Cfull, H))
    # SyGMa emits its whole applicable pool: the budgeted pool is the full pool.
    Cbud = Cfull

    def factors(idx):
        u, cf, cb, h = U[idx].sum(), Cfull[idx].sum(), Cbud[idx].sum(), H[idx].sum()
        return (cf / u if u else 0.0,
                cb / cf if cf else 0.0,
                h / cb if cb else 0.0,
                h / u if u else 0.0)

    full = np.arange(len(subs))
    cov, sel, rank, rec = factors(full)
    rng = np.random.default_rng(SEED)
    boot = np.array([factors(rng.integers(0, len(subs), len(subs))) for _ in range(N_BOOT)])
    ci = lambda j: [round(float(np.quantile(boot[:, j], 0.025)), 4),
                    round(float(np.quantile(boot[:, j], 0.975)), 4)]

    rep = {"method": "SyGMa", "match": MATCH, "k": K, "n_substrates": len(subs),
           "n_boot": N_BOOT, "seed": SEED,
           "factors": {"coverage_bank": {"point": round(cov, 4), "ci95": ci(0)},
                       "selection_retention": {"point": round(sel, 4), "ci95": ci(1),
                                               "note": "1 by construction: SyGMa has no selection stage"},
                       "ranking_conversion": {"point": round(rank, 4), "ci95": ci(2)}},
           "micro_recall": round(rec, 4),
           "mean_output": round(float(np.mean([len(keyed(preds[s], table)) for s in subs])), 2)}
    print(json.dumps(rep["factors"], indent=1), flush=True)
    print(f"micro recall@{K} = {rec:.4f}   product = {cov*sel*rank:.4f}", flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
