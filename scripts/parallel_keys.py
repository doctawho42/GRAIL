#!/usr/bin/env python3
"""Parallel structure-key precomputation for the matching criteria.

Keying is a pure function of a SMILES string, so it can be computed once per unique structure and
looked up. Tautomer canonicalisation dominates the cost and the prediction sets barely repeat
(SyGMa: 90,400 unique of 95,758), so the per-process cache does not help and the work is spread
across processes instead.

Scoring semantics are those of scripts.gloryx_rank_flip_ci.per_substrate_recall and must stay
identical: keys are taken in rank order, deduplicated by key, truncated at k, then intersected
with the reference key set. verify_against_serial() checks that on a sample.
"""
from __future__ import annotations
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _key_worker(arg):
    from grail_metabolism.metrics import _match_keys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles, mode = arg
    return smiles, next(iter(_match_keys([smiles], mode)), None)


def build_key_table(smiles: Iterable[str], mode: str, n_proc: int | None = None) -> Dict[str, str]:
    uniq = sorted({s for s in smiles if s})
    n_proc = n_proc or max(1, cpu_count() - 2)
    if len(uniq) < 500 or n_proc == 1:
        return dict(_key_worker((s, mode)) for s in uniq)
    with Pool(n_proc) as pool:
        pairs = pool.map(_key_worker, [(s, mode) for s in uniq], chunksize=128)
    return {s: k for s, k in pairs}


def recall_vector(preds: Dict[str, List[str]], truth: Dict[str, List[str]],
                  subs: Sequence[str], table: Dict[str, str], k: int) -> np.ndarray:
    """Same semantics as per_substrate_recall, reading keys from a precomputed table."""
    v = np.empty(len(subs))
    for i, s in enumerate(subs):
        ranked, seen = [], set()
        for item in preds.get(s, []):
            key = table.get(item)
            if key and key not in seen:
                seen.add(key)
                ranked.append(key)
        real = {table[r] for r in truth[s] if table.get(r)}
        v[i] = (len(set(ranked[:k]) & real) / len(real)) if real else 0.0
    return v


def verify_against_serial(preds, truth, subs, table, mode, k, sample=40, seed=0) -> None:
    """Fail loudly if the parallel path disagrees with the paper's own scorer."""
    from scripts.gloryx_rank_flip_ci import per_substrate_recall
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(subs), size=min(sample, len(subs)), replace=False)
    sub = [subs[i] for i in idx]
    mine = recall_vector(preds, truth, sub, table, k)
    theirs = per_substrate_recall(preds, truth, sub, mode, k)
    if not np.allclose(mine, theirs, atol=1e-12):
        bad = int(np.argmax(np.abs(mine - theirs)))
        raise SystemExit(
            f"ERROR: parallel keying disagrees with the serial scorer on '{mode}' "
            f"(substrate {sub[bad]!r}: {mine[bad]} vs {theirs[bad]}). Not a speed-up, a bug.")
    print(f"    verified against serial scorer on {len(sub)} substrates ({mode})", flush=True)
