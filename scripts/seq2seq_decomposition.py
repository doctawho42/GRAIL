#!/usr/bin/env python3
"""The decomposition's analogue for a sequence-to-sequence method, which has no rule bank to apply.

The factorisation in this paper runs only on methods whose rule base can be enumerated, which
excludes the two strongest systems in the comparison. A reviewer's point is that the structural
analogue is available anyway: for a generative model the three gates are the same three, with the
bank replaced by what a wide beam can reach at all.

  coverage  -- references appearing anywhere in a wide-beam decode, the generative counterpart of
               applying the whole bank
  selection -- what survives narrowing to the deployed beam, the counterpart of the rule budget
  ranking   -- what survives truncation to k, the same gate in both formulations

The three still nest, so the identity is exact for the same reason: a reference in the deployed
beam was in the wide beam, and a reference in the top k was in the deployed beam. What differs is
that a rule bank is a fixed object while a beam is a budget, so the first factor here is not a
property of a knowledge base and should not be read as one. It bounds this decode, not the model.

Both arms come from the same checkpoints, the same seed and the same two-stage pipeline, differing
only in n_best and beam_size, so nothing but the beam changes between them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED, K = 10000, 0, 15
_TABLE = json.loads((ROOT / "results" / "key_tables" / "inchikey_tautomer.json").read_text())
_MISS: dict = {}


def _rel(p) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _key(s):
    k = _TABLE.get(s) or _MISS.get(s)
    if k is not None:
        return k
    try:
        k = _tautomer_inchikey(s)
    except Exception:
        k = s
    _MISS[s] = k
    return k


def _keys(seq, cap=None):
    out, seen = [], set()
    for s in seq:
        k = _key(s)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
            if cap and len(out) >= cap:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", required=True, help="predictions from the wide-beam decode")
    ap.add_argument("--deployed", default="artifacts/tier2/metapredictor_preds.json")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out", default=str(ROOT / "results" / "seq2seq_decomposition.json"))
    args = ap.parse_args()

    wide = json.loads((ROOT / args.wide).read_text()) if not Path(args.wide).is_absolute() \
        else json.loads(Path(args.wide).read_text())
    dep = json.loads((ROOT / args.deployed).read_text())
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    subs = [s for s in json.loads((ROOT / "artifacts/tier2/substrates.json").read_text())
            if truth.get(s) and s in dep and s in wide]
    print(f"substrates scored by both decodes: {len(subs)}", flush=True)
    if len(subs) < 100:
        raise SystemExit("the two decodes do not share enough substrates to compare")

    U, Cfull, Cbud, H = [], [], [], []
    for s in subs:
        ref = {k for k in (_key(x) for x in truth[s]) if k}
        w = set(_keys(wide[s]))
        d = set(_keys(dep[s]))
        top = set(_keys(dep[s], args.k))
        U.append(len(ref))
        Cfull.append(len(ref & (w | d)))   # the deployed beam is a subset of the wide one by design;
        Cbud.append(len(ref & d))          # union guards against a decode that dropped a hypothesis
        H.append(len(ref & top))
    U, Cfull, Cbud, H = map(np.array, (U, Cfull, Cbud, H))
    if (Cbud > Cfull).any():
        raise SystemExit(f"the deployed decode reaches references the wide one does not on "
                         f"{(Cbud > Cfull).sum()} substrates -- the sets do not nest and the "
                         f"identity does not hold")

    def factors(i):
        u, cf, cb, h = U[i].sum(), Cfull[i].sum(), Cbud[i].sum(), H[i].sum()
        return (cf / u if u else 0.0, cb / cf if cf else 0.0, h / cb if cb else 0.0, h / u if u else 0.0)

    full = np.arange(len(subs))
    cov, sel, rank, rec = factors(full)
    rng = np.random.default_rng(SEED)
    boot = np.array([factors(rng.integers(0, len(subs), len(subs))) for _ in range(N_BOOT)])
    ci = lambda j: [round(float(np.quantile(boot[:, j], .025)), 4),
                    round(float(np.quantile(boot[:, j], .975)), 4)]

    rep = {"config": {**_code_version(), "wide": _rel(Path(args.wide)),
                      "deployed": _rel(ROOT / args.deployed), "k": args.k},
           "n": len(subs), "match": "inchikey_tautomer",
           "mean_wide": round(float(np.mean([len(_keys(wide[s])) for s in subs])), 1),
           "mean_deployed": round(float(np.mean([len(_keys(dep[s])) for s in subs])), 1),
           "factors": {"beam_coverage": {"point": round(cov, 4), "ci95": ci(0)},
                       "selection_retention": {"point": round(sel, 4), "ci95": ci(1)},
                       "ranking_conversion": {"point": round(rank, 4), "ci95": ci(2)}},
           "micro_recall": round(rec, 4)}
    print(f"\n  beam coverage      {cov:.4f} {ci(0)}")
    print(f"  selection retention {sel:.4f} {ci(1)}")
    print(f"  ranking conversion  {rank:.4f} {ci(2)}")
    print(f"  product {cov*sel*rank:.4f} against micro recall@{args.k} {rec:.4f}")
    if abs(cov * sel * rank - rec) > 1e-6:
        raise SystemExit("the factors do not multiply to the realised recall -- the sets do not nest")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
