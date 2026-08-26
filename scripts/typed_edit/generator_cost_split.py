"""Where the generator's time goes, split three ways.

The envelope sweep shows the generator, not the filter, is what grows with substrate size: at 39
heavy atoms it spends 211 seconds against the filter's 4.4. That rules out the pool cap as the
guard for it, because the pool has to be enumerated before it can be capped, so the guard has to
go somewhere inside the enumeration. Which place depends on which of three costs dominates:

  forward       one model pass that encodes every rule graph, constant in the substrate
  reactants     RDKit applying 7,581 templates, growing with the number of matching sites
  normalize     standardising each product, which runs tautomer canonicalisation

The three are measured rather than argued about: the forward pass is timed on its own, and the
normalisation is timed by wrapping the module-level function the generator calls, so what is
left over is the reaction execution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

GEN = ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(GEN))
    ap.add_argument("--pools", default=str(ROOT / "results/wide_pools.json"))
    ap.add_argument("--n", type=int, default=8, help="substrates, spread over the size range")
    ap.add_argument("--out", default=str(ROOT / "results/generator_cost_split.json"))
    ap.add_argument("--repeat", type=int, default=2,
                    help="times to run each substrate; the last run is the one recorded, so the "
                         "normalisation cache is in the state a repeated query would find")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _load
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.model import generator as genmod
    from grail_metabolism.workflows.factory import build_generator

    subs = list(json.loads(Path(args.pools).read_text())["pools"])
    sized = sorted((Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s) for s in subs
                   if Chem.MolFromSmiles(s))
    step = max(1, len(sized) // args.n)
    sample = sized[::step][:args.n]

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))

    # wrap the normalisation the enumeration loop calls, counting calls and seconds
    original = genmod._normalize_smiles_cached
    box = {"n": 0, "t": 0.0}

    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return original(*a, **k)
        finally:
            box["n"] += 1
            box["t"] += time.perf_counter() - t

    genmod._normalize_smiles_cached = timed

    # The first forward pass encodes and caches every rule graph, so timing it cold makes the
    # forward share exceed the total and the remainder go negative. Warm the cache on a trivial
    # molecule first; every timing below is then of a warm system, which is also the state a
    # service runs in.
    generator._prepare_generation("CCO", 7581, None)
    generator.generate_scored_with_details("CCO", top_k=7581, threshold=None,
                                           compute_sites=False)

    rows = []
    for heavy, s in sample:
        reps = []
        for rep in range(args.repeat):
            t0 = time.perf_counter()
            generator._prepare_generation(s, 7581, None)
            t_forward = time.perf_counter() - t0
            box["n"], box["t"] = 0, 0.0
            t1 = time.perf_counter()
            det = generator.generate_scored_with_details(s, top_k=7581, threshold=None,
                                                         compute_sites=False)
            t_total = time.perf_counter() - t1
            t_norm, n_norm = box["t"], box["n"]
            reps.append({"t_total": round(t_total, 3), "t_forward": round(t_forward, 3),
                         "t_normalize": round(t_norm, 3), "n_normalize_calls": n_norm})
        t_react = t_total - t_forward - t_norm
        rows.append({"heavy": heavy, "n_cands": len(det), "repeat": args.repeat,
                     "t_total": round(t_total, 2), "t_forward": round(t_forward, 2),
                     "t_reactants": round(t_react, 2), "t_normalize": round(t_norm, 2),
                     "n_normalize_calls": n_norm,
                     "share_forward": round(t_forward / t_total, 3) if t_total else None,
                     "share_reactants": round(t_react / t_total, 3) if t_total else None,
                     "share_normalize": round(t_norm / t_total, 3) if t_total else None,
                     # the first pass pays standardisation for products nothing has seen; the
                     # last finds them all cached. A service meets molecules cold, so the gap
                     # between these two is the part of the cost a cache can remove and the
                     # part it cannot.
                     "cold": reps[0], "warm": reps[-1],
                     "cold_over_warm": round(reps[0]["t_total"] / reps[-1]["t_total"], 1)
                     if reps[-1]["t_total"] else None})
        r = rows[-1]
        print(f"  heavy={heavy:3d}  cands={len(det):5d}  total={r['t_total']:8.2f}s   "
              f"forward={r['share_forward']:.0%}  reactants={r['share_reactants']:.0%}  "
              f"normalize={r['share_normalize']:.0%}   cold/warm="
              f"{r['cold_over_warm']}x  ({r['cold']['n_normalize_calls']} normalisations cold)",
              file=sys.stderr, flush=True)
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__), "checkpoint": args.gen_ckpt,
             "note": "forward is timed alone; normalize is timed by wrapping the function the "
                     "enumeration calls; reactants is the remainder",
             "rows": rows}, indent=1))

    genmod._normalize_smiles_cached = original
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
