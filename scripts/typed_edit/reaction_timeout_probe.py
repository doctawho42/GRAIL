"""Does the five-second cap on a single rule application actually hold?

`safe_run_reactants` arms `signal.setitimer(ITIMER_REAL, 5.0)` around `rxn.RunReactants`. On
CPython a signal that arrives during a C extension call does not run its Python handler until
the interpreter regains control, so a template that spends a minute inside RDKit is not
interrupted at five seconds -- the alarm fires into a flag that nobody reads until the call
returns. Whether that matters is an empirical question about this bank and these substrates,
and it is answered by timing every one of the 7,581 applications and looking at the tail.

The answer decides where the input envelope has to be enforced. If no single application ever
exceeds the cap, the generator's cost is the sum of many small ones and bounding the rule budget
bounds it. If applications routinely run past it, the cap is decorative and the bound has to be
somewhere that can actually stop the work.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/wide_pools.json"))
    ap.add_argument("--n-largest", type=int, default=3)
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/reaction_timeout_probe.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _load
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.utils.preparation import (_REACTION_PRODUCT_CAP,
                                                    _REACTION_TIMEOUT_SECONDS,
                                                    safe_run_reactants)
    from grail_metabolism.workflows.factory import build_generator

    # the generator's own initialised templates, so the probe times what the pipeline runs
    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    reactions = [(str(i), rx) for i, rx in enumerate(generator.rule_reactions) if rx is not None]
    print(f"{len(reactions)} templates from the checkpoint", file=sys.stderr, flush=True)

    subs = list(json.loads(Path(args.pools).read_text())["pools"])
    sized = sorted(((Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s) for s in subs
                    if Chem.MolFromSmiles(s)), reverse=True)[:args.n_largest]

    rows = []
    for heavy, smi in sized:
        mol = Chem.MolFromSmiles(smi)
        times, n_prod, over = [], 0, []
        t_all = time.perf_counter()
        for i, (label, rx) in enumerate(reactions):
            t0 = time.perf_counter()
            out = safe_run_reactants(rx, mol)
            dt = time.perf_counter() - t0
            times.append(dt)
            n_prod += sum(len(t) for t in out)
            if dt > _REACTION_TIMEOUT_SECONDS:
                over.append({"rule_index": i, "seconds": round(dt, 2)})
        total = time.perf_counter() - t_all
        times.sort()
        rows.append({
            "heavy": heavy, "smiles": smi[:120],
            "total_s": round(total, 1), "n_products": n_prod,
            "per_rule_s": {"mean": round(sum(times) / len(times), 4),
                           "p50": round(times[len(times) // 2], 4),
                           "p99": round(times[int(0.99 * len(times))], 4),
                           "max": round(times[-1], 3)},
            "applications_over_the_cap": len(over),
            "worst_offenders": sorted(over, key=lambda x: -x["seconds"])[:5]})
        r = rows[-1]
        print(f"  heavy={heavy:4d}  total={r['total_s']:8.1f}s  products={n_prod:6d}  "
              f"per-rule max={r['per_rule_s']['max']:.2f}s  "
              f"over the {_REACTION_TIMEOUT_SECONDS:.0f}s cap: {len(over)}",
              file=sys.stderr, flush=True)
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__), "n_rules": len(reactions),
             "cap_seconds": _REACTION_TIMEOUT_SECONDS, "cap_products": _REACTION_PRODUCT_CAP,
             "note": "times only rule application; normalisation of the products is not "
                     "included, so these are a lower bound on the generator's cost",
             "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
