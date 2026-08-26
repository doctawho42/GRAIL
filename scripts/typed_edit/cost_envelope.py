"""What a substrate costs, and where the pipeline stops terminating.

A validation substrate of 291 heavy atoms did not finish in over three hours, which makes the
supported input envelope an open question rather than a footnote. The pair filter already caps
each MCS at five seconds, so the unbounded quantity is not one alignment but their number: the
pool grows with the substrate and every pair in it can pay the full cap.

This measures the three costs separately -- rule application, generator scoring, filter scoring
-- against heavy-atom count, and records which substrates do not finish at all. A substrate that
exceeds the deadline kills its worker, so the sweep continues past cases the pipeline cannot
complete; that is the point, and it is why the worker is a subprocess rather than a function.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GEN = ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"
FILT = ROOT / "artifacts/full5000_implicit/checkpoints/filter.pt"


def worker(gen_ckpt: str, filt_ckpt: str) -> int:
    """Read one SMILES per line, print one JSON timing record per line."""
    from bank_without_selection import _load
    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.workflows.factory import build_filter, build_generator

    generator = _load(Path(gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(filt_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        t0 = time.perf_counter()
        det = generator.generate_scored_with_details(s, top_k=7581, threshold=None,
                                                     compute_sites=False)
        t1 = time.perf_counter()
        cands = [d[0] for d in det]
        filt.score_batch(s, cands) if cands else []
        t2 = time.perf_counter()
        print(json.dumps({"smiles": s, "n_cands": len(cands),
                          "t_generate": round(t1 - t0, 3),
                          "t_filter": round(t2 - t1, 3)}), flush=True)
    return 0


def spawn(gen_ckpt, filt_ckpt):
    p = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker",
         "--gen-ckpt", gen_ckpt, "--filter-ckpt", filt_ckpt],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = p.stdout.readline()          # the ready handshake, which also pays the model load
    if not line or "ready" not in line:
        raise RuntimeError("worker did not come up")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gen-ckpt", default=str(GEN))
    ap.add_argument("--filter-ckpt", default=str(FILT))
    ap.add_argument("--deadline", type=float, default=600.0)
    ap.add_argument("--every", type=int, default=3, help="take every nth substrate by size")
    ap.add_argument("--out", default=str(ROOT / "results/cost_envelope.json"))
    args = ap.parse_args()

    if args.worker:
        return worker(args.gen_ckpt, args.filter_ckpt)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from _provenance import stamp
    from build_val_pools import population

    subs, _ = population()
    sized = sorted(((Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s) for s in subs
                    if Chem.MolFromSmiles(s)))
    # every nth by size, plus the whole top of the range where the failures live
    sample = sized[::args.every] + sized[-12:]
    seen, todo = set(), []
    for n, s in sample:
        if s not in seen:
            seen.add(s); todo.append((n, s))
    todo.sort()
    print(f"{len(todo)} substrates, {todo[0][0]}..{todo[-1][0]} heavy atoms", file=sys.stderr)

    def dump(a, rs):
        """Persist after every substrate: a sweep whose point is non-termination must not
        lose its record when the run is interrupted."""
        Path(a.out).write_text(json.dumps(
            {"provenance": stamp(__file__), "split": "validation",
             "deadline_s": a.deadline, "sample_every": a.every,
             "n_done": len(rs), "rows": rs}, indent=1))

    rows, p = [], spawn(args.gen_ckpt, args.filter_ckpt)
    for i, (n, s) in enumerate(todo, 1):
        t0 = time.perf_counter()
        p.stdin.write(s + "\n"); p.stdin.flush()
        # readline cannot be given a deadline, so a reader thread carries it
        box = {}

        def read():
            box["line"] = p.stdout.readline()

        th = threading.Thread(target=read, daemon=True)
        th.start(); th.join(args.deadline)
        if th.is_alive() or not box.get("line"):
            p.kill(); p.wait()
            rows.append({"smiles": s, "heavy": n, "finished": False,
                         "deadline_s": args.deadline})
            print(f"  {i}/{len(todo)} heavy={n:3d}  DID NOT FINISH in {args.deadline:.0f}s",
                  file=sys.stderr, flush=True)
            p = spawn(args.gen_ckpt, args.filter_ckpt)
            dump(args, rows)
            continue
        r = json.loads(box["line"])
        r.update({"heavy": n, "finished": True, "t_total": round(time.perf_counter() - t0, 3)})
        r["ms_per_candidate"] = round(1000 * r["t_filter"] / r["n_cands"], 2) if r["n_cands"] else None
        rows.append(r)
        print(f"  {i}/{len(todo)} heavy={n:3d}  pool={r['n_cands']:5d}  "
              f"gen={r['t_generate']:7.1f}s  filt={r['t_filter']:8.1f}s  "
              f"{r['ms_per_candidate']}ms/cand", file=sys.stderr, flush=True)
        dump(args, rows)
    p.kill()
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
