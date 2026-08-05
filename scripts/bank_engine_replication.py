#!/usr/bin/env python3
"""Does the one switch that carries the engine effect carry it for banks we did not author?

scripts/engine_knobs.py isolates the effect to a single decision -- whether the substrate is
expanded with explicit hydrogens before the templates fire -- on the 152 rules our bank and SyGMa's
share. Measured on one rule set, that is a fact about those 152 rules. The claim the paper makes is
larger: that a reported ceiling is a property of the system rather than of the knowledge base, for
knowledge bases in general. So the same switch is flipped here against three banks assembled by
three groups with no common authorship:

    grail_full        our own 7581 SMIRKS -- and the bank the paper's headline ceiling is measured on
    sygma_175         SyGMa's complete bank, both phases
    biotransformer    the SMIRKS shipped in BioTransformer's reaction databases

Same substrates, same matcher, same engine, one line different. The grail_full arm doubles as a
gate: with the switch in its deployed position it must reproduce the committed
results/reach_engine_vs_bank.json ceiling on these substrates, or this is not the engine the paper
reports. It also answers a question the paper currently leaves open -- whether the ceiling we
publish understates what our own bank holds, and by how much.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sygma
from rdkit import Chem, RDLogger

from engine_knobs import DEFAULT, apply_with
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
SYGMA_RULES = Path(os.environ.get("SYGMA_RULES") or (Path(sygma.__file__).parent / "rules"))
# the same three files scripts/decompose_biotransformer.py reads, so the bank here is the 994
# the appendix reports and not a subset of it
BT_DB = [ROOT / "artifacts/tier2/biotransformer/database/metabolicReactions.json",
         ROOT / "artifacts/tier2/biotransformer/database/ENVMICRO/metabolicReactions.json",
         ROOT / "artifacts/tier2/biotransformer/database/standardizationReactions.json"]
_CTX: dict = {}


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


def _dedup(seq):
    out, seen = [], set()
    for s in seq:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_bank(name: str) -> list[str]:
    if name == "grail_full":
        return _dedup(l.split("\t")[0] for l in
                      (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines()
                      if l.strip() and not l.startswith("#"))
    if name == "sygma_175":
        return _dedup(l.split("\t")[0] for f in ("phase1.txt", "phase2.txt")
                      for l in (SYGMA_RULES / f).read_text().splitlines()
                      if l.strip() and not l.startswith("#"))
    if name == "biotransformer":
        # BioTransformer's databases carry // comments before the JSON, so the SMIRKS are pulled
        # by pattern exactly as scripts/decompose_biotransformer.py pulls them.
        return _dedup(t.replace("\\\\", "\\") for p in BT_DB if p.exists()
                      for t in re.findall(r'"smirks"\s*:\s*"([^"]{5,})"', p.read_text(errors="ignore")))
    raise SystemExit(f"unknown bank {name}")


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, {True: (0, 0), False: (0, 0)}
    out = {}
    for add_hs in (True, False):
        prods = apply_with(mol, _CTX["rules"], add_hs, DEFAULT["norm"], DEFAULT["drop_invalid"])
        u, hit, _ = _tautomer_recovered(trues, prods, audit=False)
        out[add_hs] = (int(u), int(hit))
    return sub, out


EXPECTED_RULES = {"grail_full": 7581, "sygma_175": 175, "biotransformer": 994}


def run_bank(name: str, items, workers: int) -> dict:
    rules = load_bank(name)
    if len(rules) != EXPECTED_RULES[name]:
        raise SystemExit(f"{name}: parsed {len(rules)} rules, the committed record says "
                         f"{EXPECTED_RULES[name]} -- this is not the bank the paper reports")
    print(f"\n{name}: {len(rules)} rules over {len(items)} substrates", flush=True)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {name} {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])           # imap_unordered: fix the order before resampling

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    # stamped per bank: a partial re-run replaces one block and leaves the others, so a single
    # top-level config would claim this invocation's provenance for numbers it did not produce
    out: dict = {"n_rules": len(rules), "provenance": _code_version()}
    ratios, counts, Uref = {}, {}, None
    for add_hs in (True, False):
        U = np.array([r[1][add_hs][0] for r in rows])
        H = np.array([r[1][add_hs][1] for r in rows])
        if Uref is None:
            Uref = U
        elif not np.array_equal(U, Uref):
            raise SystemExit("the reference denominator differs between arms")
        bt = np.array([H[j].sum() / max(U[j].sum(), 1) for j in idx])
        ratios[add_hs], counts[add_hs] = H / np.maximum(U, 1), H
        out["deployed" if add_hs else "without_explicit_h"] = {
            "reach": round(float(H.sum() / max(U.sum(), 1)), 4),
            "ci95": [round(float(np.quantile(bt, .025)), 4),
                     round(float(np.quantile(bt, .975)), 4)]}
    # micro is the estimator the committed engine term uses -- a difference of ratio-of-sums, not a
    # mean of per-substrate ratios. The two disagree materially, so both are written out, labelled.
    D = counts[False] - counts[True]
    mic = float(D.sum() / max(Uref.sum(), 1))
    mbt = np.array([D[j].sum() / max(Uref[j].sum(), 1) for j in idx])
    mlo, mhi = float(np.quantile(mbt, .025)), float(np.quantile(mbt, .975))
    d = ratios[False] - ratios[True]
    abt = d[idx].mean(axis=1)
    alo, ahi = float(np.quantile(abt, .025)), float(np.quantile(abt, .975))
    out["gain_from_dropping_explicit_h"] = {
        "delta": round(mic, 4), "ci95": [round(mlo, 4), round(mhi, 4)],
        "excludes_zero": bool(mlo > 0 or mhi < 0), "estimator": "micro",
        "macro": {"delta": round(float(d.mean()), 4), "ci95": [round(alo, 4), round(ahi, 4)],
                  "excludes_zero": bool(alo > 0 or ahi < 0)}}
    g = out["gain_from_dropping_explicit_h"]
    print(f"  {name}: {out['deployed']['reach']:.4f} deployed -> "
          f"{out['without_explicit_h']['reach']:.4f} without explicit H  "
          f"{g['delta']:+.4f} {g['ci95']} micro / {g['macro']['delta']:+.4f} macro "
          f"{'certified' if g['excludes_zero'] else 'n.s.'}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", default=["sygma_175", "biotransformer", "grail_full"])
    ap.add_argument("--out", default=str(ROOT / "results" / "bank_engine_replication.json"))
    args = ap.parse_args()

    subs = [r["sub"] for r in
            json.loads((ROOT / "results/filter_vs_prior_ci.json").read_text())["per_substrate"]]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    items = [(s, truth[s]) for s in subs if truth.get(s)]
    workers = max(1, (os.cpu_count() or 4) - 2)

    out_path = Path(args.out)
    banks = (json.loads(out_path.read_text()).get("banks", {}) if out_path.exists() else {})
    for name in args.banks:
        banks[name] = run_bank(name, items, workers)
        # written after each bank so a long grail_full arm never loses the cheap ones
        out_path.write_text(json.dumps(
            {"config": {**_code_version(), "banks_this_invocation": list(args.banks),
                        "n_substrates": len(items), "match": "inchikey_tautomer",
                        "n_boot": N_BOOT, "seed": SEED, "knobs_held": {k: v for k, v in DEFAULT.items()
                                                                       if k != "add_hs"},
                        "apply_loop": "scripts/engine_knobs.apply_with (gated there against arm A)"},
             "banks": banks}, indent=1))

    if "grail_full" in banks:
        committed = json.loads((ROOT / "results/reach_engine_vs_bank.json").read_text())
        ceiling = committed["gates"]["grail_full_bank_same_substrates"]
        got = banks["grail_full"]["deployed"]["reach"]
        print(f"\ngate: grail_full deployed {got} against the committed ceiling {ceiling}")
        if abs(got - ceiling) > 1e-4:
            raise SystemExit("this is not the engine the published ceiling was measured with")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
