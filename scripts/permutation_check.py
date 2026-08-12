#!/usr/bin/env python3
r"""Do the translation boards' certified reversals survive a test that assumes nothing?

Where a board scores each item a hit or a miss, the paired difference is three-valued and the
exact conditional test is available, so that is what decides multiplicity there. The translation
boards are not like that: an \textsc{mqm} or \textsc{esa} segment score is continuous, zero-inflated
and long-tailed on the left, and their $p$ comes from the normal approximation to a paired mean. At
the thresholds Holm compares against --- $10^{-5}$ to $10^{-3}$ --- that approximation is doing
work no central limit theorem promises on a few hundred items with one catastrophic outlier.

There is a test that assumes nothing about the shape: under the null that the two systems are
exchangeable item by item, flipping the sign of each paired difference independently leaves the
distribution alone, so the sign-flip permutation distribution of the mean is exact by construction.
It is expensive, which is why it is not the estimator; but the question is not whether it agrees
everywhere, it is whether it changes a verdict. So it is computed exactly where a verdict lives:
every reversal test its board certifies, and every test within a factor of the board's own Holm
cutoff, which is where a change of test could move something across.

Reports, per test, the analytic $p$, the permutation $p$ and whether the board's verdict moves.
"""
from __future__ import annotations

import argparse
import ast
import math
import importlib.util
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

DRAWS, SEED = 200_000, 0
NEAR = 10.0  # a test within this factor of the cutoff is close enough for a change of test to matter


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


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(f"_pc_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"_pc_{name}"] = m
    spec.loader.exec_module(m)
    return m


def hits_for(board_key: str) -> dict:
    """The per-item vectors a board was built from, rebuilt by the script that built it."""
    if board_key in ("en-de", "ja-zh"):
        wb = _mod("wmt_board")
        tsv = ("mqm_generalMT2024_ende.tsv" if board_key == "en-de"
               else "mqm_generalMT2024_jazh.tsv")
        d = wb.load(ROOT / "data/external/wmt24/humeval" / tsv)
        hits, systems, items, cells = wb.build_hits(d, "macro")
        return hits
    if board_key.startswith("esa:"):
        eb = _mod("wmt_esa_boards")
        per = eb.load(ROOT / "data/external/wmt24/txt/wmt24-genmt-humeval.jsonl")
        hits, systems, items, cells = eb.build_board(per[board_key.split(":", 1)[1]])
        return hits
    if board_key.startswith("wmt23:"):
        w3 = _mod("wmt23_boards")
        d = w3.load(ROOT / "data/external/wmt23/humaneval/DA+SQM/WMT23.scores_all.csv")
        src, tgt = board_key.split(":", 1)[1].split("-")
        sub = d[(d["src"] == src) & (d["tgt"] == tgt)]
        hits, systems, items, cells = w3.build_board(sub)
        return hits
    raise KeyError(board_key)


def permutation_p_mc(d: np.ndarray, draws: int, rng) -> float:
    """Monte Carlo sign-flip $p$, kept only to check the analytic tail below.

    Its resolution is $1/B$, which is the whole difficulty: the thresholds Holm compares against
    here are $10^{-6}$ to $10^{-3}$, so a draw count that could resolve them is a draw count that
    cannot be run. It is used where the tail is coarse enough for it to have something to say.
    """
    obs = abs(float(d.mean()))
    n = len(d)
    hit, done = 0, 0
    block = max(1, min(draws, int(4e7 // max(n, 1))))
    while done < draws:
        k = min(block, draws - done)
        signs = rng.integers(0, 2, size=(k, n)) * 2 - 1
        hit += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-15).sum())
        done += k
    return (hit + 1) / (draws + 1)


def permutation_p(d: np.ndarray) -> float:
    r"""Two-sided sign-flip $p$ from the saddlepoint approximation to its own exact null.

    Under exchangeability the null distribution of $T=\sum_i arepsilon_i d_i$ with
    $arepsilon_i=\pm1$ is exact by construction, and its cumulant generating function is
    $K(t)=\sum_i \log\cosh(t d_i)$ --- no assumption about the shape of a segment score enters. What
    is hard is its tail, which is exactly where a multiplicity correction reads it and exactly where
    Monte Carlo cannot go. Lugannani--Rice gives that tail to a relative error of order $1/n$
    uniformly, including far out, which is why the saddlepoint and not the resample is the
    instrument here.

    One caveat, stated because it is checkable: on a lattice --- every $|d_i|$ equal, which is what
    a hit-or-miss board would give --- the continuous form runs about a fifth low against the exact
    binomial, since it omits the continuity correction. That is why the hit boards use the exact
    conditional test instead and this is applied only where the scores are continuous, and why
    every value it returns here is checked against a resample wherever the resample can resolve it.
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0.0]                      # a zero difference contributes nothing under a sign flip
    n = len(d)
    s = abs(float(d.sum()))
    if n == 0 or s == 0.0:
        return 1.0
    if s >= float(np.abs(d).sum()) - 1e-12:      # the statistic is at its own maximum
        return 2.0 ** (1 - n)

    def kp(t):                                   # K'(t)
        return float(np.sum(d * np.tanh(t * d)))

    lo, hi = 0.0, 1.0
    while kp(hi) < s:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):                         # bisection: K' is increasing in t
        mid = 0.5 * (lo + hi)
        if kp(mid) < s:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    if t <= 0.0:
        return 1.0
    k = float(np.sum(np.log(np.cosh(t * d))))
    k2 = float(np.sum((d / np.cosh(t * d)) ** 2))
    w = math.sqrt(max(2.0 * (t * s - k), 0.0))
    u = t * math.sqrt(k2)
    if w <= 0.0 or u <= 0.0:
        return 1.0
    tail = 0.5 * math.erfc(w / math.sqrt(2.0)) + _phi(w) * (1.0 / w - 1.0 / u)
    return float(min(1.0, max(0.0, 2.0 * tail)))


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=str(ROOT / "results" / "permutation_check.json"))
    args = ap.parse_args()

    um = _mod("union_multiplicity")
    boards = {}
    for name, key in (("robust_order_wmt24_en-de.json", "en-de"),
                      ("robust_order_wmt24_ja-zh.json", "ja-zh")):
        q = ROOT / "results" / name
        if q.exists():
            boards[key] = json.loads(q.read_text())
    for src, prefix in (("robust_order_wmt24_esa.json", "esa"), ("robust_order_wmt23.json", "wmt23")):
        q = ROOT / "results" / src
        if q.exists():
            for lp, b in json.loads(q.read_text())["boards"].items():
                boards[f"{prefix}:{lp}"] = b

    rng = np.random.default_rng(SEED)
    rows, moved = [], 0
    for key, b in boards.items():
        cut = um.cutoff(b["multiplicity"]["p_values"], 0.05)
        interesting = [r for r in b["multiplicity"]["reversal_tests"]
                       if r["p"] <= cut * NEAR]
        if not interesting:
            continue
        hits = hits_for(key)
        for r in interesting:
            hi, lo = r["pair"].split(" over ")
            cell = ast.literal_eval(r["cell"])
            d = np.asarray(hits[(hi, cell)] - hits[(lo, cell)], dtype=float)
            pp = permutation_p(d)
            # where the tail is coarse enough for a resample to see it, the two are compared
            mc = (permutation_p_mc(d, args.draws, rng)
                  if pp > 20.0 / args.draws else None)
            certified_now = r["p"] <= cut
            certified_then = pp <= cut
            moved += int(certified_now != certified_then)
            rows.append({"board": key, "pair": r["pair"], "cell": r["cell"], "n_items": len(d),
                         "analytic_p": r["p"], "permutation_p": pp, "monte_carlo_p": mc,
                         "board_cutoff": cut,
                         "certified_analytic": certified_now, "certified_permutation":
                             certified_then, "ratio": pp / r["p"] if r["p"] else float("inf")})
            print(f"  {key:22s} {r['pair'][:34]:34s} n={len(d):5d} "
                  f"analytic {r['p']:.3e}  permutation {pp:.3e}  "
                  f"{'certified' if certified_now else 'not certified'}"
                  f"{' -> CHANGES' if certified_now != certified_then else ''}", flush=True)

    cert = [r for r in rows if r["certified_analytic"]]
    _val = [abs(math.log(r["monte_carlo_p"] / r["permutation_p"]))
            for r in rows if r.get("monte_carlo_p") and r["permutation_p"] > 0]
    rep = {"config": {**_code_version(), "draws": args.draws, "seed": SEED,
                      "near_factor": NEAR,
                      "note": "the sign-flip permutation is exact under exchangeability and "
                              "assumes nothing about the shape of a segment score; it is computed "
                              "on every certified reversal and every test within a factor of ten "
                              "of its board's cutoff"},
           "n_tests_checked": len(rows),
           "n_certified_checked": len(cert),
           "n_verdicts_that_move": moved,
           "largest_ratio": round(max((r["ratio"] for r in rows), default=0.0), 3),
           "largest_ratio_among_certified": round(max((r["ratio"] for r in cert), default=0.0), 3),
           "n_validated_against_a_resample": len(_val),
           "largest_log_gap_against_the_resample": round(max(_val, default=0.0), 3),
           "tests": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n  {len(rows)} tests checked, {len(cert)} of them certified; "
          f"{moved} verdicts move")
    print(f"  the permutation p runs to {rep['largest_ratio']}x the analytic one, and to "
          f"{rep['largest_ratio_among_certified']}x among the certified")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
