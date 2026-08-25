"""Round-robin decoding over formula groups, against the orderings that ignore them.

The oracle decomposition says almost the whole ordering loss is between formula groups and
almost none of it is within: on the shipped generator a perfect choice of which group to spend
budget on is worth +0.314 of micro recall@15, a perfect choice of isomer inside a group +0.011.
A ranker that emits fifteen candidates from three groups is spending its budget the way the
oracle says is wrong, so this measures the family of cheapest possible corrections: cap how many
candidates any one group may take in the early slots, at m = 1, 2, 3 and 5, plus round-robin,
which is the m = 1 extreme that also discards the score order between groups. None of them fits
a parameter and the m -> infinity limit is the uncapped ranking itself.

The answer is that the whole family hurts, monotonically in how tight the cap is. The oracle's
headroom is selective, not diversifying: it comes from spending the budget on the groups that
contain a reference, and spreading the budget evenly across groups is not an approximation to
that, it is the opposite of it. This closes the cheap route to the between-group headroom.

A rule that had won here would still have been an upper bound rather than a result, having been
chosen after reading the oracle decomposition on these same substrates, which is the trap H7 was
written for. The artifact records that condition whichever way the number falls.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402  (the one implementation of the registered rule)

from bank_without_selection import _dedup  # noqa: E402  (keys the SMILES as it dedups)

RDLogger.DisableLog("rdApp.*")

METATOX = ROOT / "results/metatox_smirks_preds.json"
FOUR = ROOT / "results/four_method_291.json"
BUDGETS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED, RRF_K = 10000, 0, 60

_FORM: dict[str, str] = {}


def formula(smiles: str) -> str:
    if smiles not in _FORM:
        m = Chem.MolFromSmiles(smiles)
        _FORM[smiles] = rdMolDescriptors.CalcMolFormula(m) if m else smiles
    return _FORM[smiles]




def round_robin(ordered):
    """One candidate per formula group in the order the groups first appear, then the next."""
    groups, order = defaultdict(list), []
    for c in ordered:
        g = formula(c["smiles"])
        if g not in groups:
            order.append(g)
        groups[g].append(c)
    out, depth = [], 0
    while len(out) < len(ordered):
        for g in order:
            if depth < len(groups[g]):
                out.append(groups[g][depth])
        depth += 1
    return out


def capped(ordered, m):
    """Score order, but no group may take more than m of the early slots.

    Round-robin is the extreme: it caps every group at one and also throws away the score order
    between groups. This keeps the score order and caps depth only, so the family separates
    "spreading the budget hurts" from "spreading it all the way hurts".
    """
    kept, spill, seen = [], [], defaultdict(int)
    for c in ordered:
        g = formula(c["smiles"])
        if seen[g] < m:
            seen[g] += 1
            kept.append(c)
        else:
            spill.append(c)
    return kept + spill


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--out", default=str(ROOT / "results/group_decode.json"))
    args = ap.parse_args()

    pools, refs = {}, {}
    for p in sorted(glob.glob(args.pools)):
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    print(f"{len(subs)} substrates", file=sys.stderr, flush=True)

    arms = {}
    for n, s in enumerate(subs, 1):
        if n % 50 == 0:
            print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
        prod = sorted(pools[s], key=lambda c: -c["combined"])
        fused = rrf_order(pools[s])
        seqs = [("product", prod), ("rrf", fused),
                ("rrf+round_robin", round_robin(fused))]
        seqs += [(f"rrf+cap{m}", capped(fused, m)) for m in (1, 2, 3, 5)]
        for name, seq in seqs:
            arms.setdefault(name, {})[s] = [c["key"] for c in seq]

    mtx = json.loads(METATOX.read_text())["predictions"]
    arms["metatox"] = {s: _dedup(mtx.get(s, []), max(BUDGETS)) for s in subs}

    U = np.array([len(real[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(arm, b):
        return np.array([len(set(arms[arm][s][:b]) & real[s]) for s in subs], dtype=float)

    table, contrasts = {}, {}
    for b in BUDGETS:
        h = {a: hits(a, b) for a in arms}
        table[str(b)] = {a: round(float(v.sum() / U.sum()), 4) for a, v in h.items()}
        row = {}
        for a in [x for x in arms if x not in ("product", "rrf", "metatox")]:
            for ref in ("rrf", "metatox"):
                d = h[a] - h[ref]
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                row[f"{a} - {ref}"] = {"gap": round(float(d.sum() / U.sum()), 4),
                                       "ci95": [round(lo, 4), round(hi, 4)],
                                       "excludes_zero": bool(lo > 0 or hi < 0)}
        contrasts[str(b)] = row

    four = json.loads(FOUR.read_text())["per_method"]["MetaTox"]["recall"]
    mism = [f"k={b}: {table[str(b)]['metatox']} vs committed {four[str(b)]}"
            for b in BUDGETS if str(b) in four
            and abs(table[str(b)]["metatox"] - four[str(b)]) > 1e-9]

    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "status": "UPPER BOUND. The round-robin rule was chosen after reading the oracle "
                     "decomposition on these same 291 substrates. It must be registered and "
                     "checked on the validation split before it is quoted as a result, on the "
                     "same terms as H7.",
           "gate": {"reproduces_four_method_291_metatox": not mism, "mismatches": mism},
           "recall_by_budget": table, "contrasts": contrasts,
           "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    order = ["product", "rrf", "rrf+cap1", "rrf+cap2", "rrf+cap3", "rrf+cap5",
             "rrf+round_robin", "metatox"]
    print(f"\n{'k':>4}" + "".join(f"{a:>15}" for a in order))
    for b in BUDGETS:
        print(f"{b:>4}" + "".join(f"{table[str(b)][a]:>15.4f}" for a in order))
    print(f"\ngate reproduces the committed MetaTox column: {not mism}")
    print("\ncontrasts (micro, paired bootstrap):")
    for b in BUDGETS:
        r = contrasts[str(b)]["rrf+round_robin - rrf"]
        m = contrasts[str(b)]["rrf+round_robin - metatox"]
        print(f"  k={b:<3} vs rrf {r['gap']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
              f"{'*' if r['excludes_zero'] else ' '}   "
              f"vs metatox {m['gap']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]"
              f"{'*' if m['excludes_zero'] else ' '}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
