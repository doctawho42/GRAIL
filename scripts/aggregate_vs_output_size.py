#!/usr/bin/env python3
"""Does the aggregate reorder methods that emit the same number of candidates?

The paper presents three undeclared parameters as three axes. Two reviewers, independently, said
two of them collapse into one: the aggregate and the budget both act through output size, and every
certified reordering the paper shows runs through SyGMa, the one method emitting seventy-five to
eighty-two candidates against everyone else's eight to eleven. The paper acknowledges the asymmetry
but never measures whether the aggregate can reorder anything without it, and acknowledging a
confound does not license attributing an effect to one of its arms. A real ICLR review killed a
comparable paper on exactly this: architecture and pretraining strategy confounded, so that no
difference could be attributed to either.

So this asks the question directly. Partition the method pairs by how similar their output sizes
are, and count certified reversals in each partition. A reversal is the paper's own standard and
not a swap of point estimates: A must lead B under one aggregate with a paired interval excluding
zero, and B must lead A under another with a paired interval excluding zero. Both directions, or it
is not a reversal.

The answer decides what the paper may claim. If comparable emitters reverse, the aggregate is an
axis in its own right. If they do not, then the aggregate and the budget are two windows onto one
uncontrolled variable, and only the matching criterion is independent -- which is a sharper claim
than three axes, and the one the data supports.

Reads only frozen artifacts; no model is run and nothing is re-scored.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "set_metrics_by_criterion.json"
OUT = ROOT / "results" / "aggregate_vs_output_size.json"
# Methods count as comparable emitters when neither emits more than this multiple of the other.
COMPARABLE_RATIO = 1.5


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


def point(v):
    return v.get("point") if isinstance(v, dict) else v


def analyse(block: dict, metrics: list[str]) -> dict:
    methods = [k for k in block if not k.startswith("_")]
    out = {m: point(block[m].get("mean_output")) for m in methods}
    diffs = block.get("_paired_diffs", {})

    def cell(a, b, metric):
        """Signed difference a-b with its interval, taking either stored orientation."""
        k1, k2 = f"{a}-{b}|{metric}", f"{b}-{a}|{metric}"
        if k1 in diffs:
            v = diffs[k1]
            d = v.get("delta", v.get("point"))
            return d, v.get("ci95"), bool(v.get("excludes_zero"))
        if k2 in diffs:
            v = diffs[k2]
            d = v.get("delta", v.get("point"))
            ci = v.get("ci95")
            return (None if d is None else -d,
                    None if not ci else [-ci[1], -ci[0]], bool(v.get("excludes_zero")))
        return None, None, False

    rows = []
    for a, b in itertools.combinations(sorted(methods), 2):
        oa, ob = out.get(a), out.get(b)
        ratio = None
        if oa and ob:
            ratio = round(max(oa, ob) / min(oa, ob), 2)
        found = []
        for m1, m2 in itertools.combinations(metrics, 2):
            d1, c1, e1 = cell(a, b, m1)
            d2, c2, e2 = cell(a, b, m2)
            if d1 is None or d2 is None:
                continue
            # a reversal: the sign of the a-b difference flips between the two aggregates
            if (d1 > 0) != (d2 > 0):
                found.append({"metrics": [m1, m2], "delta_1": d1, "ci_1": c1, "certified_1": e1,
                              "delta_2": d2, "ci_2": c2, "certified_2": e2,
                              "certified_both_directions": bool(e1 and e2)})
        rows.append({"pair": [a, b], "mean_output": [oa, ob], "output_ratio": ratio,
                     "comparable": bool(ratio is not None and ratio <= COMPARABLE_RATIO),
                     "sign_flips": found,
                     "certified_reversals": sum(1 for f in found if f["certified_both_directions"])})
    return {"methods": methods, "mean_output": out, "pairs": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="inchikey_tautomer")
    ap.add_argument("--metrics", default="recall,f1,jaccard,precision")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    metrics = args.metrics.split(",")

    src = json.loads(SRC.read_text())
    rep = {"config": {**_code_version(), "source": str(SRC.relative_to(ROOT)),
                      "criterion": args.criterion, "metrics": metrics,
                      "comparable_ratio": COMPARABLE_RATIO},
           "populations": {}}
    verdicts = []
    for pop, pd_ in src["populations"].items():
        block = pd_["by_mode"][args.criterion]
        a = analyse(block, metrics)
        comp = [r for r in a["pairs"] if r["comparable"]]
        wide = [r for r in a["pairs"] if not r["comparable"]]
        a["summary"] = {
            "n_comparable_pairs": len(comp),
            "certified_reversals_among_comparable": sum(r["certified_reversals"] for r in comp),
            "n_disparate_pairs": len(wide),
            "certified_reversals_among_disparate": sum(r["certified_reversals"] for r in wide),
        }
        rep["populations"][pop] = a
        verdicts.append((pop, a["summary"]))
        print(f"\n=== {pop} ({args.criterion}) ===")
        print("  mean output: " + ", ".join(f"{m} {a['mean_output'][m]}" for m in a["methods"]))
        for r in a["pairs"]:
            tag = "comparable" if r["comparable"] else "disparate "
            print(f"  {tag} x{str(r['output_ratio']):>5}  {r['pair'][0]:14} vs {r['pair'][1]:14}"
                  f"  sign flips {len(r['sign_flips'])}, certified {r['certified_reversals']}")
        s = a["summary"]
        print(f"  -> certified reversals: {s['certified_reversals_among_comparable']} among "
              f"{s['n_comparable_pairs']} comparable pairs, "
              f"{s['certified_reversals_among_disparate']} among {s['n_disparate_pairs']} disparate")

    # Third leg: is the matching criterion an axis in its own right, or does it track output size
    # too? Sensitivity is a within-method quantity, so the test is whether it orders with emission.
    ms = json.loads((ROOT / "results" / "match_sensitivity_fulln_paired.json").read_text())
    em = json.loads((ROOT / "results" / "budget_curves.json").read_text())["mean_emitted"]
    crit = {m: {"emitted": em.get(m), "gain": v["gain"], "ci95": v["ci95"]}
            for m, v in ms["sensitivity"].items() if m in em}
    order = sorted(crit, key=lambda m: crit[m]["emitted"])
    gains = [crit[m]["gain"] for m in order]
    monotone = all(x <= y for x, y in zip(gains, gains[1:])) or all(x >= y for x, y in zip(gains, gains[1:]))
    def overlap(a, b):
        return not (crit[a]["ci95"][1] < crit[b]["ci95"][0] or crit[b]["ci95"][1] < crit[a]["ci95"][0])
    rep["criterion_axis"] = {
        "strict": ms["strict"], "tolerant": ms["tolerant"], "n": ms["n_substrates"], "k": ms["k"],
        "by_method": crit, "ordered_by_output": order,
        "monotone_in_output_size": bool(monotone),
        "note": ("criterion sensitivity is a within-method gain; if it were a proxy for output size "
                 "it would order with emission"),
    }
    print(f"\ncriterion sensitivity against output size ({ms['strict']} -> {ms['tolerant']}):")
    for m in order:
        c = crit[m]
        print(f"  {m:16} emits {str(c['emitted']):>6}  gain {c['gain']} {c['ci95']}")
    print(f"  monotone in output size: {monotone}")

    tot_c = sum(v["certified_reversals_among_comparable"] for _, v in verdicts)
    tot_d = sum(v["certified_reversals_among_disparate"] for _, v in verdicts)
    rep["verdict"] = {
        "certified_reversals_among_comparable_emitters": tot_c,
        "certified_reversals_among_disparate_emitters": tot_d,
        "reading": ("the aggregate reorders methods of equal output size, so it is an axis in its "
                    "own right" if tot_c else
                    "no certified reversal survives between methods of comparable output size, so "
                    "the aggregate axis is a window onto output size rather than a separate "
                    "parameter"),
    }
    print(f"\nVERDICT: {tot_c} certified reversal(s) among comparable emitters, "
          f"{tot_d} among disparate ones")
    print(f"  {rep['verdict']['reading']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
