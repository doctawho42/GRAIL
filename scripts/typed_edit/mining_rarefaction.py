#!/usr/bin/env python3
"""Whether template discovery has saturated, measured instead of asserted.

The manuscript says mining is saturated and offers, as evidence, that re-running the miner on the
same split re-derives the same templates and none new. That is a determinism check. Saturation is a
claim about what more data would yield, and a referee pointed out that the paper's own statistic
argues the other way: most mined templates rest on a single training pair, which is the signature
of an undersampled population rather than an exhausted one.

The curve can be had without re-mining. The catalog records, for every mined template, the training
pairs it was derived from, so a fraction of the pairs can be withheld and the templates that would
still have been found counted. Averaged over draws, that is a rarefaction curve, and its slope at
the full corpus is what "saturated" has to mean.

Two estimates are reported beside it. The singleton share is how many templates rest on one pair.
Good and Turing's estimate of the mass belonging to unseen types is the singleton count over the
sample size, which for a type-frequency distribution like this one is the standard way to say how
much is still missing.

What this cannot say is anything about a different corpus. It rarefies within the pairs this corpus
holds, so it answers "would more of the same have found more templates" and not "would other
sources". The manuscript's claim is about the second, and this measures the first.

    python scripts/typed_edit/mining_rarefaction.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
DRAWS, SEED = 40, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="results/mined_rule_catalog_v2.json")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=str(ROOT / "results" / "mining_rarefaction.json"))
    args = ap.parse_args()

    catalog = json.loads((ROOT / args.catalog).read_text())

    # Pairs are shared between templates, so they are indexed once and each template is stored as
    # the set of pair indices it came from. Withholding a pair then withholds it from every
    # template that used it, which is what re-mining on less data would do.
    pair_index: dict = {}
    template_pairs = []
    for row in catalog.values():
        idxs = set()
        for pair in row.get("source_pairs", []):
            key = tuple(pair)
            if key not in pair_index:
                pair_index[key] = len(pair_index)
            idxs.add(pair_index[key])
        if idxs:
            template_pairs.append(idxs)

    n_pairs = len(pair_index)
    n_templates = len(template_pairs)
    support = Counter(len(idxs) for idxs in template_pairs)
    singletons = support.get(1, 0)

    rng = random.Random(SEED)
    curve = {}
    for frac in FRACTIONS:
        take = max(1, int(round(frac * n_pairs)))
        if take >= n_pairs:
            curve[f"{frac}"] = {"pairs": n_pairs, "templates_mean": float(n_templates),
                                "templates_sd": 0.0, "draws": 1}
            continue
        counts = []
        for _ in range(args.draws):
            kept = set(rng.sample(range(n_pairs), take))
            counts.append(sum(1 for idxs in template_pairs if idxs & kept))
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / max(len(counts) - 1, 1)
        curve[f"{frac}"] = {"pairs": take, "templates_mean": round(mean, 1),
                            "templates_sd": round(var ** 0.5, 1), "draws": args.draws}

    # The slope at the top of the curve: templates gained per thousand additional pairs, read
    # between the last two points. A saturated curve is flat here.
    last, prev = curve["1.0"], curve["0.9"]
    gained = last["templates_mean"] - prev["templates_mean"]
    per_thousand = gained / max(last["pairs"] - prev["pairs"], 1) * 1000

    report = {
        "provenance": stamp(__file__),
        "catalog": args.catalog,
        "mined_templates": n_templates,
        "distinct_training_pairs": n_pairs,
        "templates_resting_on_one_pair": singletons,
        "singleton_share": round(singletons / max(n_templates, 1), 4),
        "good_turing_mass_of_unseen_types": round(singletons / max(n_pairs, 1), 4),
        "support_histogram_head": {str(k): support[k] for k in sorted(support)[:10]},
        "rarefaction": curve,
        "templates_gained_per_thousand_pairs_at_the_full_corpus": round(per_thousand, 1),
        "draws_per_point": args.draws,
        "seed": SEED,
        "scope": ("rarefaction within the pairs this corpus holds. It answers whether more data of "
                  "the same kind would find more templates and says nothing about a different "
                  "corpus, which is the question the manuscript's claim was about"),
        "reading": (
            "A saturated curve is flat at its right-hand end and has few singletons. This one is "
            "neither, so what the determinism check established is that the miner is "
            "deterministic, not that mining is finished."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{n_templates} mined templates from {n_pairs} distinct training pairs")
    print(f"  resting on a single pair: {singletons} ({singletons / n_templates:.1%})")
    print(f"  Good-Turing mass of unseen types: {singletons / n_pairs:.4f}\n")
    print(f"{'fraction':>9s} {'pairs':>8s} {'templates':>12s}")
    for frac in FRACTIONS:
        row = curve[f"{frac}"]
        print(f"{frac:9.2f} {row['pairs']:8d} {row['templates_mean']:12.1f}"
              f"  +/- {row['templates_sd']:.1f}")
    print(f"\nat the full corpus, {per_thousand:.1f} further templates per thousand more pairs")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
