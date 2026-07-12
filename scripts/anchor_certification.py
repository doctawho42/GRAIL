#!/usr/bin/env python
"""Certify the honest anchor: SyGMa > GRAIL on the COMMON substrate set. Paired bootstrap on
d_i = recall_GRAIL_i - recall_SyGMa_i (continuous per-substrate recall gap) + exact McNemar on
any-hit@15 (binary). Tautomer-InChIKey throughout. Emits results/anchor_certification.json.

This certifies EVALUATION variance -- the spread of the recall gap under resampling of substrates
from the FIXED test set -- NOT training-seed variance (a single deployed checkpoint is scored). The
common subset's rule-bank ceiling (SigmaCfull/SigmaU) is reported to show the common set is
representative of the full 1170-substrate test.

Per the 2026-07-12 controller correction, GRAIL's per-substrate deployed top-15 and per-substrate
recall come from Task 2's results/recall_factorization.json (H_i/U_i -- the ACTUAL deployed pipeline
run on all 1170 substrates), NOT the 291-substrate artifacts/.../test_predictions.csv export (a
non-representative ensemble.py export cap). SyGMa's per-substrate top-15 comes from
run_benchmark.sygma_topk (the DRY generation sygma_baseline scores) on the same substrates, joined on
the substrate SMILES string.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.stats import mcnemar_exact_p, paired_diff_bootstrap_ci
from run_benchmark import load_test_map, sygma_topk
from factorize_recall import tautomer_hits

K = 15
FACTORIZATION_JSON = ROOT / "results" / "recall_factorization.json"
OUT = ROOT / "results" / "anchor_certification.json"


def per_substrate_recall(preds, trues) -> float:
    """Tautomer-InChIKey recall for ONE substrate: (# tautomer-distinct trues recovered by any
    pred) / (# tautomer-distinct trues). The denominator is keyed by tautomer-InChIKey so it
    equals Task 2's U_i, keeping GRAIL and SyGMa on the same denominator."""
    tk = set()
    for t in trues:
        try:
            tk.add(_tautomer_inchikey(t))
        except Exception:
            tk.add(t)
    return tautomer_hits(preds, trues) / len(tk) if tk else 0.0


def any_hit(preds, trues) -> bool:
    """True iff preds recover at least one tautomer-distinct true metabolite (any-hit@k)."""
    return tautomer_hits(preds, trues) > 0


def main() -> int:
    fact = json.loads(FACTORIZATION_JSON.read_text())
    # GRAIL per-substrate H_i / U_i / Cfull_i (deployed pipeline, all 1170; Task 2), keyed by SMILES.
    grail_by_sub = {r["sub"]: r for r in fact["per_substrate"]}

    test_map = load_test_map(sample=None, seed=0)  # {sub_smiles: set(true_smiles)}
    sygma = sygma_topk(test_map, K)  # {sub_smiles: [top-15 tautomer-distinct SMILES]}

    diffs = []
    b = c = 0  # McNemar discordants: b = GRAIL hit & SyGMa miss, c = SyGMa hit & GRAIL miss
    common = 0
    ceil_num = ceil_den = 0
    sum_rg = sum_rs = 0.0
    for sub, prods in test_map.items():
        g = grail_by_sub.get(sub)
        if g is None or sub not in sygma:
            continue
        common += 1
        trues = list(prods)
        u_i = g["U"]

        # GRAIL: reuse Task 2's deployed H_i / U_i directly (H_i = tautomer_hits(deployed_top15, trues)).
        rg = g["H"] / u_i if u_i else 0.0
        g_hit = g["H"] > 0

        # SyGMa: score its top-15 with the SAME tautomer keying and the SAME U_i denominator.
        s_hits = tautomer_hits(sygma[sub], trues)
        rs = s_hits / u_i if u_i else 0.0
        s_hit = s_hits > 0

        diffs.append(rg - rs)
        sum_rg += rg
        sum_rs += rs
        if g_hit and not s_hit:
            b += 1
        elif s_hit and not g_hit:
            c += 1

        # Representativeness: full-bank rule ceiling over the common set (ratio-of-sums of Task 2 Cfull, U).
        ceil_num += g["Cfull"]
        ceil_den += u_i

    dp, dlo, dhi = paired_diff_bootstrap_ci(diffs, n_boot=10000, seed=0)
    mean_rg = sum_rg / common if common else 0.0
    mean_rs = sum_rs / common if common else 0.0
    # Identity sanity: the paired-diff point IS mean(recall_GRAIL) - mean(recall_SyGMa).
    assert abs(dp - (mean_rg - mean_rs)) < 1e-9, (dp, mean_rg - mean_rs)

    report = {
        "match": "inchikey_tautomer",
        "k": K,
        "common_n": common,
        "variance_certified": (
            "evaluation variance (substrate resampling of the fixed 1170-substrate test set); "
            "NOT training-seed variance -- a single deployed checkpoint is scored"
        ),
        "mean_recall_GRAIL": mean_rg,
        "mean_recall_SyGMa": mean_rs,
        "delta_mean_recall": {
            "point": dp,
            "lo": dlo,
            "hi": dhi,
            "definition": (
                "mean_i (recall_GRAIL_i - recall_SyGMa_i); point < 0 with the 95% CI wholly "
                "below 0 certifies GRAIL loses to SyGMa on the common set"
            ),
        },
        "mcnemar": {
            "b_grail_only": b,
            "c_sygma_only": c,
            "p": mcnemar_exact_p(b, c),
            "definition": (
                "exact two-sided McNemar on any-hit@15 discordance; b = GRAIL hit & SyGMa miss, "
                "c = SyGMa hit & GRAIL miss"
            ),
        },
        "common_subset_ceiling": ceil_num / ceil_den if ceil_den else 0.0,
        "full_ceiling_reference": fact["factors"]["coverage_bank"]["point"],
        "provenance": {
            "resampling_unit": "substrate",
            "n_boot": 10000,
            "seed": 0,
            "match": "inchikey_tautomer",
            "join_key": "substrate SMILES string",
            "grail_source": (
                "results/recall_factorization.json per_substrate H_i/U_i "
                "(deployed generator.pt x filter.pt on all 1170 substrates; Task 2)"
            ),
            "sygma_source": (
                "run_benchmark.sygma_topk (phase1+phase2 scenario): first-15 tautomer-distinct, "
                "parent-dropped predictions in SyGMa score order"
            ),
            "common_subset_ceiling_def": "ratio-of-sums SigmaCfull/SigmaU over the common set (Task 2 Cfull, U)",
            "full_ceiling_reference_def": "Task 2 micro coverage_bank (SigmaCfull/SigmaU over all 1170)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    summary = {
        k: report[k]
        for k in (
            "common_n",
            "mean_recall_GRAIL",
            "mean_recall_SyGMa",
            "delta_mean_recall",
            "mcnemar",
            "common_subset_ceiling",
            "full_ceiling_reference",
        )
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
