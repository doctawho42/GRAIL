import math
from grail_metabolism.stats import (
    ratio_of_sums, ratio_of_sums_ci, factor_bootstrap_ci,
    paired_diff_bootstrap_ci, mcnemar_exact_p,
)

def test_ratio_of_sums_pools_not_averages_ratios():
    # substrate A: 1/1, substrate B: 0/3 -> ratio-of-sums = 1/4 = 0.25 (NOT mean(1.0,0.0)=0.5)
    assert ratio_of_sums([(1, 1), (0, 3)]) == 0.25
    assert ratio_of_sums([(0, 0)]) == 0.0  # empty denominator guard

def test_factor_bootstrap_closure_and_bounds():
    # two substrates; factors are exact on the point estimate
    records = [
        {"U": 2, "Cfull": 2, "Cbud": 2, "H": 1},
        {"U": 3, "Cfull": 2, "Cbud": 1, "H": 1},
    ]
    specs = {
        "coverage_bank": ("Cfull", "U"),
        "selection_retention": ("Cbud", "Cfull"),
        "ranking_conversion": ("H", "Cbud"),
    }
    res = factor_bootstrap_ci(records, specs, n_boot=200, seed=0)
    cb = res["coverage_bank"]["point"]
    sr = res["selection_retention"]["point"]
    rc = res["ranking_conversion"]["point"]
    assert cb == 4 / 5 and sr == 3 / 4 and rc == 2 / 3
    # product of factors == micro recall = sum(H)/sum(U) = 2/5
    assert math.isclose(cb * sr * rc, 2 / 5)
    for name in specs:
        assert 0.0 <= res[name]["lo"] <= res[name]["point"] <= res[name]["hi"] <= 1.0

def test_paired_diff_ci_sign_and_determinism():
    diffs = [-0.2, -0.3, -0.1, -0.25]
    p, lo, hi = paired_diff_bootstrap_ci(diffs, n_boot=500, seed=0)
    assert p < 0 and hi < 0                     # certifies a loss
    assert (p, lo, hi) == paired_diff_bootstrap_ci(diffs, n_boot=500, seed=0)  # seeded, reproducible

def test_mcnemar_exact_two_sided():
    assert mcnemar_exact_p(0, 0) == 1.0
    # 10 vs 0 discordant -> 2 * 0.5^10 = 0.001953...
    assert math.isclose(mcnemar_exact_p(10, 0), 2 * (0.5 ** 10))
    assert mcnemar_exact_p(5, 5) == 1.0
