"""Pure statistical estimators for the theory-spine analyses. No I/O, no RDKit,
no torch — so they unit-test under `make test` without the dataset. Cluster =
substrate; all bootstraps resample substrates (clusters), never individual pairs."""
from __future__ import annotations

import random
from math import comb
from typing import Dict, List, Sequence, Tuple


def ratio_of_sums(pairs: Sequence[Tuple[float, float]]) -> float:
    """Sum(numerator)/Sum(denominator) over per-cluster (num, den) pairs; 0.0 if
    total denominator is 0. This is the correct estimator when pairs within a
    substrate are dependent (do NOT average per-substrate ratios)."""
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return num / den if den else 0.0


def ratio_of_sums_ci(
    pairs: Sequence[Tuple[float, float]], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Percentile CI for a ratio-of-sums estimator via cluster (substrate)
    resampling with replacement. Returns (point, lo, hi)."""
    rng = random.Random(seed)
    n = len(pairs)
    point = ratio_of_sums(pairs)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(ratio_of_sums(sample))
    boots.sort()
    return point, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot)]


def factor_bootstrap_ci(
    records: Sequence[Dict[str, float]],
    factor_specs: Dict[str, Tuple[str, str]],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Joint cluster bootstrap for several ratio-of-sums factors. `records` is one
    dict per substrate; `factor_specs[name] = (numerator_field, denominator_field)`.
    Each bootstrap resamples substrates ONCE and recomputes every factor on that
    same resample, so the factor CIs are mutually consistent. Returns
    {name: {"point": .., "lo": .., "hi": ..}}."""
    rng = random.Random(seed)
    n = len(records)

    def factors(sample: Sequence[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, (nf, df) in factor_specs.items():
            num = sum(r[nf] for r in sample)
            den = sum(r[df] for r in sample)
            out[name] = num / den if den else 0.0
        return out

    point = factors(records)
    acc: Dict[str, List[float]] = {name: [] for name in factor_specs}
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        f = factors(sample)
        for name in factor_specs:
            acc[name].append(f[name])
    res: Dict[str, Dict[str, float]] = {}
    for name in factor_specs:
        b = sorted(acc[name])
        res[name] = {
            "point": point[name],
            "lo": b[int((alpha / 2) * n_boot)],
            "hi": b[int((1 - alpha / 2) * n_boot)],
        }
    return res


def paired_diff_bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Percentile CI for the mean of paired per-substrate differences d_i. Returns
    (point, lo, hi). A wholly-below-0 CI certifies a loss."""
    rng = random.Random(seed)
    n = len(diffs)
    point = sum(diffs) / n if n else 0.0
    boots: List[float] = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return point, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot)]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (GRAIL hit, other
    miss) and c (GRAIL miss, other hit), under Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
