"""A paired contrast that refuses the two ways this project has produced a confident nothing.

Both were found by reading output, not by a test failing, and both looked like results.

**A comparator absent is not a comparator beaten.** MetaTox was run on the 291 and nowhere else.
Asked for the gap on validation, the difference of hit counts returned this arm's own recall --
+0.4901, interval excluding zero -- because an empty list scores zero exactly as a very bad
method does. Subtraction cannot tell those apart, and a bootstrap over the difference will put a
tight interval around the confusion. So a contrast now states the population its comparator
actually covers, and refuses when that is none.

**An effect near its ceiling is a question, not a triumph.** The same figure was 0.4901 against a
maximum of 0.4901: the gap equalled the arm's whole score, which is only possible if the other
side contributed nothing at all. A contrast that consumes almost all of the room available to it
is flagged, because the honest reading of that is usually that the room was measured wrong.
"""
from __future__ import annotations

import numpy as np

NEAR_CEILING = 0.95     # a gap this close to the arm's own score is reported for inspection


class EmptyComparator(ValueError):
    """The comparator has no predictions on this population, so no contrast exists."""


def paired_contrast(a, b, weights, *, n_boot=10000, seed=0,
                    comparator_covers=None, name="comparator"):
    """Micro gap a - b with a paired bootstrap over items.

    `weights` is the per-item denominator -- references per substrate -- so the statistic is the
    ratio of sums, resampled by item.

    `comparator_covers` is the number of items on which `b` had anything to say. Pass it. When it
    is zero the call raises rather than returning a number, because the number would be `a`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (a.shape == b.shape == w.shape):
        raise ValueError("arms and weights must have one entry per item")
    if comparator_covers is None:
        raise ValueError("comparator_covers is required: a contrast has to know whether the "
                         "thing it is comparing against exists on this population")
    if comparator_covers == 0:
        raise EmptyComparator(
            f"{name} has no predictions on any of the {len(a)} items; the gap would be this "
            f"arm's own score. Report the arm, not a contrast.")

    total = w.sum()
    gap = float((a - b).sum() / total) if total else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    denom = np.maximum(w[idx].sum(axis=1), 1)
    bt = (a - b)[idx].sum(axis=1) / denom
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))

    own = float(a.sum() / total) if total else 0.0
    near = bool(own > 0 and gap / own >= NEAR_CEILING)
    out = {"gap": round(gap, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "excludes_zero": bool(lo > 0 or hi < 0),
           "comparator_covers": int(comparator_covers), "n_items": int(len(a))}
    if near:
        out["suspect"] = (f"the gap is {gap / own:.1%} of this arm's own score, so the other side "
                          f"contributed almost nothing; check that the comparator is present and "
                          f"scored on the same population before reading this as a result")
    return out
