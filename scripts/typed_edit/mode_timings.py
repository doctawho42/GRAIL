"""What each operating mode costs per substrate, on one population and with the tail shown.

The draft was going to report 1.85 s for the interactive mode. That figure is a mean over a
forty-substrate draw from the comparison set, and the exhaustive mode's 5.28 s is a median over
the 293 validation substrates: two statistics, two populations, printed as a pair. This measures
both modes as medians on the same population and reports the tail, because the mean and the
median differ by six times here and the difference is one peptide.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def summarise(pattern):
    t = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        t += json.loads(Path(f).read_text()).get("generator_seconds", [])
    if not t:
        return None
    s = sorted(x["seconds"] for x in t)
    return {"n": len(s), "median_s": round(st.median(s), 2), "mean_s": round(st.mean(s), 2),
            "p90_s": round(s[int(0.9 * len(s))], 2), "max_s": round(s[-1], 2)}


def main() -> int:
    inter = summarise("results/k30timed/s*.json")
    h15 = json.loads((ROOT / "results/h15_verdict.json").read_text())["time"]
    rep = {"provenance": stamp(__file__), "split": "validation",
           "measures": "seconds per substrate before the filter, the same boundary in both modes",
           "interactive": {"top_k": 30, **(inter or {})},
           "exhaustive": {"top_k": 7581,
                          "median_s": h15["h15_median_s"],
                          "note": "the survivors arm at tautomer budget 200; its median under "
                                  "the load the comparison arm ran at is "
                                  f"{h15['load_correction']['median_under_the_other_arms_load']} s"},
           "why_the_mean_is_not_the_median": (
               "the interactive mode's mean is "
               f"{inter['mean_s']} s against a median of {inter['median_s']} s, and its maximum is "
               f"{inter['max_s']} s: one 291-heavy-atom peptide, which the exhaustive arm has "
               "never finished at any budget, moves the mean by six times")}
    (ROOT / "results/mode_timings.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
