#!/usr/bin/env python3
"""Every quantity the manuscript derives from the decomposition, recomputed and compared.

SELF_CLAIMS row 11 says every numeric passage traces to its artifact, and nothing enforced it. The
cost of that showed up when the ceiling was corrected: the macro was updated, and seventeen values
derived from it were not -- a conversion ratio, a count of references lost to truncation, a paired
difference that became arithmetically impossible, an external ceiling measured in the other
convention, and a figure caption. Every one was found by a reader rather than by a check.

This is that check. It does not scan for numbers that happen to appear in both a paper and an
artifact -- with tens of thousands of values in the record, any three-decimal figure matches
something, and a test that cannot fail is worse than none. It recomputes each derived quantity from
the canonical artifact and compares it to what the manuscript prints, by name.

Exit status is non-zero when anything disagrees, so this belongs in front of a submission the way a
test suite belongs in front of a commit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "grail_iclr.tex"
TEX = [PAPER] + sorted((ROOT / "paper" / "app").glob("*.tex"))


def macro(name: str, text: str):
    """The macro's value with LaTeX digit grouping removed, e.g. 1{,}170 -> 1170."""
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.+?)\}\s*(?:%|\n)", text)
    if m is None:
        return None
    return m.group(1).replace("{,}", "").replace(",", "").strip()


def close(a, b, tol=5e-4):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    art = json.loads((ROOT / "results/recall_factorization.json").read_text())
    f = art["factors"]
    rows = art["per_substrate"]
    S = {k: sum(r[k] for r in rows) for k in ("U", "Cfull", "Cbud", "H")}
    text = PAPER.read_text()
    whole = "".join(p.read_text() for p in TEX)

    checks = []

    def check(name, printed, computed, note=""):
        ok = close(printed, computed, args.tol)
        checks.append((ok, name, printed, computed, note))

    # 1. the macros against the artifact they summarise
    check("\\ceiling", macro("ceiling", text), f["coverage_bank"]["point"])
    check("\\ceilingmacro", macro("ceilingmacro", text), art["macro_coverage_bank"])
    check("\\realised", macro("realised", text), art["micro_recall"])
    check("\\selret", macro("selret", text), f["selection_retention"]["point"])
    check("\\grailmacro", macro("grailmacro", text), art["macro_recall"])
    check("\\ntest", macro("ntest", text), art["n_substrates"])

    # 2. the identity, which is the whole point of the instrument
    prod = (f["coverage_bank"]["point"] * f["selection_retention"]["point"]
            * f["ranking_conversion"]["point"])
    check("identity closes", prod, art["micro_recall"], "coverage x selection x ranking")

    # 3. the factors against the per-substrate record they are pooled from
    check("coverage from rows", f["coverage_bank"]["point"], S["Cfull"] / S["U"])
    check("selection from rows", f["selection_retention"]["point"], S["Cbud"] / S["Cfull"])
    check("ranking from rows", f["ranking_conversion"]["point"], S["H"] / S["Cbud"])

    # 4. quantities the prose derives and prints in words
    conv = re.search(r"converts only \$([\d.]+)\\%\$ of its own ceiling", whole)
    check("conversion percentage", conv and float(conv.group(1)) / 100, S["H"] / S["Cfull"],
          "H / Cfull")
    lost = re.search(r"and all \$(\d+)\$\s*\n?references lost between the budgeted pool", whole)
    check("references lost to truncation", lost and lost.group(1), S["Cbud"] - S["H"],
          "Cbud - H")
    cap = sum(1 for r in rows if len(r.get("deployed_top15") or []) >= 15)
    atcap = re.search(r"emits at the cap on \$(\d+)\$ of", whole)
    check("substrates at the cap", atcap and atcap.group(1), cap)

    # 5. the nesting the identity needs, which a clamp used to supply silently
    nests = sum(1 for r in rows if r.get("nests", True))
    check("nesting holds on every row", nests, len(rows), "no clamp applied")

    # 6. every OTHER artifact that records a full-bank coverage must be in the same convention as
    # the canonical one, or be named here as a deliberate reference to the other. A checker whose
    # scope is narrower than its claim reports a pass it has not earned, which is how five stale
    # values survived the last sweep while this script printed "all agree".
    EXPANDED_ON_PURPOSE = {
        "recall_factorization_expanded_convention.json",   # kept as the measure of the cost
        "recall_factorization_rerun.json",                 # the same run under its old name
    }
    canonical = f["coverage_bank"]["point"]
    for path in sorted((ROOT / "results").glob("*.json")):
        if path.name in EXPANDED_ON_PURPOSE or path.name == "recall_factorization.json":
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for key in ("coverage", "internal_ceiling", "ceiling"):
            v = d.get(key)
            v = v.get("point") if isinstance(v, dict) else v
            if isinstance(v, (int, float)) and 0.5 < v < 1.0:
                checks.append((close(v, canonical, 2e-3), f"{path.name}:{key}", round(v, 4),
                               round(canonical, 4), "full-bank coverage, must match the convention"))

    # 7. the external ceiling and its split, which live in their own artifacts
    ext = ROOT / "results/ceiling_external_validity.json"
    if ext.exists():
        e = json.loads(ext.read_text())
        for label, printed_pat, value in (
                ("external ceiling", r"the external ceiling is ([\d.]+)",
                 e["external_ceiling_uncapped"]["point"]),):
            m = re.search(printed_pat, whole)
            check(label, m and m.group(1), value)
    spl = ROOT / "results/external_ceiling_split.json"
    if spl.exists():
        sp = json.loads(spl.read_text())
        for label, key, pat in (
                ("external, seen in training", "seen_in_training",
                 r"\$([\d.]+)\$[^.]{0,80}?on the twenty-four seen"),
                ("external, unseen", "unseen", r"against \$([\d.]+)\$ on the thirteen unseen")):
            m = re.search(pat, whole)
            check(label, m and m.group(1), sp[key]["coverage"])

    # 8. intervals printed beside the factors
    for label, key in (("ceiling interval", "coverage_bank"), ("ranking interval", "ranking_conversion")):
        lo, hi = f[key]["lo"], f[key]["hi"]
        pat = re.compile(r"\$\[" + f"{lo:.3f}" + r",\s*" + f"{hi:.3f}" + r"\]\$")
        checks.append((bool(pat.search(whole.replace(" ", ""))
                            or re.search(r"\[" + f"{lo:.3f}" + r"," + f"{hi:.3f}" + r"\]",
                                         whole.replace(" ", ""))),
                       label, f"[{lo:.3f},{hi:.3f}]", "present in the manuscript", ""))

    width = max(len(c[1]) for c in checks)
    bad = 0
    for ok, name, printed, computed, note in checks:
        if not ok:
            bad += 1
        mark = "ok  " if ok else "FAIL"
        extra = f"   ({note})" if note else ""
        print(f"  {mark} {name:{width}}  manuscript {str(printed):>10}   artifact {str(computed)[:12]:>12}{extra}")
    print(f"\n  {len(checks) - bad} of {len(checks)} agree")
    if bad:
        print("  a manuscript number does not follow from the artifact it cites")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
