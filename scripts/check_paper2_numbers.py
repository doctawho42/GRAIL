"""Hold the manuscript to its artifacts: every macro defined, every number a macro.

SELF_CLAIMS section 11a asks that no number in the main text be an orphan. Checking prose after
the fact caught five errors in an hour of writing here, all of them from generalising over the
middle of a sample. This checks the other direction, which cannot be got wrong by carelessness: a
figure reaches the page only through a macro generated from an artifact, and a literal that is not
on the small allow-list below fails the build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper2/grail_service.tex"
NUMS = ROOT / "paper2/numbers.tex"

# Numbers that are part of the language rather than a result: a constant published elsewhere, a
# section number, a percentile, a font size. Each is here because it was looked at.
ALLOWED = {
    "11", "1", "8", "0", "2",              # documentclass, geometry, itemize plumbing
    "60",                                  # the RRF constant, published by Cormack et al.
    "95",                                  # the confidence level
    "10", "000",                           # 10,000 bootstrap resamples
    "5.2", "109",                          # the timing example, cited in prose with its artifact
    "5",                                   # a section cross-reference
    "2009",                                # a citation year
    # The budget column of the comparison table names the budgets; those are the grid the
    # comparators are read at, not measurements, and every cell beside them is a macro.
    "3", "15", "20", "30", "50",
}


def main() -> int:
    tex = TEX.read_text()
    body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliographystyle")]

    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", NUMS.read_text()))
    used = set(re.findall(r"\\(num[A-Za-z]+)", body))
    missing = sorted(used - defined)
    unused = sorted(defined - used)

    # a numeric literal in the body that is not inside a macro definition or an allowed constant
    stripped = re.sub(r"\\num[A-Za-z]+", " ", body)
    stripped = re.sub(r"\\(?:label|ref|cite[a-z]*|input|includegraphics)\{[^}]*\}", " ", stripped)
    stripped = re.sub(r"%.*", " ", stripped)
    literals = [t for t in re.findall(r"(?<![A-Za-z\\])\d+(?:[.,]\d+)?", stripped)
                if t.replace("{,}", "") not in ALLOWED and t not in ALLOWED]

    ok = True
    print(f"macros defined {len(defined)}, used {len(used)}")
    if missing:
        ok = False
        print(f"FAIL: {len(missing)} macros used and not defined: {missing[:10]}")
    if literals:
        ok = False
        print(f"FAIL: {len(literals)} numeric literals in the body that are not macros: "
              f"{sorted(set(literals))[:15]}")
    if unused:
        print(f"note: {len(unused)} generated macros the manuscript does not cite")
    print("check_paper2_numbers: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
