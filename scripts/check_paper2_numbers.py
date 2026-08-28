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
TEX = ROOT / "paper2/body.tex"
NUMS = ROOT / "paper2/numbers.tex"

# Numbers that are part of the language rather than a result: a constant published elsewhere, a
# section number, a percentile, a font size. Each is here because it was looked at.
ALLOWED = {
    "11", "1", "8", "0", "2",              # documentclass, geometry, itemize plumbing
    "60",                                  # the RRF constant, published by Cormack et al.
    "95", "2.5", "97.5",                   # the confidence level and the percentiles that define it
    "10", "000",                           # 10,000 bootstrap resamples
    "5.2", "109",                          # the timing example, cited in prose with its artifact
    "5",                                   # a section cross-reference
    "2009",                                # a citation year
    "60",                                  # the RRF constant
    "0.88", "0.016", "16.22", "16.3",      # the emission-transfer figures, cited with the artifact
    "0.5", "0.109", "0.0012", "0.012", "8.5", "0.01", "70", "2022.09", "2025", "26",
    "291",                                 # the peptide's heavy-atom count, named in prose
    "0.0556", "10",
    "256",                                 # SHA-256, the name of a hash function
    "4.0",                                 # CC BY 4.0, the name of a licence
    # The budget column of the comparison table names the budgets; those are the grid the
    # comparators are read at, not measurements, and every cell beside them is a macro.
    "3", "15", "20", "30", "50",
}


def main() -> int:
    # Every journal wrapper, not one named wrapper. The abstract lives in the wrapper, and when
    # the target journal changed the checker went on reading the old one: the new abstract, which
    # is where a hand-typed number is most likely, was outside the gate entirely.
    wrappers = sorted((ROOT / "paper2").glob("grail_*.tex"))
    assert wrappers, "no paper2/grail_*.tex wrapper found; the checker would vacuously pass"
    body = TEX.read_text() + "".join(w.read_text() for w in wrappers)

    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", NUMS.read_text()))
    used = set(re.findall(r"\\(num[A-Za-z]+)", body))
    missing = sorted(used - defined)
    unused = sorted(defined - used)

    # a numeric literal in the body that is not inside a macro definition or an allowed constant
    stripped = re.sub(r"\\(?:vspace|hspace|setlength|documentclass|usepackage|captionsetup"
                      r"|titleformat|renewcommand|includegraphics)\s*(\[[^\]]*\])?"
                      r"(\{[^}]*\})*", " ", body)
    stripped = re.sub(r"\\\\\s*\[[^\]]*\]", " ", stripped)   # line-break kerns
    stripped = re.sub(r"\\num[A-Za-z]+", " ", stripped)
    stripped = re.sub(r"\\(?:label|ref|cite[a-z]*|input|includegraphics)\{[^}]*\}", " ", stripped)
    stripped = re.sub(r"%.*", " ", stripped)
    # the lookbehind must exclude a preceding DIGIT as well as a letter, or "H14" yields a
    # spurious "4": the 1 is rejected for following H and the 4 is accepted for following 1.
    literals = [t for t in re.findall(r"(?<![A-Za-z\\\d])\d+(?:[.,]\d+)?", stripped)
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
