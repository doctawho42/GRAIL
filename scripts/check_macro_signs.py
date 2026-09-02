#!/usr/bin/env python3
"""No sentence writes a sign by hand that the macro beside it contradicts.

The manuscript writes intervals and differences as `$+\\numSomething$`, with the sign in the prose
and the magnitude in the macro. That works while the macro stays positive. When a re-measurement
turns it negative the prose does not follow, and the page prints `+ -0.0045` for a difference that
runs the other way -- which is how a trail was printed as a lead in two places at once, in the
manuscript and in the Supporting Information, and survived every other check here.

Every other gate in this project asks whether a number came from an artifact. This one asks
whether the sentence around it still says what the number says.

    python scripts/check_macro_signs.py

Exit status is non-zero when a hand-written sign disagrees with its macro's value.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper2"
DOCS = ("body.tex", "si.tex", "grail_jcim.tex")
MACRO = re.compile(r"\\newcommand\{\\(num[A-Za-z]+)\}\{([^}]*)\}")
USE = re.compile(r"([+-])\\(num[A-Za-z]+)")


def values() -> dict:
    out = {}
    for line in (PAPER / "numbers.tex").read_text().splitlines():
        m = MACRO.match(line)
        if not m:
            continue
        raw = m.group(2).replace("\\%", "").replace("%", "").replace("$", "")
        raw = raw.replace("{,}", "").strip()
        try:
            out[m.group(1)] = float(raw)
        except ValueError:
            continue          # a word, a date or a verdict; no sign to contradict
    return out


def main() -> int:
    vals = values()
    problems = []
    for name in DOCS:
        path = PAPER / name
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for sign, macro in USE.findall(line):
                value = vals.get(macro)
                if value is None or value == 0:
                    continue
                if (value < 0) != (sign == "-"):
                    problems.append(f"{name}:{n}  {sign}\\{macro} = {value}")
    print(f"macros with a numeric value: {len(vals)}")
    for row in problems:
        print(f"    {row}")
    print("check_macro_signs: " + ("OK" if not problems else "FAIL"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
