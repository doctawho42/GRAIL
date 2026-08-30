#!/usr/bin/env python3
"""Both documents compile, resolve every reference and overrun no column.

The cross-references into the Supporting Information now go through xr, so a pointer at the wrong
table cannot resolve and prints ??. That only helps if something counts them, which is what this
does. It also counts overfull boxes, because ACS's column widths are a submission requirement and
a table that overruns one is rejected before it is read.

    python scripts/check_paper2_build.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("si", "grail_jcim")
SUFFIX = ".log"

ERROR = re.compile(r"^! .*", re.M)
# The page number is not always digits: the Supporting Information numbers its pages S1, S2 and
# so on, so a pattern requiring \d+ here matched nothing in that document and every undefined
# reference in it passed. A referee found one printing ?? on page S10.
UNDEF = re.compile(r"(?:Reference|Citation) `([^']+)' on page \S+ undefined")
OVERFULL = re.compile(r"Overfull \\[hv]box")
PAGES = re.compile(r"\((\d+) pages")


def main() -> int:
    ok = True
    for name in DOCS:
        record = ROOT / "paper2" / (name + SUFFIX)
        if not record.exists():
            print(f"FAIL: the build record for {name} is missing; run scripts/build_paper2.sh")
            return 1
        text = record.read_text(errors="replace")
        errors = ERROR.findall(text)
        undefined = UNDEF.findall(text)
        overfull = OVERFULL.findall(text)
        pages = PAGES.findall(text)
        print(f"{name}: errors={len(errors)} undefined={len(undefined)} "
              f"overfull={len(overfull)} pages={pages[-1] if pages else '?'}")
        for item in errors[:5]:
            print(f"    {item}")
        for item in undefined[:8]:
            print(f"    undefined: {item}")
        if errors or undefined or overfull:
            ok = False
    print("check_paper2_build: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
