#!/usr/bin/env python3
"""Refuse the ICLR submission if its main text runs past the strict nine pages.

The limit is a desk-reject condition, and the paper sits at it: the last sentence of the
conclusion is the last line of page nine. In that state any later edit that adds a clause moves
the text onto page ten silently, because nothing about the PDF looks different until someone
counts. So the count is a gate rather than a note in a checklist.

What counts, per iclr2027_conference.tex: nine pages of main text, with references, the AI use
statement, the ethics statement and the reproducibility statement all exempt. The main text
therefore ends where the first of those statements begins.

Two details this got wrong before, both worth keeping in the code:

  * The style prints a line number in the margin of every line. `pdftotext` returns those as text,
    so a page carrying only the statements still yields a token that looks like a line of prose.
    A first version of this check reported one line of overflow that did not exist. A line counts
    only once its margin number is stripped and something is left.

  * The heading is matched case-insensitively against the rendered page, not against the source,
    because what matters is where the statement lands after floats have moved.

    python scripts/check_page_limit.py                   # the ICLR submission
    python scripts/check_page_limit.py --pdf other.pdf --limit 8
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The first of these to appear ends the counted main text. All three are exempt from the limit.
EXEMPT_HEADINGS = ("AI USE STATEMENT", "ETHICS STATEMENT", "REPRODUCIBILITY STATEMENT", "REFERENCES")


def prose_lines(page: str) -> list:
    """The lines of a page that carry words, once the style's margin line numbers are removed."""
    out = []
    for line in page.split("\n"):
        stripped = re.sub(r"^\s*\d+\s*", "", line).strip()
        if stripped and "ICLR 2027" not in stripped and len(stripped) > 3:
            out.append(stripped)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "paper" / "grail_iclr.pdf"))
    ap.add_argument("--limit", type=int, default=9)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"no PDF at {pdf}; build it first", file=sys.stderr)
        return 2
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                          capture_output=True, text=True, check=True).stdout
    pages = text.split("\f")

    end = None
    for i, page in enumerate(pages, 1):
        upper = page.upper()
        if any(h in upper for h in EXEMPT_HEADINGS):
            end = (i, next(h for h in EXEMPT_HEADINGS if h in upper))
            break
    if end is None:
        print("found none of the exempt headings, so the main text has no measurable end",
              file=sys.stderr)
        return 2
    page_no, heading = end

    # Whatever prose sits above that heading on its own page is main text on that page.
    page = pages[page_no - 1]
    cut = next(i for i, l in enumerate(page.split("\n")) if heading in l.upper())
    spill = prose_lines("\n".join(page.split("\n")[:cut]))

    last = page_no - 1 + (1 if spill else 0)
    print(f"{heading.lower()} opens on page {page_no}; main text ends on page {last} "
          f"of a {args.limit}-page limit")
    if last > args.limit:
        print(f"\nREFUSING: {last - args.limit} page(s) over, {len(spill)} line(s) of it here:")
        for l in spill[:12]:
            print("   " + l[:96])
        return 1
    # "How full is the last page" cannot be a line count: a page carrying a float holds fewer
    # lines of prose while being just as full. Compare it against the other main-text pages
    # instead, which are set by the same style at the same measure.
    if last == args.limit:
        counts = [len(prose_lines(pages[i])) for i in range(1, last - 1)]
        here = len(prose_lines(pages[last - 1]))
        typical = sorted(counts)[len(counts) // 2] if counts else here
        print(f"at the limit: {here} lines of prose on page {last} against a median of {typical} "
              f"across pages 2--{last - 1}. Any addition risks pushing the text over, so re-run "
              f"this check after every edit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
