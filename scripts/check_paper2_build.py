#!/usr/bin/env python3
"""Both documents compile, resolve every reference and overrun no column.

The cross-references into the Supporting Information now go through xr, so a pointer at the wrong
table cannot resolve and prints ??. That only helps if something counts them, which is what this
does. It also counts overfull boxes, because ACS's column widths are a submission requirement and
a table that overruns one is rejected before it is read.

    python scripts/check_paper2_build.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
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
# A float that does not fit is not an overfull box and was not counted. LaTeX reports it
# separately, and the document still compiles: the table is pushed to a page of its own or
# runs past the text block, and nothing in this checker said so. si_table_macro.tex was over
# by 75.7pt through every build this gate has ever passed.
FLOAT = re.compile(r"Float too large for page by ([0-9.]+)pt")
PAGES = re.compile(r"\((\d+) pages")


STAMP = ROOT / "paper2" / ".build_sources.json"


def _source_digests() -> dict:
    """What every source of both documents contains, as a digest per file."""
    watched = sorted((ROOT / "paper2").glob("*.tex"))
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in watched}


def _sources_changed_since_build() -> list:
    """Any source that has changed in CONTENT since the build the records describe.

    This file reads a log rather than running a build, which makes it a check that passes on a
    record from an hour ago while the sources it describes have moved underneath. That happened:
    several rounds of edits were reported clean against a log written before any of them.

    Modification time is not the signal. A generator that rewrites a file with identical bytes
    moves its mtime and changes nothing, and so does a checkout; running the table generators
    before the test suite was enough to fail this check against a document that had not moved.
    What the build promises is that these bytes produced those logs, so the bytes are what is
    recorded and compared.
    """
    if not STAMP.exists():
        return ["(no record of which sources were built; run scripts/build_paper2.sh)"]
    was = json.loads(STAMP.read_text())["sources"]
    now = _source_digests()
    return sorted(set(k for k in set(was) | set(now) if was.get(k) != now.get(k)))


def write_stamp() -> None:
    """Record what the build just compiled, called by scripts/build_paper2.sh."""
    # The compiled documents' own digests go in beside the sources, because a submission has to be
    # a fixed pair: three distinct builds of the manuscript stood at this path during one round of
    # review and three referees reported against different files. These are the digests to quote.
    built = {}
    for name in DOCS:
        pdf = ROOT / "paper2" / (name + ".pdf")
        if pdf.exists():
            built[f"paper2/{name}.pdf"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    STAMP.write_text(json.dumps({
        "what_this_is": "the digest of every .tex source at the moment the documents were built, "
                        "so a later check can tell a real edit from a rewrite that changed "
                        "nothing, and the digest of each document this build produced",
        "sources": _source_digests(),
        "built": built}, indent=1))


def main() -> int:
    if "--stamp" in sys.argv:
        write_stamp()
        print(f"recorded {len(_source_digests())} sources as built")
        return 0
    ok = True
    for name in DOCS:
        record = ROOT / "paper2" / (name + SUFFIX)
        if not record.exists():
            print(f"FAIL: the build record for {name} is missing; run scripts/build_paper2.sh")
            return 1
        stale = _sources_changed_since_build()
        if stale:
            print(f"FAIL: {len(stale)} source(s) have changed since the build these records "
                  f"describe ({', '.join(stale[:4])}{'...' if len(stale) > 4 else ''}); "
                  f"run scripts/build_paper2.sh")
            return 1
        # ACS production returns a PDF whose figure text is Type 3: the glyphs are drawing
        # programs with no character map, so a label is neither searchable nor extractable. The
        # build gate is where that is caught, because nothing else looks at the compiled file.
        pdf = ROOT / "paper2" / (name + ".pdf")
        if pdf.exists() and shutil.which("pdffonts"):
            fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
            bad = [l.split()[0] for l in fonts.splitlines() if "Type 3" in l]
            if bad:
                print(f"FAIL: {name}.pdf embeds {len(bad)} Type 3 font(s) "
                      f"({', '.join(sorted(set(bad))[:3])}); regenerate the figures with "
                      f"matplotlib's pdf.fonttype set to 42")
                return 1

        text = record.read_text(errors="replace")
        errors = ERROR.findall(text)
        undefined = UNDEF.findall(text)
        overfull = OVERFULL.findall(text)
        floats = FLOAT.findall(text)
        pages = PAGES.findall(text)
        print(f"{name}: errors={len(errors)} undefined={len(undefined)} "
              f"overfull={len(overfull)} floatstoolarge={len(floats)} "
              f"pages={pages[-1] if pages else '?'}")
        for item in errors[:5]:
            print(f"    {item}")
        for item in undefined[:8]:
            print(f"    undefined: {item}")
        if errors or undefined or overfull or floats:
            ok = False
    print("check_paper2_build: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
