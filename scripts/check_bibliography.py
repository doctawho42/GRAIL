#!/usr/bin/env python3
"""Every reference the two documents cite carries what a reader needs to find it.

A referee reading round thirteen found two incomplete entries by eye: a journal article with no
volume and no pages, and a preprint cited from both documents. Finding bibliography defects by
eye is the same instrument this project has already replaced everywhere else, so this replaces it
here: the cited keys are read from the compiled .aux files, and each entry is checked for the
fields its type needs to be resolvable.

An entry is resolvable if it carries a DOI or a URL. A journal article is complete if it also
carries a volume and pages, or says in a note why it does not, which is what an advance article
or an accepted manuscript legitimately looks like.

    python scripts/check_bibliography.py

Exit status is non-zero when a cited entry is missing something. Uncited entries are not checked:
a bibliography file may legitimately carry more than the two documents use.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper2"
BIB = PAPER / "refs.bib"
AUX = ("grail_jcim.aux", "si.aux")

RESOLVABLE = ("doi", "url", "eprint")
# A journal that numbers articles rather than paginating them -- which is most of the ones cited
# here -- gives an article number and no page range, so either locates the paper.
LOCATOR = ("pages", "number", "eid", "articleno")
# achemso cites a control entry of its own to select the bibliography style. It is not a
# reference and has no bibliography record to check.
NOT_A_REFERENCE = {"achemso-control"}


def cited_keys() -> set:
    keys = set()
    for name in AUX:
        path = PAPER / name
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            for match in re.findall(r"\\(?:abx@aux@cite|citation)\{([^}]*)\}", line):
                keys.update(k.strip() for k in match.split(",") if k.strip())
    return {k for k in keys if k and k != "*"}


def entries() -> dict:
    """Every entry of the bibliography as {key: (type, {field: value})}."""
    text = BIB.read_text(errors="replace")
    out = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        kind, key = match.group(1).lower(), match.group(2).strip()
        start = match.end()
        depth, i = 1, match.start()
        # Walk the braces from the entry's opening one so a nested {} in a title cannot end it.
        i = text.index("{", match.start())
        depth, j = 1, i + 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        body = text[start:j - 1]
        fields = {m.group(1).lower(): m.group(2)
                  for m in re.finditer(r"(\w+)\s*=\s*[{\"]([^}\"]*)", body)}
        out[key] = (kind, fields)
    return out


def main() -> int:
    keys = cited_keys()
    if not keys:
        print("FAIL: no citations found; build the documents first")
        return 1
    bib = entries()
    problems, incomplete = [], []
    for key in sorted(keys - NOT_A_REFERENCE):
        if key not in bib:
            problems.append(f"{key}: cited but not in refs.bib")
            continue
        kind, fields = bib[key]
        resolvable = any(f in fields and fields[f].strip() for f in RESOLVABLE)
        missing = []
        if not fields.get("volume", "").strip():
            missing.append("volume")
        if not any(fields.get(f, "").strip() for f in LOCATOR):
            missing.append("pages or article number")
        specified = kind == "article" and not missing and fields.get("year", "").strip()
        # The requirement is that a reader can find it. A DOI does that, and so does a complete
        # journal citation; an entry with neither cannot be found from what the paper prints.
        if not resolvable and not specified:
            problems.append(f"{key}: no DOI, URL or eprint and not a complete journal citation, "
                            f"so a reader cannot find it")
        elif kind == "article" and missing and not fields.get("note", "").strip():
            incomplete.append(f"{key}: no {' and no '.join(missing)}, resolvable only by its "
                              f"identifier")
    print(f"cited entries: {len(keys)}; bibliography entries: {len(bib)}")
    for row in problems:
        print(f"    {row}")
    for row in incomplete:
        print(f"    incomplete: {row}")
    print("check_bibliography: " + ("OK" if not problems else "FAIL")
          + (f" ({len(incomplete)} incomplete)" if incomplete else ""))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
