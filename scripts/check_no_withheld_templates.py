#!/usr/bin/env python3
"""A document may not claim non-redistribution while a tracked file carries the templates.

The released bank drops 611 templates that BioTransformer's published set also contains. That
removal is a courtesy rather than an obligation: `artifacts/tier2/biotransformer/LICENSE.md` grants
copying and redistribution on four conditions -- credit, a link to the licence, an indication of
changes, and the notices retained -- and reserves permission for COMMERCIAL use or redistribution.
An earlier reading applied the commercial clause to all redistribution, and the removal was built
on it.

What the removal did not do is make "those templates are not redistributed" true. A trained
generator's checkpoint persists the bank it was built against, `arch` and `rules` together, so a
loader can refuse a mismatched pair; one such checkpoint is tracked and carries all 7,581 templates
in plain text. So do the curated collections the bank was mined from. Removing a file from the
index does not remove what other files also hold, and every check here asked about the bank's own
path, so none of them could see the difference.

The invariant is therefore not "no tracked file carries a withheld template", which the corrected
reading makes an odd thing to demand. It is that no document claims otherwise. This counts the
carriers, writes the census the manuscript's number comes from, and refuses when a document still
asserts non-redistribution while carriers exist.

    python scripts/check_no_withheld_templates.py
    python scripts/check_no_withheld_templates.py --list   # name every carrier and what it holds
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import stamp  # noqa: E402

MEASURED = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
RELEASED = ROOT / "grail_metabolism" / "resources" / "extended_smirks_released.txt"
TEXT_SUFFIXES = {".txt", ".json", ".tex", ".md", ".py", ".csv", ".yaml", ".yml", ".smi"}

# Where a claim of non-redistribution would live, and the shapes such a claim takes. The patterns
# are deliberately narrow: they match an assertion ABOUT THE TEMPLATES, not the word "distribute".
CLAIMANTS = ("NOTICE.md", "README.md", "pyproject.toml",
             "paper2/body.tex", "paper2/si.tex", "paper2/grail_jcim.tex")
CLAIMS = (
    re.compile(r"those templates are not redistributed", re.I),
    re.compile(r"templates?[^.\n]{0,60}\bnever distributed\b", re.I),
    re.compile(r"\bnot distributed\b[^.\n]{0,40}templates?", re.I),
    re.compile(r"bank[^.\n]{0,30}\bis\s+\*{0,2}not distributed", re.I),
)


def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [p for p in out.stdout.split("\n") if p.strip()]


def _withheld() -> set:
    if not MEASURED.exists():
        # The measured bank is exactly what the release does not carry, so a clone reaches this
        # every time. Refusing there would report the release as broken for working as documented;
        # passing silently would look as though the check had run. It says which, and stops.
        print("  not checkable in this tree: the measured bank "
              f"({MEASURED.relative_to(ROOT)}) is not redistributed, so what the released bank "
              "withholds cannot be computed here. This is a clone of the release, not a defect.")
        raise SystemExit(0)
    if not RELEASED.exists():
        raise SystemExit(
            "REFUSING: the measured bank is here and the released one is not, so the release was "
            "never built. Run scripts/build_released_bank.py.")
    measured = {l.strip() for l in MEASURED.read_text().splitlines() if l.strip()}
    released = {l.strip() for l in RELEASED.read_text().splitlines() if l.strip()}
    return measured - released


def _templates_in(path: Path) -> set:
    """Every rule string a file carries, whether it is text or a pickled checkpoint."""
    if path.suffix == ".pt":
        try:
            import torch
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return set()
        rules = blob.get("rules") if isinstance(blob, dict) else None
        return {str(r).strip() for r in rules} if rules else set()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return set()
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return set()
    return {l.strip() for l in text.splitlines() if l.strip()}


def _claims() -> list:
    found = []
    for rel in CLAIMANTS:
        p = ROOT / rel
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            for pat in CLAIMS:
                if pat.search(line):
                    found.append((rel, n, line.strip()[:110]))
                    break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="name every carrier and what it holds")
    # This script had no way to run without writing, and the paper-gate suite calls it with no
    # arguments -- so running the tests rewrote a tracked artifact as a side effect. That cost an
    # hour of cross-checking a census nobody had deliberately regenerated, and the worse form is
    # obvious once seen: the suite could re-baseline an artifact immediately before another gate
    # verifies against it. The verdict here is computed before the write and does not read the
    # file back, so redirecting the output changes nothing the check concludes.
    ap.add_argument("--out", default=str(ROOT / "results" / "withheld_template_carriers.json"),
                    help="where to write the census; point it elsewhere to leave the tracked "
                         "artifact untouched, which is what a test run should do")
    args = ap.parse_args()

    withheld = _withheld()
    if not withheld:
        print("REFUSING: the two banks are identical, so nothing is withheld and this check "
              "asserts nothing. Either the released bank was not built or it was built wrong.",
              file=sys.stderr)
        return 1

    tracked = _tracked()
    carriers = []
    for rel in tracked:
        path = ROOT / rel
        if not path.exists() or path.is_dir():
            continue
        hit = _templates_in(path) & withheld
        if hit:
            carriers.append((rel, len(hit), sorted(hit)[:3]))

    claims = _claims()
    Path(args.out).write_text(json.dumps({
        "provenance": stamp(__file__),
        "question": ("which tracked files carry a template the released bank removes, since "
                     "removing the bank's own file does not remove what other files hold"),
        "withheld": len(withheld),
        "tracked_files_scanned": len(tracked),
        "carriers": [{"path": p, "withheld_templates_it_holds": n} for p, n, _ in carriers],
        "n_carriers": len(carriers),
        "documents_claiming_non_redistribution": [
            {"file": f, "line": n, "text": t} for f, n, t in claims],
        "reading": ("a carrier is not a licence problem under BioTransformer's terms, which grant "
                    "redistribution with attribution; it is a fact the documents have to state "
                    "correctly, and the removal is a courtesy rather than a requirement"),
    }, indent=1))

    # Name the path written. "Who wrote this file?" came up three times in one evening here and
    # once cost an hour of two people cross-checking a census neither had asked for; a producer
    # that says where it put its output answers that question before it is asked.
    print(f"  wrote {args.out}")
    print(f"  {len(withheld)} withheld templates, {len(tracked)} tracked files scanned")
    print(f"  {len(carriers)} tracked file(s) carry at least one of them:")
    for rel, n, sample in carriers:
        print(f"    {rel}  ({n} of the {len(withheld)})")
        if args.list:
            for s in sample:
                print(f"        {s[:88]}")

    if carriers and claims:
        print(f"\nREFUSING: {len(claims)} document line(s) assert that these templates are not "
              f"redistributed, while {len(carriers)} tracked file(s) carry them:", file=sys.stderr)
        for f, n, t in claims:
            print(f"    {f}:{n}  {t}", file=sys.stderr)
        print("\n  Either stop shipping the carriers, or state what the repository actually "
              "carries. BioTransformer's licence grants redistribution with attribution, so the "
              "second is available and is the cheaper of the two.", file=sys.stderr)
        return 1

    print("  no document asserts non-redistribution, so the census stands as a statement of fact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
