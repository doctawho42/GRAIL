#!/usr/bin/env python3
"""No released artifact names the machine it was written on.

Several artifacts record the absolute path of every file they read. That is worse than untidy in
a submission: it names the author in an archive meant to be anonymous, it does not resolve on any
other machine, and it makes a manifest offered as a verification device unusable as one. The
paths carry no information the repository-relative form does not.

This rewrites them in place, everywhere they appear, and reports what it touched. It is
idempotent: a second run finds nothing.

    python scripts/scrub_absolute_paths.py --check     # non-zero if any remain
    python scripts/scrub_absolute_paths.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ("results", "paper2", "artifacts", "strata", "configs", "docs")
SUFFIXES = (".json", ".txt", ".md", ".csv")

# Any absolute path whose tail is inside a checkout, in any worktree, becomes the relative form.
ABSOLUTE = re.compile(r"(/[^\s\"']*?/GRAIL(?:/\.claude/worktrees/[^/\s\"']+)?)/")
# Paths to things OUTSIDE the checkout -- a sibling baselines tree, an installed package -- have
# no relative form. They are replaced by a name that says what the thing is, because that is the
# whole of what the artifact needed to record.
EXTERNAL = (
    (re.compile(r"/[^\s\"']*/site-packages/([A-Za-z0-9_]+)"), r"<installed package \1>"),
    (re.compile(r"/[^\s\"']*/GRAIL_baselines/([A-Za-z0-9_.-]+)"), r"<external baseline \1>"),
    (re.compile(r"/Users/[^\s\"']+"), "<local path>"),
)


def scrub(text: str) -> tuple[str, int]:
    out, n = ABSOLUTE.subn("", text)
    for pattern, replacement in EXTERNAL:
        out, k = pattern.subn(replacement, out)
        n += k
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report and exit non-zero without writing")
    args = ap.parse_args()

    touched, total = [], 0
    for folder in SCAN:
        base = ROOT / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            new, n = scrub(text)
            if not n:
                continue
            total += n
            touched.append((path.relative_to(ROOT), n))
            if not args.check:
                path.write_text(new)

    for rel, n in touched:
        print(f"  {n:>4}  {rel}")
    if args.check:
        print(f"{len(touched)} files still carry an absolute path ({total} occurrences)")
        return 1 if touched else 0
    print(f"rewrote {total} absolute paths in {len(touched)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
