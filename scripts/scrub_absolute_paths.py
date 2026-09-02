#!/usr/bin/env python3
"""No released artifact names the machine it was written on.

Several artifacts record the absolute path of every file they read. That is worse than untidy in
a submission: it names the author in an archive meant to be anonymous, it does not resolve on any
other machine, and it makes a manifest offered as a verification device unusable as one. The
paths carry no information the repository-relative form does not.

This rewrites them in place, everywhere they appear, and reports what it touched. It is
idempotent: a second run finds nothing.

"Everywhere" means every file git tracks, which is the set that ships. It used to mean six
directories named by hand, and the difference was not academic: an audit found the author's home
path in three tracked files under `.review/`, one of them extensionless, while this script reported
zero and exited clean. A walk narrower than the claim above it is worse than no walk, because the
clean exit is what gets believed.

    python scripts/scrub_absolute_paths.py --check     # non-zero if any remain
    python scripts/scrub_absolute_paths.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# This file carries the patterns it searches for, so it matches itself. It is named rather than
# pattern-exempted: an exemption by pattern would silently cover the next file that resembles it.
# Two files carry the patterns as source code and so match themselves. They are named, with the
# pattern each must still contain: if one stops being a scanner the exemption fails loudly instead
# of quietly covering a file that has become an ordinary one.
SCANNERS = {
    "scripts/scrub_absolute_paths.py": "EXTERNAL",
    "scripts/build_anonymous_archive.py": "an absolute home directory",
}

# Any absolute path whose tail is inside a checkout, in any worktree, becomes the relative form.
# It must start at a home directory. Anchoring on "ends in /GRAIL" instead matched /root/GRAIL in
# the two Modal scripts, which is a path inside a container the job creates: it identifies nobody,
# and rewriting it would have broken the scripts to fix a leak that was not there.
ABSOLUTE = re.compile(r"((?:/Users|/home)/[^/\s\"']+/[^\s\"']*?/GRAIL"
                      r"(?:/\.claude/worktrees/[^/\s\"']+)?)/")
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

    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\0")
    touched, total = [], 0
    for name in sorted(n for n in tracked if n):
        path = ROOT / name
        # A suffix list would miss an extensionless file, which is where the audit found one.
        # Anything that decodes as text is read instead.
        if not path.is_file():
            continue
        if name in SCANNERS:
            body = path.read_text(errors="replace")
            if SCANNERS[name] not in body:
                print(f"  REFUSING: {name} is exempt as a scanner but no longer contains "
                      f"{SCANNERS[name]!r}, so the exemption covers an ordinary file")
                return 2
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if True:
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
