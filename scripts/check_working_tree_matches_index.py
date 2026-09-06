"""An input's digest must be the digest a reader gets, not the one this machine happens to hold.

Artifacts record the files they read by path and sha256, and the provenance sweep re-hashes those
files to say whether an input has moved. That is only a guarantee for a reader if the bytes on this
machine are the bytes a clone materialises. They were not.

`core.autocrlf = input` normalises CRLF to LF when a file is committed and does nothing on
checkout. A file written with CRLF and then committed therefore lives in the repository with LF,
while the working copy keeps its carriage returns forever, since git compares normalised and sees
nothing to update. Eight tracked data files were in that state here. Every digest recorded against
one of them was a digest of bytes no reader would ever see, so three artifacts reported their
inputs as MOVED in a fresh clone while verifying here -- the provenance guarantee the manuscript
quotes could not be reproduced by the only people it is for.

This compares, for every tracked file, the bytes on disk against the bytes in HEAD, and refuses on
a difference that is not a live edit. It is cheap and it is the only thing that makes an input
digest mean the same on both sides.

    python scripts/check_working_tree_matches_index.py
    python scripts/check_working_tree_matches_index.py --fix   # re-materialise the divergent files
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [p for p in out.stdout.split("\n") if p.strip()]


def _staged_or_modified() -> set:
    """Paths git itself reports as changed, which are live edits and not the defect."""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True)
    return {line[3:].strip() for line in out.stdout.splitlines() if line.strip()}


def _committed(rel: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="delete and re-checkout the divergent files so they match HEAD")
    args = ap.parse_args()

    live = _staged_or_modified()
    divergent = []
    for rel in _tracked():
        if rel in live:
            continue
        p = ROOT / rel
        if not p.exists() or p.is_dir():
            continue
        blob = _committed(rel)
        if blob is None:
            continue
        try:
            work = p.read_bytes()
        except Exception:
            continue
        if hashlib.sha256(work).hexdigest() != hashlib.sha256(blob).hexdigest():
            divergent.append((rel, len(work), len(blob)))

    print(f"  {len(_tracked())} tracked files, {len(live)} with live edits (skipped)")
    if not divergent:
        print("  every other tracked file is byte-identical to HEAD, so a digest recorded here "
              "is the digest a clone computes")
        return 0

    if args.fix:
        for rel, _, _ in divergent:
            (ROOT / rel).unlink()
            subprocess.run(["git", "checkout", "--", rel], cwd=ROOT, check=True)
            print(f"    re-materialised {rel}")
        print("\n  Re-run any producer that recorded one of these as an input; its digest is now "
              "the one a reader computes and the old one is not.")
        return 0

    print(f"\nREFUSING: {len(divergent)} tracked file(s) differ on disk from what HEAD holds, "
          f"with no live edit to explain it:", file=sys.stderr)
    for rel, w, b in divergent:
        print(f"    {rel}  ({w} bytes here, {b} in HEAD)", file=sys.stderr)
    print("\n  A digest recorded against one of these describes bytes no reader will ever have. "
          "git's line-ending normalisation is the usual cause and git will not report the file as "
          "modified, because it compares normalised. Run with --fix, then re-run the producers "
          "that name these as inputs.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
