#!/usr/bin/env python3
"""No tracked file is large enough to be one of the deposit's artifacts.

`results/` is gitignored in full, so every artifact the paper pins has to be force-added by name.
A force-add with a glob does not distinguish the artifacts from the two candidate pools, which are
45 MB each and are the whole content of the Zenodo deposit. One such glob put 220 MB into the
history, and the only symptom was a push that hung up on the sideband.

This is the check that would have caught it before the commit. A tracked file above the limit is
either a mistake or a deliberate exception, and a deliberate exception belongs on the list below
where the next person can see it.

    python scripts/check_tracked_sizes.py
    python scripts/check_tracked_sizes.py --limit 2000000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that are legitimately large and are tracked on purpose. Each needs a reason.
ALLOWED = {
    # the deployed rule bank: the paper is about it, and a reader cannot check anything without it
    "grail_metabolism/resources/extended_smirks.txt",
    "grail_metabolism/resources/mined_only.txt",
    "grail_metabolism/resources/mined_only_v2.txt",
    "grail_metabolism/data/smirks.txt",
    "grail_metabolism/data/merged_smirks.txt",
    "grail_metabolism/data/xtracted.txt",
    "grail_metabolism/data/reactions.txt",
}

# 20 MB. The repository already tracks the released pool shards and both checkpoints, the
# largest of which is 12 MB, and those are deliberate. The accident this exists to catch
# is a 45 MB candidate pool, so the limit sits between the two rather than at a round
# number that would either pass the accident or fail eleven files that belong here.
DEFAULT_LIMIT = 20_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="bytes; a tracked file above this fails unless it is on the allow-list")
    args = ap.parse_args()

    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            capture_output=True, text=True)
    if listed.returncode != 0:
        print("not a git checkout; nothing to check")
        return 0

    offenders = []
    for rel in listed.stdout.split("\0"):
        if not rel or rel in ALLOWED:
            continue
        path = ROOT / rel
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > args.limit:
            offenders.append((size, rel))

    for size, rel in sorted(offenders, reverse=True):
        print(f"  {size / 1e6:8.1f} MB  {rel}")
    if offenders:
        print(f"FAIL: {len(offenders)} tracked files above {args.limit / 1e6:.1f} MB. "
              "If one belongs in the repository, add it to ALLOWED with a reason; if it does not, "
              "it was probably force-added by a glob over a gitignored directory.")
        return 1
    print(f"check_tracked_sizes: OK, no tracked file above {args.limit / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
