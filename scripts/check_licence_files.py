#!/usr/bin/env python3
"""LICENSE and NOTICE.md describe the bank this repository actually ships.

Both files were written when the bank redistributed BioTransformer's templates, and both went on
saying so after the release stopped carrying them: LICENSE gave BioTransformer's LGPL as half the
reason for the GPL, and NOTICE.md's per-rightsholder table listed 611 of its templates as present.
A licence file that describes a different distribution from the one in the tree is worse than a
missing one, because a reader has no reason to doubt it.

So the counts in those two files are held against the artifact the release is built from, and the
package's own build configuration is held against the same thing: a wheel must carry the released
bank and must not carry the measured one.

    python scripts/check_licence_files.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "released_bank.json"
LICENCE = ROOT / "LICENSE"
NOTICE = ROOT / "NOTICE.md"
PYPROJECT = ROOT / "pyproject.toml"


def _thousands(n: int) -> list:
    """Both ways a count is written in prose here, so a comma is not a miss."""
    return [str(n), f"{n:,}"]


def main() -> int:
    if not RECORD.exists():
        print(f"REFUSING: {RECORD.relative_to(ROOT)} is missing, so nothing states what ships; "
              f"run scripts/build_released_bank.py", file=sys.stderr)
        return 1
    rec = json.loads(RECORD.read_text())
    measured = rec["measured_bank"]["templates"]
    released = rec["released_bank"]["templates"]
    removed = rec["removed"]
    released_name = Path(rec["released_bank"]["path"]).name
    measured_name = Path(rec["measured_bank"]["path"]).name

    bad = []

    # 1. The removed count is what both files must say, and neither may still present those
    #    templates as shipped.
    notice = NOTICE.read_text()
    if not any(t in notice for t in _thousands(removed)):
        bad.append(f"NOTICE.md never mentions the {removed} templates the release removes")
    if not any(t in notice for t in _thousands(released)):
        bad.append(f"NOTICE.md never mentions the released bank's {released} templates")
    if released_name not in notice:
        bad.append(f"NOTICE.md does not name {released_name}, the file that ships")

    # The counts being present somewhere is not the claim. NOTICE.md's per-rightsholder table is
    # what a reader reads, and its released column for the rightsholder whose templates were
    # removed has to be zero. Looking only for the numerals passed a table that said all 611 still
    # ship, which is the whole defect this file exists to prevent.
    row = re.search(r"^\|\s*BioTransformer\s*\|([^|]*)\|([^|]*)\|", notice, re.M)
    if not row:
        bad.append("NOTICE.md has no BioTransformer row in a per-rightsholder table, so nothing "
                   "there states how many of its templates ship")
    else:
        shipped = re.sub(r"[^0-9]", "", row.group(2))
        if shipped != "0":
            bad.append(f"NOTICE.md's rightsholder table says {shipped or 'an unreadable number'} "
                       f"of BioTransformer's templates are in the released bank; the release "
                       f"removes all {removed}")

    licence = LICENCE.read_text()
    if not any(t in licence for t in _thousands(removed)):
        bad.append(f"LICENSE never mentions the {removed} templates the release removes")

    # 2. A wheel must carry the released bank and must not carry the measured one. The second is
    #    the one that matters: a glob over the resources directory would have shipped it.
    toml = PYPROJECT.read_text()
    inc = re.search(r"^include = \[(.*?)^\]", toml, re.S | re.M)
    exc = re.search(r"^exclude = \[(.*?)^\]", toml, re.S | re.M)
    inc_s = inc.group(1) if inc else ""
    exc_s = exc.group(1) if exc else ""
    if released_name not in inc_s:
        bad.append(f"pyproject.toml does not include {released_name}, so a wheel would not carry "
                   f"the bank the release is supposed to ship")
    if re.search(r"resources/\*\.txt", inc_s):
        bad.append("pyproject.toml globs the resources directory, so a wheel built in a checkout "
                   f"holding {measured_name} would ship it")
    if measured_name not in exc_s:
        bad.append(f"pyproject.toml does not exclude {measured_name}, which this project has no "
                   f"permission to redistribute")

    # 3. And the licence the package declares has to be the one the file grants.
    if not re.search(r'^license = "GPL-3\.0', toml, re.M):
        bad.append("pyproject.toml declares no GPL-3 license field, so tooling reads the package "
                   "as unlicensed while LICENSE grants the GPL")

    if bad:
        print(f"REFUSING: {len(bad)} statement(s) about what this repository distributes do not "
              f"match what it distributes:")
        for line in bad:
            print("   " + line)
        return 1

    print(f"the licence files and the package configuration describe the released bank: "
          f"{released} of {measured} templates, {removed} removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
