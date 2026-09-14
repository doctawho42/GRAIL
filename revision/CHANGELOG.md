# Changelog of quoted numbers

Every value the manuscript can print, diffed key by key from `results/paper2_numbers.json`. A number absent from that file cannot appear in the paper, so this is the whole population of quoted values rather than a sample of it.

- changed: 0
- moved beyond its own interval half-width: 0
- moved within it: 0
- moved but cannot be judged (no interval, or not a number): 0
- added: 0
- removed: 0

- generator inputs checked: 99
- generator inputs that moved: 0

No input to `scripts/paper2_numbers.py` moved, so no quoted value could have. That is what makes the table above a result rather than a check that never looked.

A value with no interval is reported as moved and marked as one this check cannot judge. It is not reported as unflagged: 2,198 of the 2,459 keys carry no interval, and calling those unflagged would read as checked and within tolerance.

No quoted number changed.
