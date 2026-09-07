"""The prose counts the register's verdicts, so the count has to be taken from the verdicts.

The manuscript says how many of the registered predictions were confirmed and how many failed. It
said seven and three for a while after the table said six, three and one: P5's verdict had been
weakened from a confirmation to "measured, not adjudicated in advance", because its threshold was
fixed before the result was visible and its population was not, and the two sentences that count
confirmations were not brought back into line with it. Both sentences and the table are typeset a
page apart, so the manuscript contradicted itself in view of the reader, and every check passed:
the numbers gate cannot see a spelled-out word, and nothing else read the table.

The tally is a hand-typed word rather than a macro, because the adjudication lives in the table's
generator rather than in the number chain, and wiring the generator into the chain would create an
ordering the repository does not otherwise enforce. This is the compensating control: it reads the
generated table, counts the verdict column, reads the two sentences, and refuses when they differ.

    python scripts/check_register_tally.py
    python scripts/check_register_tally.py --self-test   # the check fails on a wrong count
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "paper2" / "table_hypotheses.tex"
BODY = ROOT / "paper2" / "body.tex"

WORDS = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16}
NUM = r"(?:" + "|".join(WORDS) + r"|\d+)"


def _word(s: str) -> int:
    s = s.strip().lower()
    return int(s) if s.isdigit() else WORDS[s]


def tally(table_tex: str) -> dict:
    """What the table's own verdict column says, one row at a time."""
    body = table_tex.split("\\midrule")[1].split("\\bottomrule")[0]
    rows = [r.strip() for r in body.split("\\\\") if r.strip()]
    out = {"rows": len(rows), "confirmed": 0, "failed": 0, "other": 0}
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        verdict = cells[-2].lower()
        # The first word of the cell is the verdict; the rest qualifies it. "confirmed on
        # validation" is a confirmation and "measured, not adjudicated in advance" is not, and a
        # cell that begins with neither word is counted separately rather than guessed at.
        if verdict.startswith("confirmed"):
            out["confirmed"] += 1
        elif verdict.startswith("failed"):
            out["failed"] += 1
        else:
            out["other"] += 1
    return out


# The two sentences that count. Each is matched on its own shape rather than on a fixed string, so
# rewording that keeps the arithmetic passes and rewording that changes it does not.
CLAIMS = (
    ("checked", re.compile(rf"({NUM})\s+of\s+the\s+({NUM})\s+registered\s+are\s+checked", re.I)),
    ("confirmed_failed",
     re.compile(rf"({NUM})\s+are\s+confirmed\s+and\s+({NUM})\s+failed", re.I)),
    ("of_the_confirmed",
     re.compile(rf"({NUM})\s+of\s+the\s+({NUM})\s+confirmed\s+predictions", re.I)),
)


def read_claims(body_tex: str) -> dict:
    found = {}
    for name, pat in CLAIMS:
        m = pat.search(body_tex)
        if m:
            found[name] = tuple(_word(g) for g in m.groups())
    return found


def check(table_tex: str, body_tex: str) -> list:
    t = tally(table_tex)
    c = read_claims(body_tex)
    bad = []
    for name, _ in CLAIMS:
        if name not in c:
            bad.append(f"the sentence carrying '{name}' is not in the manuscript any more, so the "
                       f"tally is no longer checked; update the pattern or restore the sentence")
    if "checked" in c and c["checked"][0] != t["rows"]:
        bad.append(f"the manuscript says {c['checked'][0]} predictions are checked here and "
                   f"Table~\\ref{{tab:hyp}} has {t['rows']} rows")
    if "confirmed_failed" in c:
        said_c, said_f = c["confirmed_failed"]
        if said_c != t["confirmed"]:
            bad.append(f"the manuscript says {said_c} are confirmed and the table's verdict column "
                       f"says {t['confirmed']}")
        if said_f != t["failed"]:
            bad.append(f"the manuscript says {said_f} failed and the table's verdict column says "
                       f"{t['failed']}")
    if "of_the_confirmed" in c and c["of_the_confirmed"][1] != t["confirmed"]:
        bad.append(f"the manuscript says '{c['of_the_confirmed'][0]} of the "
                   f"{c['of_the_confirmed'][1]} confirmed predictions' and the table's verdict "
                   f"column carries {t['confirmed']} confirmations")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="check that this check fails when the manuscript's count is wrong")
    args = ap.parse_args()

    if not TABLE.exists() or not BODY.exists():
        print("  not checkable in this tree: the manuscript is not here")
        return 0
    table_tex, body_tex = TABLE.read_text(), BODY.read_text()

    if args.self_test:
        # A gate that cannot fail is not a gate. Move the count by one and require a refusal.
        broken = re.sub(r"\b(Six|six) are confirmed", "Seven are confirmed", body_tex, count=1)
        if broken == body_tex:
            print("REFUSING: the self-test could not perturb the manuscript, so it proves "
                  "nothing about whether this check can fail.", file=sys.stderr)
            return 1
        if not check(table_tex, broken):
            print("REFUSING: the count was moved by one and this check still passed, so it is "
                  "not reading the table.", file=sys.stderr)
            return 1
        if check(table_tex, body_tex):
            print("self-test: OK on the perturbation but the manuscript itself does not pass; "
                  "run without --self-test for the reason.", file=sys.stderr)
            return 1
        print("self-test: OK (fails on a count moved by one, passes on the manuscript)")
        return 0

    t = tally(table_tex)
    print(f"  Table~\\ref{{tab:hyp}}: {t['rows']} rows, {t['confirmed']} confirmed, "
          f"{t['failed']} failed, {t['other']} neither")
    bad = check(table_tex, body_tex)
    if bad:
        print("\nREFUSING: the manuscript's count of the register disagrees with the register:",
              file=sys.stderr)
        for x in bad:
            print(f"    {x}", file=sys.stderr)
        print("\n  A verdict cell can be weakened without anyone noticing that two sentences a "
              "page away count it. Whichever is right, they have to say the same thing.",
              file=sys.stderr)
        return 1
    print("  the manuscript's count matches the verdict column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
