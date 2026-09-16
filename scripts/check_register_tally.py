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
generated table, counts the verdict column and the marked rows, reads the sentences that state
those counts, and refuses when they differ. The marked rows are here because the population section
counts them a hundred lines before the table, which is the same exposure with a longer fuse.

Three ways this one guarantee has failed, all three found by reading the prose rather than by any
check, and worth keeping together because they are the same defect at three depths.

1. No check at all. The paragraph above: the numbers gate cannot see a spelled-out word and
   nothing read the table, so seven stood against six across a page for as long as it took a
   person to notice. This file is the answer to that one.
2. A pattern list with no completeness over the forms the claim takes. The manuscript states the
   count three ways, and only two were listed here. "Six are confirmed and three failed" was
   right and checked; "which of the seven confirmations", four lines later, was wrong and matched
   nothing. Both sentences were present, both were typeset, and the gate passed.
3. A form listed and never compared. Adding the third pattern to CLAIMS made the count readable
   and left check() with four comparisons for five forms, so the gate then guarded that the
   sentence still existed and not that its number was right. Presence without value reads exactly
   like a guarantee.

The self-test is built against the third: every form is perturbed twice, once so the value moves
while the pattern still matches, once so the sentence is gone, and each perturbation must produce
its own kind of complaint. A self-test that moves one form of five certifies one comparison of
five, which is defect 2 again, one level in.

    python scripts/check_register_tally.py
    python scripts/check_register_tally.py --self-test   # every form, both branches
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "paper2" / "table_hypotheses.tex"
# Both documents, because a sentence that counts the table can live in either and one of them
# moved: the shortening pass sent the count of marked rows into the Supporting Information, and a
# check that read only the manuscript reported its own subject missing rather than following it.
PROSE = (ROOT / "paper2" / "body.tex", ROOT / "paper2" / "si.tex")

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
    out = {"rows": len(rows), "confirmed": 0, "failed": 0, "other": 0, "daggered": 0}
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        verdict = cells[-2].lower()
        # The population column carries the mark for a prediction adjudicated on a population
        # fixed after the result was visible. The manuscript counts those marks too.
        if "\\dagger" in r:
            out["daggered"] += 1
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
    # A third phrasing of the same count, which this check did not cover and which was therefore
    # wrong in the manuscript while the two patterns above were right: "states which of the seven
    # confirmations a stricter reading leaves undecided", four lines after "Six are confirmed".
    # The gate is not blind to a spelled-out word -- WORDS is right there -- it was blind to a
    # sentence form, which is the same defect one level up: a pattern list with no completeness
    # over the ways the claim can be written.
    ("which_of_the_confirmations",
     re.compile(rf"which\s+of\s+the\s+({NUM})\s+confirmations", re.I)),
    # The population section counts the rows the table marks, a hundred lines before the table.
    ("adjudicated_after",
     re.compile(rf"({NUM})\s+of\s+the\s+({NUM})\s+reported\s+predictions\s+were\s+in\s+the\s+"
                rf"event\s+adjudicated", re.I)),
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
    if "adjudicated_after" in c:
        said_n, said_of = c["adjudicated_after"]
        if said_n != t["daggered"]:
            bad.append(f"the manuscript says {said_n} predictions were adjudicated on a population "
                       f"fixed afterwards and the table marks {t['daggered']}")
        if said_of != t["rows"]:
            bad.append(f"the manuscript says {said_of} predictions are reported and the table has "
                       f"{t['rows']} rows")
    if "of_the_confirmed" in c and c["of_the_confirmed"][1] != t["confirmed"]:
        bad.append(f"the manuscript says '{c['of_the_confirmed'][0]} of the "
                   f"{c['of_the_confirmed'][1]} confirmed predictions' and the table's verdict "
                   f"column carries {t['confirmed']} confirmations")
    # The fifth form, and the reason it is worth a comment. Its pattern was added to CLAIMS while
    # this branch was not, and the loop above only asks whether the sentence is still present. So
    # for one revision the gate guarded the existence of "states which of the N confirmations" and
    # not its value: the manuscript said seven four lines after saying six, both sentences were
    # present, and the check passed. Presence without value is the half of a guarantee that reads
    # as a guarantee.
    if ("which_of_the_confirmations" in c
            and c["which_of_the_confirmations"][0] != t["confirmed"]):
        bad.append(f"the manuscript says 'which of the "
                   f"{c['which_of_the_confirmations'][0]} confirmations' and the table's verdict "
                   f"column carries {t['confirmed']} confirmations")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="check that this check fails when the manuscript's count is wrong")
    args = ap.parse_args()

    if not TABLE.exists() or not all(p.exists() for p in PROSE):
        print("  not checkable in this tree: the manuscript is not here")
        return 0
    table_tex = TABLE.read_text()
    body_tex = "\n\n".join(p.read_text() for p in PROSE)

    if args.self_test:
        # A gate that cannot fail is not a gate, and a self-test that perturbs one form cannot
        # certify a list of forms. This used to move "six are confirmed" alone, one of the five
        # patterns in CLAIMS, and it stayed green through a revision in which a fifth pattern was
        # read and never compared: the sentence "which of the seven confirmations" sat four lines
        # from "Six are confirmed" and nothing refused. So every form is perturbed in turn and
        # each one has to produce a refusal of its own. A form that survives its perturbation is
        # a comparison nobody wrote.
        PERTURBATIONS = {
            "checked": (r"\b(Ten|ten) of the sixteen", "Eleven of the sixteen"),
            "confirmed_failed": (r"\b(Six|six) are confirmed", "Seven are confirmed"),
            "of_the_confirmed": (r"\b(Three|three) of the six confirmed",
                                 "Three of the seven confirmed"),
            "which_of_the_confirmations": (r"which of the six confirmations",
                                           "which of the seven confirmations"),
            "adjudicated_after": (r"\b(Three|three) of the ten reported",
                                  "Four of the ten reported"),
        }
        missing = [n for n, _ in CLAIMS if n not in PERTURBATIONS]
        if missing:
            print(f"REFUSING: {missing} are in CLAIMS with no perturbation here, so the self-test "
                  f"cannot say whether their comparison exists.", file=sys.stderr)
            return 1
        # Two perturbations per form, and the complaint's KIND is what is asserted. Checking only
        # that some complaint appeared conflates the two branches: a perturbation that happens to
        # break the pattern match refuses through "the sentence is not in the manuscript any
        # more", which reads like success while testing nothing about the comparison. Presence and
        # value have to be provoked separately, because their being indistinguishable is what let
        # a form be read and never compared.
        for name, (find, repl) in PERTURBATIONS.items():
            absent = f"the sentence carrying {name!r}"

            moved = re.sub(find, repl, body_tex, count=1)
            if moved == body_tex:
                print(f"REFUSING: the self-test could not perturb the sentence carrying {name!r}, "
                      f"so it proves nothing about that form. Either the sentence was reworded "
                      f"and this perturbation needs updating, or the form is gone.",
                      file=sys.stderr)
                return 1
            disagreements = [c for c in check(table_tex, moved) if absent not in c]
            if not disagreements:
                print(f"REFUSING: {name!r} was moved by one and no disagreement was reported. Its "
                      f"pattern is read into read_claims and never compared in check(); a form "
                      f"listed without a comparison guards the sentence's presence and not its "
                      f"value.", file=sys.stderr)
                return 1

            removed = re.sub(find, "", body_tex, count=1)
            if removed == body_tex:
                print(f"REFUSING: the self-test could not remove the sentence carrying {name!r}.",
                      file=sys.stderr)
                return 1
            if not any(absent in c for c in check(table_tex, removed)):
                print(f"REFUSING: the sentence carrying {name!r} was deleted and nothing said so, "
                      f"so this tally can stop being checked without a word of warning.",
                      file=sys.stderr)
                return 1
        if check(table_tex, body_tex):
            print("self-test: OK on every perturbation but the manuscript itself does not pass; "
                  "run without --self-test for the reason.", file=sys.stderr)
            return 1
        print(f"self-test: OK ({len(PERTURBATIONS)} forms, each perturbed by one and each "
              f"refused; the manuscript itself passes)")
        return 0

    t = tally(table_tex)
    print(f"  Table~\\ref{{tab:hyp}}: {t['rows']} rows, {t['confirmed']} confirmed, "
          f"{t['failed']} failed, {t['other']} neither, {t['daggered']} marked")
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
