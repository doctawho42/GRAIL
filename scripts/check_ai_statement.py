#!/usr/bin/env python3
"""Every task the AI use statement must place carries a verdict, and no verdict moves unnoticed.

The AI use statement is the one section of this paper no number gate can reach, and it is the
section most likely to be edited under pressure to sound better. It has already been wrong twice.
Once it described a review mechanism this repository does not implement, claiming every number is
script-generated rather than typed when the macros are hand-typed. Once a shortening pass removed
two verdicts outright, and the removals were invisible: the paragraph they left behind reads as a
run of denials, so the default reading of a missing item is "not used".

That second failure is the one this gate exists for, because it runs in the dangerous direction.
An admitted use that quietly disappears is under-disclosure, and the archive carries the revision
history, so a reviewer diffing revisions would find an admission being withdrawn.

So two things are checked:

  * every task on the declared list has exactly one verdict in the statement, and
  * the verdict each task carries is the one recorded in results/ai_statement_verdicts.json.

The second is what makes a change deliberate. Flipping a verdict is legitimate and has happened
for a good reason: the claim that AI proposed and refined hypotheses was an over-claim, since the
four evaluation choices this paper is about were found by measuring one predictor in one domain.
What is not legitimate is flipping one silently, so a flip fails here until the record is updated
in the same commit, where a reader and a reviewer can see it.

The declared list is derived from the venue's four-slot frame and from what the statement has
previously placed. The venue defers the enumeration to its policy page, which is not in this
repository, so the list is this paper's own standard rather than a transcription of the venue's;
`--list` prints it so it can be checked against the live policy before submission.

    python scripts/check_ai_statement.py            # non-zero if a task is unplaced or moved
    python scripts/check_ai_statement.py --list     # the declared list and each verdict
    python scripts/check_ai_statement.py --record   # accept the current verdicts as the record
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "grail_iclr.tex"
RECORD = ROOT / "results" / "ai_statement_verdicts.json"

# task key -> a pattern matching the clause that places it. Each pattern must anchor on wording
# specific enough that a clause about a different task cannot satisfy it: the failure this gate
# guards against is a task being read as covered by a neighbouring sentence.
TASKS = {
    "implementing methods": r"(not )?used to implement methods",
    "cleaning or reformatting data": r"(not )?used\b[^.]{0,40}\bto clean and reformat data",
    "running the experiments": r"(not )?used to run the experiments",
    "interpreting results": r"the readings of measurements",
    # The clause sits at the tail of the sentence that also places data cleaning, so the
    # window has to reach the length of that sentence; [^.] keeps it inside one sentence.
    "qualitative or thematic analysis": r"(not )?used[^.]{0,220}?to support qualitative analysis",
    "feedback on methodology": r"(not )?used to give feedback on\s*methodology",
    "conceptual frameworks": r"(not )?used to\s*develop conceptual frameworks",
    "research ideas or hypotheses": r"(not )?used to generate research ideas or hypotheses",
    "synthetic data": r"(not )?used to generate synthetic data",
    "theoretical models": r"(not )?used to develop\s*theoretical models",
    "mathematical claims": r"(not )?used[^.]{0,60}to formulate mathematical claims",
    "proofs": r"(not )?used[^.]{0,90}to write proofs",
    "drafting": r"drafted with assistance",
    "editing and other recommended tasks": r"recommended tasks of code editing",
    "translation": r"[Tt]ranslation is not applicable",
}


def statement() -> str:
    text = PAPER.read_text()
    a = text.index(r"\subsubsection*{AI use statement}")
    b = text.index(r"\subsubsection*{Ethics statement}")
    return re.sub(r"\s+", " ", text[a:b])


def verdicts() -> tuple:
    """Each task's verdict, and the tasks the statement does not place at all."""
    flat, found, missing = statement(), {}, []
    for task, pattern in TASKS.items():
        hits = list(re.finditer(pattern, flat))
        if not hits:
            missing.append(task)
            continue
        # "not applicable" is written as such; otherwise the negation carried by the clause decides.
        window = flat[max(0, hits[0].start() - 60):hits[0].end() + 60]
        if "not applicable" in window:
            found[task] = "not applicable"
        else:
            # Some patterns anchor on a phrase that carries no negation slot ("drafted with
            # assistance"); those are affirmative by construction.
            neg = hits[0].groups()[0] if hits[0].re.groups else None
            found[task] = "not used" if (neg or "").strip() == "not" else "used"
        if len(hits) > 1:
            found[task] += f" ({len(hits)} clauses)"
    return found, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the declared list and each verdict")
    ap.add_argument("--record", action="store_true",
                    help="accept the current verdicts as the recorded ones")
    args = ap.parse_args()

    found, missing = verdicts()

    if args.list:
        print(f"{len(TASKS)} tasks declared; the venue defers its own list to a policy page, so "
              f"check these against it before submitting.\n")
        for task in TASKS:
            print(f"  {found.get(task, 'NO VERDICT'):22s} {task}")
        return 0

    if args.record:
        RECORD.write_text(json.dumps(
            {"note": "the verdict the AI use statement gives each task on the declared list; a "
                     "change here must be made in the same commit as the change to the statement, "
                     "so that a verdict cannot move without the move being visible",
             "verdicts": found}, indent=1))
        print(f"recorded {len(found)} verdicts to {RECORD.relative_to(ROOT)}")
        return 0

    bad = []
    if missing:
        bad += [f"no verdict for: {t}" for t in missing]

    if RECORD.exists():
        was = json.loads(RECORD.read_text())["verdicts"]
        for task, verdict in sorted(found.items()):
            if task in was and was[task] != verdict:
                bad.append(f"{task}: recorded {was[task]!r}, statement now says {verdict!r}")
        for task in sorted(set(was) - set(found)):
            bad.append(f"{task}: recorded {was[task]!r} and is no longer placed at all")
    else:
        bad.append(f"{RECORD.relative_to(ROOT)} does not exist, so no verdict change can be "
                   f"detected; run --record")

    if bad:
        print("REFUSING: the AI use statement does not place every task, or a verdict moved "
              "without the record moving with it:")
        for line in bad:
            print("   " + line)
        print("\nIf a change is intended, make it and re-run with --record in the same commit.")
        return 1
    print(f"all {len(TASKS)} declared tasks carry a verdict, and each is the one on record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
