#!/usr/bin/env python3
"""The word around a number makes a claim too, and until now nothing read it.

Every figure in these manuscripts is a macro generated from an artifact, and a checker holds each
one against the artifact that produced it. Nearly three thousand of those checks agree. What none
of them reads is the sentence the macro sits in, and a round of blind review found sixteen defects
of exactly that shape: the figure was right and the quantifier around it was wrong.

    "the blend beats the deployed rule by up to +0.0391"

+0.0391 is the difference at a budget of three, correctly generated from the artifact. It is not
the maximum: the same artifact holds +0.0526 at a budget of five. The sentence claims an extremum
over a series and names a member of it, and the number gate cannot tell the difference because the
number is right.

    "the conditional effect ... separates from zero, as the pooled one also does"

Both intervals are [-0.0706, 0.0000] and [-0.0215, 0.0000]. The artifact records excludes_zero as
false for both, and the bound is a true zero rather than a rounded one. Again each printed figure
is correct and the verdict placed on them is not.

Two families are checked here, both mechanical:

  SUPERLATIVE  a sentence saying "up to", "as much as", "at most", "at least" beside a macro must
               name the extremum of that macro's series, not a member of it. A range the sentence
               declares ("through ten") restricts the series before the comparison.

  SEPARATION   a sentence claiming an interval separates from zero, or does not, must agree with
               the artifact's own verdict, or with its bounds where no verdict was recorded.

A third family the review found -- counts over a table, "of the eleven separating cells" -- is not
checked here and is listed by `--todo`, because deciding what a count ranges over needs the table's
own structure rather than the sentence's.

    python scripts/check_quantifiers.py
    python scripts/check_quantifiers.py --todo    # the claims this cannot yet reach
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# claims whose set this cannot resolve; reported by --todo, never as violations
UNRESOLVED: list = []
NUMBERS = ROOT / "results" / "paper2_numbers.json"
MACROS = ROOT / "paper2" / "numbers.tex"
DOCS = ("paper2/body.tex", "paper2/si.tex", "paper2/grail_jcim.tex")

# how far after the quantifier the macro may sit and still be the thing it quantifies
WINDOW = 120

MAX_WORDS = (r"up\s+to|as\s+much\s+as|at\s+most|no\s+more\s+than|as\s+high\s+as|"
             r"peaking\s+at|at\s+its\s+(?:largest|highest)")
MIN_WORDS = (r"at\s+least|no\s+less\s+than|as\s+low\s+as|as\s+little\s+as|"
             r"at\s+its\s+(?:smallest|lowest)")

# a declared restriction on the series the extremum ranges over
THROUGH = {"one": 1, "three": 3, "five": 5, "eight": 8, "ten": 10, "fifteen": 15,
           "twenty": 20, "thirty": 30, "fifty": 50}

# Every space here is \s+ because these phrases wrap across lines in the source, and "does not
# exclude zero" split by a newline was read as the positive "exclude zero" on the first run.
def _ws(p: str) -> str:
    return p.replace(" ", r"\s+")


SEPARATES = _ws(r"separates? from zero|excludes? zero|separating from zero")
# The negatives are matched FIRST, because each contains a positive as a substring: scanning
# "does not exclude zero" left to right, an alternation that tries the positive first matches
# "exclude zero" and reports the sentence as claiming the opposite of what it says.
NOT_SEPARATES = _ws(r"does not separate|do not separate|does not exclude zero|"
                    r"do not exclude zero|includes zero|include zero|"
                    r"indistinguishable from zero|neither separates|does not separate from zero")


def macro_keys() -> dict:
    """macro name -> the numbers key it was generated from, read from the comment beside it."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{[^}]*\}\s*%\s*([\w.]+)", MACROS.read_text()):
        out[m.group(1)] = m.group(2)
    return out


def series_of(key: str, numbers: dict) -> list:
    """Every (budget, value) sharing this key's prefix, so an extremum can be taken over them."""
    m = re.match(r"^(.*?)(\d+)$", key)
    if not m:
        return []
    prefix = m.group(1)
    out = []
    for k, v in numbers.items():
        km = re.match(r"^" + re.escape(prefix) + r"(\d+)$", k)
        if km and isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((int(km.group(1)), v))
    return sorted(out)


def flat(path: Path) -> str:
    return re.sub(r"[ \t]+", " ", path.read_text())


def check_superlatives(numbers: dict, keys: dict) -> list:
    bad = []
    for doc in DOCS:
        text = flat(ROOT / doc)
        for m in re.finditer(rf"\b({MAX_WORDS}|{MIN_WORDS})\b(.{{0,{WINDOW}}}?)\\num(\w+)",
                             text, re.S | re.I):
            word, between, macro = m.group(1).lower(), m.group(2), m.group(3)
            # a macro reached across a sentence boundary is not the one being quantified
            if "." in between.replace("0.", "").replace("+.", "").replace("-.", ""):
                continue
            key = keys.get("num" + macro)
            if not key:
                continue
            members = series_of(key, numbers)
            if len(members) < 2:
                continue
            here = numbers.get(key)
            if not isinstance(here, (int, float)) or isinstance(here, bool):
                continue

            # "through ten" and the like restrict the series before the extremum is taken
            tail = text[m.end():m.end() + 160]
            head = text[max(0, m.start() - 160):m.start()]
            limit = None
            for w, n in THROUGH.items():
                if re.search(rf"through {w}\b", head + " " + tail, re.I):
                    limit = n
            ranged = [(b, v) for b, v in members if limit is None or b <= limit]
            if len(ranged) < 2:
                continue

            # A superlative can range over the OTHER axis. "ahead of every comparator by at least
            # X" fixes the budget and ranges over comparators, and comparing X against the budget
            # series answers a question the sentence did not ask. Where the sentence names such a
            # set, this cannot resolve the axis and says so under --todo rather than reporting a
            # violation: a gate that is wrong two times in five stops being read.
            if re.search(r"\b(?:every|each|any|all|either) (?:comparator|method|system|arm|source|"
                         r"population|criterion|tool)\b", head + " " + tail, re.I):
                UNRESOLVED.append(f"{doc}: \"{word}\" at \\num{macro} ranges over a named set of "
                                  f"methods, not over the budget series; axis not resolvable here")
                continue

            want_max = bool(re.match(MAX_WORDS, word))
            extreme = max(ranged, key=lambda x: x[1]) if want_max else min(ranged, key=lambda x: x[1])
            if abs(extreme[1] - here) > 1e-12:
                line = text.count("\n", 0, m.start()) + 1
                bad.append(
                    f"{doc}:{line}  \"{word}\" names \\num{macro} = {here} ({key}), but the "
                    f"{'largest' if want_max else 'smallest'} of that series"
                    f"{f' through {limit}' if limit else ''} is {extreme[1]} at {extreme[0]} "
                    f"({re.sub(r'[0-9]+$', '', key)}{extreme[0]})")
    return bad


# A verdict phrase names the budget it governs more often than not, in words or in $k=..$ form.
BUDGET_WORDS = {"one": 1, "three": 3, "five": 5, "eight": 8, "ten": 10, "fifteen": 15,
                "twenty": 20, "thirty": 30, "fifty": 50}


def _budget_named(span: str):
    """The budget a verdict phrase names, if it names one."""
    m = re.search(r"\bk\s*=\s*(\d+)", span)
    if m:
        return int(m.group(1))
    m = re.search(r"\bat (" + "|".join(BUDGET_WORDS) + r")\b", span, re.I)
    if m:
        return BUDGET_WORDS[m.group(1).lower()]
    m = re.search(r"\bat \$?(\d+)\$?\b", span)
    return int(m.group(1)) if m else None


def check_separations(numbers: dict, keys: dict) -> list:
    """A separation verdict is checked against the interval it governs, not the nearest one.

    Attaching a verdict to whatever interval sits within a fixed window produced three false
    reports on the first run, all of the same shape: a sentence reading "... and still excludes
    zero. At fifteen it does not: [interval]" or "... at k=10 ..., it does not separate at k=15,
    and then trails ... [interval at 20]". The phrase was right and the interval it was pinned to
    was a different budget. A gate that reports those gets switched off, so the phrase must earn
    its interval: nothing else may sit between them, and where the phrase names a budget the
    interval has to be that budget's.
    """
    bad = []
    for doc in DOCS:
        text = flat(ROOT / doc)

        intervals = []
        for m in re.finditer(r"\\num(\w+?)Lo\b.{0,40}?\\num(\w+?)Hi\b", text, re.S):
            if m.group(1) != m.group(2):
                continue
            base = keys.get("num" + m.group(1) + "Lo")
            if base and base.endswith(".lo"):
                intervals.append((m.start(), m.end(), m.group(1), base[:-3]))
        if not intervals:
            continue

        for vm in re.finditer(rf"({NOT_SEPARATES}|{SEPARATES})", text, re.I):
            # negatives first, and the verdict is decided by which family fully matched
            claim_yes = not re.fullmatch(NOT_SEPARATES, vm.group(0), re.I)
            # the interval this phrase governs: the nearest one with no other between
            before = [iv for iv in intervals if iv[1] <= vm.start()]
            after = [iv for iv in intervals if iv[0] >= vm.end()]
            cands = []
            if before:
                cands.append((vm.start() - before[-1][1], before[-1]))
            if after:
                cands.append((after[0][0] - vm.end(), after[0]))
            if not cands:
                continue
            gap, iv = min(cands, key=lambda x: x[0])
            if gap > 200:
                continue
            _, _, macro, stem = iv

            # where the phrase names a budget, the interval must be that budget's
            named = _budget_named(text[max(0, vm.start() - 60):vm.end() + 60])
            if named is not None:
                km = re.search(r"(\d+)$", stem)
                if not km or int(km.group(1)) != named:
                    continue

            # A population marker between the phrase and the interval means the phrase is about a
            # different measurement of the same quantity. "both intervals excluding zero, where on
            # the comparison set neither separates" puts the negative on the comparison-set figures
            # while the nearest interval is the whole-set one.
            span = text[min(vm.end(), iv[0]):max(vm.start(), iv[1])]
            if re.search(r"\b(?:on|over|against) the (?:comparison|whole|validation|full|other|"
                         r"unselected) (?:set|split|population)\b", span, re.I):
                UNRESOLVED.append(f"{doc}: a verdict and an interval separated by a population "
                                  f"marker; which measurement the verdict governs is not resolvable")
                continue

            lo, hi = numbers.get(stem + ".lo"), numbers.get(stem + ".hi")
            if lo is None or hi is None:
                continue
            recorded = numbers.get(stem + ".sep")
            separates = bool(recorded) if isinstance(recorded, bool) else (lo > 0 or hi < 0)
            if claim_yes == separates:
                continue
            line = text.count("\n", 0, vm.start()) + 1
            src = "the artifact's own verdict" if isinstance(recorded, bool) else "its bounds"
            bad.append(f"{doc}:{line}  \"{' '.join(vm.group(0).split())}\" is placed on "
                       f"\\num{macro}, and {src} says the opposite: [{lo}, {hi}]")
    return bad


TODO = """Claims this check cannot yet reach, listed so the gap is stated rather than implied:

  counts over a table   "of the eleven separating cells", "the four cells that separate", "fifteen,
                        not eleven". Deciding what such a count ranges over needs the table's own
                        structure -- which rows, which comparators, which budgets -- and the
                        sentence does not carry it. Round 14 found three of these.

  scope words           "a large part of the bank", "almost nothing", "most of". These compare a
                        measured share against an unstated threshold, so there is nothing to check
                        them against until the threshold is written down.

  cross-document        a figure in the manuscript contradicting one in the supporting information
                        when the two are generated from different artifacts measuring the same
                        quantity under different instruments. Round 14 found four; each needs the
                        two instruments named beside the two figures, which is prose work.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true", help="what this check cannot reach")
    args = ap.parse_args()
    if args.todo:
        print(TODO)
        numbers = json.loads(NUMBERS.read_text())["numbers"]
        keys = macro_keys()
        check_superlatives(numbers, keys)
        check_separations(numbers, keys)
        if UNRESOLVED:
            print("  claims found but not decidable by this check:")
            for u in dict.fromkeys(UNRESOLVED):
                print("    " + u)
        return 0

    numbers = json.loads(NUMBERS.read_text())["numbers"]
    keys = macro_keys()
    bad = check_superlatives(numbers, keys) + check_separations(numbers, keys)
    if bad:
        print(f"REFUSING: {len(bad)} sentence(s) make a claim their own artifact does not support:")
        for line in bad:
            print("   " + line)
        print("\nThese are claims about a SET, not about a figure. The number in each is correct.")
        return 1
    print("every superlative names its series' extremum, and every separation verdict matches "
          "the artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
