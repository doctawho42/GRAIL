#!/usr/bin/env python3
r"""How often this manuscript reaches for two constructions, held against what the journal does.

A reviewer said the manuscript and its Supporting Information read as machine-written. The reply
that each hedge is individually defensible is true and answers a different question: the complaint
is about a rate, and a set of individually defensible sentences can still sit an order of
magnitude outside what the literature does. So the rate is measured rather than argued.

The comparison is eight papers, 49,637 words of body prose, obtained as PMC open-access XML: six
JCIM method comparisons and two adjacent-journal ones chosen because they are the closest in
subject, one of them running SyGMa, GLORYx and BioTransformer, which is this work's own comparator
set. References, captions and tables are excluded.

    rather than            pooled 0.34 per 1000 words, per-paper range 0.00-0.68
    is/are not a           pooled 0.02, range 0.00-0.11    see ceiling_caveats
    epistemic self-grading pooled 0.02, range 0.00-0.26    see ceiling_caveats
    worth stating/saying   0.00 in all eight
    which is why           0.00 in all eight
    the objection          0.00 in all eight

THOSE SIX COUNTS NO LONGER CARRY CEILINGS, and the reason is a defect in them. They overlap by
construction: "rather than a" is a subset of "rather than" and both sit inside the self-grading
probe, so the counts are not independent, two residues cannot be summed or compared, and a reword
clears one probe while the construction stays on the page. That is not a worry, it is what
happened. Rewording 39 occurrences of "rather than" out of the manuscript moved 25 of them into
"and not" -- a phrase occurring ZERO times in the 49,637 words of comparison corpus -- and the
per-phrase counter reported 62 -> 23 while the share of sentences reaching for any of the
constructions moved only 98 -> 81. So the criterion is now that share, which cannot double-count
and does not migrate under rewording, and the six counts below it are printed to localise the
work and to be read, never to be passed. The corpus puts that share at 17 sentences in 1,841,
0.92 per cent, against 18.75 per cent here.

Those two caveats are in the artifact rather than restated here, because a figure copied into a
comment is a figure that can drift from the file it was copied from -- the fault this file has
already committed once. In short: a per-paper maximum computed from a SINGLE occurrence estimates
the length of the paper carrying it and not how often the construction is used. "is/are not a"
and epistemic self-grading each occur once in the whole corpus, so their ceilings rest on the
shortest carrying paper, 8,816 and 3,853 words. Those ceilings are still taken from data instead
of chosen, which is the point, and they are not frequencies. The three zeros carry no such
caveat: zero in all eight, with the planted-case control showing the probes see real occurrences.

Those figures are measured here and not transcribed, and the difference mattered. A report that
established this corpus put "rather than" at 0.28 pooled and epistemic self-grading at "zero in
50,139 words". Measured with the probes below, which count across a line break, the first is 0.34
and the second is one occurrence at 0.26 per 1000 words in a single paper. A ceiling of zero would
have been the pooled rate mistaken for the bound.

A BASELINE IS A MEASUREMENT ONLY IF ITS PROBE CAN FAIL. A zero and a probe that cannot see the
construction produce the same number, and this file had already been burned by that once: a
"not X but Y" pattern written with the two words adjacent returned 0 on a manuscript containing
five, which would have certified the absence of whatever it could not match. So every probe is
presented with a known answer in both directions. Against its own fixture it must find the
planted count; against the comparison corpus, real occurrences taken verbatim from the manuscript
are injected and the probe must find exactly as many as were injected. All six pass that, the
three zeros included, which is what makes a zero here a fact about the literature rather than a
fact about a regex.

Two measured constructions are deliberately NOT here, and the omission is the point: "not X but Y"
runs at 0.36 and 0.17 against a corpus range of 0.00-0.77, with one of the eight papers above us,
and the mean sentence length runs at 25.4 and 27.4 against a corpus range of 23.2-28.5. Both are
inside the range these papers occupy. Editing either would be damage dressed as polish, and a
later pass that "fixes" them would be working from an impression rather than a measurement.

One paper could not be measured and is named rather than estimated: the benchmark of metabolite
predictors against human radiolabelled ADME data, which this work cites, has no PMCID and is not
in the open-access subset. Its rates are unknown and no substitute was used for them.

Every probe here counts across a line break, because the manuscript is hard-wrapped and a pattern
written with a literal space misses "rather\nthan" -- 3 such in the manuscript and 5 in the
Supporting Information, which is enough to move a headline count. And every probe is run first
against a fixture whose answer is known, and the check refuses if a probe fails its own fixture:
a regex that silently matches nothing reports a clean document, which is the failure this file
exists to avoid.

    python scripts/check_prose_rate.py
    python scripts/check_prose_rate.py --show    # print the occurrences, not the counts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_prose_density as cpd  # noqa: E402  the one splitter this repository has

# \s+ everywhere a space would do, so a hard-wrapped occurrence counts once rather than never.
PROBES = {
    "rather than": r"\brather\s+than\b",
    "is/are not a": r"\b(?:is|are)\s+not\s+a\b",
    "epistemic self-grading": (r"\brather\s+than\s+a\b|\band\s+not\s+a\b|\brather\s+than\s+of\b"
                               r"|\bis\s+reported\b|\bis\s+stated\b"),
    # naming, printing, recording and setting out were missing from the first version of this
    # probe, which undercounted the Supporting Information by six: "worth naming", "worth
    # printing", "worth recording", "worth setting out". The occurrences are printed with --show
    # rather than trusted as a count, which is how the gap was found.
    "worth V-ing": (r"\bworth\s+(?:stating|saying|naming|printing|recording|having|carrying"
                    r"|putting|setting)\b"),
    "which is why": r"\bwhich\s+is\s+why\b",
    "the objection": r"\bthe\s+objection\b",
}

# A fixture per probe, with the answer written down. The line break in each is deliberate.
FIXTURES = {
    "rather than": ("the sweep is reported rather\nthan described, and rather than one cell", 2),
    "is/are not a": ("that is not a\nresult, and these are not a family", 2),
    "epistemic self-grading": ("it is stated\nhere, and reported as a direction rather than a "
                               "measurement", 2),
    "worth V-ing": ("worth\nstating exactly, and worth carrying through", 2),
    "which is why": ("which is\nwhy it is printed", 1),
    "the objection": ("the\nobjection is answered by measurement", 1),
}

# The construction, not the phrase. See the docstring for why the six above are diagnostics: they
# overlap, so rewording moves an occurrence between them without removing anything from the page.
# "and not" is carried here and in no ceiling of its own because the corpus gives it zero in all
# eight papers, so there is no rate to hold it against -- only an absence to report.
#
# "instead of" is the same habit in a third spelling and was added on the author's decision, with
# its cost stated first rather than sold as a free tightening: the corpus DOES use it (one
# occurrence in one of the eight, pooled 0.02 per 1000 words), so unlike "and not" it has a real
# rate. Including it moves our count AND the corpus norm together -- body 81 to 84 carriers, the
# corpus 17 to 18, 0.92 to 0.98 per cent -- so the ceiling moves too and the gap does not change
# the way intuition suggests. What it buys is completeness: without it, a pass that repaired every
# carrier would leave three manuscript sentences carrying the habit untouched, which is a hole in
# the pass rather than in the text.
CARRIER_PROBES = dict(PROBES, **{"and not": r"\band\s+not\b",
                                 "instead of": r"\binstead\s+of\b"})
CARRIER_ANY = "|".join("(?:" + p + ")" for p in CARRIER_PROBES.values())

# A planted case whose answer is known by construction: five sentences, three of them carriers.
# Every probe in the carrier set needs a sentence here that it alone would catch, so that adding a
# probe without testing its liveness is not possible: the Epsilon sentence is the one "instead of"
# must find, and a probe that matches nothing would drop the count to 2 and refuse the run.
CARRIER_FIXTURE = ("Alpha is measured on the test split.\n\nBeta is a property of the budget and "
                   "not of the system.\n\nGamma is swept rather than disclosed.\n\nDelta carries "
                   "no such construction at all.\n\nEpsilon reports the median instead of the "
                   "tail.\n", 5, 3)


def carriers(rel: str) -> tuple:
    """Sentences reaching for at least one construction, over every sentence in the file.

    Measured on the RAW file through the density gate's splitter, which is the splitter the
    ceiling is quoted against. The same quantity over the disclosure inventory's splitter gives
    142 where this gives 144 on the Supporting Information, so the number is not meaningful
    without the instrument named beside it, and the artifact records that requirement rather than
    leaving it to whoever reads the output.

    The first version of this measurement fed prose() -- already split and re-joined -- back into
    the splitter. Re-blocking collapsed 432 sentences into 155 and the word denominator lost more
    than half the file, so both columns were wrong and the headline ratio with them, while the
    corpus rows came through the correct path: target from one instrument, reading from another.
    The self-test below is what catches that, by requiring a sentence count established elsewhere.
    """
    p = ROOT / rel
    if not p.exists():
        return 0, 0, 0
    sents = [s for blk in cpd.blocks(p.read_text()) for s in cpd.sentences(blk)]
    car = [s for s in sents if re.search(CARRIER_ANY, s, re.I)]
    return len(sents), len(car), len(" ".join(sents).split())


def carrier_ceiling(words: int) -> tuple:
    """The carrier ceiling and the corpus block it came from, read at every run, never typed."""
    art = json.loads(CORPUS.read_text())
    c = art["carrier_sentences"]
    return int(c["per_paper_max_per_1000"] * words / 1000), c


# Counts, not rates: a rate invites rounding an argument, and these are small integers a person
# can check by eye. The manuscript's ceiling is the corpus's generous per-paper bound carried
# across 11.5k words; the Supporting Information's is the same bound across 29.7k.
# Each ceiling is the highest rate any single comparison paper reaches, carried across this
# file's word count, and never the count the file happens to show: an earlier draft set the
# manuscript's "is/are not a" ceiling to 4 when the file held 4, which is a ceiling drawn around
# the state it is meant to judge.
#
# The rates come from results/prose_rate_corpus.json, which this gate reads rather than restating,
# and which records the planted-case control described above. Two of these ceilings moved once
# that control was run and the baseline was measured here instead of transcribed. "rather than"
# allows 7.4 in the manuscript at the highest paper's 0.68 per 1000 words, so 7 and not 8. And
# epistemic self-grading is NOT zero in the corpus: one paper carries it at 0.26 per 1000 words,
# which allows 2.8 here and 7.6 in the Supporting Information. A ceiling of zero was the pooled
# rate mistaken for the bound, and it would have driven an edit toward a number no paper meets.
# The three that remain zero are zero in all eight papers, and the control proves the probes can
# see them when they are there.
FILES = ("paper2/body.tex", "paper2/si.tex")
CORPUS = ROOT / "results" / "prose_rate_corpus.json"

# The length target, gated rather than remembered. A reviewer asked for a much shorter manuscript;
# a word count that lives only in a conversation is a target nobody can be held to, and the repo's
# own convention is that an accepted miss is a RED GATE with a written reason, not a note. So the
# baseline and the target are here, the gate refuses while the file is above target, and why the
# remaining distance was not taken is in DECLARED_DEBT beside the refusal.
#
# The baselines are the word counts at the start of the shortening pass, measured with prose()
# below -- not with either of the two other word-counting conventions in this repository, which
# disagree with it by up to five per cent. A target quoted without its instrument is not a target.
VOLUME = {
    "paper2/body.tex": {"baseline": 11131, "target": 8349, "cut": 0.25},
}


def ceiling(probe: str, words: int) -> int:
    """The highest rate any one comparison paper reaches, carried across this file's length.

    Derived rather than typed, because the first version of this file carried the six numbers as
    literals under a docstring saying they came from the artifact. They did not: they were
    hand-edited, and two of them were already wrong. A comment asserting what the code does not do
    is the defect this repository has now produced four times -- a list whose comment promises it
    follows the data, a counter whose comment promises symmetry, a gate whose docstring promises a
    comparison that was never written, and this. So the number is computed from the recorded
    corpus at every run and cannot drift from it.

    Scaling by the file's current length is deliberate: the ceiling is a rate, and a file that
    gets shorter is allowed proportionally fewer. It tightens as the pass proceeds.
    """
    mx, _ = bounds(probe)
    return int(mx * words / 1000)


def bounds(probe: str) -> tuple:
    """The two rates the corpus supports for this probe: the per-paper maximum and the pooled.

    Returned together because the gate refuses on one and has to show the other. Which of the two
    is the ceiling is recorded in the artifact's ceiling_basis field, with the reason and with the
    admission that the maximum is the generous of the two.

    A probe the artifact does not record returns (None, None) rather than a number, and the caller
    says so. Silence beats a guess here for a specific reason: the first version of the
    eight-probe criterion printed "ZERO in all eight comparison papers" for every probe outside
    the original six, which was true of "and not" and false of "instead of" -- the corpus carries
    one occurrence of it. An unmeasured absence asserted as a measured zero overstates the case
    against the text this gate is run on, and an author's own instrument must never fail in that
    direction.
    """
    art = json.loads(CORPUS.read_text())
    if probe not in art["per_paper_max_per_1000"] or probe not in art["pooled"]:
        return None, None
    return art["per_paper_max_per_1000"][probe], art["pooled"][probe]["per_1000"]


def prose(rel: str) -> str:
    """The text a reader meets, through the gate's splitter so there is one de-TeXer here."""
    p = ROOT / rel
    if not p.exists():
        return ""
    return " ".join(s for blk in cpd.blocks(p.read_text()) for s in cpd.sentences(blk))


def hits(text: str, pat: str) -> list:
    return [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(pat, text, re.I)]


def locate(rel: str, pat: str) -> list:
    """Every occurrence as file:line with the words around it, for a person about to edit.

    The first version of --show printed what hits() returns, which is the matched text alone:
    forty lines reading "rather than" and nothing else. That satisfies the instruction to print
    occurrences instead of counts and defeats its purpose, because an occurrence you cannot find
    is not an occurrence you can weigh. Counting stays on the prose (tables and captions are not
    read in sequence); locating goes back to the raw file, which is the only place a line number
    exists. The two totals are printed side by side rather than reconciled: where they differ,
    the difference is occurrences inside floats, and a person should see that rather than be
    handed one number.
    """
    p = ROOT / rel
    if not p.exists():
        return []
    raw = p.read_text()
    out = []
    for m in re.finditer(pat, raw, re.I):
        line = raw.count("\n", 0, m.start()) + 1
        lo, hi = max(0, m.start() - 60), min(len(raw), m.end() + 60)
        out.append((line, re.sub(r"\s+", " ", raw[lo:hi]).strip()))
    return out


def self_test() -> list:
    """Each probe against its own fixture. A probe that cannot find a planted case is refused."""
    broken = []
    for name, pat in PROBES.items():
        fixture, expected = FIXTURES[name]
        got = len(hits(fixture, pat))
        if got != expected:
            broken.append(f"probe {name!r} found {got} in its own fixture, expected {expected}")
    text, exp_sents, exp_car = CARRIER_FIXTURE
    sents = [s for blk in cpd.blocks(text) for s in cpd.sentences(blk)]
    car = [s for s in sents if re.search(CARRIER_ANY, s, re.I)]
    if (len(sents), len(car)) != (exp_sents, exp_car):
        broken.append(f"the carrier measure found {len(sents)} sentences and {len(car)} carriers "
                      f"in its own fixture, expected {exp_sents} and {exp_car}: a measure that "
                      f"miscounts sentences reports a share of the wrong denominator")
    return broken


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print occurrences instead of counts")
    args = ap.parse_args()

    broken = self_test()
    if broken:
        print("REFUSING: a probe failed its own fixture, so a clean report would mean nothing:",
              file=sys.stderr)
        for b in broken:
            print(f"    {b}", file=sys.stderr)
        return 1

    if not CORPUS.exists():
        print(f"REFUSING: {CORPUS.relative_to(ROOT)} is not here, so there is no measured corpus "
              f"to hold this prose against and no ceiling that is not invented.", file=sys.stderr)
        return 1

    bad = []
    for rel in FILES:
        text = prose(rel)
        if not text:
            print(f"  {rel}: not in this checkout")
            continue
        words = len(text.split())
        n_sents, n_car, raw_words = carriers(rel)
        cap, cinfo = carrier_ceiling(raw_words)
        print(f"  {rel}: {words} words of prose, {n_sents} sentences")
        print(f"      {'sentences reaching for one':24} {n_car:4d}  of {n_sents} = "
              f"{100.0 * n_car / n_sents:5.2f}%  (ceiling {cap}, from "
              f"{cinfo['per_paper_max_per_1000']:.2f}/1k of {cinfo['max_from']} x {raw_words}w; "
              f"corpus pooled {100.0 * cinfo['pooled']['share']:.2f}% of sentences)"
              f"{'' if n_car <= cap else '  OVER'}")
        if n_car > cap:
            bad.append(f"{rel}: {n_car} of {n_sents} sentences reach for one of these "
                       f"constructions against a ceiling of {cap}")

        vol = VOLUME.get(rel)
        if vol:
            done = vol["baseline"] - words
            need = words - vol["target"]
            print(f"      {'length against the target':24} {words:4d}  of {vol['baseline']} "
                  f"({done} removed, {100.0 * done / vol['baseline']:.1f}%); target "
                  f"{vol['target']} at -{vol['cut']:.0%}, {need} still to remove"
                  f"{'' if need <= 0 else '  OVER'}")
            if need > 0:
                bad.append(f"{rel}: {words} words against a target of {vol['target']}, "
                           f"{need} still to remove")
        print("      -- the counts below are DIAGNOSTICS and carry no ceiling: they overlap each "
              "other, so they localise the work and cannot be summed --")
        for name, pat in CARRIER_PROBES.items():
            found = hits(text, pat)
            n = len(found)
            mark = ""
            # No verdict on this line, and that is the change. These counts overlap, so a ceiling
            # on each would be a ceiling on an unknown number of shared occurrences, and the
            # residues could be neither summed nor compared. The corpus rates are still printed
            # beside each count, because a diagnostic without a scale is an impression: what is
            # withdrawn is the pass/fail, not the comparison. "and not" has no rate to print --
            # zero in all eight papers -- so its absence is stated instead of a bound invented.
            mx, pooled = bounds(name)
            if mx is None:
                note = "not measured in the corpus, so no rate is claimed for it in either direction"
            elif mx == 0.0 and pooled == 0.0:
                note = "measured at ZERO in all eight comparison papers"
            else:
                note = (f"corpus per-paper max {mx:.2f}/1k, pooled {pooled:.2f}/1k, which over "
                        f"{words}w would be {mx * words / 1000:.1f} and "
                        f"{pooled * words / 1000:.1f}")
            print(f"      {name:24} {n:4d}  = {1000 * n / words:5.2f}/1k   ({note}){mark}")
            if args.show:
                found_at = locate(rel, pat)
                if len(found_at) != n:
                    print(f"          [{len(found_at)} in the raw file against {n} in the prose: "
                          f"the difference is inside floats]")
                for line, ctx in found_at:
                    print(f"          {rel}:{line}  ...{ctx}...")

    if bad:
        print("\nREFUSING: this text defines by exclusion in a share of its sentences that no "
              "comparison paper approaches:", file=sys.stderr)
        for x in bad:
            print(f"    {x}", file=sys.stderr)
        print("\n  Rewording is the wrong instrument, and that is measured rather than assumed: a "
              "pass over the manuscript moved occurrences out of \"rather than\" and into \"and "
              "not\", a construction the corpus never uses, and barely moved this criterion while "
              "the phrase counter reported a large fall. The integers are deliberately not "
              "restated here -- they age with every edit, and a failure message quoting a stale "
              "count is the defect this file exists to catch; results/prose_rate_corpus.json and "
              "the diagnostics above carry the current ones. What lowers this criterion is "
              "deleting a carrier that makes no claim of its own, or asserting positively what "
              "the sentence currently says by negation.\n"
              "  Two independent grounds are needed before any deletion, because the uniqueness "
              "test has failed three times here in one evening: first the inventory flag from "
              "scripts/disclosure_inventory.py, whose locator is _norm over _sentences(_plain(.)) "
              "and nothing shorter -- omitting _plain leaves the macros in and silently locates 31 "
              "of 65; second a check that every substantive claim in the sentence appears "
              "elsewhere verbatim or by norm, which is what caught a near-deletion whose twin "
              "omitted one word and a clause that explains a printed count while appearing in no "
              "inventory. A recorded concession is rewritten, never dropped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
