"""How long the sentences are, held to a ceiling, because density creeps back one edit at a time.

This manuscript's voice packs a claim, the qualification that bounds it and the evidence that
supports it into one sentence. That is why it says a great deal in few pages, and it is also why it
read as heavily as it did: the mean sentence ran to 28.8 words with 12 per cent of them over 45 and
thirteen over 60, where a chemistry journal's prose sits nearer 22.

Splitting them cost nothing. The macro, citation and cross-reference multisets came through the
pass unchanged, and every concession the manuscript makes is still in it. What splitting cannot do
is stay done: each new paragraph is written in the same voice, and the mean climbs back a sentence
at a time with no single edit ever looking wrong. So the ceiling is recorded and checked.

The thresholds are a ceiling and not a target. They are set a little above where the manuscript now
sits, so an edit that adds a genuinely long sentence passes and a drift back to where it was does
not.

    python scripts/check_prose_density.py
    python scripts/check_prose_density.py --worst   # the sentences nearest the ceiling
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The manuscript is what a reader reads first and is held tightest; the Supporting Information is
# reference material and is held loosely, because a method described once in full is not improved
# by being cut into six sentences.
LIMITS = {
    "paper2/body.tex": {"mean": 28.0, "over45_share": 0.10, "longest": 75},
    "paper2/grail_jcim.tex": {"mean": 30.0, "over45_share": 0.20, "longest": 60},
    # The Supporting Information was absent from this dict while the reviewer who prompted the
    # pass named it explicitly ("your manuscript and SI"). Between the pass that set these
    # ceilings and now, the manuscript went 31 pages to 26 and the SI went 87 to 101: what was
    # measured shrank and what was not measured grew. The mean is NOT the defect and its ceiling
    # matches the manuscript's, because eight comparable papers put the whole-document mean at
    # 23.2-28.5 (median 26.3) and this file sits at 27.4, inside that range. What sits outside is
    # the tail: 12 per cent of its sentences run past 45 words and its longest is 96. So the two
    # tail ceilings are set where the manuscript's are, this file does not meet them today, and
    # that is recorded as an open debt rather than accommodated by a ceiling drawn around it.
    "paper2/si.tex": {"mean": 28.0, "over45_share": 0.10, "longest": 75},
}

# A file-level mean cannot fail on one section. The Conclusions were reported elsewhere at 48
# words a sentence, which would have dissolved into the manuscript's 25.4 and tripped nothing;
# measured through this splitter they are 27.5, so that figure was an artefact of another
# instrument and no ceiling is set to catch it. What the same measurement does find is real and
# local: of 26 manuscript sections the highest mean is 30.5, while the Supporting Information
# runs to 42.0 and 39.9 in two of its 42. The ceiling is therefore set above every section of the
# manuscript and above the whole-document range those eight papers occupy, so it flags a genuine
# local outlier and nothing else. Sections under four sentences are not measured: a mean over
# three sentences is noise.
SECTION_LIMITS = {"mean": 32.0, "min_sentences": 4}
# A sentence that is one long list by format rather than by prose. The Supporting Information
# listing is required to enumerate every section in one sentence, and cutting it up would break the
# format the journal asks for.
EXEMPT = ("The Supporting Information is available free of charge",)


def blocks(src: str) -> list:
    """Prose cut at every structural boundary, so no measured sentence spans two of them."""
    t = re.sub(r"(?<!\\)%.*", "", src)
    # suppinfo is one sentence by the journal's format: an enumeration of every SI section. It is
    # not prose and cutting it up would break the format, so it never reaches the measurement.
    t = re.sub(r"\\begin\{(equation|align|figure|table|tabular|suppinfo)\*?\}.*?\\end\{\1\*?\}",
               "\n@@\n", t, flags=re.S)
    t = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", "\n@@\n", t)
    # The front matter is not prose and does not end in full stops, so an address, a title and the
    # abstract's opening ran together into one 61-word "sentence" that no reader ever meets. Each
    # front-matter field is its own block, and so is the boundary of the abstract.
    t = re.sub(r"\\(title|author|affiliation|email|keyword|abbreviations)\*?"
               r"(\[[^\]]*\])?\{", "\n@@\n\\1{", t)
    t = re.sub(r"\\(begin|end)\{abstract\}", "\n@@\n", t)
    t = re.sub(r"\\(cite[a-z]*|ref|label|input|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " C ", t)
    t = re.sub(r"\\num[A-Za-z]+\{?\}?", " N ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$\\~^_&]", " ", t)
    return [re.sub(r"\s+", " ", b).strip() for b in t.split("@@")]


ABBR = re.compile(r"\b(e\.g|i\.e|cf|vs|Fig|Eq|Sec|al|approx|ca)\.$")


def sentences(block: str) -> list:
    out, cur = [], []
    for tok in re.split(r"(?<=\.)\s+", block):
        cur.append(tok)
        if ABBR.search(tok.strip()) or not tok.strip().endswith("."):
            continue
        out.append(" ".join(cur).strip())
        cur = []
    if cur:
        out.append(" ".join(cur).strip())
    return [s for s in out if len(s.split()) > 3
            and not any(s.startswith(e) or e in s[:120] for e in EXEMPT)]


def measure(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.exists():
        return None
    ss = [s for b in blocks(p.read_text()) for s in sentences(b)]
    if not ss:
        return None
    L = [len(s.split()) for s in ss]
    return {"n": len(ss), "mean": st.mean(L), "median": st.median(L), "longest": max(L),
            "over45": sum(1 for x in L if x > 45),
            "over45_share": sum(1 for x in L if x > 45) / len(L),
            "worst": sorted(ss, key=lambda s: -len(s.split()))[:6]}


def by_section(rel: str) -> list:
    """The same measurement, per sectioning unit, because a file mean cannot fail on a section.

    Every ceiling above is an average over a whole document, and an average has no way to refuse
    one heavy section: a section at 42 words a sentence inside a file at 27.4 moves the file by a
    fraction of a word. Titles are kept so the refusal names the section a person has to open,
    and each unit is measured through the same blocks/sentences pair as the file, so a section
    figure and a file figure are the same quantity at two scales.
    """
    p = ROOT / rel
    if not p.exists():
        return []
    src = p.read_text()
    marks = [(m.start(), re.sub(r"\\[a-zA-Z]+\*?|[{}]", "", m.group(1)).strip())
             for m in re.finditer(r"\\(?:sub)*section\*?\{([^}]*)\}", src)]
    if not marks:
        return []
    marks.append((len(src), ""))
    out = []
    for (a, title), (b, _) in zip(marks, marks[1:]):
        ss = [s for blk in blocks(src[a:b]) for s in sentences(blk)]
        if len(ss) < SECTION_LIMITS["min_sentences"]:
            continue
        L = [len(s.split()) for s in ss]
        out.append({"title": title or "(untitled)", "n": len(ss), "mean": st.mean(L),
                    "longest": max(L)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst", action="store_true", help="print the sentences nearest the ceiling")
    args = ap.parse_args()

    bad = []
    for rel, lim in LIMITS.items():
        m = measure(rel)
        if m is None:
            print(f"  {rel}: not in this checkout")
            continue
        print(f"  {rel}: {m['n']} sentences, mean {m['mean']:.1f} (ceiling {lim['mean']}), "
              f"{m['over45']} over 45 = {100 * m['over45_share']:.0f}% "
              f"(ceiling {100 * lim['over45_share']:.0f}%), longest {m['longest']} "
              f"(ceiling {lim['longest']})")
        for key, label in (("mean", "the mean sentence"), ("over45_share", "the share over 45"),
                           ("longest", "the longest sentence")):
            if m[key] > lim[key]:
                bad.append(f"{rel}: {label} is {m[key]:.2f}, over the ceiling of {lim[key]}")

        heavy = [s for s in by_section(rel) if s["mean"] > SECTION_LIMITS["mean"]]
        for s in sorted(heavy, key=lambda s: -s["mean"]):
            bad.append(f"{rel}: section \"{s['title'][:60]}\" means {s['mean']:.1f} words over "
                       f"{s['n']} sentences, over the section ceiling of "
                       f"{SECTION_LIMITS['mean']}")

        if args.worst:
            for s in m["worst"]:
                print(f"      [{len(s.split()):3d}] {s[:150]}")

    if bad:
        print("\nREFUSING: the prose has drifted back past its ceiling:", file=sys.stderr)
        for x in bad:
            print(f"    {x}", file=sys.stderr)
        print("\n  A long sentence is not a defect on its own. A manuscript whose mean sentence "
              "climbs back is one nobody will read, and it climbs back one edit at a time with no "
              "single edit looking wrong, which is why this is a ceiling rather than a review.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
