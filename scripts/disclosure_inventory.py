"""Every sentence in which this work concedes something, listed so a shortening cannot drop one.

A cut is made for length and is judged on length, and what it costs is invisible at the moment it
is made: the sentences that go are the ones that read as slack, and a concession reads as slack.
This project has done it once already -- a shortening pass took out two verdicts, and nobody could
see it in the diff because the diff was large and the verdicts were one clause each.

So the concessions are enumerated before a cut and compared after it. A disclosure is a sentence in
which the paper says the work does not reach, does not establish, cannot separate, was not
measured, holds only under a condition, or is weaker than it looks. The extractor is a keyword
sweep and is therefore noisy in the harmless direction: a sentence that is not really a concession
costs a moment to confirm, while one that is a concession and vanishes is exactly what this exists
to catch.

Both documents are read together, and that is the point. Moving a concession from the manuscript to
the Supporting Information is a legitimate cut and this must not object to it; deleting one is not,
and this must. So a disclosure counts as present if it survives ANYWHERE, and the check reports
only what survives in neither.

    python scripts/disclosure_inventory.py            # record the inventory
    python scripts/disclosure_inventory.py --check    # refuse if a recorded one is now in neither
    python scripts/disclosure_inventory.py --refresh-text   # re-read the prose for unchanged keys

The `text` field is a snapshot taken when the inventory was last written, not the prose as it
stands. `_norm` deletes every character outside [a-z ] so that a key survives reflowing,
renumbering and a macro change -- which is what makes the guard usable during an edit, and also
what lets the stored sentence drift out of date silently. A semicolon raised to a full stop, a
capital, a rewrapped line: the key is identical, `--check` is green, and the recorded sentence is
no longer the one in the document. Anyone then reading `text` instead of the .tex reads the past.

`--refresh-text` re-reads the prose for keys whose identity is unchanged. It refuses outright if
any key has appeared or disappeared, because updating text on a tree whose disclosures have moved
would leave a green check and a freshened snapshot -- destroying both signals that a retraction
had happened. It never touches `reviewed_and_accepted`: the `text` inside those records is the
state a human was looking at when they accepted, and rewriting it would rewrite what was agreed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import stamp  # noqa: E402

# Everything typeset into either document, found by following the \input chain rather than
# listed here. The list used to be the three hand-written files, and it missed every generated
# table -- so a caption could lose a concession ("those two are lower bounds", "the two arms'
# slowest cells are not on the same substrates") and this check would report the cut clean,
# because the sentence it was protecting had never been in the inventory. A caption is typeset
# into the manuscript and a reader reads it; it is part of what the paper says.
ROOTS = ("paper2/grail_jcim.tex", "paper2/si.tex")


def _sources() -> tuple:
    """Every .tex the two documents pull in, transitively, in a stable order."""
    seen, queue = [], list(ROOTS)
    while queue:
        rel = queue.pop(0)
        if rel in seen or not (ROOT / rel).exists():
            continue
        seen.append(rel)
        for name in re.findall(r"\\input\{([^}]+)\}", (ROOT / rel).read_text()):
            nxt = f"paper2/{name}" if "/" not in name else name
            if not nxt.endswith(".tex"):
                nxt += ".tex"
            queue.append(nxt)
    return tuple(seen)


SOURCES = _sources()

# What a concession sounds like. Each is a claim the work makes ABOUT ITS OWN REACH, so the list is
# about limits and not about the chemistry: "does not" catches a limit on the work and also a fact
# about a molecule, and the noise that produces is cheaper than a missed concession.
MARKERS = (
    r"\bdoes not (?:follow|establish|reach|resolve|separate|say|show|support|transfer|cover)\b",
    r"\bcannot\b", r"\bis not (?:a|an|the|evidence|established|measured|checkable|comparable)\b",
    r"\bnot (?:measured|established|reported|checkable|available|redistributed|sought|corrected)\b",
    r"\bno (?:figure|number|claim|correction|model|evidence|producer|licence text)\b",
    r"\bwe do not\b", r"\bnothing (?:here|in this|that)\b",
    r"\bweaker\b", r"\bpessimistic\b", r"\bunresolved\b", r"\bopen question\b",
    r"\bis a bound\b", r"\bonly (?:under|at|when|holds|says|establishes)\b",
    r"\blimitation\b", r"\bcaveat\b", r"\bstated as (?:a|the) (?:bound|limit)\b",
    r"\bnot the (?:same|whole|only)\b", r"\brather than (?:a|an|the) (?:fact|result|verdict)\b",
    r"\bcould not be\b", r"\bwas not\b", r"\bdid not\b",
)
PAT = re.compile("|".join(MARKERS), re.I)


def _plain(tex: str) -> str:
    """LaTeX with its markup taken off, so a sentence is comparable across an edit that reflows."""
    # Nothing above \begin{document} is prose a reader meets: \usepackage lines, the author list,
    # affiliations and the title come through with only their macro names removed, and since none
    # of them ends in a full stop followed by a capital they do not split -- they arrive as ONE key.
    # paper2/grail_jcim.tex gave a 92-word, 688-character key of package names and affiliations run
    # into the opening of the abstract, and it was recorded as a disclosure. That made the author
    # block load-bearing: adding an author or fixing an affiliation would move the key and --check
    # would report a retraction for an edit that touched no claim. The cut is conditional because
    # the generated table files carry no \begin{document} and taking everything would take them.
    _body = re.split(r"\\begin\{document\}", tex, maxsplit=1)
    tex = _body[1] if len(_body) > 1 else tex
    t = re.sub(r"(?<!\\)%.*", "", tex)
    # Only the numeric bodies go: a tabular is rows of figures and an equation is not prose.
    # The float wrappers stay, because a caption lives inside one and a caption is prose the
    # reader reads. Stripping \begin{table} to \end{table} wholesale is how every generated
    # table's caption stayed outside this inventory, and one of them lost a concession.
    t = re.sub(r"\\begin\{(equation|align|tabular)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\(cite[a-z]*|ref|label|input|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\num[A-Za-z]+\{?\}?", " N ", t)          # a macro's value is not the sentence
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$\\~^_&]", " ", t)
    return re.sub(r"\s+", " ", t)


def _sentences(text: str) -> list:
    # Split on a full stop that ends a word and is followed by a capital or a quote. Abbreviations
    # inside a sentence ("Fig.", "et al.") therefore do not split it.
    parts = re.split(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z\"'\u201c])", text)
    return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    """A form stable under reflowing, renumbering and macro changes."""
    s = re.sub(r"\bN\b", "", s)
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return " ".join(s.split())


def collect() -> dict:
    found = {}
    for rel in SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        for s in _sentences(_plain(p.read_text())):
            if len(s) < 40 or not PAT.search(s):
                continue
            k = _norm(s)
            if len(k.split()) < 6:
                continue
            found.setdefault(k, {"text": s[:300], "in": []})
            if rel not in found[k]["in"]:
                found[k]["in"].append(rel)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the recorded inventory and refuse on a disclosure that "
                         "now survives in neither document")
    ap.add_argument("--accept", metavar="REASON",
                    help="re-baseline, recording each sentence that went and why. Use only after "
                         "reading every one the check named: the extractor cannot tell a rewrite "
                         "from a deletion, so a rewritten sentence looks exactly like a dropped "
                         "one and only a reader can say which it was")
    ap.add_argument("--refresh-text", action="store_true",
                    help="re-read the prose into the `text` field for keys whose identity is "
                         "unchanged, and re-stamp. Refuses if any key appeared or disappeared, "
                         "and never touches reviewed_and_accepted")
    args = ap.parse_args()

    out = ROOT / "results" / "disclosure_inventory.json"
    now = collect()

    if args.check:
        if not out.exists():
            print(f"REFUSING: {out.relative_to(ROOT)} has not been recorded, so there is nothing "
                  f"to compare a cut against. Run this without --check before shortening.",
                  file=sys.stderr)
            return 1
        before = json.loads(out.read_text())["disclosures"]
        gone = [k for k in before if k not in now]
        print(f"  {len(before)} disclosures recorded, {len(now)} present now, {len(gone)} gone")
        if gone:
            print(f"\nREFUSING: {len(gone)} disclosure(s) survive in neither the manuscript nor "
                  f"the Supporting Information:", file=sys.stderr)
            for k in gone[:25]:
                print(f"    was in {', '.join(before[k]['in'])}: {before[k]['text'][:150]}",
                      file=sys.stderr)
            print("\n  Moving one to the Supporting Information is a cut and passes here. "
                  "Deleting one is a retraction and has to be a decision, not a side effect of a "
                  "word count.", file=sys.stderr)
            return 1
        added = [k for k in now if k not in before]
        if added:
            print(f"  {len(added)} new disclosure(s) since the inventory was recorded")
        return 0

    if args.refresh_text:
        if not out.exists():
            print(f"REFUSING: {out.relative_to(ROOT)} has not been recorded, so there is no "
                  f"snapshot to refresh. Run this without a flag first.", file=sys.stderr)
            return 1
        prior = json.loads(out.read_text())
        before = prior.get("disclosures") or {}

        # The refusal this flag exists around. Refreshing text on a tree whose disclosures have
        # moved would leave a green --check AND a freshened snapshot, which is strictly worse than
        # either alone: it removes both of the signals a retraction leaves behind. So the key sets
        # must match exactly, and a disagreement is adjudicated by a reader through --accept.
        gone = [k for k in before if k not in now]
        added = [k for k in now if k not in before]
        if gone or added:
            print(f"\nREFUSING: the recorded keys and the documents no longer agree "
                  f"({len(gone)} gone, {len(added)} new), so this is not a text refresh.",
                  file=sys.stderr)
            for k in gone[:5]:
                print(f"    gone: {before[k]['text'][:130]}", file=sys.stderr)
            for k in added[:5]:
                print(f"    new:  {now[k]['text'][:130]}", file=sys.stderr)
            print("\n  Refreshing here would hide exactly what --check is for. Read what the "
                  "check names and settle it with --accept; then refresh.", file=sys.stderr)
            return 1

        drift = [k for k in before if before[k].get("text") != now[k]["text"]]
        moved = [k for k in before if before[k].get("in") != now[k]["in"]]
        was = (prior.get("provenance") or {}).get("source_sha256")
        fresh = stamp(__file__)
        stale_producer = was != fresh.get("source_sha256")

        if not drift and not stale_producer:
            print(f"  {len(before)} recorded, key sets agree, no text has drifted and the "
                  f"producer is unchanged: nothing to refresh")
            return 0

        # `before` is prior["disclosures"], not a copy of it, so the assignment below reaches the
        # old text through the alias and destroys it. Keep what is about to be overwritten first,
        # or the report prints the new sentence twice and claims no difference.
        was_text = {k: before[k].get("text", "") for k in drift}
        for k in drift:
            prior["disclosures"][k]["text"] = now[k]["text"]
        prior["provenance"] = fresh

        # Intending not to touch the accepted decisions is not a mechanism. Compare them.
        keep = json.dumps(json.loads(out.read_text()).get("reviewed_and_accepted") or [],
                          sort_keys=True)
        if json.dumps(prior.get("reviewed_and_accepted") or [], sort_keys=True) != keep:
            print("REFUSING: this run would have altered reviewed_and_accepted, which records "
                  "what a human agreed to and the text they agreed it against. Nothing written.",
                  file=sys.stderr)
            return 1

        out.write_text(json.dumps(prior, indent=1))
        print(f"  {len(before)} recorded, key sets agree, {len(drift)} sentence(s) refreshed"
              + (f", producer re-stamped ({str(was)[:8]} -> {fresh['source_sha256'][:8]})"
                 if stale_producer else ""))
        # Print a window around the FIRST difference, not the first 120 characters: the drift that
        # motivated this flag sat at offset 129 of its sentence, where a leading slice shows two
        # identical lines and hides the one character that moved.
        for k in drift[:10]:
            a, b = was_text[k], now[k]["text"]
            i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            lo, hi = max(0, i - 55), i + 55
            head = "" if lo == 0 else "..."
            print(f"    at offset {i}:")
            print(f"      was: {head}{a[lo:hi]}...")
            print(f"      now: {head}{b[lo:hi]}...")
        if moved:
            print(f"  {len(moved)} disclosure(s) now appear in a different set of files; the `in` "
                  f"field is left as recorded so the move stays visible, not normalised away:")
            for k in moved[:10]:
                print(f"    {before[k].get('in')} -> {now[k]['in']}")
        if list(prior.get("sources") or []) != list(SOURCES):
            print(f"  the \\input chain has changed since the inventory was recorded; `sources` is "
                  f"left as recorded. Re-record deliberately if that is real.")
        print(f"wrote {out.relative_to(ROOT)}")
        return 0

    accepted = []
    if args.accept:
        prior = json.loads(out.read_text()) if out.exists() else {}
        accepted = list(prior.get("reviewed_and_accepted") or [])
        before = prior.get("disclosures") or {}
        for k in before:
            if k not in now:
                accepted.append({"text": before[k]["text"], "was_in": before[k]["in"],
                                 "reason": args.accept})
        print(f"  {len([a for a in accepted if a['reason'] == args.accept])} sentence(s) accepted "
              f"as reworded rather than retracted, with the reason recorded")

    out.write_text(json.dumps({
        "provenance": stamp(__file__),
        "reviewed_and_accepted": accepted,
        "question": ("every sentence in which this work concedes something, recorded before a cut "
                     "so that what a cut costs is visible rather than invisible"),
        "sources": list(SOURCES),
        "n": len(now),
        "reading": ("a disclosure counts as present if it survives in ANY of the sources: moving "
                    "one to the Supporting Information shortens the manuscript without retracting "
                    "anything, and deleting one shortens it by retracting something"),
        "disclosures": now,
    }, indent=1))
    per = {}
    for v in now.values():
        for rel in v["in"]:
            per[rel] = per.get(rel, 0) + 1
    print(f"  {len(now)} disclosures recorded")
    for rel, n in sorted(per.items()):
        print(f"    {rel}: {n}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
