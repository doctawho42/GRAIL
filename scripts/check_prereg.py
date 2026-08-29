#!/usr/bin/env python3
"""Every sentence claiming an effect, matched against the hypothesis that registered it.

A preregistration is worth the commit it is timestamped in only if something later checks
that the paper stayed inside it. This is the same device the leaderboard paper uses for the
word `certified` (scripts/audit_claim_words.py): enumerate every sentence that makes the
claim, match each against a declared list, and exit non-zero on one that has no entry.

Two things are checked, and they fail separately.

  The registry. A hypothesis is registered only if it declares a prediction, a failure
  condition, and a family with a size. A prediction with no way to fail is not a
  preregistration, so a hypothesis missing any of the three is an error in the registry
  itself, before any manuscript is read.

  The manuscript, forwards. Every sentence asserting an effect has to name the hypothesis
  that entitles it, as `\\prereg{H1}` or a bare `H1`. Sentences that attribute an effect to
  someone else's system are counted separately rather than silently exempted, because an
  exemption nobody counts is where unregistered claims go to live.

  The vocabularies. This project carries two type vocabularies of nearly equal cardinality --
  4,417 bond-delta types and 4,456 signature types -- and a reader six months later will take
  them for one number rounded differently. A sentence that quotes a type count must name which
  vocabulary it counts.

  The manuscript, backwards. Every registered hypothesis has to appear, either carrying a
  claim or reported as having failed. A hypothesis that was registered, did not hold, and
  then quietly left the paper is the violation preregistration exists to prevent, and it is
  the likeliest of them: the forward direction cannot see it, because the sentence that
  would have carried it is the one that is gone.

The count of scanned sentences is printed and belongs in the report. A checker whose
matcher quietly stops matching still exits zero, and the count is what shows it.

    python scripts/check_prereg.py --prereg paper2/preregistration.md --text paper2/main.tex
    python scripts/check_prereg.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A hypothesis heading: `## H1 — title` or `## H1 - title` or `## H1: title`.
HEADING = re.compile(r"^#{1,6}\s*(H\d+)\s*[-—:–]\s*(.+?)\s*$", re.M)

# The three fields a registration has to carry. Each is matched as a bolded lead-in, in
# either language, so the registry can be drafted in one and published in the other.
FIELDS = {
    "prediction": re.compile(r"\*\*(prediction|предсказание)\.?\*\*", re.I),
    "failure": re.compile(r"\*\*(failure|fails?|провал)\.?:?\*\*", re.I),
    "family": re.compile(r"\*\*(family|семейство)\.?:?\*\*", re.I),
}
# The family has to carry a size: `m = 6`, `size 6`, `family of 168`.
FAMILY_SIZE = re.compile(r"(?:\bm\s*=\s*|\bsize\s+|\bof\s+)(\d+)")

# Verbs and comparatives that assert a change. A sentence carrying one of these is making a
# claim about an effect and needs a hypothesis behind it.
EFFECT = re.compile(
    r"\b(improve[sd]?|improving|rais(?:e|es|ed)|lift(?:s|ed)?|increase[sd]?|reduce[sd]?|"
    r"lower(?:s|ed)?|beat(?:s|en)?|outperform(?:s|ed)?|exceed(?:s|ed)?|gain(?:s|ed)?|"
    r"recover(?:s|ed)?|outrank(?:s|ed)?|dominate[sd]?|better than|worse than|higher than|"
    r"lower than|closes the gap|ahead of)\b", re.I)
# A signed effect size claims one too, even with no verb: `+0.176`, `$+0.12$`.
EFFECT_NUMBER = re.compile(r"[+\u2212-]\s*0\.\d{2,}")
# `recovers a reference' is a claim; `recovered from the commit' is a file operation that
# happens to share a verb. The sense is fixed by the object, so the object excludes it.
RETRIEVAL = re.compile(
    r"recover(?:s|ed|y)?\s+(?:\w+\s+){0,3}from\s+(?:the\s+)?"
    r"(?:commit|log|git|repository|artifact|release|pointer|blob|tree|checkout)", re.I)

TAG = re.compile(r"\\prereg\{(H\d+)\}|\b(H\d+)\b")
# The manuscript numbers the predictions in reading order, P1 upward, and carries the register's
# own identifier beside each in the hypotheses table. A checker that knows only H-identifiers
# reads such a manuscript as mentioning no hypothesis at all and reports every one of them
# absent -- sixteen failures on a paper with no defect, which is a gate that has stopped gating.
# The alias map is read from the generated table rather than hard-coded, so it cannot drift.
ALIAS_ROW = re.compile(r"^(P\d+)\s*&.*&\s*(H\d+)\s*\\\\", re.M)
ALIAS_TAG = re.compile(r"\b(P\d+)\b")


def alias_map(root: Path) -> dict:
    """P-identifier to register identifier, read from the table the manuscript prints."""
    table = root / "paper2" / "table_hypotheses.tex"
    if not table.exists():
        return {}
    return {p: h for p, h in ALIAS_ROW.findall(table.read_text())}


def tabled_outcomes(root: Path) -> set:
    """Hypotheses whose outcome the manuscript reports as a row of its own table.

    The backwards direction asks that a registered hypothesis carry a claim or a reported
    failure. A table row giving the threshold, the measured value and the population it was
    checked on is a reported outcome by any reading, and reading it as silence made the gate
    fail on six predictions the paper reports in full.
    """
    table = root / "paper2" / "table_hypotheses.tex"
    if not table.exists():
        return set()
    out = set()
    for row in table.read_text().splitlines():
        cells = [c.strip() for c in row.split("&")]
        if len(cells) < 6:
            continue
        register = re.search(r"\b(H\d+)\b", cells[-1])
        measured = re.search(r"-?\d", cells[3])
        if register and measured:
            out.add(register.group(1))
    return out
CITED = re.compile(r"\\cite[a-z]*\{|\\citet|\\citep")
# A sentence carrying this marker states a property of the estimator rather than a result about
# a system: a limit, an identity, or a rule derived from one. Such sentences are not registrable
# -- there is nothing to have predicted -- but they are effect-shaped, and exempting them
# silently is the failure this file exists to prevent. They are counted and listed instead, and
# the marker has to be written by hand next to the sentence so the exemption is deliberate.
DERIVATION = re.compile(r"%\s*no-claim:\s*derivation")
# A hypothesis may leave the paper only by being reported as not having held.
# The counts are read from the artifacts rather than typed, so a vocabulary that grows does not
# quietly stop being checked.
VOCAB_WORDS = re.compile(r"bond-delta|signature type|signature-type|signature vocabulary", re.I)
TYPE_WORD = re.compile(r"\btypes?\b", re.I)


def type_counts(root: Path) -> dict:
    """{count: vocabulary} over every type count the artifacts record."""
    out = {}
    carriers = root / "results" / "typed_edit_type_carriers.json"
    curve = root / "results" / "typed_edit_type_curve.json"
    if carriers.exists():
        d = json.loads(carriers.read_text())
        for k in ("n_types", "types_with_any_relaxable_carrier",
                  "types_with_any_curated_carrier", "types_carried_only_by_mined_rules"):
            if isinstance(d.get(k), int):
                out[d[k]] = "bond-delta"
    if curve.exists():
        d = json.loads(curve.read_text())
        for v in d.get("by_variant", {}).values():
            for k in ("n_types", "types_with_ge5_pairs"):
                if isinstance(v.get(k), int):
                    out.setdefault(v[k], "signature")
        for v in d.get("by_radius", {}).values():
            if isinstance(v.get("n_types"), int):
                out.setdefault(v["n_types"], "signature")
    return out


def _numbers(s: str):
    for m in re.finditer(r"\b\d[\d,{}]*\b", s):
        raw = m.group(0).replace(",", "").replace("{", "").replace("}", "")
        if raw.isdigit():
            yield int(raw)


# A hypothesis the paper says it does not report is accounted for. That is not the violation the
# backwards direction exists to catch -- which is a prediction that was run, did not hold, and
# then left -- and reading the two as the same made the gate fail on six declarations that are
# exactly what a register is for.
NOT_RUN = re.compile(
    r"\b(does not report|not reported|were not run|was not run|none was run|no(?:ne)? is "
    r"reported|concern work this paper does not)\b", re.I)

FAILED = re.compile(
    r"\b(fail(?:s|ed|ure)?|did not hold|does not hold|not supported|unsupported|refut(?:e|ed|es)|"
    r"returns? a null|bounded null|is a null|not confirmed|did not replicate)\b", re.I)


def sentences(text: str) -> list:
    """LaTeX split into sentences, with the constructs that fake a full stop neutralised."""
    t = re.sub(r"(?m)^\s*%.*$", "", text)                    # comments
    t = re.sub(r"\\(?:ref|eqref|label|cref)\{[^}]*\}", "REF", t)
    t = re.sub(r"\\begin\{(figure|table|tabular|equation|align|longtable)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(e\.g|i\.e|cf|vs|et al|Fig|Eq|Tab)\.", r"\1<DOT>", t)
    out = [s.strip().replace("<DOT>", ".") for s in re.split(r"(?<=[.!?])\s+", t)]
    return [s for s in out if s]


def parse_registry(path: Path) -> tuple:
    """Return (hypotheses, problems). A hypothesis with a missing field is not registered."""
    text = path.read_text()
    marks = list(HEADING.finditer(text))
    hyps, problems = {}, []
    for i, m in enumerate(marks):
        hid, title = m.group(1), m.group(2)
        body = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        entry = {"title": title, "fields": {}}
        for name, rx in FIELDS.items():
            entry["fields"][name] = bool(rx.search(body))
            if not entry["fields"][name]:
                problems.append(f"{hid} declares no {name}")
        size = FAMILY_SIZE.search(body[body.lower().find("family"):]) if \
            entry["fields"]["family"] else None
        entry["family_size"] = int(size.group(1)) if size else None
        if entry["fields"]["family"] and entry["family_size"] is None:
            problems.append(f"{hid} declares a family with no size")
        hyps[hid] = entry
    if not hyps:
        problems.append(f"no hypothesis headings found in {path}")
    return hyps, problems


def scan_vocabularies(text: str, counts: dict) -> list:
    """Sentences quoting a type count without saying which vocabulary it belongs to."""
    if not counts:
        return []
    # A markdown table row has no full stop, so sentence splitting swallows a whole table into
    # one unit and the report points at the table instead of the row. Flowing prose has the
    # opposite problem: split it by line and a sentence that names its vocabulary on one line
    # and its count on the next reads as a violation. So table rows are units, prose is
    # sentences.
    units, prose = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            units.append(line)
        else:
            prose.append(line)
    units.extend(sentences("\n".join(prose)))
    bad = []
    for s in units:
        # markdown emphasis sits between the two words of `**signature** vocabulary', so it is
        # removed before matching rather than written into the pattern
        flat = re.sub(r"[*_`]+", "", s)
        if not TYPE_WORD.search(flat) or VOCAB_WORDS.search(flat):
            continue
        hits = sorted({n for n in _numbers(flat) if n in counts})
        if hits:
            bad.append({"sentence": s[:180],
                        "counts": [[n, counts[n]] for n in hits]})
    return bad


def scan(text: str, hyps: dict, aliases: dict | None = None,
         tabled: set | None = None) -> dict:
    aliases, tabled = aliases or {}, tabled or set()
    scanned, claims, tagged, attributed, unregistered, unknown = 0, [], [], [], [], []
    derivations = []
    claimed, failed, mentioned = set(), set(), set()
    declared_unreported = set()
    for s in sentences(text):
        scanned += 1
        ids = {a or b for a, b in TAG.findall(s)}
        ids |= {aliases[p] for p in ALIAS_TAG.findall(s) if p in aliases}
        known = {i for i in ids if i in hyps}
        mentioned |= known
        if FAILED.search(s):
            failed |= known
        if NOT_RUN.search(s):
            declared_unreported |= known
        if RETRIEVAL.search(s):
            continue
        if not (EFFECT.search(s) or EFFECT_NUMBER.search(s)):
            continue
        claims.append(s)
        if ids:
            bad = sorted(i for i in ids if i not in hyps)
            (unknown if bad else tagged).append((sorted(ids), s))
            if not bad:
                claimed |= known
            continue
        if DERIVATION.search(s):
            derivations.append(s)
            continue
        (attributed if CITED.search(s) else unregistered).append(s)

    # backwards: a registered hypothesis has to carry a claim or be reported as failed
    outcome, absent, silent = {}, [], []
    for h in sorted(hyps):
        if h in claimed:
            outcome[h] = "claimed"
        elif h in failed:
            outcome[h] = "reported as failed"
        elif h in tabled:
            outcome[h] = "reported in the hypotheses table"
        elif h in declared_unreported:
            outcome[h] = "declared as not run in this paper"
        elif h in mentioned:
            outcome[h] = "mentioned with no outcome"
            silent.append(h)
        else:
            outcome[h] = "absent from the text"
            absent.append(h)

    return {"sentences_scanned": scanned, "effect_sentences": len(claims),
            "tagged": tagged, "attributed": attributed,
            "unregistered": unregistered, "unknown_hypothesis": unknown,
            "derivations": derivations,
            "outcome": outcome, "absent": absent, "no_outcome": silent}


def report(hyps: dict, problems: list, res: dict, quiet: bool = False,
           have_text: bool = True, backwards: bool = True, forward: str = "fail") -> int:
    """`forward` is "fail" or "report".

    The register in this project covers the deployed choices: which ranking rule, which cap,
    which budget, which emission policy. The paper also reports measurements that are not
    deployed choices -- ablations, baselines, transfers -- and those were never predictable in
    advance, so requiring each to name a hypothesis is a category error that turns the forward
    direction permanently red and therefore unread. Under "report" they are counted and listed
    and do not fail the run. The backwards direction stays a hard gate under both, because the
    violation it catches -- a prediction that was made, did not hold, and then left the paper --
    is the one preregistration exists to prevent.
    """
    # With no manuscript to read, the backwards direction has nothing to say: every
    # hypothesis is trivially absent. Validating the registry alone is a real check and is
    # what this reports before the paper exists; it must not masquerade as the full one.
    # The forward direction reads whatever text it is given, a section included. The backwards
    # direction is a statement about a COMPLETE manuscript: a hypothesis missing from one
    # section has not gone missing from the paper.
    ok = not (problems or (forward == "fail" and res["unregistered"]) or res["unknown_hypothesis"]
              or res.get("vocabulary")
              or (have_text and backwards and (res["absent"] or res["no_outcome"])))
    if quiet:
        return 0 if ok else 1
    print(f"  registry: {len(hyps)} hypotheses "
          f"({', '.join(sorted(hyps)) if hyps else 'none'})")
    for h, e in sorted(hyps.items()):
        print(f"    {h}  family size {e['family_size']}  {e['title'][:58]}")
    for p in problems:
        print(f"  REGISTRY FAIL: {p}")
    for v in res.get("vocabulary", []):
        named = ", ".join(f"{n} is a {voc} count" for n, voc in v["counts"])
        print(f"  FAIL: a type count with no vocabulary named ({named}):")
        print(f"        {v['sentence'][:110]}")
    if not have_text:
        print("  manuscript: none given, so the registry alone is checked")
        print("check_prereg: registry OK" if ok else "check_prereg: FAILURES ABOVE")
        return 0 if ok else 1
    print(f"  manuscript: {res['sentences_scanned']} sentences scanned, "
          f"{res['effect_sentences']} claim an effect")
    print(f"    registered {len(res['tagged'])}   attributed to others "
          f"{len(res['attributed'])}   unregistered {len(res['unregistered'])}"
          f"   derivations {len(res.get('derivations', []))}")
    for s in res.get("derivations", []):
        print(f"  exempt as a derivation: {s[:110]}")
    for ids, s in res["unknown_hypothesis"]:
        print(f"  FAIL: names {','.join(ids)}, which the registry does not: {s[:100]}")
    label = "FAIL: claims an effect with no hypothesis" if forward == "fail" else \
        "outside the register (a measurement, not a deployed choice)"
    for s in res["unregistered"]:
        print(f"  {label}: {s[:100]}")
    if backwards:
        for h in res["absent"]:
            print(f"  FAIL: {h} is registered and never appears; a hypothesis leaves the paper "
                  f"only by being reported as having failed")
        for h in res["no_outcome"]:
            print(f"  FAIL: {h} is named but carries neither a claim nor an outcome")
    else:
        print("    the text is a section, so the backwards direction is not read")
    if not quiet and backwards:
        for h, o in sorted(res["outcome"].items()):
            print(f"    {h}: {o}")
    print("check_prereg: OK" if ok else "check_prereg: FAILURES ABOVE")
    return 0 if ok else 1


FIXTURE_REGISTRY = """# Preregistration (fixture)

## H1 — soft admissibility beats the hard gate
**Prediction.** Soft admissibility beats the hard gate on macro recall@15.
**Failure:** the contrast is at most zero.
**Family:** H1 and H5 together, m = 2.

## H2 — informed node features
**Prediction.** Reactivity and accessibility improve within-type site ordering.
**Failure:** the standardised inequality does not hold.
**Family:** H2 alone, m = 1.

## H3 — learned abstention
**Prediction.** A stop logit reaches the tuned rule's macro F1 with no alpha.
**Failure:** the learned stop is worse than the tuned constant.
**Family:** H3 alone, m = 1.
"""

FIXTURE_TEXT = r"""
Soft admissibility raises macro recall@15 by $+0.031$ \prereg{H1}.
The pool holds 82 candidates per substrate on average.
Node features improve the within-type ordering (H2).
Removing the selector raises recall by $+0.176$.
\citet{Larsson_2025} report precision that beats GLORYx by a wide margin.
The learned stop did not hold against the tuned constant, so H3 is reported as failed.
"""


def self_test() -> int:
    """The checker has to fail on the things it exists to catch, so it is shown failing.

    Three perturbations, each expected to be caught by name: an untagged effect sentence, a
    tag naming a hypothesis the registry does not carry, and a registry entry whose failure
    condition has been deleted.
    """
    ok = True
    with tempfile.TemporaryDirectory() as d:
        reg = Path(d) / "prereg.md"
        reg.write_text(FIXTURE_REGISTRY)
        hyps, problems = parse_registry(reg)
        if sorted(hyps) != ["H1", "H2", "H3"] or problems:
            print(f"FAIL: the fixture registry did not parse: {sorted(hyps)} {problems}")
            ok = False
        if hyps.get("H1", {}).get("family_size") != 2:
            print(f"FAIL: family size not read: {hyps.get('H1')}"); ok = False

        res = scan(FIXTURE_TEXT, hyps)
        if len(res["tagged"]) != 2:
            print(f"FAIL: expected 2 registered claims, got {len(res['tagged'])}"); ok = False
        if len(res["unregistered"]) != 1 or "0.176" not in res["unregistered"][0]:
            print(f"FAIL: the untagged +0.176 sentence was not caught: "
                  f"{res['unregistered']}")
            ok = False
        if len(res["attributed"]) != 1:
            print(f"FAIL: the cited sentence was not counted as attributed: "
                  f"{len(res['attributed'])}")
            ok = False
        if report(hyps, problems, res, quiet=True) == 0:
            print("FAIL: the checker passed a manuscript with an unregistered claim"); ok = False

        res_h9 = scan(r"Typing lifts recall \prereg{H9}.", hyps)
        if not res_h9["unknown_hypothesis"]:
            print("FAIL: a tag naming an unregistered hypothesis was accepted"); ok = False

        # backwards: H3 is registered, did not hold, and says so -- that is allowed
        if res["outcome"].get("H3") != "reported as failed":
            print(f"FAIL: a hypothesis reported as failed was not recognised: "
                  f"{res['outcome'].get('H3')}")
            ok = False
        # and a hypothesis that simply vanished is not
        gone = scan(FIXTURE_TEXT.replace(
            "The learned stop did not hold against the tuned constant, so H3 is reported "
            "as failed.\n", ""), hyps)
        if gone["absent"] != ["H3"]:
            print(f"FAIL: a registered hypothesis that vanished was not caught: "
                  f"{gone['absent']}")
            ok = False
        # named in passing, with neither a claim nor an outcome, is not either
        passing = scan("The design of H3 follows the same shape.", hyps)
        if passing["no_outcome"] != ["H3"]:
            print(f"FAIL: a hypothesis named with no outcome was accepted: "
                  f"{passing['no_outcome']}")
            ok = False

        reg.write_text(FIXTURE_REGISTRY.replace("**Failure:** the contrast is at most zero.\n",
                                                ""))
        _, problems2 = parse_registry(reg)
        if not any("H1 declares no failure" in p for p in problems2):
            print(f"FAIL: a hypothesis with no failure condition was accepted: {problems2}")
            ok = False

        reg.write_text(FIXTURE_REGISTRY.replace("**Family:** H2 alone, m = 1.",
                                                "**Family:** H2 alone."))
        _, problems3 = parse_registry(reg)
        if not any("no size" in p for p in problems3):
            print(f"FAIL: a family with no size was accepted: {problems3}"); ok = False

        # the vocabulary check, on counts declared here rather than read from artifacts
        counts = {4417: "bond-delta", 4456: "signature"}
        named = scan_vocabularies(
            "The bond-delta vocabulary holds 4,417 types.\n"
            "| the **signature** vocabulary | 4,456 types |", counts)
        if named:
            print(f"FAIL: a count that names its vocabulary was flagged: {named}"); ok = False
        unnamed = scan_vocabularies("The bank holds 4,417 types.", counts)
        if len(unnamed) != 1 or unnamed[0]["counts"] != [[4417, "bond-delta"]]:
            print(f"FAIL: an unnamed type count was not caught: {unnamed}"); ok = False
        # a number that is not a type count is not the checker's business
        if scan_vocabularies("The split holds 1,170 types of thing.", counts):
            print("FAIL: a number that is not a type count was flagged"); ok = False

        # `recovered from the commit' is a file operation wearing an effect verb
        retrieval = scan("The producer is recovered from the commit that added the artifact.",
                         hyps)
        if retrieval["unregistered"]:
            print(f"FAIL: a retrieval was read as an effect claim: "
                  f"{retrieval['unregistered']}")
            ok = False
        if not scan("Typing recovers 40 references the bank missed.", hyps)["unregistered"]:
            print("FAIL: a real recovery claim was excluded with the retrievals"); ok = False

        # the backwards direction is a statement about a whole manuscript
        section = scan("Soft admissibility raises recall \\prereg{H1}.", hyps)
        if report(hyps, [], section, quiet=True, backwards=False) != 0:
            print("FAIL: a section was judged for hypotheses it does not mention"); ok = False
        if report(hyps, [], section, quiet=True, backwards=True) == 0:
            print("FAIL: a whole manuscript missing two hypotheses was accepted"); ok = False

    print("self-test: OK" if ok else "self-test: FAILURES ABOVE")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", help="the preregistration markdown")
    ap.add_argument("--text", nargs="*", default=[], help="manuscript files to scan")
    ap.add_argument("--partial", action="store_true",
                    help="the text is a section and not the whole manuscript, so a hypothesis "
                         "it does not mention is not a hypothesis that has gone missing")
    ap.add_argument("--json", help="where to write the report")
    ap.add_argument("--forward", choices=("fail", "report"), default="fail",
                    help="whether an effect sentence naming no hypothesis fails the run or is "
                         "reported; the register covers the deployed choices and the paper also "
                         "reports measurements that were never deployed choices")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.prereg:
        ap.error("--prereg is required (or --self-test)")

    hyps, problems = parse_registry(Path(args.prereg))
    text = "\n".join(Path(t).read_text() for t in args.text)
    aliases = alias_map(ROOT)
    if aliases:
        print(f"reading {len(aliases)} P-identifiers as their register entries "
              f"({', '.join(f'{p}={h}' for p, h in sorted(aliases.items(), key=lambda x: int(x[0][1:])))})")
    res = scan(text, hyps, aliases, tabled_outcomes(ROOT))
    counts = type_counts(ROOT)
    # the registration is scanned for vocabulary too: it quotes these counts itself
    res["vocabulary"] = scan_vocabularies(
        text + "\n" + Path(args.prereg).read_text(), counts)
    code = report(hyps, problems, res, have_text=bool(args.text),
                  backwards=not args.partial, forward=args.forward)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"prereg": args.prereg, "text": args.text, "hypotheses": hyps,
             "registry_problems": problems,
             "counts": {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()},
             "unregistered": res["unregistered"],
             "unknown_hypothesis": [[i, s] for i, s in res["unknown_hypothesis"]]}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
