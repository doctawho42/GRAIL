#!/usr/bin/env python3
"""Phase 3: every quoted number's old and new value, and which of them moved beyond its interval.

Writes revision/CHANGELOG.md.

The manuscript reaches its numbers only through macros generated from results/paper2_numbers.json,
so this is a key-by-key diff of that file rather than a scan of prose: a number that is not in it
cannot appear in the paper, which makes the file the whole population of quoted values.

What this can and cannot judge. Of its 2,459 keys, 261 carry `.lo` and `.hi` siblings and the rest
carry no interval at all. A value with an interval is flagged when it moves by more than the
half-width of its own interval. A value without one is reported as moved and marked as impossible
to judge -- not as unflagged, which would read as checked and within tolerance across nine tenths
of the file. Values that are not numbers, the "yes"/"no" verdicts and the ISO dates, are reported
as changed and never compared arithmetically.

    python revision/phase3_changelog.py                       # against HEAD
    python revision/phase3_changelog.py --old-rev 8999435~1    # against the pre-revision state
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = "results/paper2_numbers.json"
OUT = ROOT / "revision" / "CHANGELOG.md"


def load_numbers(path):
    """The flat dotted-key map, read through the envelope that holds it.

    The file is {n_numbers, numbers, provenance}; the values are under `numbers`. Reading the top
    level would yield a count, a dict and a provenance block as though they were three quoted
    numbers, which is the envelope-for-substrate-map mistake in a different costume.
    """
    blob = json.loads(Path(path).read_text())
    inner = blob.get("numbers") if isinstance(blob, dict) else None
    return inner if isinstance(inner, dict) else blob


GENERATOR = "scripts/paper2_numbers.py"


def generator_inputs(source=None):
    """Every artifact the number generator reads, enumerated from its own source.

    Read from the generator rather than from a list kept beside it: a list next to data that has
    its own keys goes false the first time the generator gains an input.

    Two of its `art()` call sites take a variable instead of a literal -- both loop over literal
    tuples of filenames -- so a pattern matching only `art("name")` undercounts, and
    `case_study_drawn.json` is named nowhere else. Every JSON literal in the source that resolves
    under results/ is therefore included too. The authoritative enumeration is the runtime
    instrumentation in scripts/check_number_provenance.py, which counts what was actually read
    rather than what a pattern found; this is the static approximation of it, deliberately erring
    towards including too much rather than too little.
    """
    src = Path(source or (ROOT / GENERATOR)).read_text()
    direct = set(re.findall(r'art\(\s*"([^"]+)"', src))
    literals = set(re.findall(r'"([A-Za-z0-9_./-]+\.json)"', src))
    resolved = {s for s in literals if (ROOT / "results" / s).exists()}
    # The generator's own output is a literal in its source as well, and counting it as an input
    # would have this changelog report that it checked the very file it diffs -- circular, and an
    # over-count that reads as thoroughness. Widening the net for the loop-tuple names catches it
    # too, so it is removed by name rather than by hoping the pattern misses it.
    output = {NUMBERS, Path(NUMBERS).name}
    return sorted((direct | resolved) - output)


def inputs_changed_since(rev, inputs=None):
    """Which of the generator's inputs moved since a named revision, as repository paths."""
    inputs = generator_inputs() if inputs is None else inputs
    moved = []
    for name in inputs:
        rel = name if name.startswith(("results/", "artifacts/")) else f"results/{name}"
        out = subprocess.run(["git", "log", "--oneline", f"{rev}..HEAD", "--", rel],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            moved.append(rel)
    return moved


def _numeric(v):
    """True for a value an arithmetic comparison means something for. Bools are not numbers here."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _half_width(key, new, old):
    """Half the width of this key's own interval, or None when it has no interval.

    The siblings are `<key>.lo` and `<key>.hi`. The new map is consulted first so a re-run's own
    interval governs, falling back to the old one when the key has gone.
    """
    for src in (new, old):
        lo, hi = src.get(f"{key}.lo"), src.get(f"{key}.hi")
        if _numeric(lo) and _numeric(hi):
            return abs(hi - lo) / 2.0
    return None


def compare(old, new):
    """Every key that changed, arrived or went, with what can be said about the size of the move.

    Unchanged keys are omitted. `moved_beyond_interval` is True, False, or None for "cannot be
    judged", and the third case is a real answer rather than a missing one.
    """
    entries = []
    for key in sorted(set(old) | set(new)):
        was, now = old.get(key), new.get(key)
        if key in old and key in new:
            if was == now:
                continue
            kind = "changed"
        elif key in new:
            kind = "added"
        else:
            kind = "removed"

        comparable = kind == "changed" and _numeric(was) and _numeric(now)
        half = _half_width(key, new, old) if comparable else None
        delta = (now - was) if comparable else None
        entries.append({
            "key": key,
            "kind": kind,
            "old": was if key in old else None,
            "new": now if key in new else None,
            "comparable": bool(comparable),
            "delta": delta,
            "half_width": half,
            "interval": ([old.get(f"{key}.lo"), old.get(f"{key}.hi")]
                         if half is not None else None),
            "moved_beyond_interval": (None if (half is None or delta is None)
                                      else bool(abs(delta) > half)),
        })
    return entries


def render(entries, inputs_checked=None, inputs_changed=None):
    """The changelog, with nothing dropped and the unjudgeable counted rather than omitted.

    `inputs_checked` and `inputs_changed` turn an empty table into a result: "no quoted number
    changed" on its own cannot be told apart from a diff that never read anything, and the
    stronger statement is that every input to the generator was checked and none of them moved.
    """
    flagged = [e for e in entries if e["moved_beyond_interval"] is True]
    unjudged = [e for e in entries if e["kind"] == "changed"
                and e["moved_beyond_interval"] is None]
    within = [e for e in entries if e["moved_beyond_interval"] is False]
    added = [e for e in entries if e["kind"] == "added"]
    removed = [e for e in entries if e["kind"] == "removed"]

    lines = [
        "# Changelog of quoted numbers",
        "",
        "Every value the manuscript can print, diffed key by key from "
        f"`{NUMBERS}`. A number absent from that file cannot appear in the paper, so this is the "
        "whole population of quoted values rather than a sample of it.",
        "",
        f"- changed: {len([e for e in entries if e['kind'] == 'changed'])}",
        f"- moved beyond its own interval half-width: {len(flagged)}",
        f"- moved within it: {len(within)}",
        f"- moved but cannot be judged (no interval, or not a number): {len(unjudged)}",
        f"- added: {len(added)}",
        f"- removed: {len(removed)}",
        "",
    ]
    if inputs_checked is not None:
        lines += [
            f"- generator inputs checked: {inputs_checked}",
            f"- generator inputs that moved: {len(inputs_changed or [])}",
            "",
        ]
        if inputs_changed:
            lines.append("Inputs that moved:")
            lines += [f"- `{p}`" for p in inputs_changed]
            lines.append("")
        else:
            lines += [
                f"No input to `{GENERATOR}` moved, so no quoted value could have. That is what "
                "makes the table above a result rather than a check that never looked.",
                "",
            ]
    lines += [
        "A value with no interval is reported as moved and marked as one this check cannot judge. "
        "It is not reported as unflagged: 2,198 of the 2,459 keys carry no interval, and calling "
        "those unflagged would read as checked and within tolerance.",
        "",
    ]

    def table(title, rows, note=None):
        if not rows:
            return
        lines.append(f"## {title}")
        lines.append("")
        if note:
            lines.append(note)
            lines.append("")
        lines.append("| key | old | new | delta | half-width | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for e in rows:
            hw = "-" if e["half_width"] is None else f"{e['half_width']:.4g}"
            dl = "-" if e["delta"] is None else f"{e['delta']:+.4g}"
            verdict = {True: "beyond interval", False: "within interval",
                       None: ("not a number" if not e["comparable"] and e["kind"] == "changed"
                              else "cannot be judged")}[e["moved_beyond_interval"]]
            if e["kind"] != "changed":
                verdict = e["kind"]
            lines.append(f"| `{e['key']}` | {e['old']} | {e['new']} | {dl} | {hw} | {verdict} |")
        lines.append("")

    table("Moved beyond its own interval", flagged,
          "These are the values whose movement exceeds the half-width of their own confidence "
          "interval, which is the threshold this phase was asked to flag.")
    table("Moved, but cannot be judged against an interval", unjudged)
    table("Moved within its own interval", within)
    table("Added", added)
    table("Removed", removed)
    if not entries:
        lines += ["No quoted number changed.", ""]
    return "\n".join(lines)


def _numbers_at(rev):
    """The number map as of a git revision, so the baseline is named rather than assumed."""
    out = subprocess.run(["git", "show", f"{rev}:{NUMBERS}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"REFUSING: cannot read {NUMBERS} at {rev}: {out.stderr.strip()}")
    blob = json.loads(out.stdout)
    inner = blob.get("numbers") if isinstance(blob, dict) else None
    return inner if isinstance(inner, dict) else blob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rev", default="HEAD",
                    help="the revision to compare against; the baseline is named, not assumed")
    ap.add_argument("--old", default=None, help="a path, overriding --old-rev")
    ap.add_argument("--new", default=str(ROOT / NUMBERS))
    args = ap.parse_args()

    old = load_numbers(args.old) if args.old else _numbers_at(args.old_rev)
    new = load_numbers(args.new)
    entries = compare(old, new)
    inputs = generator_inputs()
    moved = inputs_changed_since(args.old_rev, inputs) if not args.old else None
    OUT.write_text(render(entries, inputs_checked=len(inputs), inputs_changed=moved))

    print(f"baseline: {args.old or args.old_rev}")
    print(f"  generator inputs {len(inputs)}, of which moved "
          f"{'not checked (--old given)' if moved is None else len(moved)}")
    print(f"  old keys {len(old)}  new keys {len(new)}  entries {len(entries)}")
    for kind in ("changed", "added", "removed"):
        print(f"  {kind:8} {len([e for e in entries if e['kind'] == kind])}")
    print(f"  beyond interval {len([e for e in entries if e['moved_beyond_interval'] is True])}")
    print(f"  cannot be judged "
          f"{len([e for e in entries if e['kind'] == 'changed' and e['moved_beyond_interval'] is None])}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
