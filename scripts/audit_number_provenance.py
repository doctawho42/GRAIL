#!/usr/bin/env python3
"""Does the table that says where each number appears actually point at the number?

The paper carries a census of every recall figure it reports for the full-split methods, one row
per figure, saying which float or section states it. That table is the answer to "which $0.34$ is
this", and it is the one place a reader goes when two printings of the same method disagree. It is
also prose: nothing checked that a row's target contained its figure, and three rows did not --- all
three named a section anchor that had been defined twice, in two files, so the reference resolved to
whichever came last and landed in a subsection about something else. The figures were right and
unfindable, which is the same defect as a wrong figure for a reader trying to reproduce one.

So the check is mechanical. For each row, resolve the target label, take the span of source it
denotes --- the float environment for a table or figure, and everything up to the next sectioning
command at the same or a higher level for a section --- and require the row's figure to appear
literally inside it. Macros are expanded first, since the row and the text often disagree about
whether a number is spelled out or named.

The failure this cannot catch is a figure that appears in the right section by coincidence, and it
is left uncaught deliberately: tightening it would need to know which sentence is the statement of
that number rather than a mention of it, which is a judgement and not a check.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
LEVEL = {"section": 0, "subsection": 1, "subsubsection": 2}


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def flatten(path: Path, depth: int = 0) -> str:
    """The document as one string, with \\input resolved.

    A section can be opened in the main file and its body supplied by an \\input, which is how
    every appendix here is written. Reading the files separately puts the label in one string and
    the text it names in another, and every such target then reads as empty.
    """
    text = strip_comments(path.read_text())
    if depth > 6:
        return text
    def sub(m):
        child = (PAPER / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex")))
        return flatten(child, depth + 1) if child.exists() else m.group(0)
    return re.sub(r"\\input\{([^}]+)\}", sub, text)


def macros(text: str) -> dict:
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text)}


def expand(s: str, table: dict) -> str:
    for _ in range(4):
        new = re.sub(r"\\(\w+)\\?\s?", lambda m: table.get(m.group(1), m.group(0)), s)
        if new == s:
            break
        s = new
    return s


def spans(sources: dict) -> dict:
    """label -> the source text it denotes: a float's body, or a section down to the next one."""
    out = {}
    for text in sources.values():
        # floats: the label belongs to the environment that encloses it
        for m in re.finditer(r"\\begin\{(table|figure)\*?\}(.*?)\\end\{\1\*?\}", text, re.S):
            for lab in re.findall(r"\\label\{([^}]+)\}", m.group(2)):
                out[lab] = m.group(2)
        # sectioning: the label belongs to the unit it opens, which ends at the next unit of the
        # same or a higher level -- a subsection does not carry the section that follows it
        heads = [(m.start(), m.end(), LEVEL[m.group(1)])
                 for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{", text)]
        units = []
        for i, (_, end, lvl) in enumerate(heads):
            stop = len(text)
            for start2, _, lvl2 in heads[i + 1:]:
                if lvl2 <= lvl:
                    stop = start2
                    break
            units.append((lvl, text[end:stop]))
        # a label inside a subsection is inside its section too; the innermost unit is the one the
        # reference sends a reader to, so deeper units claim their labels first
        for _, body in sorted(units, key=lambda u: -u[0]):
            for lab in re.findall(r"\\label\{([^}]+)\}", body):
                out.setdefault(lab, body)
    return out


def _leaves_equal(obj, wanted: str, path: str = "") -> list:
    """Paths of numeric leaves equal to the printed figure at the precision it is printed to."""
    try:
        target = float(wanted)
    except ValueError:
        return []
    places = len(wanted.split(".")[1]) if "." in wanted else 0
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _leaves_equal(v, wanted, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _leaves_equal(v, wanted, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if round(float(obj), places) == target:
            out.append(path)
    return out


def census_rows(text: str, caption_label: str) -> list:
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text[text.index(caption_label):], re.S)
    if m is None:
        raise SystemExit(f"no tabular follows {caption_label}")
    rows = []
    for line in m.group(0).split("\\\\"):
        line = re.sub(r"\\(top|mid|bottom)rule", " ", line).strip()
        if "&" not in line or not re.search(r"\\(ref|texttt)\{", line):
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append((cells[0], re.findall(r"\\ref\{([^}]+)\}", cells[1]),
                     [n.replace("\\_", "_") for n in re.findall(r"\\texttt\{([^}]+)\}", cells[1])]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "number_provenance_audit.json"))
    args = ap.parse_args()

    sources = {"grail_iclr.tex": flatten(PAPER / "grail_iclr.tex")}
    table = macros("\n".join(sources.values()))
    where = spans(sources)
    rows = census_rows(sources["grail_iclr.tex"], "\\label{tab:provenance}")

    findings, bad = [], 0
    for figure, targets, artifacts in rows:
        method, _, value = figure.partition(" ")
        wanted = expand(value, table).strip("$ ")
        # the census prints a mean and its spread; the text states the same pair
        wanted = wanted.split("\\pm")[0].strip("$ ")
        seen, where_found = [], []
        for t in targets:
            body = where.get(t)
            if body is None:
                seen.append((t, "no such label"))
                continue
            seen.append((t, "found" if wanted in expand(body, table) else "absent"))
        # A row may source a figure to a released artifact instead, when the paper does not print
        # it. Substring containment was the obvious check and is close to vacuous: a three-decimal
        # string appears somewhere in a large JSON about a third of the time by chance, which makes
        # it far weaker than the section branch and not stronger. So the value has to equal a
        # numeric leaf at the precision the row prints it to, and the paths where it does are
        # recorded, because a reader who cannot find it has not been given provenance either.
        for name in artifacts:
            f = ROOT / "results" / f"{name}.json"
            if not f.exists():
                seen.append((name, "no such artifact"))
                continue
            paths = _leaves_equal(json.loads(f.read_text()), wanted)
            # Equality with some leaf is still not provenance: a large artifact holds many numbers
            # and several may round to the same three decimals. The row names a method, so require
            # the leaf to sit under it and to be the only such leaf. Perturbing the row to another
            # method's figure then fails, which substring containment and bare equality both allow.
            paths = [q for q in paths if method.lower() in q.lower()]
            found = len(paths) == 1
            seen.append((name, "found" if found else "absent"))
            if found:
                where_found.append({"artifact": name, "json_paths": paths[:8],
                                    "n_paths": len(paths)})
        # every target a row names has to hold the figure: a row naming two and carrying it in one
        # still sends half its readers to text that states something else
        ok = bool(seen) and all(s == "found" for _, s in seen)
        bad += not ok
        findings.append({"figure": figure, "method": method, "value": wanted,
                         "targets": [{"label": t, "verdict": s} for t, s in seen],
                         "artifact_paths": where_found, "ok": ok})

    rep = {"config": {**_code_version(), "table": "tab:provenance, in paper/app/xdomain.tex as inputted",
                      "rule": "the row's figure must appear literally inside the span its target "
                              "denotes: a float's body, or a sectioning unit down to the next unit "
                              "at the same or a higher level"},
           "n_rows": len(findings), "n_unreachable": bad, "rows": findings}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    for f in findings:
        if not f["ok"]:
            print(f"  UNREACHABLE {f['figure']:28s} -> "
                  + ", ".join(f"{t['label']} ({t['verdict']})" for t in f["targets"]))
    print(f"\n  {len(findings) - bad} of {len(findings)} census rows point at text or an artifact stating them")
    print(f"wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
