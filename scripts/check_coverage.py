#!/usr/bin/env python3
r"""Which printed numbers does no check read?

``verify_paper_numbers.py`` answers "does this figure follow from the artifact it cites". It cannot
answer the prior question --- whether any check looks at a given figure at all --- and that is where
the expensive defects have lived: a median printed as $0.85$ in a sentence while the figure drawn
from the same artifacts labelled itself $0.83$, three printings of one decomposition with three
different values, a table row for nine boards that stopped summing to its own total. Every one of
them sat in a green suite because no check's pattern reached it.

So this instruments the checker rather than the manuscript. Every regular-expression call the
checker makes against the flattened manuscript is recorded with the span it matched, the spans are
unioned, and every numeral in the body that falls outside the union is reported. A numeral inside
the union is not thereby verified --- a pattern can match a sentence and test a different number in
it --- but a numeral outside it is certainly unchecked, which is the question being asked.

Reports the body by default, since that is what a reader quotes; ``--appendix`` covers the rest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "grail_iclr.tex"


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


class Recorder:
    """Wraps ``re`` so every match against the flattened manuscript leaves a span behind."""

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []
        self._orig = {k: getattr(re, k) for k in ("search", "finditer", "findall", "match")}

    def _note(self, string: str, m) -> None:
        # only calls against a long haystack are the manuscript; a regex over a filename is not
        if m is not None and isinstance(string, str) and len(string) > 20000:
            self.spans.append(m.span())

    def install(self) -> None:
        rec = self

        def search(pattern, string, flags=0):
            m = rec._orig["search"](pattern, string, flags)
            rec._note(string, m)
            return m

        def match(pattern, string, flags=0):
            m = rec._orig["match"](pattern, string, flags)
            rec._note(string, m)
            return m

        def finditer(pattern, string, flags=0):
            for m in rec._orig["finditer"](pattern, string, flags):
                rec._note(string, m)
                yield m

        def findall(pattern, string, flags=0):
            # findall discards spans, so it is served from finditer and rebuilt the way re does
            out = []
            for m in rec._orig["finditer"](pattern, string, flags):
                rec._note(string, m)
                g = m.groups()
                out.append(m.group(0) if not g else (g[0] if len(g) == 1 else g))
            return out

        re.search, re.match, re.finditer, re.findall = search, match, finditer, findall

    def restore(self) -> None:
        for k, v in self._orig.items():
            setattr(re, k, v)

    def covered(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for a, b in sorted(self.spans):
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix", action="store_true", help="report the appendices too")
    ap.add_argument("--context", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "results" / "check_coverage.json"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("_vpn", ROOT / "scripts"
                                                  / "verify_paper_numbers.py")
    vpn = importlib.util.module_from_spec(spec)
    sys.modules["_vpn"] = vpn

    rec = Recorder()
    rec.install()
    try:
        spec.loader.exec_module(vpn)
        argv = sys.argv
        sys.argv = ["verify_paper_numbers.py"]
        try:
            vpn.main()
        finally:
            sys.argv = argv
    finally:
        rec.restore()

    # the same flattening the checker uses, so the spans it left are indices into this string
    tex = [PAPER] + sorted((ROOT / "paper" / "app").glob("*.tex"))
    whole = "\n".join(p.read_text() for p in tex)
    flat = re.sub(r"\s+", " ", whole)
    # everything before \appendix in the main file is what a reader reads without turning to the back
    body_end = flat.index("\\appendix") if "\\appendix" in flat else len(flat)

    covered = rec.covered()

    def is_covered(i: int) -> bool:
        for a, b in covered:
            if a <= i < b:
                return True
            if a > i:
                break
        return False

    rows = []
    for m in re.finditer(r"\$([+-]?\d[\d.,{}]*)\$", flat):
        start = m.start()
        in_body = start < body_end
        if not in_body and not args.appendix:
            continue
        if is_covered(start):
            continue
        rows.append({"value": m.group(1), "where": "body" if in_body else "appendix",
                     "context": flat[max(0, start - args.context):start + args.context]})

    total_body = sum(1 for m in re.finditer(r"\$([+-]?\d[\d.,{}]*)\$", flat)
                     if m.start() < body_end)
    rep = {"config": {**_code_version(),
                      "note": "a numeral inside a matched span is not thereby verified; a numeral "
                              "outside every span is certainly unread by any check",
                      "checker_regex_calls_recorded": len(rec.spans),
                      "characters_covered": sum(b - a for a, b in covered)},
           "n_body_numerals": total_body,
           "n_unread": len(rows),
           "unread": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(rec.spans)} regex matches recorded over the manuscript")
    print(f"  {len(rows)} of {total_body} body numerals fall outside every one of them")
    for r in rows[:40]:
        print(f"     ${r['value']}$  ...{r['context'][args.context - 34:args.context + 34]}...")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
