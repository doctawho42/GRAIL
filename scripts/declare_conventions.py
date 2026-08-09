#!/usr/bin/env python3
"""What have you not declared? A reader-facing check for the four choices this paper measures.

The paper's recommendation is one sentence -- declare the conventions -- and a recommendation that
takes a day to act on is a recommendation nobody acts on. Everything needed to act on this one is a
property of files that already exist, so this reads them and says what is undeclared. It trains
nothing, runs no model, and needs no reference data.

Two modes, matching the two kinds of file a paper in this area releases.

  --bank RULES.txt
      A reaction-template file. Reports which templates are convention-dependent and by which of the
      three mechanisms, and whether the bank is MIXED -- containing templates that want opposite
      conventions, so that no single global setting expresses it and any published coverage figure
      is the value of a (bank, convention) pair rather than of the bank.

  --predictions FILE [FILE ...]
      Two or more prediction files. Reports whether they are even on the same population, because a
      leaderboard is only a leaderboard within one, and how large a criterion's differential effect
      would have to be to reorder them -- the gap structure that decides whether an undeclared
      matching rule can move the table.

Neither mode tells you which convention is right. That is the point: the paper's finding is that the
choice is consequential and undeclared, not that one setting is correct. What this prints is what a
reader would otherwise have to take on trust.

Every test here is the one the paper measured with, imported rather than reimplemented, so a bank
this reports as mixed is mixed in exactly the sense Appendix "Released negative results" repairs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from explicit_h_mechanism import needs_explicit_hydrogen
from retro_template_convention import COUNTS_EXPLICIT

_ATOM_TOKEN = re.compile(r"\[[^\]]*\]")
_RECURSIVE = re.compile(r"\$\(")


def classify_template(rule: str) -> set[str]:
    """Which of the three convention mechanisms this one template is exposed to.

    Read off the reactant side only: what the product side spells changes what is built, not what
    matches, and conflating the two is how a census over-reports.
    """
    reactants = rule.split(">>")[0]
    tags = set()
    if any(needs_explicit_hydrogen(t) for t in _ATOM_TOKEN.findall(reactants)):
        tags.add("wants_expanded")          # [H] or [#1] matches nothing unless hydrogens are drawn
    if COUNTS_EXPLICIT.search(reactants):
        tags.add("breaks_under_expansion")  # D<n> counts connections, and expansion changes them all
    # Only a template the tests cannot DECIDE is unclassifiable. One carrying both [H] and a
    # recursive SMARTS is already decided by the first test, and counting it here would inflate the
    # residual category -- which is the category a dispatch policy has to guess at. This matches
    # explicit_h_mechanism.hydrogen_convention, so the tool and the paper census cannot disagree.
    if not tags and _RECURSIVE.search(reactants):
        tags.add("unclassifiable")
    return tags


def report_bank(path: Path) -> dict:
    rules = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    tags = [classify_template(r) for r in rules]
    n_exp = sum("wants_expanded" in t for t in tags)
    n_deg = sum("breaks_under_expansion" in t for t in tags)
    n_unk = sum("unclassifiable" in t for t in tags)
    plain = len(rules) - sum(1 for t in tags if t - {"unclassifiable"})
    mixed = n_exp > 0 and (n_deg > 0 or plain > 0)

    print(f"\n{path.name}: {len(rules)} templates")
    print(f"  {n_exp:>6} require the substrate's hydrogens to be DRAWN "
          f"(carry [H] or [#1]; they match nothing otherwise)")
    print(f"  {n_deg:>6} break when hydrogens ARE drawn "
          f"(constrain a degree, which every added hydrogen changes)")
    print(f"  {plain:>6} constrain neither, and are read the same way under either convention")
    print(f"  {n_unk:>6} carry recursive SMARTS, which no syntactic test can see inside")
    if mixed:
        print(f"\n  MIXED. This bank contains templates that want opposite conventions, so no single\n"
              f"  global setting expresses it: one group is mismeasured whichever is chosen. A\n"
              f"  coverage figure for this bank is the value of a (bank, convention) pair. Declare\n"
              f"  which, or dispatch the convention per template.")
    elif n_exp:
        print(f"\n  SINGLE CONVENTION: expanded. Every constrained template wants hydrogens drawn;\n"
              f"  applying this bank implicitly silently disables {n_exp} of its templates.")
    elif n_deg:
        print(f"\n  SINGLE CONVENTION: implicit. {n_deg} templates lose their match if hydrogens are\n"
              f"  drawn, and none needs them drawn.")
    else:
        print(f"\n  NO EXPOSURE FOUND by these tests. The hydrogen convention is inert for this file,\n"
              f"  which is worth declaring rather than assuming.")
    return {"file": str(path), "templates": len(rules), "wants_expanded": n_exp,
            "breaks_under_expansion": n_deg, "unconstrained": plain, "unclassifiable": n_unk,
            "mixed": bool(mixed)}


def _load_predictions(path: Path):
    """(key, ranked predictions) per row, from the shapes a released file usually takes."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        if isinstance(d, dict):
            return [(k, list(v)) for k, v in d.items()]
        if d and isinstance(d[0], dict):
            key = "product" if "product" in d[0] else ("sub" if "sub" in d[0] else None)
            if key:
                return [(r[key], list(r.get("preds") or r.get("candidates") or [])) for r in d]
        raise SystemExit(f"{path.name}: JSON shape not recognised; expected a mapping or a list of "
                         f"records carrying a target and its ranked predictions")
    rows, cur, target = [], [], None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t = (r.get("target") or r.get("PRODUCT") or "").strip()
            if not t:
                continue
            if t != target:
                if target is not None:
                    rows.append((target, cur))
                cur, target = [], t
            cur.append((r.get("reactants") or "").strip())
    if target is not None:
        rows.append((target, cur))
    return rows


def report_predictions(paths: list[Path]) -> dict:
    sets, sizes = {}, {}
    for p in paths:
        rows = _load_predictions(p)
        sets[p.name] = {t for t, _ in rows}
        sizes[p.name] = round(sum(len(v) for _, v in rows) / max(len(rows), 1), 1)
        print(f"  {p.name:34} {len(rows):>6} targets, {sizes[p.name]:>6} predictions each on average")

    print("\n  do these files describe the same population?")
    groups: list[list[str]] = []
    for n in sets:
        for g in groups:
            ov = len(sets[n] & sets[g[0]]) / max(1, min(len(sets[n]), len(sets[g[0]])))
            if ov >= 0.90:
                g.append(n)
                break
        else:
            groups.append([n])
    for a, b in itertools.combinations(sets, 2):
        ov = len(sets[a] & sets[b]) / max(1, min(len(sets[a]), len(sets[b])))
        flag = "" if ov >= 0.90 else "   <-- NOT the same population"
        print(f"    {a[:22]:24} vs {b[:22]:24} overlap {ov:.2f}{flag}")
    if len(groups) > 1:
        print(f"\n  {len(groups)} DISTINCT POPULATIONS among {len(paths)} files. Scores from different\n"
              f"  groups are not comparable and a table containing both is not a leaderboard.\n"
              f"  Groups: " + " | ".join(",".join(g) for g in groups))
    else:
        print("\n  one population; the files are comparable on that axis.")

    print("\n  output size is the second undeclared variable: a method emitting more is rewarded by\n"
          "  recall and punished by precision, so the spread below decides part of any ranking.")
    lo, hi = min(sizes.values()), max(sizes.values())
    print(f"    smallest {lo}, largest {hi}, ratio {hi / max(lo, 1e-9):.1f}x")
    if hi / max(lo, 1e-9) > 1.5:
        print("    Declare the budget these are compared at, or the comparison measures output size.")
    return {"files": {p.name: {"targets": len(sets[p.name]), "mean_predictions": sizes[p.name]}
                      for p in paths},
            "populations": groups, "mean_output_ratio": round(hi / max(lo, 1e-9), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", nargs="*", default=[], help="reaction-template files to inspect")
    ap.add_argument("--predictions", nargs="*", default=[], help="prediction files to compare")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    args = ap.parse_args()
    if not args.bank and not args.predictions:
        ap.error("give --bank, --predictions, or both")

    rep = {"banks": [], "predictions": None}
    for b in args.bank:
        rep["banks"].append(report_bank(Path(b)))
    if args.predictions:
        print(f"\n{len(args.predictions)} prediction files")
        rep["predictions"] = report_predictions([Path(p) for p in args.predictions])
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
