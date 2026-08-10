#!/usr/bin/env python3
"""How widely does the hydrogen convention divide published reaction-template libraries?

The engine result rests on three banks, which is enough to show that reach depends on the procedure
and not enough to say anything about the literature. This widens the census to every template
library reachable without re-deriving one: the rule sets two published tools ship with their own
distributions, the templates extracted from a standard reaction corpus, and the partitions of this
work's own bank.

Three SMARTS constructs are counted separately, because they behave differently under the one
preprocessing step at issue and the paper's earlier phrasing ran them together:

    atom      the hydrogen ATOM primitive, [H] or [#1] as a node. Matches only a drawn hydrogen,
              so a template carrying one cannot fire on a substrate whose hydrogens are implicit.
    count     the H<n> primitive inside a bracket, as in [CH3]. Pins a total hydrogen count on the
              matched heavy atom and is unchanged by expansion.
    degree    the D<n> primitive, which pins explicit connections. Expansion adds connections, so
              a template pinning degree stops matching once hydrogens are drawn.

Only the first two are about hydrogen at all, and only the first and third move under the step. A
library can therefore be damaged by expansion, by its absence, or by neither, and which one is a
property of how its templates were written rather than of the chemistry they encode.

No rule set is downloaded. Libraries that ship with an installed tool are read from that
installation, and the report records where each came from so a rerun can be checked.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from explicit_h_mechanism import needs_explicit_hydrogen
from provenance_knob_attribution import pins_degree, pins_hydrogen

_ATOM_TOKEN = re.compile(r"\[[^\]]*\]")


def _reactant_side(rule: str) -> str:
    return rule.split(">>")[0]


def census(rules: list[str]) -> dict:
    atom = count = degree = both = neither = recursive = 0
    for r in rules:
        left = _reactant_side(r)
        a = any(needs_explicit_hydrogen(t) for t in _ATOM_TOKEN.findall(left))
        c = pins_hydrogen(r)
        d = pins_degree(r)
        atom += a
        count += c
        degree += d
        both += a and d
        neither += not (a or c or d)
        recursive += "$(" in left
    n = max(len(rules), 1)
    return {"templates": len(rules),
            "hydrogen_atom_primitive": atom, "share_atom": round(atom / n, 4),
            "hydrogen_count_primitive": count, "share_count": round(count / n, 4),
            "degree_primitive": degree, "share_degree": round(degree / n, 4),
            "atom_and_degree": both, "none_of_the_three": neither,
            "recursive_smarts": recursive,
            # A library is exposed to the expansion when a template either needs a drawn hydrogen
            # to match or stops matching once hydrogens are drawn. The two point opposite ways.
            "exposed_needs_expansion": atom, "exposed_broken_by_expansion": degree}


def _sygma_rules() -> dict:
    """SyGMa ships its rule set inside the installed package; nothing is fetched."""
    spec = importlib.util.find_spec("sygma")
    if spec is None or not spec.submodule_search_locations:
        return {}
    base = Path(spec.submodule_search_locations[0]) / "rules"
    out = {}
    for f in sorted(base.glob("*.txt")):
        rules = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split("\t")[0].split()[0]
            if ">>" in first:
                rules.append(first)
        if rules:
            out[f"SyGMa {f.stem}"] = (rules, f"installed package: sygma/rules/{f.name}")
    return out


def _uspto(path: Path) -> tuple[list[str], str] | None:
    if not path.exists():
        return None
    rules = []
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i == 0 and ">>" not in line:
                continue
            for field in line.rstrip("\n").split(","):
                if ">>" in field:
                    rules.append(field.strip().strip('"'))
                    break
    return (rules, str(path.relative_to(ROOT))) if rules else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "convention_census.json"))
    args = ap.parse_args()

    def read(path: Path) -> list[str]:
        return [l.strip() for l in path.read_text().splitlines() if l.strip() and ">>" in l]

    banks: dict[str, tuple[list[str], str]] = {}
    res = ROOT / "grail_metabolism" / "resources"
    full = read(res / "extended_smirks.txt")
    mined = set(read(res / "mined_only.txt"))
    banks["this work, full bank"] = (full, "resources/extended_smirks.txt")
    banks["this work, mined half"] = ([r for r in full if r in mined], "mined partition")
    banks["this work, curated half"] = ([r for r in full if r not in mined], "curated partition")
    banks.update(_sygma_rules())
    u = _uspto(ROOT / "grail_metabolism" / "uspto_templates.csv.gz")
    if u:
        banks["USPTO extracted templates"] = u
    # Three further files in this repository hold the hand-written collections the curated half was
    # assembled from. They are nested: the smallest sits entirely inside the other two, and all
    # three sit entirely inside the curated partition already counted. Counting them as libraries
    # would report the same templates up to four times, so they are recorded as nested and
    # excluded from the totals.
    nested = {}
    curated_set = {r for r in full if r not in mined}
    for name, rel in (("hand-written collection, largest", "resources/notebooks_rules.txt"),
                      ("hand-written collection, middle", "data/merged_smirks.txt"),
                      ("hand-written collection, smallest", "data/smirks.txt")):
        q = ROOT / "grail_metabolism" / rel
        if q.exists():
            r = read(q)
            if r:
                nested[name] = {"templates": len(r), "source": rel,
                                "share_inside_curated_partition": round(
                                    len(set(r) & curated_set) / len(r), 4)}

    # BioTransformer ships its templates with the tool, which is not installed here; its census was
    # taken from that distribution and is carried forward rather than recomputed.
    prior = ROOT / "results" / "explicit_h_mechanism.json"
    if prior.exists():
        bt = json.loads(prior.read_text())["hydrogen_convention_by_bank"].get("biotransformer")
        if bt:
            banks["BioTransformer"] = (None, "third-party distribution, census carried forward")

    rep = {"config": {"script": pathlib.Path(__file__).name,
                      "note": "no rule set is downloaded; installed tools are read in place"},
           "banks": {}}
    print(f"{'library':30}{'templates':>10}{'atom [H]':>10}{'count H<n>':>12}{'degree D<n>':>13}")
    for name, (rules, source) in banks.items():
        if rules is None:
            bt = json.loads((ROOT / "results" / "explicit_h_mechanism.json").read_text())
            v = bt["hydrogen_convention_by_bank"]["biotransformer"]
            c = {"templates": v["rules"], "hydrogen_atom_primitive": v["with_explicit_hydrogen"],
                 "share_atom": round(v["with_explicit_hydrogen"] / v["rules"], 4),
                 "hydrogen_count_primitive": None, "share_count": None,
                 "degree_primitive": None, "share_degree": None,
                 "recursive_smarts": v.get("unclassified_recursive_smarts"),
                 "partial": "atom primitive only; the tool is not installed here"}
        else:
            c = census(rules)
        c["source"] = source
        rep["banks"][name] = c
        fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else "n/a"
        print(f"{name:30}{c['templates']:>10}{fmt(c['share_atom']):>10}"
              f"{fmt(c['share_count']):>12}{fmt(c['share_degree']):>13}")

    # Independent sources, not rows: this work's two partitions are inside its own bank, and
    # SyGMa ships its rule set as two phase files that together are the 175 it publishes.
    INDEPENDENT = {"this work, full bank": "this work",
                   "SyGMa phase1": "SyGMa", "SyGMa phase2": "SyGMa",
                   "USPTO extracted templates": "USPTO", "BioTransformer": "BioTransformer"}
    sources = sorted(set(INDEPENDENT.values()))
    n_lib = len(sources)
    n_tpl = sum(v["templates"] for k, v in rep["banks"].items() if k in INDEPENDENT)
    rep["nested_not_counted"] = nested
    exposed_up = [k for k, v in rep["banks"].items() if (v["share_atom"] or 0) > 0.05]
    exposed_dn = [k for k, v in rep["banks"].items() if (v["share_degree"] or 0) > 0.05]
    print(f"\n  {n_lib} independent libraries ({', '.join(sources)}), {n_tpl} templates")
    print(f"  {len(exposed_up)} carry the atom primitive in more than a twentieth of their "
          f"templates: {exposed_up}")
    print(f"  {len(exposed_dn)} pin degree in more than a twentieth: {exposed_dn}")
    rep["summary"] = {"independent_libraries": n_lib, "sources": sources,
                      "templates": n_tpl,
                      "need_expansion": exposed_up, "broken_by_expansion": exposed_dn}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
