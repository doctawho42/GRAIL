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

    # Libraries retrieved from their own public distributions. Each is recorded with the file it
    # came from so the census can be re-taken against the same release.
    ext = res / "external"

    def _strip_comments(txt: str) -> str:
        return "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("//"))

    def _reactions(obj, out):
        if isinstance(obj, str) and ">>" in obj:
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _reactions(v, out)
        elif isinstance(obj, list):
            for v in obj:
                _reactions(v, out)

    bt = []
    for f in ("bt_database_metabolicReactions.json", "bt_database_ENVMICRO_metabolicReactions.json",
              "bt_database_standardizationReactions.json"):
        q = ext / f
        if q.exists():
            try:
                _reactions(json.loads(_strip_comments(q.read_text())), bt)
            except Exception:
                pass
    if bt:
        # The reach figures elsewhere were computed over the snapshot that shipped 994 templates,
        # 978 of them outside commented-out records. The release retrieved here holds 992 and 976,
        # and the 16-template gap between the two counts reproduces exactly, so the structure is
        # confirmed and the difference is version drift. The count is not substituted into figures
        # measured on the earlier snapshot.
        import re as _re
        raw = set()
        for f in ("bt_database_metabolicReactions.json",
                  "bt_database_ENVMICRO_metabolicReactions.json",
                  "bt_database_standardizationReactions.json"):
            q = ext / f
            if q.exists():
                raw |= set(_re.findall(r'"([^"]*>>[^"]*)"', q.read_text()))
        rep_bt = {"active": len(set(bt)), "including_commented_out": len(raw),
                  "commented_out_only": len(raw) - len(set(bt))}
        banks["BioTransformer"] = (sorted(set(bt)),
                                   "its own distribution, metabolic and standardisation rules")

    # GLORYx publishes the provenance of each of its rules, so the parts that are not SyGMa's can
    # be separated instead of being counted as a fresh library.
    g = ext / "gloryx_reactionrules.csv"
    if g.exists():
        import csv as _csv
        rows = list(_csv.DictReader(g.open()))
        by_src: dict[str, list[str]] = {}
        for r in rows:
            by_src.setdefault(r["Rule source"].strip(), []).append(r["SMIRKS"].strip())
        if by_src.get("GLORY"):
            banks["GLORY, CYP rules"] = (by_src["GLORY"], "GLORYx release, rules attributed to GLORY")
        if by_src.get("Current work"):
            banks["GLORYx, own rules"] = (by_src["Current work"],
                                          "GLORYx release, rules new in that work")
        if by_src.get("SyGMa"):
            banks["GLORYx, SyGMa portion"] = (by_src["SyGMa"],
                                              "GLORYx release, rules attributed to SyGMa")

    rs = ext / "retrosim_templates_general.json"
    if rs.exists():
        try:
            banks["RetroSim extracted templates"] = (
                sorted(json.loads(rs.read_text())),
                "a second extraction from the same reaction corpus as the USPTO row")
        except Exception:
            pass

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
    # An independent source is one whose templates were written or extracted separately. GLORYx's
    # SyGMa portion and RetroSim's re-extraction of the USPTO corpus are neither, and are reported
    # beside the sources they derive from instead of being added to them.
    INDEPENDENT = {"this work, full bank": "this work",
                   "SyGMa phase1": "SyGMa", "SyGMa phase2": "SyGMa",
                   "USPTO extracted templates": "USPTO", "BioTransformer": "BioTransformer",
                   "GLORY, CYP rules": "GLORY", "GLORYx, own rules": "GLORYx"}
    DERIVED = {"GLORYx, SyGMa portion": "SyGMa", "RetroSim extracted templates": "USPTO"}
    sources = sorted(set(INDEPENDENT.values()))
    n_lib = len(sources)
    n_tpl = sum(v["templates"] for k, v in rep["banks"].items() if k in INDEPENDENT)
    # Within one release, rules copied verbatim from another tool against rules re-typed from it.
    # The split is a control on the claim: the chemistry is the same by attribution, and only the
    # transcription differs.
    if (ext / "gloryx_reactionrules.csv").exists():
        sy_rules = set()
        for k in ("SyGMa phase1", "SyGMa phase2"):
            if k in banks:
                sy_rules |= set(banks[k][0])
        attributed = banks.get("GLORYx, SyGMa portion", ([], ""))[0]
        ident = [r for r in attributed if r in sy_rules]
        rew = [r for r in attributed if r not in sy_rules]
        has = lambda r: any(needs_explicit_hydrogen(t) for t in _ATOM_TOKEN.findall(_reactant_side(r)))
        rep["transcription"] = {
            "attributed": len(attributed), "identical": len(ident), "rewritten": len(rew),
            "identical_with_atom": sum(has(r) for r in ident),
            "rewritten_with_atom": sum(has(r) for r in rew),
            "source_rules": len(sy_rules), "source_with_atom": sum(has(r) for r in sy_rules),
            "note": "same attributed source; only whether the string was copied or re-typed differs"}
        tr = rep["transcription"]
        print(f"\n  transcription control: {tr['identical']} copied verbatim "
              f"({tr['identical_with_atom']} carry the atom primitive), {tr['rewritten']} re-typed "
              f"({tr['rewritten_with_atom']} carry it); the source itself has "
              f"{tr['source_with_atom']} of {tr['source_rules']}")

    if "rep_bt" in dir():
        rep["biotransformer_release"] = rep_bt
    rep["nested_not_counted"] = nested
    rep["derived_not_counted"] = {k: {"derives_from": v, **rep["banks"][k]}
                                  for k, v in DERIVED.items() if k in rep["banks"]}
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
