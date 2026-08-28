"""How many distinct transformations the novel-type gap is, and how its mass is distributed.

The coverage decomposition says 337 of 475 uncovered references need a reaction type the bank does
not contain. That count has been carried for months and it does not decide anything on its own: 337
misses over 40 types, twenty of which carry half the mass, is a fortnight of hand-written rules and
a closed problem. 337 misses over 300 types seen once each is a tail no corpus closes, and 0.817 is
a boundary to be declared rather than a gap to be worked on.

This asks that question. The classifier has always known the type of each miss -- it is how a miss
is called novel -- and has always kept only the count, so the artifact could not be asked. The types
are now recorded and this counts them.

It also crosses the missing types against the reaction libraries already on disk. Synthetic organic
chemistry and metabolism overlap more than their vocabularies suggest: oxidation, hydrolysis,
reduction and conjugation appear in both under different conditions. Conditions do not matter to a
coverage ceiling, which asks only whether a structure is produced at all, so a type present in a
library on this machine is a type that could be added without new data.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

COMMITTED = {"novel_type": 337, "known_type": 98, "untypeable": 40, "uncovered": 475}


def library_types():
    """Reaction types available in libraries already on this machine, by canonical_type.

    Each entry is (name, path glob, how to pull SMIRKS/SMARTS out of the file). A library that is
    not present is reported absent rather than skipped silently, because 'we checked' and 'it was
    not there' are different statements.
    """
    from grail_metabolism.model.reaction_types import canonical_type

    out, absent = {}, []
    sources = [
        ("bank (extended_smirks)", "grail_metabolism/resources/extended_smirks.txt", "lines"),
        ("example_rules", "grail_metabolism/resources/example_rules.txt", "lines"),
    ]
    for name, pat, mode in sources:
        paths = sorted(glob.glob(str(ROOT / pat)))
        if not paths:
            absent.append(name)
            continue
        rules = []
        for p in paths:
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rules.append(line.split()[0] if mode == "lines" else line)
        types = {t for t in (canonical_type(r) for r in rules) if t is not None}
        out[name] = {"n_rules": len(rules), "n_types": len(types), "types": types}
    return out, absent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="results/gaptypes/a*.json")
    ap.add_argument("--out", default=str(ROOT / "results/novel_type_census.json"))
    args = ap.parse_args()

    gap, cov, novel = Counter(), Counter(), []
    for f in sorted(glob.glob(args.shards)):
        d = json.loads(Path(f).read_text())["phase_a"]
        gap.update(d["gap"]); cov.update(d["cov"])
        novel.extend(d.get("novel_pairs", []))

    # the gate: the recomputation must reproduce the committed decomposition exactly, or the
    # types it recorded belong to a different classification and cannot be read
    got = {"novel_type": gap["novel_type"], "known_type": gap["known_type"],
           "untypeable": gap["untypeable"], "uncovered": cov["uncovered"]}
    mismatches = [f"{k}: {got[k]} vs committed {v}" for k, v in COMMITTED.items()
                  if got[k] != v]

    # canonical_type returns a nested tuple; JSON stores it as nested lists, which are not
    # hashable. The key is the serialisation, which is injective on the same structure.
    def key(t):
        return json.dumps(t, sort_keys=True)

    counts = Counter(key(r["type"]) for r in novel)
    ranked = counts.most_common()
    n_types, n_miss = len(counts), sum(counts.values())
    cum, half, ninety = 0, None, None
    for i, (_, c) in enumerate(ranked, 1):
        cum += c
        if half is None and cum >= n_miss / 2:
            half = i
        if ninety is None and cum >= 0.9 * n_miss:
            ninety = i
    singles = sum(1 for _, c in ranked if c == 1)

    # The type is a multiset of changed bonds WITH counts, which is nearly a fingerprint of a
    # whole transformation, so near-uniqueness may be the granularity rather than the chemistry.
    # Coarsening is free and the tail at each level is the question actually being asked: a level
    # at which the tail collapses is only useful if a type there still determines a product.
    def bond_classes(t):
        return frozenset(tuple(map(tuple, e[0][:1])) + tuple(e[0][1:]) for e in t)

    def elements_only(t):
        return frozenset(tuple(e[0][0]) for e in t)

    levels = [
        ("exact multiset, with counts", key),
        ("set of changed-bond classes, counts dropped",
         lambda t: str(sorted(bond_classes(t)))),
        ("element pairs involved", lambda t: str(sorted(elements_only(t)))),
        ("number of bonds changed", lambda t: str(sum(e[1] for e in t))),
    ]
    granularity = []
    for name, fn in levels:
        c = Counter(fn(r["type"]) for r in novel)
        rk = c.most_common()
        sing = sum(1 for _, x in rk if x == 1)
        cum, h = 0, None
        for i, (_, x) in enumerate(rk, 1):
            cum += x
            if h is None and cum >= n_miss / 2:
                h = i
        granularity.append({"granularity": name, "types": len(c), "seen_once": sing,
                            "share_of_mass_in_singletons": round(sing / n_miss, 4),
                            "types_carrying_half_the_mass": h,
                            "determines_a_product": name.startswith(("exact", "set of"))})

    libs, absent = library_types()
    have = set()
    for v in libs.values():
        have |= v["types"]
    covered_by_libraries = [t for t in counts if t in {key(x) for x in have}]

    rep = {"provenance": stamp(__file__),
           "gate": {"reproduces_committed_decomposition": not mismatches,
                    "mismatches": mismatches, "computed": got, "committed": COMMITTED},
           "novel_misses": n_miss, "distinct_types": n_types,
           "types_carrying_half_the_mass": half,
           "types_carrying_ninety_per_cent": ninety,
           "types_seen_once": singles,
           "share_of_misses_in_types_seen_once": round(singles / n_miss, 4) if n_miss else None,
           "top_types": [{"type": json.loads(t), "misses": c} for t, c in ranked[:30]],
           "granularity_curve": granularity,
           "libraries_on_disk": {k: {"n_rules": v["n_rules"], "n_types": v["n_types"]}
                                 for k, v in libs.items()},
           "libraries_absent": absent,
           "missing_types_already_in_a_library_on_disk": len(covered_by_libraries),
           "cross_library": "results/uspto_type_overlap.json",
           "reading": ("a small head means hand-written rules close it; a long tail of singletons "
                       "means the ceiling is a property of the corpus and should be declared")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"gate reproduces the committed decomposition: {not mismatches}"
          + (f"  {mismatches}" if mismatches else ""))
    print(f"\n{n_miss} novel-type misses over {n_types} distinct types")
    print(f"  half the mass sits in {half} types, ninety per cent in {ninety}")
    print(f"  {singles} types are seen once, carrying "
          f"{singles / n_miss:.1%} of the misses")
    print(f"\n  the ten heaviest types:")
    for t, c in ranked[:10]:
        print(f"    {c:>4}  {t[:100]}")
    print(f"\n  the tail against granularity:")
    print(f"    {'level':<44}{'types':>7}{'once':>7}{'mass':>8}{'half in':>9}  usable")
    for g in granularity:
        print(f"    {g['granularity']:<44}{g['types']:>7}{g['seen_once']:>7}"
              f"{g['share_of_mass_in_singletons']:>8.1%}{g['types_carrying_half_the_mass']:>9}"
              f"  {'yes' if g['determines_a_product'] else 'no'}")
    print(f"\n  missing types already present in a library on disk: "
          f"{len(covered_by_libraries)} of {n_types}")
    if absent:
        print(f"  libraries not on this machine: {absent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
