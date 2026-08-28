"""Are the missing metabolic types already present in a synthetic-chemistry library on disk?

The coverage decomposition says 337 uncovered references need a reaction type the bank does not
contain, and the census says those collapse into 312 distinct types at the granularity the bank's
coverage is defined on. The next question costs nothing to ask: 42,555 USPTO retro templates are
already in this repository, and synthetic organic chemistry and metabolism overlap more than their
vocabularies suggest. Oxidation, hydrolysis, reduction and conjugation appear in both under
different conditions, and conditions do not matter to a coverage ceiling, which asks only whether
a structure is produced at all.

USPTO templates are retrosynthetic -- product on the left, reactants on the right -- so the
metabolic direction is the reversed template. Both directions are typed and both are reported,
because a type that appears only as written is a disconnection and not a transformation of the
substrate.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from grail_metabolism.model.reaction_types import canonical_type  # noqa: E402


def key(t):
    return json.dumps(t, sort_keys=True)


def flip(template: str):
    if ">>" not in template:
        return None
    a, b = template.split(">>", 1)
    return f"{b}>>{a}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="results/gaptypes/a*.json")
    ap.add_argument("--uspto", default="grail_metabolism/uspto_templates.csv.gz")
    ap.add_argument("--out", default=str(ROOT / "results/uspto_type_overlap.json"))
    args = ap.parse_args()

    novel = []
    for f in sorted(glob.glob(args.shards)):
        novel += json.loads(Path(f).read_text())["phase_a"].get("novel_pairs", [])
    missing = Counter(key(r["type"]) for r in novel)
    print(f"{len(novel)} novel-type misses over {len(missing)} distinct types",
          file=sys.stderr, flush=True)

    bank_rules = [ln.split()[0] for ln in
                  (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    bank_types = {key(t) for t in (canonical_type(r) for r in bank_rules) if t is not None}

    path = ROOT / args.uspto
    forward, reverse, n, t0 = set(), set(), 0, time.time()
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tmpl = parts[1]
            n += 1
            if n % 5000 == 0:
                print(f"  {n} templates ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            t = canonical_type(tmpl)
            if t is not None:
                forward.add(key(t))
            f = flip(tmpl)
            if f:
                t = canonical_type(f)
                if t is not None:
                    reverse.add(key(t))

    both = forward | reverse
    hit_f = [t for t in missing if t in forward]
    hit_r = [t for t in missing if t in reverse]
    hit_b = [t for t in missing if t in both]
    mass = sum(missing[t] for t in hit_b)

    rep = {"provenance": stamp(__file__),
           "uspto": {"path": args.uspto, "templates": n,
                     "distinct_types_as_written": len(forward),
                     "distinct_types_reversed": len(reverse),
                     "distinct_types_either_direction": len(both),
                     "note": "USPTO templates are retrosynthetic; the metabolic direction is the "
                             "reversed template, and both are reported"},
           "bank": {"rules": len(bank_rules), "distinct_types": len(bank_types)},
           "missing": {"misses": len(novel), "distinct_types": len(missing)},
           "overlap": {"types_in_uspto_as_written": len(hit_f),
                       "types_in_uspto_reversed": len(hit_r),
                       "types_in_uspto_either_direction": len(hit_b),
                       "misses_those_types_carry": mass,
                       "share_of_the_novel_gap": round(mass / len(novel), 4) if novel else None},
           "sanity": {"bank_types_uspto_also_has": len(bank_types & both),
                      "note": "the bank and USPTO must share something; a zero here would mean "
                              "the two type vocabularies do not meet and the intersection above "
                              "measures incomparability rather than absence"},
           "examples_recoverable": [json.loads(t) for t in hit_b[:8]]}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\nUSPTO: {n} templates -> {len(forward)} types as written, {len(reverse)} reversed, "
          f"{len(both)} either way")
    print(f"bank:  {len(bank_rules)} rules -> {len(bank_types)} types")
    print(f"\nof the {len(missing)} missing types, USPTO has:")
    print(f"  as written        {len(hit_f)}")
    print(f"  reversed          {len(hit_r)}")
    print(f"  either direction  {len(hit_b)}   carrying {mass} of {len(novel)} misses "
          f"({mass / len(novel):.1%})")
    print(f"\nsanity: the bank and USPTO share {len(bank_types & both)} types "
          f"({len(bank_types & both) / len(bank_types):.1%} of the bank's)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
