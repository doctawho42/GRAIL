#!/usr/bin/env python3
"""What one pass of GLORYxR's rule bank actually does, measured rather than read off the source.

Writes results/gloryxr_mechanics.json.

This producer exists because two committed artifacts published a false statement about someone
else's code. revision/T_gloryxr_provenance.json and revision/phase2_gloryxr_table.py both said
"Reactor.react_one applies each rule once and the package has no recursion". The second clause is
true. The first is not: reactions.py:140 calls `reaction.RunReactants([AddHs(educt)])`, which
enumerates every match of the template, so one firing of one rule can return several products.
A reader could download that record and carry the wrong sentence into print.

Three quantities go in, and each is here because the table asserted it with no producer behind it:

  1. products per rule firing, over the whole evaluated population, which is what refutes the
     retracted wording;
  2. the 0.2 factor applied to rules of "uncommon" priority, quoted with the file and line it was
     read from rather than asserted as a number;
  3. how many of the rule table's rules carry that priority.

The measurement needs gloryxr, which requires Python >= 3.13 while this repository runs 3.10, so
`import gloryxr` is deliberately lazy: the module must import under either interpreter or its
tests cannot run at all. Run it under the same locked environment the published column was
computed in, not a fresh resolve:

    <scratchpad>/gloryxr_src/GLORYxR/.venv/bin/python revision/phase2_gloryxr_mechanics.py \
        --rule-table <scratchpad>/gloryxr_src/GLORYxR/src/gloryxr/rules_data/\
gloryx_reactionrules_connect.csv

No model dumps are needed: scoring is what loads the forests, and one pass of the reactor does not
score. That is why this costs about a minute over all 1,170 substrates while the prediction columns
cost forty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POPULATION_FILE = ROOT / "results" / "test_references.json"
OUT_PATH = ROOT / "results" / "gloryxr_mechanics.json"

# One phase setting, the same one the published columns used, so this describes that run and not a
# configuration nobody reported.
PHASE = "1+2"

# Where the 0.2 factor is written in the package. Quoted with its location because it is read out
# of someone else's source: a bare 0.2 in a record of ours is exactly the defect this file fixes.
PRIORITY_FACTOR = 0.2
PRIORITY_FACTOR_SOURCE = "gloryxr/models/fame3r.py:59-62, applied at :76"
PRIORITY_FACTOR_PROVIDER = "MultiFAME3RModelProvider"

RETRACTS = (
    "the wording 'Reactor.react_one applies each rule once and the package has no recursion', "
    "published in revision/T_gloryxr_provenance.json (limitations, id=generation) and in "
    "revision/phase2_gloryxr_table.py. The second clause stands; the first is false, and the "
    "measurement below is what refutes it"
)


def population() -> list:
    """The evaluated substrates, as the corpus stores them, sorted.

    The corpus drawing rather than a re-tautomerised one, because that is the drawing every
    comparator was handed and the one the published columns were computed on.
    """
    return sorted(json.loads(POPULATION_FILE.read_text()))


def summarise(rows: list) -> dict:
    """Reduce per-substrate rule firings to the quantities the retracted sentence got wrong.

    Each row is {"substrate": smiles, "firings": {rule_name: n_products}}. Kept pure and free of
    any gloryxr import so it is testable under the repository's own interpreter.

    An empty measurement is refused. Zero substrates would report "one application per rule holds"
    vacuously, which is how a run that failed becomes a claim that succeeded.
    """
    if not rows:
        raise ValueError("empty measurement: no substrate was measured, so nothing can be "
                         "summarised; a failed run must not report a holding claim")

    counts = [n for row in rows for n in (row.get("firings") or {}).values()]
    n_multi = sum(1 for n in counts if n > 1)
    worst = max(counts) if counts else 0
    per_substrate_max = sorted(max((row.get("firings") or {}).values(), default=0)
                               for row in rows)
    median_max = per_substrate_max[len(per_substrate_max) // 2] if per_substrate_max else 0
    return {
        "n_substrates": len(rows),
        "n_rule_firings": len(counts),
        "n_firings_yielding_more_than_one_product": n_multi,
        "share_of_firings_yielding_more_than_one_product": (
            round(n_multi / len(counts), 4) if counts else None),
        "max_products_from_one_rule_firing": worst,
        "median_of_the_per_substrate_worst_rule": median_max,
        "one_application_per_rule_holds": worst <= 1,
        "total_reactions_returned": sum(counts),
    }


def rule_table(path) -> dict:
    """Counts from GLORYxR's own rule table, or a recorded absence.

    The table ships with GLORYxR and not with this repository, so a clean checkout has no copy.
    Absence is recorded rather than fatal, and never filled in from memory.
    """
    p = Path(path)
    if not p.exists():
        return {"present": False, "checked_path": str(p), "sha256": None,
                "n_rules": None, "n_uncommon": None, "n_subsets": None,
                "why": "the rule table ships with GLORYxR, not with this repository"}

    import csv
    import hashlib
    from collections import Counter

    with open(p, newline="") as handle:
        rows = list(csv.DictReader(handle))
    priority = Counter(r.get("Priority level", "") for r in rows)
    subsets = Counter(r.get("Name of rule subset", "") for r in rows)
    source = Counter(r.get("Rule source", "") for r in rows)
    return {"present": True,
            "identified_by": ("digest rather than path: the table is read out of a GLORYxR clone, "
                              "which is as transient as the session that made it"),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "file_name": p.name,
            "n_rules": len(rows),
            "n_uncommon": priority.get("uncommon", 0),
            "n_common": priority.get("common", 0),
            "n_subsets": len(subsets),
            "by_subset": dict(subsets),
            "by_rule_source": dict(source)}


def priority_factor() -> dict:
    """The 0.2 multiplier, with the location it was read from.

    It is quoted rather than measured because it is a literal in the package. The same block is
    duplicated in the unused single-model provider, so the provider is named: the factor reported
    here is the one on the path the published columns took.
    """
    return {"factor": PRIORITY_FACTOR,
            "source": PRIORITY_FACTOR_SOURCE,
            "provider": PRIORITY_FACTOR_PROVIDER,
            "applies_to": "rules whose Priority level is 'uncommon'",
            "read_not_measured": True}


def _package_versions() -> dict:
    """The build this measurement describes. cdpkit recomputes descriptors at run time, so a
    mechanics record without versions cannot be placed against a column."""
    import importlib.metadata as md
    out = {}
    for name in ("gloryxr", "fame3r", "cdpkit", "rdkit", "scikit-learn", "numpy"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def provenance() -> dict:
    """What this record is, what it retracts, and which build produced it."""
    stamp = {}
    try:
        from _provenance import stamp as _stamp
        stamp = _stamp(__file__)
    except Exception:
        stamp = {"script": Path(__file__).name}
    return {**stamp,
            "what_this_is": ("one pass of GLORYxR's rule bank over the evaluated population, "
                             "measured: how many products one firing of one rule returns"),
            "retracts": RETRACTS,
            "python": sys.version.split()[0],
            "packages": _package_versions(),
            "phase": PHASE,
            "no_models_needed": ("scoring loads the forests; one pass of the reactor does not "
                                 "score, so this measurement needs no model dumps")}


def measure(rule_table_path, limit: int = 0) -> dict:
    """Run the reactor once per substrate and count products per rule firing.

    Lazy imports: this is the only function that needs the 3.13 environment.
    """
    from rdkit import RDLogger
    from rdkit.Chem.rdmolfiles import MolFromSmiles
    from gloryxr.reactions import Reactor

    RDLogger.DisableLog("rdApp.*")
    reactor = Reactor.load_builtin(phase=PHASE, strict_soms=False)

    subs = population()
    if limit:
        subs = subs[:limit]

    rows, unparsed = [], []
    for smiles in subs:
        mol = MolFromSmiles(smiles)
        if mol is None:
            unparsed.append(smiles)
            continue
        firings: dict = {}
        for rxn in reactor.react_one(mol):
            try:
                name = rxn.GetProp("_Name")
            except Exception:
                name = "<unnamed>"
            firings[name] = firings.get(name, 0) + 1
        rows.append({"substrate": smiles, "firings": firings})

    report = {"provenance": provenance(),
              "measured": summarise(rows),
              "rule_table": rule_table(rule_table_path),
              "priority_factor": priority_factor(),
              "n_substrates_requested": len(subs),
              "n_substrates_unparsed": len(unparsed),
              "unparsed": unparsed}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-table", required=True,
                    help="path to GLORYxR's gloryx_reactionrules_connect.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    report = measure(args.rule_table, limit=args.limit)
    OUT_PATH.write_text(json.dumps(report, indent=1))

    m = report["measured"]
    print(f"substrates measured        {m['n_substrates']}")
    print(f"rule firings               {m['n_rule_firings']}")
    print(f"firings with >1 product    {m['n_firings_yielding_more_than_one_product']} "
          f"({m['share_of_firings_yielding_more_than_one_product']})")
    print(f"worst single rule firing   {m['max_products_from_one_rule_firing']} products")
    print(f"one application per rule   {m['one_application_per_rule_holds']}")
    rt = report["rule_table"]
    if rt["present"]:
        print(f"rule table                 {rt['n_rules']} rules, {rt['n_uncommon']} uncommon")
    else:
        print(f"rule table                 ABSENT at {rt['checked_path']}")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
