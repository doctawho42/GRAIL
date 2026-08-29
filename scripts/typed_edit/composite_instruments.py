#!/usr/bin/env python3
"""How much of the mined bank is more than one enzymatic step, measured two ways.

The first instrument counts the loci of a template's reaction centre and reports 8.9%. It is a
floor and the paper called it a bound: two edits that share or neighbour centre atoms arrive as
one component and are counted once.

The second instrument is registered as H16 and its threshold was fixed before this file existed.
It counts core-incident edits: bonds between matched atoms whose order or presence moves, plus
attachment points at which a substituent is gained or lost. Bonds inside an added fragment are
not edits to the substrate and are not counted, so a glucuronide costs one, not twelve. A
template is composite by this instrument when that count reaches five, which is above every
single enzymatic turnover the register enumerates.

Both instruments run on the same pairs, the same catalogue and the same MCS settings, so the
union is a union and not two populations.

    python scripts/typed_edit/composite_instruments.py --limit 0
"""
from __future__ import annotations

import argparse
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

# Registered in paper2/preregistration.md before this file was written.
E_THRESHOLD = 5
UNION_BAR = 0.134


def _loci(sub, centre) -> int:
    """Instrument 1: components of the centre, joined only through centre atoms."""
    seen, comps = set(), 0
    for a in centre:
        if a in seen:
            continue
        comps += 1
        stack = [a]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for nb in sub.GetAtomWithIdx(u).GetNeighbors():
                if nb.GetIdx() in centre and nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
    return comps


def _core_incident_edits(sub, prod, sm, pm) -> int:
    """Instrument 2: E, the core-incident edit count registered as H16."""
    s_of_core = {core_i: s_i for core_i, s_i in enumerate(sm)}
    p_of_core = {core_i: p_i for core_i, p_i in enumerate(pm)}
    s_core, p_core = set(sm), set(pm)

    edits = 0
    # (a) bonds between two core atoms whose presence or order differs
    n_core = len(sm)
    for i in range(n_core):
        for j in range(i + 1, n_core):
            sb = sub.GetBondBetweenAtoms(s_of_core[i], s_of_core[j])
            pb = prod.GetBondBetweenAtoms(p_of_core[i], p_of_core[j])
            if (sb is None) != (pb is None):
                edits += 1
            elif sb is not None and sb.GetBondType() != pb.GetBondType():
                edits += 1

    # (b) attachment points gained or lost: a core atom's bonds to non-core atoms, counted as
    # the change in how many such bonds it carries, so one substituent swap costs one on each
    # side and a fragment of any size costs one
    for i in range(n_core):
        s_out = sum(1 for nb in sub.GetAtomWithIdx(s_of_core[i]).GetNeighbors()
                    if nb.GetIdx() not in s_core)
        p_out = sum(1 for nb in prod.GetAtomWithIdx(p_of_core[i]).GetNeighbors()
                    if nb.GetIdx() not in p_core)
        edits += abs(s_out - p_out)
    return edits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="results/mined_rule_catalog_v2.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/composite_instruments.json"))
    args = ap.parse_args()

    import random

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    RDLogger.DisableLog("rdApp.*")
    from scripts.mine_rules import MCS_TIMEOUT_SECONDS, find_reaction_center

    catalog = json.loads((ROOT / args.catalog).read_text())
    keys = sorted(catalog)
    if args.limit:
        random.Random(args.seed).shuffle(keys)
        keys = keys[:args.limit]

    stats = Counter()
    hist_e, hist_loci = Counter(), Counter()
    only_1, only_2, both, examples = [], [], [], []
    for smk in keys:
        pair = (catalog[smk].get("source_pairs") or [None])[0]
        if not pair:
            stats["no_pair"] += 1
            continue
        sub, prod = Chem.MolFromSmiles(pair[0]), Chem.MolFromSmiles(pair[1])
        if sub is None or prod is None:
            stats["unparseable_pair"] += 1
            continue
        try:
            mcs = rdFMCS.FindMCS([sub, prod], timeout=MCS_TIMEOUT_SECONDS, matchValences=False,
                                 ringMatchesRingOnly=True, completeRingsOnly=True,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 atomCompare=rdFMCS.AtomCompare.CompareElements)
            if mcs.canceled or mcs.numAtoms == 0:
                stats["mcs_failed"] += 1
                continue
            core = Chem.MolFromSmarts(mcs.smartsString)
            sm = sub.GetSubstructMatch(core)
            pm = prod.GetSubstructMatch(core)
            if not sm or not pm:
                stats["no_match"] += 1
                continue
            cs, _ = find_reaction_center(sub, prod, sm, pm)
            if not cs:
                stats["no_centre"] += 1
                continue
            loci = _loci(sub, set(cs))
            e = _core_incident_edits(sub, prod, sm, pm)
        except Exception:
            stats["error"] += 1
            continue

        stats["scored"] += 1
        hist_loci[min(loci, 5)] += 1
        hist_e[min(e, 10)] += 1
        i1, i2 = loci > 1, e >= E_THRESHOLD
        stats["instrument_1"] += i1
        stats["instrument_2"] += i2
        stats["union"] += (i1 or i2)
        stats["both"] += (i1 and i2)
        if i1 and not i2:
            only_1.append(smk)
        if i2 and not i1:
            only_2.append(smk)
            if len(examples) < 15:
                examples.append({"loci": loci, "E": e, "substrate": pair[0][:100],
                                 "product": pair[1][:100], "smirks": smk[:160]})
        if i1 and i2:
            both.append(smk)

    n = max(stats["scored"], 1)
    shares = {k: round(stats[k] / n, 4) for k in ("instrument_1", "instrument_2", "union", "both")}
    verdict = "confirmed" if shares["union"] >= UNION_BAR else "failed"
    rep = {
        "provenance": stamp(__file__),
        "register": "H16",
        "n_rules_considered": len(keys), "limit": args.limit, "seed": args.seed,
        "counts": dict(stats),
        "shares": shares,
        "threshold_E": E_THRESHOLD,
        "registered_bar": UNION_BAR,
        "verdict": verdict,
        "loci_histogram": {str(k): v for k, v in sorted(hist_loci.items())},
        "E_histogram": {str(k): v for k, v in sorted(hist_e.items())},
        "flagged_by_instrument_2_only": len(only_2),
        "flagged_by_instrument_1_only": len(only_1),
        "examples_instrument_2_only": examples,
        "instrument_1": ("the reaction centre of the pair the template was mined from, split into "
                         "components joined only through centre atoms; more than one component "
                         "is more than one locus"),
        "instrument_2": ("E, the core-incident edit count: bonds between matched atoms whose "
                         "order or presence differs, plus attachment points gained or lost, "
                         "each counted once and bonds internal to an added or removed fragment "
                         f"not counted; composite at E >= {E_THRESHOLD}"),
        "caveat": ("both instruments run under the mining MCS wall-clock timeout, so the counts "
                   "are load-dependent at the margin, and a template is scored on one of its "
                   "source pairs rather than all of them"),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"scored {stats['scored']} of {len(keys)} mined templates")
    print(f"  instrument 1, loci > 1            {stats['instrument_1']:>5}  {shares['instrument_1']:.2%}")
    print(f"  instrument 2, E >= {E_THRESHOLD}              {stats['instrument_2']:>5}  {shares['instrument_2']:.2%}")
    print(f"  both                              {stats['both']:>5}  {shares['both']:.2%}")
    print(f"  union                             {stats['union']:>5}  {shares['union']:.2%}")
    print(f"\nH16 registered bar {UNION_BAR:.1%}: {verdict}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
