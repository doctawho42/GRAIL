"""Free first stage of queue point 6: how much same-type-wrong-site signal is there, and its PU risk.

The filter trains on every rule-applicable unannotated product as a negative. The informative
negatives, the idea says, are products of a positive's transformation type on a different site.
This measures how many there are -- and, because the annotation is positive-unlabelled, the same
count is the PU risk: a same-type product on a different site is exactly what an unannotated true
metabolite looks like, so weighting these up as hard negatives is weighting up candidate true
positives.

Per substrate: type each reference (a positive) and each high-ranked non-reference candidate (a
would-be negative) by the MCS route, and count the non-references whose type matches a reference's
type on that substrate. Only the top of the deployed ranking is typed, because those are the
negatives that actually cost recall and the ones a hard-negative scheme would reach.

    python scripts/typed_edit/hard_negative_signal.py --topn 20
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
import bank_without_selection as B  # noqa: E402
import coverage_gap_types as CG  # noqa: E402
from _rrf import rrf_order  # noqa: E402
from _provenance import stamp  # noqa: E402

CAP = 100


def deployed_order(pool, self_key):
    cands = sorted(pool, key=lambda c: -(c["filter"] * c["generator"]))
    seen, dedup = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        dedup.append(c)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:CAP]
    return [c for c in rrf_order(keep) if c["key"] != self_key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=20, help="rank depth to type non-references at")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "hard_negative_signal.json"))
    args = ap.parse_args()

    wide, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        b = json.loads(Path(f).read_text())
        wide.update(b["pools"])
        refs_raw.update(b["references"])
    subs = sorted(set(wide) & set(refs_raw))
    if args.limit:
        subs = subs[: args.limit]

    def typ(sub_mol, smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        try:
            return json.dumps(CG.pair_to_type(sub_mol, m), sort_keys=True)
        except Exception:
            return None

    tot_neg = hard_neg = typed_neg = tot_ref = 0
    t0 = time.time()
    for i, s in enumerate(subs, 1):
        if i % 25 == 0:
            print(f"  {i}/{len(subs)} ({time.time()-t0:.0f}s) neg={tot_neg} hard={hard_neg}",
                  flush=True)
        sm = Chem.MolFromSmiles(s)
        if sm is None:
            continue
        self_key = B._key(s)
        refs = set(refs_raw[s])
        # positive types: type each reference by its own structure
        ref_struct = {c["smiles"]: c for c in wide[s] if c["key"] in refs}
        pos_types = set()
        for smi in ref_struct:
            t = typ(sm, smi)
            if t:
                pos_types.add(t)
                tot_ref += 1
        if not pos_types:
            continue
        order = deployed_order(wide[s], self_key)
        for c in order[: args.topn]:
            if c["key"] in refs:
                continue
            tot_neg += 1
            t = typ(sm, c["smiles"])
            if t is None:
                continue
            typed_neg += 1
            if t in pos_types:
                hard_neg += 1

    rep = {"provenance": stamp(__file__),
           "population": len(subs), "rank_depth": args.topn,
           "references_typed": tot_ref,
           "top_ranked_non_references": tot_neg,
           "of_those_typed": typed_neg,
           "same_type_as_a_reference_on_the_substrate": hard_neg,
           "hard_share_of_typed_negatives": round(hard_neg / typed_neg, 4) if typed_neg else None,
           "reading": (
               f"of {typed_neg} typed non-reference candidates in the deployed top-{args.topn}, "
               f"{hard_neg} carry the same transformation type as a true metabolite on their own "
               f"substrate. That is the hard-negative signal and, identically, the PU risk: each is "
               f"a same-type-different-site product, which is what an unannotated true metabolite "
               f"looks like. A large share means a hard-negative scheme has much to reject and much "
               f"to wrongly reject; a small one means little of either.")}
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n  top-{args.topn} non-references typed: {typed_neg}")
    print(f"  same type as a reference on the substrate: {hard_neg} "
          f"({hard_neg/typed_neg:.1%} of typed)" if typed_neg else "  none typed")
    print(f"\n  {rep['reading']}")
    print(f"\nWrote {rep and args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
