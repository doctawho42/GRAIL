#!/usr/bin/env python3
"""How far the candidate pool grows as the environment is relaxed, one primitive at a time.

Step 0 measures two endpoints: the reactant SMARTS as written, and the skeleton with the
whole environment dropped. On the real bank those are 4.5k and 31k matches per substrate,
so the endpoints alone say the hard gate becomes nearly vacuous and leave nowhere to stand
in between. This measures the rungs.

The rungs are chosen to be the constructs the leaderboard paper's Table 7 identifies as
convention-dependent, stripped in that order:

  as_written   the reactant template unchanged
  no_H         hydrogen-count primitives removed (H<n> inside a bracket atom, and h<n>)
  no_H_no_deg  also the connectivity primitives (D<n>, X<n>)
  skeleton     every environment atom dropped, atom queries reduced to element (step 0)

Stripping a constraint can only add matches, so the counts must be non-decreasing along the
ladder. That is asserted per substrate rather than assumed: a rung that goes down means the
rewrite changed the query's topology, not just its constraints.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rdkit import Chem, RDLogger  # noqa: E402

from step0 import load_rules, reaction_centre, skeleton_query  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
SUBS = ROOT / "results" / "match_sensitivity_fulln.json"

# A primitive is stripped only inside a bracket atom, and only where it really is a
# primitive. `H` is a hydrogen COUNT when something precedes it in the expression
# (`[cH]`, `[C;H1]`, `[CH3]`); it is an ATOM when it starts the expression (`[H:2]`,
# `[H+]`), and dropping that would change the query's topology rather than relax it. The
# two-letter elements beginning with H or X (He, Hf, Hg, Ho, Hs, Xe) are left alone.
_TWO_LETTER = re.compile(r"^[HX][efgos]")


def _strip_expr(head: str, drop_h: bool, drop_deg: bool) -> str:
    out, i = [], 0
    while i < len(head):
        c = head[i]
        if c == "$" and i + 1 < len(head) and head[i + 1] == "(":
            # a recursive SMARTS: dropping a primitive inside `!$(...)` EXCLUDES more, so
            # the rung would tighten instead of relaxing. Copy the group through untouched.
            depth, j = 0, i + 1
            while j < len(head):
                if head[j] == "(":
                    depth += 1
                elif head[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(head[i:j + 1])
            i = j + 1
            continue
        if _TWO_LETTER.match(head[i:]):          # He, Hf, Hg, Ho, Hs, Xe
            out.append(head[i:i + 2])
            i += 2
            continue
        prev = head[i - 1] if i else ""
        at_start = not any(ch.isalnum() or ch in "*#" for ch in head[:i])
        primitive = (drop_h and c in "Hh") or (drop_deg and c in "DX")
        if primitive and prev != "#" and not at_start:
            i += 1
            while i < len(head) and head[i].isdigit():
                i += 1
            continue
        out.append(c)
        i += 1
    head = "".join(out)
    head = re.sub(r"[;&,]{2,}", lambda m: m.group(0)[0], head).strip(";&,")
    return head or "*"


def _strip(smarts: str, drop_h: bool, drop_deg: bool) -> str:
    out, i = [], 0
    while i < len(smarts):
        c = smarts[i]
        if c != "[":
            out.append(c)
            i += 1
            continue
        j = smarts.index("]", i)
        head, _, mapnum = smarts[i + 1:j].partition(":")
        head = _strip_expr(head, drop_h, drop_deg)
        out.append("[" + head + (":" + mapnum if mapnum else "") + "]")
        i = j + 1
    return "".join(out)


def rungs(rxn):
    """Return {rung: query mol or None} for one reaction."""
    tmpl = rxn.GetReactantTemplate(0)
    raw = Chem.MolToSmarts(tmpl)
    out = {"as_written": tmpl}
    for name, (dh, dd) in (("no_H", (True, False)), ("no_H_no_deg", (True, True))):
        try:
            out[name] = Chem.MolFromSmarts(_strip(raw, dh, dd))
        except Exception:
            out[name] = None
    out["skeleton"] = skeleton_query(rxn)[0]
    return out


def self_test(probe_smiles=("CC(=O)Nc1ccc(O)cc1", "CC(C)NCC(O)COc1cccc2ccccc12",
                            "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1"
                                                          "[N+](=O)[O-])C(=O)OC")) -> int:
    """Two invariants over the whole bank, not over hand-written cases.

    1. Relaxing a constraint can only add matches, so `no_H` and `no_H_no_deg` must match
       a SUPERSET of the rung below, compared as sets of atom tuples. Comparing counts
       would pass a rewrite that swapped one match for another.
    2. The skeleton is the admissibility gate the typed model would use, so every site the
       full template matches has to survive it. The skeleton drops atoms, so its tuples
       are shorter: the full match is projected onto the centre atoms and compared as a
       set of frozensets, which is how step 0 deduplicates (type, site) pairs anyway.
    """
    rules, _ = load_rules(str(BANK))
    mols = [Chem.MolFromSmiles(s) for s in probe_smiles]
    names = ["as_written", "no_H", "no_H_no_deg"]
    bad_rung, bad_skel, checked, unparsed = [], [], 0, 0

    for smirks, rxn in rules:
        lad = rungs(rxn)
        if lad["no_H"] is None or lad["no_H_no_deg"] is None:
            unparsed += 1
            continue
        centre = reaction_centre(rxn)[0]
        tmpl = rxn.GetReactantTemplate(0)
        centre_pos = [a.GetIdx() for a in tmpl.GetAtoms() if a.GetAtomMapNum() in centre]

        for mol in mols:
            raw = {nm: mol.GetSubstructMatches(lad[nm], uniquify=True, maxMatches=5000)
                   for nm in names}
            # uniquify returns ONE representative tuple per atom set, and which
            # representative it picks changes as a relaxed query gains symmetry. The
            # invariant is about sites, so compare atom sets.
            sets = {nm: {frozenset(m) for m in raw[nm]} for nm in names}
            for a, b in zip(names, names[1:]):
                if not sets[a] <= sets[b]:
                    bad_rung.append((smirks, a, b, len(sets[a]), len(sets[b])))
                checked += 1
            sk = lad["skeleton"]
            if sk is None or not centre_pos:
                continue
            projected = {frozenset(m[i] for i in centre_pos) for m in raw["as_written"]}
            skel = {frozenset(m) for m in
                    mol.GetSubstructMatches(sk, uniquify=True, maxMatches=5000)}
            if not projected <= skel:
                bad_skel.append((smirks, len(projected), len(skel)))
            checked += 1

    print(f"{checked} comparisons over {len(rules)} templates x {len(mols)} molecules")
    print(f"{unparsed} rewrites did not parse")
    ok = True
    if bad_rung:
        ok = False
        print(f"FAIL: {len(bad_rung)} rungs are not a superset of the rung below")
        for row in bad_rung[:5]:
            print("   ", row[0][:70], row[1], "->", row[2], row[3], row[4])
    if bad_skel:
        ok = False
        print(f"FAIL: {len(bad_skel)} skeletons drop a site the full template matched")
        for row in bad_skel[:5]:
            print("   ", row[0][:70], "full-projected", row[1], "skeleton", row[2])
    print("self-test: OK" if ok else "self-test: FAILURES ABOVE")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_relaxation.json"))
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    rules, load_stats = load_rules(str(BANK))
    ladders, broken = [], 0
    for _, rxn in rules:
        r = rungs(rxn)
        if r["no_H"] is None or r["no_H_no_deg"] is None:
            broken += 1
        ladders.append(r)
    print(f"bank: {load_stats['parsed']} rules, {broken} whose rewrite did not parse",
          file=sys.stderr, flush=True)

    pool = json.loads(SUBS.read_text())["substrates"]
    picked = random.Random(args.seed).sample(pool, args.cap)
    names = ["as_written", "no_H", "no_H_no_deg", "skeleton"]

    rows, violations = [], []
    for n, smi in enumerate(picked, 1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = {"substrate": smi, "heavy": mol.GetNumHeavyAtoms()}
        for name in names:
            tot = 0
            for lad in ladders:
                q = lad[name]
                if q is None:
                    continue
                tot += len(mol.GetSubstructMatches(q, uniquify=True, maxMatches=5000))
            row[name] = tot
        for a, b in zip(names, names[1:]):
            if row[b] < row[a]:
                violations.append({"substrate": smi, "rung": b, a: row[a], b: row[b]})
        rows.append(row)
        print(f"  {n}/{len(picked)}  " + "  ".join(f"{k}={row[k]}" for k in names),
              file=sys.stderr, flush=True)

    mean = {k: round(sum(r[k] for r in rows) / max(len(rows), 1), 1) for k in names}
    out = {
        "rule_bank": {"path": str(BANK.relative_to(ROOT)), **load_stats,
                      "rewrites_that_did_not_parse": broken},
        "substrate_sample": {"source": str(SUBS.relative_to(ROOT)), "population": len(pool),
                             "cap": args.cap, "seed": args.seed, "n": len(rows)},
        "mean_matches_per_substrate": mean,
        "relative_to_as_written": {k: round(mean[k] / max(mean["as_written"], 1e-9), 2)
                                   for k in names},
        "monotonicity_violations": violations,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if violations:
        print(f"WARNING: {len(violations)} non-monotone rungs; a rewrite changed topology",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
