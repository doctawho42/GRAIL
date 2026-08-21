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
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

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


def _split_top(expr: str, sep: str) -> list:
    """Split on `sep` at the top level, ignoring anything inside [] or ()."""
    out, depth, cur = [], 0, []
    for c in expr:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return out


def _strip_chain(chunk: str, drop_h: bool, drop_deg: bool, first: bool = True) -> str:
    """Strip primitives from a chunk with no top-level `,`, so every term is conjunctive.

    `first` says whether this chunk can hold the element symbol, which only the first chunk
    of an atom expression can. `H` opening the whole expression is the hydrogen ATOM (`[H:2]`,
    `[H+]`) and dropping it would change the query's topology; `H` opening a later chunk
    (`[C;!H3]`) is always a count.
    """
    out, i = [], 0
    while i < len(chunk):
        c = chunk[i]
        if c == "$" and i + 1 < len(chunk) and chunk[i + 1] == "(":
            # a recursive SMARTS: dropping a primitive inside `!$(...)` EXCLUDES more, so
            # the rung would tighten instead of relaxing. Copy the group through untouched.
            depth, j = 0, i + 1
            while j < len(chunk):
                if chunk[j] == "(":
                    depth += 1
                elif chunk[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(chunk[i:j + 1])
            i = j + 1
            continue
        if _TWO_LETTER.match(chunk[i:]):          # He, Hf, Hg, Ho, Hs, Xe
            out.append(chunk[i:i + 2])
            i += 2
            continue
        prev = chunk[i - 1] if i else ""
        at_start = first and not any(ch.isalnum() or ch in "*#" for ch in chunk[:i])
        primitive = (drop_h and c in "Hh") or (drop_deg and c in "DX")
        if primitive and prev != "#" and not at_start:
            # a negation belongs to the primitive it negates: leaving `!` behind produces
            # `[#6;!:4]`, which does not parse, and keeping `!H3` while dropping `H3`
            # would invert the direction of the change
            if out and out[-1] == "!":
                out.pop()
            i += 1
            while i < len(chunk) and chunk[i].isdigit():
                i += 1
            continue
        out.append(c)
        i += 1
    # `MolToSmarts` renders a conjunction with `&`, so removing a primitive from the middle
    # of `#6&A&X4&!$(...)` leaves `&&` behind, which does not parse. Collapse the runs and
    # trim the ends rather than leaving the separator the primitive used to sit between.
    text = re.sub(r"[&;,]{2,}", lambda m: m.group(0)[0], "".join(out))
    return text.strip("&;,")


def _strip_expr(head: str, drop_h: bool, drop_deg: bool) -> str:
    """Relax one bracket atom expression, or leave it alone where relaxing would tighten.

    SMARTS `,` is a disjunction, and dropping ONE of its alternatives narrows the expression
    rather than widening it: `[C;!H3,X2]` becomes `[C;X2]`, which matches strictly fewer
    atoms. A comma group is therefore relaxed only when every one of its alternatives would
    disappear, in which case the whole group goes and the result is a superset.
    """
    chunks = []
    for n, chunk in enumerate(_split_top(head, ";")):
        if not chunk:
            continue
        first = (n == 0)
        terms = _split_top(chunk, ",")
        if len(terms) > 1:
            if all(t.lstrip("!") and
                   _strip_chain(t.lstrip("!"), drop_h, drop_deg, first) == ""
                   for t in terms):
                continue                      # every alternative goes, so the group goes
            chunks.append(chunk)              # otherwise leave the disjunction alone
            continue
        stripped = _strip_chain(chunk, drop_h, drop_deg, first)
        if stripped:
            chunks.append(stripped)
    head = ";".join(chunks)
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
    # the rewrite is textual, so the cases that decide its direction are pinned here before
    # the bank-wide invariants run. Each is a construct where a careless strip inverts the
    # change: a negation orphaned from its primitive, a disjunction losing one alternative,
    # a hydrogen that is an atom rather than a count, an element whose symbol starts with H.
    CASES = [
        ("[C;$(C[#6!H3]):2](=[O:3])O[#6;!H3:4]", 1, 1,
         "[C;$(C[#6!H3]):2](=[O:3])O[#6:4]"),
        ("[C;!H3:1]", 1, 1, "[C:1]"),
        ("[C;!H3,X2:1]", 1, 1, "[C:1]"),
        ("[C;!H3,X2:1]", 1, 0, "[C;!H3,X2:1]"),
        ("[c;H1,H0:1]", 1, 0, "[c:1]"),
        ("[C;X4:1][H:2]", 1, 0, "[C;X4:1][H:2]"),
        ("[C;X4:1][H:2]", 1, 1, "[C:1][H:2]"),
        ("[cH:1]", 1, 0, "[c:1]"),
        ("[N;X3:1][CH3:2]", 1, 1, "[N:1][C:2]"),
        ("[He:1]", 1, 1, "[He:1]"), ("[H+:1]", 1, 1, "[H+:1]"),
        ("[nH:1]c1ccccc1", 1, 0, "[n:1]c1ccccc1"),
    ]
    ok_cases = True
    for expr, dh, dd, want in CASES:
        got = _strip(expr, bool(dh), bool(dd))
        if got != want:
            print(f"FAIL: {expr} h={dh} d={dd} -> {got}, expected {want}")
            ok_cases = False

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
    ok = ok_cases
    print(f"{unparsed} of {len(rules)} rewrites did not parse")
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
        "provenance": stamp(__file__),
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
