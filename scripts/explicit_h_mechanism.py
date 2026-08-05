#!/usr/bin/env python3
"""Why one call to AddHs moves a bank's reach, and where in the pipeline the reach goes.

scripts/engine_knobs.py shows that expanding the substrate with explicit hydrogens before the
templates fire costs the shared bank the whole of its engine term. That is a measurement, not an
explanation, and a reader is entitled to ask why a chemically inert operation changes what a rule
set reaches.

The obvious account -- that patterns stop matching an H-expanded molecule -- is wrong, and this
script is what rules it out: the templates fire essentially as often either way, and every firing
writes a SMILES either way. The loss is one step later, and it is a valence violation.

A substitution template written against implicit hydrogens does not mention the hydrogen it
replaces; it attaches the new group and lets RDKit decrement an implicit count that is not stored as
an atom. Expand the substrate first and that hydrogen is an atom, nothing removes it, and the
product carries an aromatic carbon holding both its hydrogen and the new substituent. Benzoic acid
under aromatic hydroxylation gives exactly that:

    [H]OC(=O)c1c([H])c([H])c([H])(O)c([H])c1[H]

which RDKit will not parse back, so the fragment never reaches the matcher. The template is not
wrong and the engine is not wrong; they disagree about how a hydrogen is spelled.

The measurement is therefore staged along the pipeline -- firings isolate matching, SMILES written
isolate generation, and fragments surviving the validity check isolate the failure. If the account
holds, the first two are flat and the third is not, with the drop landing on fragments RDKit refuses
rather than on fragments below the heavy-atom floor.

The last section counts, for each of the three banks, how many templates spell a hydrogen atom out
on the reactant side. That is the convention a bank was authored in, it is readable from the rule
file without running anything, and it lines up with the sign of the effect in
results/bank_engine_replication.json. Two banks with opposite conventions moving in opposite
directions is what shows the direction belongs to the bank rather than to our engine; calling the
share PREDICTIVE would be a stronger claim than three banks can carry, and it is not made.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

from engine_knobs import DEFAULT, apply_with, shared_rules
from grail_metabolism.utils.preparation import _iter_reaction_products

RDLogger.DisableLog("rdApp.*")
_ATOM_TOKEN = re.compile(r"\[[^\]]*\]")
_MAP = re.compile(r":\d+$")
_H_ATOM = re.compile(r"(?:#1|H)(?:[+-]\d*)?")
_RECURSIVE = re.compile(r"\$\(")
_STAGES = ("fired", "written", "kept", "unparseable", "below_heavy_atom_floor")
_CTX: dict = {}


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def needs_explicit_hydrogen(token: str) -> bool:
    """True iff this bracket atom is the hydrogen ATOM primitive, [H] or [#1].

    RDKit matches both only against a hydrogen atom -- [H] finds nothing in ethanol and six atoms in
    its H-expanded form -- so a template carrying one cannot fire on a substrate whose hydrogens are
    implicit. Three things that look similar are not this and are excluded: the hydrogen-COUNT
    primitive inside [CH3] or [C;H2], which is unaffected by the expansion; a negation such as
    [!#1], which asserts the opposite; and a multi-digit atomic number such as [#16], which merely
    begins with the same two characters. A substring search gets all three wrong.
    """
    body = _MAP.sub("", token[1:-1])
    for primitive in re.split(r"[;,&]", body):
        primitive = primitive.strip()
        if primitive and not primitive.startswith("!") and _H_ATOM.fullmatch(primitive):
            return True
    return False


def hydrogen_convention() -> dict:
    """How each bank spells hydrogens on the reactant side of its templates."""
    from bank_engine_replication import load_bank
    out = {}
    for bank in ("grail_full", "sygma_175", "biotransformer"):
        rules = load_bank(bank)
        n_exp = n_rec = 0
        for rule in rules:
            reactants = rule.split(">>")[0]
            if any(needs_explicit_hydrogen(t) for t in _ATOM_TOKEN.findall(reactants)):
                n_exp += 1
            elif _RECURSIVE.search(reactants):
                n_rec += 1          # recursive SMARTS this token test does not look inside
        out[bank] = {"rules": len(rules), "with_explicit_hydrogen": n_exp,
                     "share": round(n_exp / max(len(rules), 1), 3),
                     "unclassified_recursive_smarts": n_rec}
    return out


def _fate(smiles: str) -> str:
    """Why a written fragment does or does not survive the validity check."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable"                    # a valence RDKit will not accept
    if mol.GetNumHeavyAtoms() < 2:
        return "below_heavy_atom_floor"
    return "kept"


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = rules


def _worker(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    stage: dict = {}
    for add_hs in (True, False):
        substrate = Chem.AddHs(Chem.Mol(mol)) if add_hs else Chem.Mol(mol)
        c = dict.fromkeys(_STAGES, 0)
        for rule in _CTX["rules"]:
            try:
                generated = list(_iter_reaction_products(substrate, rule))
            except Exception:
                continue
            c["fired"] += len(generated)
            for product in generated:
                try:
                    out = Chem.MolToSmiles(product)
                except Exception:
                    continue
                c["written"] += 1
                for fragment in (f.strip() for f in out.split(".") if f.strip()):
                    c[_fate(fragment)] += 1
        stage[add_hs] = c
    prods = {add_hs: set(apply_with(mol, _CTX["rules"], add_hs, DEFAULT["norm"],
                                    DEFAULT["drop_invalid"]))
             for add_hs in (True, False)}
    return {"stage": stage, "distinct": {k: len(v) for k, v in prods.items()},
            "shared": len(prods[True] & prods[False])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "explicit_h_mechanism.json"))
    args = ap.parse_args()

    rules = shared_rules()
    subs = [r["sub"] for r in
            json.loads((ROOT / "results/filter_vs_prior_ci.json").read_text())["per_substrate"]]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = [s for s in subs if truth.get(s)]
    if args.limit:
        subs = subs[: args.limit]
    print(f"{len(rules)} rules over {len(subs)} substrates", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules,)) as pool:
        rows = [r for r in pool.map(_worker, subs, 4) if r]

    def tot(fn):
        return sum(fn(r) for r in rows)

    pipeline = {}
    for add_hs in (True, False):
        key = "deployed" if add_hs else "without_explicit_h"
        pipeline[key] = {k: tot(lambda r, h=add_hs, k=k: r["stage"][h][k]) for k in _STAGES}
        # the denominator for the validity split is fragments, not products: one product can
        # separate into several, so quoting the product count against it would be the wrong ratio
        pipeline[key]["fragments"] = sum(pipeline[key][k] for k in
                                         ("kept", "unparseable", "below_heavy_atom_floor"))
        pipeline[key]["distinct_after_normalisation"] = tot(lambda r, h=add_hs: r["distinct"][h])
    d, w = pipeline["deployed"], pipeline["without_explicit_h"]

    unparseable_extra = d["unparseable"] - w["unparseable"]
    floor_extra = d["below_heavy_atom_floor"] - w["below_heavy_atom_floor"]
    rep = {"config": {**_code_version(), "n_rules": len(rules), "n_substrates": len(rows),
                      "knobs_held": {k: v for k, v in DEFAULT.items() if k != "add_hs"}},
           "pipeline": pipeline,
           "products_in_both": tot(lambda r: r["shared"]),
           "where_it_is_lost": {
               "firing_ratio": round(d["fired"] / max(w["fired"], 1), 3),
               "written_ratio": round(d["written"] / max(w["written"], 1), 3),
               "kept_ratio": round(d["kept"] / max(w["kept"], 1), 3),
               "extra_fragments_rdkit_refuses": unparseable_extra,
               "extra_fragments_below_the_floor": floor_extra,
               "share_of_the_drop_that_is_unparseable": round(
                   unparseable_extra / max(unparseable_extra + floor_extra, 1), 3)},
           "hydrogen_convention_by_bank": hydrogen_convention()}

    print(f"\n  {'stage':30} {'deployed':>10} {'implicit H':>12}")
    for k in (*_STAGES, "fragments", "distinct_after_normalisation"):
        print(f"  {k:30} {d[k]:>10} {w[k]:>12}")
    loss = rep["where_it_is_lost"]
    print(f"\nratios deployed/implicit: firings {loss['firing_ratio']}, "
          f"SMILES written {loss['written_ratio']}, fragments kept {loss['kept_ratio']}")
    print(f"unparseable as a share of fragments: "
          f"{100 * d['unparseable'] / max(d['fragments'], 1):.1f}% deployed against "
          f"{100 * w['unparseable'] / max(w['fragments'], 1):.1f}% with hydrogens implicit")
    print(f"the deployed setting writes {unparseable_extra} more fragments RDKit refuses to parse "
          f"and {floor_extra} more below the heavy-atom floor: "
          f"{100 * loss['share_of_the_drop_that_is_unparseable']:.1f}% of the excess is valence")
    print("\nhow each bank spells hydrogens on the reactant side:")
    for bank, v in rep["hydrogen_convention_by_bank"].items():
        print(f"  {bank:16} {v['with_explicit_hydrogen']:>5} of {v['rules']:>5} templates "
              f"({100 * v['share']:.1f}%)")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
