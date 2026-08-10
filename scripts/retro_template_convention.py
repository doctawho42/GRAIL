#!/usr/bin/env python3
"""The same undeclared choice, a different field, and the other of its two mechanisms.

Everything the paper says about hydrogen convention is measured on metabolite rule banks, so a
reader may fairly read it as a fact about one small literature. This applies the same instrument to
a template library from another field entirely: the USPTO retrosynthesis templates that ship with
this repository, extracted by other people from other reactions for another task.

The result is not a repetition, because the mechanism is the other one. In the metabolite banks the
templates fire just as often under either convention and the LOSS IS IN THE PRODUCT -- an unmapped
hydrogen stays on the reacting atom and the product will not parse back
(results/explicit_h_mechanism.json). Retrosynthesis templates fail one step earlier: every one of
them constrains atoms by DEGREE, `D<n>` counts explicit connections, and expanding a molecule with
its hydrogens changes every degree in it, so THE MATCH IS LOST BEFORE ANY PRODUCT IS BUILT.

That gives the claim its general form. What is convention-dependent is not a domain but a class of
SMARTS construct: a primitive that counts explicit structure. Where a library is built on those, the
expansion breaks matching; where a template rewrites without naming the hydrogen it consumes, the
expansion breaks the product. Both are readable from the rule file, and neither is ever declared.

The causal claim is gated rather than asserted. If degree is the mechanism, then deleting the degree
constraints from the very templates that stopped matching must bring them back; the run fails if it
does not.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
import os
import pathlib
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
# shipped with the package; a worktree symlinks it as the datasets are symlinked
TEMPLATES = ROOT / "grail_metabolism" / "uspto_templates.csv.gz"
PRODUCTS = ROOT / "grail_metabolism" / "data" / "USPTO_50k" / "test.csv"
# A primitive is a primitive however it is separated. Requiring a semicolon before D matched
# [C;D1:1] and missed [CD1:1], [#6D2:1] and [C&D1:1], all of which RDKit reads as the same degree
# constraint and all of which lose every match under AddHs. No element symbol is a capital D, X or
# a bare lowercase h followed by a digit, so scanning inside the brackets is unambiguous.
# Detection and deletion are different patterns and were the same one. The detector asks whether a
# bracket carries the primitive anywhere; the deletion has to remove the primitive and leave the
# bracket standing, so it matches the primitive together with the separator in front of it.
COUNTS_EXPLICIT = re.compile(r"\[[^\]]*D\d")
DEGREE = re.compile(r"[;&]?D\d+")
# the implicit-hydrogen count. It reads the hydrogens the expansion just took away, so unlike the
# bracketed H count it is not inert: [c;h1] matches five sites before AddHs and none after.
IMPLICIT_H_COUNT = re.compile(r"\[[^\]]*(?<![A-Za-z])h\d")
# total connections, the inert twin of the degree primitive: X counts neighbours including the
# implicit hydrogens, so drawing them changes nothing. Distinguished here because Daylight calls
# X the connectivity and D the degree, and a census that conflates them reads backwards.
TOTAL_CONNECTIONS = re.compile(r"\[[^\]]*X\d")
# [H] or [#1] as an atom, the construct the metabolite banks use and this library does not
H_ATOM = re.compile(r"\[(?:#1|H)[;:\]+-]")
# a wildcard, which matches the drawn hydrogens once they exist and so fires more often, not less
WILDCARD = re.compile(r"(?:^|[^\[])\*|\[\*")
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


def load_templates() -> list[str]:
    """The side of a retro template that MATCHES: the product, before the arrow."""
    out = []
    with gzip.open(TEMPLATES, "rt") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1 and ">>" in parts[1]:
                out.append(parts[1].split(">>")[0])
    return out


def _init(templates):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["q"] = [(t, Chem.MolFromSmarts(t)) for t in templates]
    _CTX["q"] = [(t, q) for t, q in _CTX["q"] if q is not None]
    # the same patterns with the degree constraint deleted and nothing else changed
    _CTX["stripped"] = [Chem.MolFromSmarts(DEGREE.sub("", t)) for t, _ in _CTX["q"]]


def _worker(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    expanded = Chem.AddHs(Chem.Mol(mol))
    implicit = explicit = recovered = 0
    for i, (_, q) in enumerate(_CTX["q"]):
        hit_i = mol.HasSubstructMatch(q)
        hit_e = expanded.HasSubstructMatch(q)
        implicit += hit_i
        explicit += hit_e
        if hit_i and not hit_e:
            s = _CTX["stripped"][i]
            recovered += bool(s is not None and expanded.HasSubstructMatch(s))
    return implicit, explicit, recovered, int(implicit > 0), int(explicit > 0)


def convention_census(templates) -> dict:
    n = len(templates)
    return {"templates": n,
            "with_a_degree_primitive": sum(1 for t in templates if COUNTS_EXPLICIT.search(t)),
            "with_a_hydrogen_atom": sum(1 for t in templates if H_ATOM.search(t))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=int, default=500)
    ap.add_argument("--templates", type=int, default=0, help="0 uses the whole library")
    ap.add_argument("--out", default=str(ROOT / "results" / "retro_template_convention.json"))
    args = ap.parse_args()

    if not TEMPLATES.exists():
        raise SystemExit(f"{TEMPLATES} is not here; symlink it from the checkout that ships it")
    templates = load_templates()
    census = convention_census(templates)
    print(f"library: {census['templates']} retro templates, "
          f"{census['with_a_degree_primitive']} carry a degree primitive, "
          f"{census['with_a_hydrogen_atom']} carry a hydrogen atom", flush=True)
    if args.templates:
        random.Random(0).shuffle(templates)
        templates = templates[: args.templates]

    products = []
    with open(PRODUCTS) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            products.append(row[1])
            if len(products) >= args.products:
                break
    print(f"products: {len(products)} from the USPTO-50k test split", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (templates,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, products, 8), 1):
            if r:
                rows.append(r)
            if n % 100 == 0 or n == len(products):
                print(f"  {n}/{len(products)}", flush=True)

    implicit = sum(r[0] for r in rows)
    explicit = sum(r[1] for r in rows)
    recovered = sum(r[2] for r in rows)
    any_i = sum(r[3] for r in rows)
    any_e = sum(r[4] for r in rows)
    lost = implicit - explicit

    rep = {"config": {**_code_version(), "n_products": len(rows), "n_templates": len(templates),
                      "library": "grail_metabolism/uspto_templates.csv.gz",
                      "products": "grail_metabolism/data/USPTO_50k/test.csv",
                      "comparison": "as parsed against Chem.AddHs, nothing else changed"},
           "convention_census": census,
           "applicability": {
               "matches_hydrogens_implicit": implicit,
               "matches_hydrogens_explicit": explicit,
               "ratio": round(explicit / max(implicit, 1), 4),
               "products_with_any_applicable_template": {"implicit": any_i, "explicit": any_e}},
           "cause": {
               "matches_lost_to_the_expansion": lost,
               "restored_by_deleting_the_degree_constraint": recovered,
               "share_restored": round(recovered / max(lost, 1), 4)}}

    a = rep["applicability"]
    c = rep["cause"]
    print(f"\napplicability: {implicit} template-product matches with hydrogens implicit against "
          f"{explicit} with them explicit, a ratio of {a['ratio']}")
    print(f"products with any applicable template: {any_i} against {any_e} of {len(rows)}")
    print(f"cause: of {lost} matches the expansion costs, {recovered} come back when the degree "
          f"constraint alone is deleted ({100 * c['share_restored']:.1f}%)")
    if c["share_restored"] < 0.9:
        raise SystemExit("degree does not account for the loss; the mechanism claimed here is wrong")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
