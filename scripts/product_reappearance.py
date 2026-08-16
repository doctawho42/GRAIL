#!/usr/bin/env python3
r"""How often an annotated product is itself annotated as a substrate.

The design appendix argues that supervision here is dense at every node rather than terminal, and
the fact it rests on is that the annotation graph has depth: a metabolite that is itself
metabolised appears again on the left of a triple. That share was printed with nothing behind it.

It is counted here on the clean training split, with two identities rather than one, because the
answer depends on the identity and this paper's whole subject is that such a dependence has to be
declared:

    canonical SMILES   two structures are the same molecule when their canonical strings match
    tautomer key       the same, up to the tautomer-aware key the evaluation criterion uses

The looser key merges more structures, so it can only find at least as many substrates for a given
product --- and it finds fewer reappearances, because merging also collapses the product side.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


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


def _edges(smiles: list, triples: Path):
    for line in triples.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[2] != "1":
            continue
        a, b = int(parts[0]), int(parts[1])
        if a >= len(smiles) or b >= len(smiles) or smiles[a] is None or smiles[b] is None:
            continue
        yield smiles[a], smiles[b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--triples", default=None,
                    help="defaults to the clean triples the deployed configuration trains on")
    ap.add_argument("--out", default=str(ROOT / "results" / "product_reappearance.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    sdf = ROOT / f"grail_metabolism/data/{args.split}.sdf"
    triples = Path(args.triples) if args.triples else (
        ROOT / f"grail_metabolism/data/{args.split}_triples_clean.txt")
    smiles = [Chem.MolToSmiles(m) if m is not None else None
              for m in Chem.SDMolSupplier(str(sdf))]

    canon_subs, canon_prods = set(), set()
    pairs = []
    for a, b in _edges(smiles, triples):
        canon_subs.add(a)
        canon_prods.add(b)
        pairs.append((a, b))
    reappear = canon_prods & canon_subs

    # the same count under the criterion's own identity, so the dependence is visible
    from grail_metabolism.metrics import _tautomer_inchikey

    cache: dict = {}

    def key(s):
        if s not in cache:
            try:
                cache[s] = _tautomer_inchikey(s)
            except Exception:
                cache[s] = None
        return cache[s]

    tsubs = {k for k in (key(a) for a, _ in pairs) if k}
    tprods = {k for k in (key(b) for _, b in pairs) if k}
    treappear = tprods & tsubs

    rep = {"config": {**_code_version(), "split": args.split,
                      "triples": str(triples.relative_to(ROOT)),
                      "note": "a product reappears when it is annotated as a substrate anywhere "
                              "in the same split; only annotated (is_real=1) triples are edges"},
           "canonical_smiles": {"substrates": len(canon_subs), "products": len(canon_prods),
                                "products_that_reappear": len(reappear),
                                "share_percent": round(100.0 * len(reappear)
                                                       / max(len(canon_prods), 1), 4)},
           "tautomer_key": {"substrates": len(tsubs), "products": len(tprods),
                            "products_that_reappear": len(treappear),
                            "share_percent": round(100.0 * len(treappear)
                                                   / max(len(tprods), 1), 4)},
           "annotated_edges": len(pairs)}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    c, t = rep["canonical_smiles"], rep["tautomer_key"]
    print(f"  {rep['annotated_edges']} annotated edges in the {args.split} split")
    print(f"  canonical SMILES: {c['products_that_reappear']} of {c['products']} products "
          f"reappear as substrates, {c['share_percent']:.2f}%")
    print(f"  tautomer key:     {t['products_that_reappear']} of {t['products']}, "
          f"{t['share_percent']:.2f}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
