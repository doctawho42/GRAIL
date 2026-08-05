#!/usr/bin/env python3
"""How much do two independent curations of the same drug's metabolites agree?

Three architecturally unrelated systems converge at recall@15 of 0.55 to 0.585 on this benchmark --
a pure rule engine, a sequence-to-sequence model and a transformer with a filter. When unrelated
methods land in the same place the binding constraint is usually a property of the data rather than
of the models, and the candidate that can be tested with what is already on disk is the reference
itself: annotated metabolite sets average 1.9 to 2.4 per substrate, where a real drug produces
dozens, most of them minor. If the reference is a publication-biased sample of the true map, recall
measures agreement with which metabolites got reported, not whether the map was predicted.

The measurement that separates those is inter-source agreement, and it needs no model. GLORYx ships
a curated set of 37 drugs with literature-traced metabolites, and this repository's corpus curates
its own. Where the two name the same parent, their metabolite sets can be compared to each other
under the same matching criteria the paper uses for predictions.

The reading is binary and both outcomes matter. Agreement near the models' own scores means the
benchmark is saturated: the missing 41% is curator disagreement rather than model error, and no
method can go higher without predicting which source a reference came from. Agreement near one means
the plateau is a modelling failure and the headroom is real.

Reported as: each source's recall of the other, their Jaccard, and the same under all five criteria,
since a disagreement that is only tautomeric is not a disagreement about chemistry.
"""
from __future__ import annotations
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")

GLORYX = ROOT / "docs" / "benchmark" / "data" / "gloryx_test.json"
DATA = ROOT / "grail_metabolism" / "data"
MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
OUT = ROOT / "results" / "annotation_agreement.json"


def key_of(smiles: str, mode: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if mode == "canonical":
            # isomericSmiles=False, matching grail_metabolism.metrics._canonical_key. RDKit's
            # default is stereo-AWARE, and keying this rung that way while every method table in
            # the paper goes through the harness made the curator ladder's strictest endpoint a
            # different match rule from Table 1's canonical column: 0.075 against 0.241.
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        if mode == "inchikey":
            return Chem.MolToInchiKey(mol)
        if mode == "inchi_no_stereo":
            return Chem.MolToInchiKey(mol).split("-")[0]
        if mode == "tanimoto1":
            from rdkit.Chem import rdFingerprintGenerator
            g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            return tuple(g.GetFingerprint(mol).GetOnBits())
        if mode == "inchikey_tautomer":
            return _tautomer_inchikey(smiles)
    except Exception:
        return None
    return None


def load_gloryx() -> dict:
    """GLORYx parents to their flattened metabolite lists, all generations."""
    raw = GLORYX.read_text()
    fixed = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", raw)  # the file carries unescaped backslashes
    data = json.loads(fixed)

    def flatten(nodes):
        out = []
        for n in nodes or []:
            if n.get("smiles"):
                out.append(n["smiles"])
            out.extend(flatten(n.get("metabolites")))
        return out

    return {p["smiles"]: flatten(p.get("metabolites")) for p in data if p.get("smiles")}


def load_corpus() -> dict:
    """This repository's own curation: substrate SMILES to annotated metabolite SMILES."""
    out: dict = {}
    for split in ("train", "val", "test"):
        sdf, trip = DATA / f"{split}.sdf", DATA / f"{split}_triples_clean.txt"
        if not sdf.exists() or not trip.exists():
            continue
        by_id = {}
        for mol in Chem.SDMolSupplier(str(sdf)):
            if mol is None:
                continue
            p = mol.GetPropsAsDict()
            idx = str(p.get("Index", ""))
            smi = p.get("SMILES") or Chem.MolToSmiles(mol)
            if idx and smi:
                by_id[idx] = smi
        with open(trip) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 3 or parts[2] != "1":
                    continue
                s, m = by_id.get(parts[0]), by_id.get(parts[1])
                if s and m:
                    out.setdefault(s, set()).add(m)
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    gl, corp = load_gloryx(), load_corpus()
    print(f"GLORYx {len(gl)} parents; corpus {len(corp)} substrates", flush=True)

    # Match parents across sources on the tautomer-aware key, the paper's default.
    corp_by_key = {}
    for s in corp:
        k = key_of(s, "inchikey_tautomer")
        if k:
            corp_by_key.setdefault(k, s)
    pairs = []
    for gs in gl:
        k = key_of(gs, "inchikey_tautomer")
        if k and k in corp_by_key:
            pairs.append((gs, corp_by_key[k]))
    print(f"{len(pairs)} parents curated by BOTH sources\n", flush=True)
    if not pairs:
        raise SystemExit("no overlapping parents -- the two curations share no drug")

    rep = {"n_gloryx": len(gl), "n_corpus": len(corp), "n_shared": len(pairs), "by_mode": {}}
    print(f"{'criterion':>20}{'|GLORYx|':>10}{'|corpus|':>10}{'G rec C':>9}{'C rec G':>9}{'Jaccard':>9}")
    for mode in MODES:
        rg, rc, jac, sg, sc = [], [], [], [], []
        for gs, cs in pairs:
            a = {k for k in (key_of(x, mode) for x in gl[gs]) if k}
            b = {k for k in (key_of(x, mode) for x in corp[cs]) if k}
            if not a or not b:
                continue
            inter = len(a & b)
            rg.append(inter / len(a))          # what fraction of GLORYx's set the corpus recovers
            rc.append(inter / len(b))
            jac.append(inter / len(a | b))
            sg.append(len(a))
            sc.append(len(b))
        rep["by_mode"][mode] = {
            "n_pairs_scored": len(jac),
            "mean_gloryx_set": round(st.mean(sg), 2), "mean_corpus_set": round(st.mean(sc), 2),
            "gloryx_recovered_by_corpus": round(st.mean(rg), 4),
            "corpus_recovered_by_gloryx": round(st.mean(rc), 4),
            "jaccard": round(st.mean(jac), 4),
        }
        m = rep["by_mode"][mode]
        print(f"{mode:>20}{m['mean_gloryx_set']:10.2f}{m['mean_corpus_set']:10.2f}"
              f"{m['gloryx_recovered_by_corpus']:9.4f}{m['corpus_recovered_by_gloryx']:9.4f}"
              f"{m['jaccard']:9.4f}")

    best = rep["by_mode"]["inchikey_tautomer"]
    print(f"\nunder the paper's default criterion, two independent curations of the same drugs agree")
    print(f"  Jaccard {best['jaccard']:.3f}; each recovers {best['gloryx_recovered_by_corpus']:.3f} "
          f"and {best['corpus_recovered_by_gloryx']:.3f} of the other")
    print(f"  for comparison, the best published method reaches recall@15 0.585 on this benchmark")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
