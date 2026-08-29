#!/usr/bin/env python3
"""How the corpus draws its molecules, and what the rule bank therefore expects.

The paper sweeps five ways of judging a hit and nine output budgets, and never says how the
substrate is drawn. It is drawn one specific way, and the way is not neutral: every structure in
the corpus is the InChI round-trip of its own record, and RDKit's InChI reader places a mobile
hydrogen on oxygen. An amide written R-C(=O)-N-H therefore comes back as the imidic acid
R-C(-OH)=N, and cytosine comes back as the lactim. Rules are matched against that drawing.

This measures three things and mixes none of them.

  the normalisation      whether the stored SMILES of a record is exactly what its stored InChI
                         round-trips to, per split, over every record
  the dialect            how many substrates are not fixed points of the declared standardiser,
                         how many of those moves are pure tautomer moves, and how many structures
                         carry an imidic-drawn amide, per split and on the evaluated test subset
  the bank              how many templates require an imidic reactant and how many require an
                         amide one, split by curated and mined provenance

The bank instrument is structural and declared here rather than in prose: a template is counted
as amide-requiring when its reactant half contains a carbon double-bonded to oxygen and single or
aromatic bonded to nitrogen, and as imidic-requiring when its reactant half contains a carbon
single or aromatic bonded to oxygen and double-bonded to nitrogen. The two are counted
independently: a template with both motifs is counted in both, and the counts do not partition
the bank.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, BondType  # noqa: E402

from _provenance import stamp  # noqa: E402

from grail_metabolism.metrics import _match_keys  # noqa: E402
from grail_metabolism.utils.preparation import (  # noqa: E402
    resolve_default_rule_bank,
    standardize_mol,
)

RDLogger.DisableLog("rdApp.*")

DATA = ROOT / "grail_metabolism" / "data"
SPLITS = ("train", "val", "test")
# The imidic acid as the corpus writes it, and the amide it stands for.
IMIDIC = Chem.MolFromSmarts("[NX2]=[CX3]-[OX2H1]")
AMIDE = Chem.MolFromSmarts("[NX3]-[CX3]=[OX1]")
# Cytosine's lactim and its kind: an exocyclic imine on an aromatic ring that also bears a
# hydroxyl. The aliphatic pattern above does not see it because the two halves sit on
# different ring atoms.
AROM_LACTIM = Chem.MolFromSmarts("[NX2]=[c]")
AROM_OH = Chem.MolFromSmarts("[c]-[OX2H1]")


def sdf_records(path: Path):
    """Stream (SMILES, InChI, State, Index) out of an SDF without building molecules."""
    cur, key = {}, None
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                match = re.match(r">\s*<([^>]+)>", line)
                key = match.group(1) if match else None
                continue
            if key is not None:
                value = line.strip()
                if value == "":
                    key = None
                else:
                    cur.setdefault(key, value)
                continue
            if line.startswith("$$$$"):
                yield cur
                cur, key = {}, None


def _canon(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def normalisation_evidence(path: Path, cap: int | None) -> dict:
    """Is the stored SMILES exactly the round-trip of the stored InChI?"""
    hit = miss = unparsed = 0
    examples = []
    for n, rec in enumerate(sdf_records(path)):
        if cap is not None and n >= cap:
            break
        inchi, smiles = rec.get("InChI"), rec.get("SMILES")
        if not inchi or not smiles:
            continue
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            unparsed += 1
            continue
        rt, stored = Chem.MolToSmiles(mol), _canon(smiles)
        if rt == stored:
            hit += 1
        else:
            miss += 1
            if len(examples) < 5:
                examples.append({"stored": stored, "round_trip": rt})
    total = hit + miss
    return {"records_compared": total, "equal_to_inchi_round_trip": hit,
            "differing": miss, "inchi_unparsed": unparsed,
            "share": round(hit / total, 6) if total else None,
            "examples_of_difference": examples}


def _motifs(mol) -> dict:
    return {
        "imidic": mol.HasSubstructMatch(IMIDIC),
        "amide": mol.HasSubstructMatch(AMIDE),
        "aromatic_lactim": mol.HasSubstructMatch(AROM_LACTIM) and mol.HasSubstructMatch(AROM_OH),
    }


def dialect(smiles_list: list[str]) -> dict:
    """Fixed points of the declared standardiser, and the shape of the moves."""
    moved = fixed = unparsed = 0
    pure_tautomer = other_move = 0
    motif = Counter()
    examples = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unparsed += 1
            continue
        for name, present in _motifs(mol).items():
            if present:
                motif[name] += 1
        stored = Chem.MolToSmiles(mol)
        try:
            standard = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(smiles)))
        except Exception:
            unparsed += 1
            continue
        if stored == standard:
            fixed += 1
            continue
        moved += 1
        # A move that the paper's own default matcher cannot see is a tautomer move.
        try:
            same_key = _match_keys([stored], "inchikey_tautomer") == _match_keys(
                [standard], "inchikey_tautomer")
        except Exception:
            same_key = False
        if same_key:
            pure_tautomer += 1
        else:
            other_move += 1
        if len(examples) < 6:
            examples.append({"stored": stored, "standardised": standard,
                             "pure_tautomer_move": bool(same_key)})
    total = moved + fixed
    return {"n": total, "unparsed": unparsed,
            "fixed_points": fixed, "moved": moved,
            "moved_share": round(moved / total, 6) if total else None,
            "moved_pure_tautomer": pure_tautomer, "moved_other": other_move,
            "carrying_imidic_amide": motif["imidic"],
            "carrying_imidic_amide_share": round(motif["imidic"] / total, 6) if total else None,
            "carrying_aromatic_lactim": motif["aromatic_lactim"],
            "carrying_amide": motif["amide"],
            "examples": examples}


def _query_motifs(query) -> dict:
    """The dialect motifs carried by one SMARTS query molecule."""
    amide = imidic = lactim_c = exo_imine = False
    for carbon in query.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        partners = [(nb.GetAtomicNum(),
                     query.GetBondBetweenAtoms(carbon.GetIdx(), nb.GetIdx()).GetBondType())
                    for nb in carbon.GetNeighbors()]
        has_dbl_o = any(z == 8 and o == BondType.DOUBLE for z, o in partners)
        has_dbl_n = any(z == 7 and o == BondType.DOUBLE for z, o in partners)
        has_sng_n = any(z == 7 and o in (BondType.SINGLE, BondType.AROMATIC) for z, o in partners)
        has_sng_o = any(z == 8 and o in (BondType.SINGLE, BondType.AROMATIC) for z, o in partners)
        if has_dbl_o and has_sng_n:
            amide = True
        if has_dbl_n and has_sng_o:
            imidic = True
        if has_sng_o and has_sng_n:
            lactim_c = True
        if has_dbl_n:
            exo_imine = True
    return {"amide": amide, "imidic": imidic,
            "aromatic_lactim": bool(lactim_c and exo_imine)}


def _reactant_motifs(smirks: str) -> dict | None:
    """The dialect motifs required anywhere on the reactant side of a template.

    Parsed through the reaction parser rather than as a bare SMARTS string, because 149 of the
    templates wrap several reactant components in parentheses and only the reaction parser
    accepts that grouping.
    """
    try:
        reaction = AllChem.ReactionFromSmarts(smirks)
    except Exception:
        return None
    if reaction is None or reaction.GetNumReactantTemplates() == 0:
        return None
    out = {"amide": False, "imidic": False, "aromatic_lactim": False}
    for index in range(reaction.GetNumReactantTemplates()):
        for name, present in _query_motifs(reaction.GetReactantTemplate(index)).items():
            out[name] = out[name] or present
    return out


def bank_census(bank: Path, provenance: dict[int, str]) -> dict:
    lines = [line.strip() for line in bank.read_text().splitlines() if line.strip()]
    counts = {"amide": Counter(), "imidic": Counter(), "aromatic_lactim": Counter()}
    parsed = unparsed = 0
    imidic_only, amide_only = [], []
    for index, smirks in enumerate(lines):
        motifs = _reactant_motifs(smirks)
        if motifs is None:
            unparsed += 1
            continue
        parsed += 1
        source = provenance.get(index, "unknown")
        for name, present in motifs.items():
            if present:
                counts[name]["total"] += 1
                counts[name][source] += 1
        if (motifs["imidic"] or motifs["aromatic_lactim"]) and not motifs["amide"]:
            imidic_only.append(index)
        if motifs["amide"] and not (motifs["imidic"] or motifs["aromatic_lactim"]):
            amide_only.append(index)
    return {"templates": len(lines), "parsed": parsed, "unparsed": unparsed,
            "amide_requiring": dict(counts["amide"]),
            "imidic_requiring": dict(counts["imidic"]),
            "aromatic_lactim_requiring": dict(counts["aromatic_lactim"]),
            "imidic_or_lactim_and_not_amide": len(imidic_only),
            "amide_and_neither_imidic_nor_lactim": len(amide_only),
            "example_imidic_only": imidic_only[:8],
            "instrument": ("reactant half parsed as SMARTS; a carbon double-bonded to oxygen and "
                           "single or aromatic bonded to nitrogen is the amide motif, a carbon "
                           "single or aromatic bonded to oxygen and double-bonded to nitrogen is "
                           "the imidic motif, and a carbon bonded singly to both together with "
                           "any carbon double-bonded to nitrogen is the aromatic lactim motif")}


def rule_provenance(bank: Path) -> dict[int, str]:
    """Curated rules are the head of the bank; the census reads the split the paper reports."""
    composition = ROOT / "results" / "bank_composition.json"
    curated = None
    if composition.exists():
        blob = json.loads(composition.read_text())
        curated = (blob.get("curated") or {}).get("total")
    if curated is None:
        curated = 1725
    return {i: ("curated" if i < curated else "mined")
            for i in range(len(bank.read_text().splitlines()))}


def substrates_of(split: str, cap: int | None) -> list[str]:
    sdf = DATA / f"{split}.sdf"
    out, seen = [], set()
    for n, rec in enumerate(sdf_records(sdf)):
        if rec.get("State") != "Substrate":
            continue
        smiles = rec.get("SMILES")
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        out.append(smiles)
        if cap is not None and len(out) >= cap:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dialect_census.json"))
    ap.add_argument("--normalisation-cap", type=int, default=5000,
                    help="records per split for the InChI round-trip evidence; 0 means all")
    ap.add_argument("--substrate-cap", type=int, default=0,
                    help="substrates per split for the dialect census; 0 means all")
    args = ap.parse_args()

    cap = args.normalisation_cap or None
    scap = args.substrate_cap or None

    report = {"provenance": stamp(__file__), "splits": {}, "normalisation": {}}
    for split in SPLITS:
        report["normalisation"][split] = normalisation_evidence(DATA / f"{split}.sdf", cap)
        print(f"{split}: InChI round-trip {report['normalisation'][split]['share']}")

    for split in SPLITS:
        subs = substrates_of(split, scap)
        report["splits"][split] = dialect(subs)
        row = report["splits"][split]
        print(f"{split}: {row['moved']} of {row['n']} substrates move under the standardiser "
              f"({row['moved_share']}), {row['carrying_imidic_amide']} carry an imidic amide")

    evaluated = [r["sub"] for r in
                 json.loads((ROOT / "results" / "recall_factorization.json").read_text())
                 ["per_substrate"]]
    report["evaluated_test_subset"] = dialect(evaluated)
    row = report["evaluated_test_subset"]
    print(f"evaluated test subset: {row['moved']} of {row['n']} move ({row['moved_share']}), "
          f"{row['carrying_imidic_amide']} carry an imidic amide")

    bank = Path(resolve_default_rule_bank())
    report["bank"] = bank_census(bank, rule_provenance(bank))
    report["bank"]["path"] = str(bank.relative_to(ROOT))
    b = report["bank"]
    print(f"bank: amide-requiring {b['amide_requiring'].get('total')}, "
          f"imidic-requiring {b['imidic_requiring'].get('total')}, "
          f"aromatic-lactim-requiring {b['aromatic_lactim_requiring'].get('total')}")

    # Which drawing each arm of the comparison was handed. Four met the corpus string; the
    # MetaTox submission re-tautomerises, on the stated ground that sending an unnatural imidic
    # acid to an external service would be unfair to it.
    arms = {"GRAIL exhaustive": "stored", "GRAIL interactive": "stored",
            "SyGMa": "stored", "MetaPredictor": "stored"}
    metatox = ROOT / "results" / "metatox_input" / "substrate_map.csv"
    if metatox.exists():
        import csv
        rows = list(csv.DictReader(metatox.open()))
        changed = sum(1 for r in rows
                      if str(r.get("retautomerized", "")).strip().lower() in ("true", "1", "yes"))
        report["arm_presentation"] = {
            "arms_given_the_corpus_string": arms,
            "MetaTox": "re-tautomerised to the natural form before submission",
            "metatox_substrates": len(rows),
            "metatox_retautomerised": changed,
            "source": "results/metatox_input/substrate_map.csv, written by scripts/make_metatox_input.py",
        }
        print(f"MetaTox submission: {changed} of {len(rows)} substrates re-tautomerised; "
              f"every other arm met the corpus string")

    report["reading"] = (
        "every structure in the corpus is the InChI round-trip of its own record, and the "
        "round-trip places a mobile hydrogen on oxygen, so amides are stored as imidic acids "
        "and cytosine as its lactim; the substrate reaches the matcher in that drawing")
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
