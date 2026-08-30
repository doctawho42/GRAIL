#!/usr/bin/env python3
"""What each method recovers, split by the chemistry of the transformation rather than by budget.

Every recall figure in this paper aggregates over transformations of very different kinds. A
reader who wants to know whether a system is usable on their compound class cannot read that off a
single number, and a referee is entitled to ask which chemistry each method misses. This splits the
comparison set's annotated references into named biotransformation classes and reports recall
within each class for every arm.

The class of a (substrate, metabolite) pair is read from the change in molecular formula, which is
what the annotation actually determines: the corpus records a substrate and a product and not a
mechanism, so a mechanism cannot be recovered from it without guessing. A formula delta is
therefore a class of transformations and not a single reaction -- an added oxygen is aliphatic
hydroxylation, aromatic hydroxylation, N-oxidation or S-oxidation, and this cannot tell them
apart. The classes are named for what the delta is, the ambiguity is stated with each, and pairs
whose delta matches nothing in the vocabulary are reported as a class of their own rather than
dropped.

    python scripts/typed_edit/error_by_chemistry.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

CAP = 100
BUDGETS = (5, 15, 30)

# Formula deltas as {element: change}, with the name each one carries in this literature and what
# the name does not distinguish. Ordered: the first match wins, so a composite conjugation is
# tested before the plain addition it contains.
CLASSES = [
    ("glucuronidation", {"C": 6, "H": 8, "O": 6},
     "addition of glucuronic acid, at any acceptor"),
    ("glutathione conjugation", {"C": 10, "H": 15, "N": 3, "O": 6, "S": 1},
     "addition of glutathione"),
    ("sulfation", {"O": 3, "S": 1}, "addition of a sulfo group, at any acceptor"),
    ("acetylation", {"C": 2, "H": 2, "O": 1}, "addition of an acetyl group"),
    ("methylation", {"C": 1, "H": 2}, "addition of a methyl group"),
    ("glycine conjugation", {"C": 2, "H": 3, "N": 1, "O": 1}, "addition of glycine, minus water"),
    ("oxidation, one oxygen added", {"O": 1},
     "aliphatic or aromatic hydroxylation, N-oxidation or S-oxidation; these are not distinguished"),
    ("oxidation, two oxygens added", {"O": 2}, "two of the above, or a dihydrodiol"),
    ("dehydrogenation", {"H": -2},
     "alcohol to carbonyl, or an introduced double bond"),
    ("oxidation to a carbonyl", {"O": 1, "H": -2},
     "hydroxylation followed by dehydrogenation, counted as one step"),
    ("demethylation", {"C": -1, "H": -2}, "N-, O- or S-demethylation; these are not distinguished"),
    ("oxidative demethylation", {"C": -1, "H": -2, "O": -1},
     "loss of a methoxy group as methanol"),
    ("hydrolysis", {"H": 2, "O": 1},
     "amide, ester or epoxide hydrolysis; these are not distinguished"),
    ("reduction", {"H": 2}, "carbonyl to alcohol, or a reduced double bond"),
    ("dehalogenation, chlorine", {"Cl": -1, "H": 1}, "replacement of chlorine by hydrogen"),
    ("dehalogenation, fluorine", {"F": -1, "H": 1}, "replacement of fluorine by hydrogen"),
    ("deamination", {"N": -1, "H": -1, "O": 1}, "amine to carbonyl or alcohol"),
    ("desaturation and hydroxylation", {"O": 1, "H": -2, "C": 0}, "one oxygen added, two hydrogens lost"),
]


def formula(mol) -> Counter:
    from rdkit import Chem

    counts = Counter()
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return counts


def classify(sub_mol, met_mol) -> tuple:
    """(class name, the delta) for one pair. Fragmentation and unnamed deltas get their own class."""
    a, b = formula(sub_mol), formula(met_mol)
    delta = {el: b.get(el, 0) - a.get(el, 0) for el in set(a) | set(b)}
    delta = {el: n for el, n in delta.items() if n}
    if not delta:
        return "isomerisation, no formula change", delta
    for name, want, _ in CLASSES:
        if delta == want:
            return name, delta
    # A metabolite smaller than its substrate in carbon is a cleavage of some kind; the annotation
    # does not say which bond, so the class says only that.
    if delta.get("C", 0) <= -2:
        return "cleavage, two or more carbons lost", delta
    return "other", delta


def main() -> int:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey  # noqa: F401

    refs_by_smiles = json.loads((ROOT / "results/test_references.json").read_text())

    def load(spec):
        pools, refs = {}, {}
        for f in sorted(glob.glob(str(ROOT / spec))) or [str(ROOT / spec)]:
            blob = json.loads(Path(f).read_text())
            pools.update(blob["pools"]); refs.update(blob["references"])
        return pools, refs

    big, refs_b = load("results/widepools_implicit/w*.json")
    small, refs_s = load("results/widepools_k30/all.json")
    refs = {**refs_b, **refs_s}
    subs = sorted(s for s in set(big) & set(small) if refs.get(s))

    comparators = {"metatox": ("results/metatox_smirks_preds.json", "predictions"),
                   "sygma": ("results/sygma_fulltest_predictions.json", None),
                   "metapredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None)}
    arms = {}
    parent = {s: tautkey(s) for s in subs}

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    def ranked(pool, s):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        return drop_parent([c["key"] for c in rrf_order(keep)], s)

    arms["GRAIL exhaustive"] = {s: ranked(big[s], s) for s in subs}
    arms["GRAIL interactive"] = {s: ranked(small[s], s) for s in subs}
    for name, (rel, key) in comparators.items():
        path = ROOT / rel
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        preds = blob[key] if key else blob
        arms[name] = {s: drop_parent(_dedup(preds.get(s, []), max(BUDGETS) + 5), s) for s in subs}

    # Join each reference key back to the structure it came from, so the pair can be classified.
    # A reference whose structure is not recoverable is counted and excluded rather than guessed.
    # Two flat tables rather than one nested record: the counts and the hits are indexed the same
    # way and never read together, and keeping them apart keeps each one a single type.
    refs_in_class = Counter()
    hits_in_class = defaultdict(Counter)          # (class, arm) -> {budget: hits}
    unresolved = 0
    for s in subs:
        sub_mol = Chem.MolFromSmiles(s)
        wanted = set(refs[s])
        if sub_mol is None:
            unresolved += len(wanted)
            continue
        structures = {}
        for met in refs_by_smiles.get(s, []):
            met_mol = Chem.MolFromSmiles(met)
            if met_mol is None:
                continue
            try:
                structures[tautkey(met)] = met_mol
            except Exception:
                continue
        for key in wanted:
            met_mol = structures.get(key)
            if met_mol is None:
                unresolved += 1
                continue
            name, _ = classify(sub_mol, met_mol)
            refs_in_class[name] += 1
            for arm, lists in arms.items():
                got = lists[s]
                for k in BUDGETS:
                    if key in set(got[:k]):
                        hits_in_class[(name, arm)][k] += 1

    table = {}
    for name, n_refs in sorted(refs_in_class.items(), key=lambda kv: -kv[1]):
        entry = {"references": n_refs, "recall": {}}
        for arm in arms:
            hits = hits_in_class[(name, arm)]
            entry["recall"][arm] = {str(k): round(hits[k] / n_refs, 4) for k in BUDGETS}
        entry["what_the_name_does_not_distinguish"] = next(
            (note for cname, _, note in CLASSES if cname == name), None)
        table[name] = entry

    total = sum(v["references"] for v in table.values())
    report = {
        "provenance": stamp(__file__),
        "population": {"substrates": len(subs), "references_classified": total,
                       "references_whose_structure_could_not_be_recovered": unresolved,
                       "note": "the comparison set, the population every other comparison uses"},
        "criterion": "tautomer-aware InChIKey, as everywhere else",
        "budgets": list(BUDGETS),
        "classes": table,
        "reading": (
            "A class is a change in molecular formula, which is what the annotation determines. "
            "It groups mechanisms the corpus does not separate, and each row says which. Recall "
            "within a class is over that class's references only, so classes with few references "
            "carry wide uncertainty and no interval is quoted for them here."),
    }
    (ROOT / "results/error_by_chemistry.json").write_text(json.dumps(report, indent=1))
    print(f"{len(subs)} substrates, {total} references classified, {unresolved} unresolved\n")
    head = "class".ljust(38) + "n".rjust(5) + "".join(
        f"  {a[:12]:>12s}" for a in arms)
    print(head)
    for name, entry in table.items():
        cells = "".join(f"  {entry['recall'][a]['15']:12.3f}" for a in arms)
        print(f"{name[:38]:38s}{entry['references']:5d}{cells}")
    print("\n(recall at a budget of 15)")
    print("wrote results/error_by_chemistry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
