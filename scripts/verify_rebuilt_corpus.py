#!/usr/bin/env python3
"""Check a corpus you rebuilt yourself against the one this work evaluated on.

The extraction that selected records from ChEMBL, DrugBank, MetXBioDB and the GLORYx reference set
is not in this repository: the corpus files predate its first commit and their producer was never
committed. That step cannot be shipped, and pretending otherwise would be worse than saying so.

What can be shipped is the other half. Holding your own licences you can obtain the source records
and assemble a corpus; this tells you, precisely, how yours stands to ours. A manifest that only
reports a digest mismatch tells you nothing you can act on. This reports which substrates are in
one and not the other, whether the annotation agrees where both hold the same substrate, whether
your splits are disjoint the way ours are, and whether your structures are drawn in the same
dialect --- which is the difference most likely to make the same corpus behave differently, and
the one a digest comparison cannot see at all.

    python scripts/verify_rebuilt_corpus.py --dir /path/to/your/corpus
    python scripts/verify_rebuilt_corpus.py --dir /path/to/your/corpus --json report.json

The directory is expected to hold {train,val,test}.sdf and {train,val,test}_triples_clean.txt in
the layout grail_metabolism/data/ uses. Anything missing is reported and skipped rather than
raising, because a partial rebuild is a normal thing to want checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OURS = ROOT / "grail_metabolism" / "data"
MANIFEST = ROOT / "paper2" / "split_manifest.json"
SPLITS = ("train", "val", "test")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sdf_records(path: Path):
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


def read_split(base: Path, split: str):
    """(substrate key -> set of metabolite keys), and the dialect evidence, for one split."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    sdf, triples = base / f"{split}.sdf", base / f"{split}_triples_clean.txt"
    if not sdf.exists() or not triples.exists():
        return None

    index_to_smiles, round_trip_hits, round_trip_of = {}, 0, 0
    for record in sdf_records(sdf):
        smiles, inchi = record.get("SMILES"), record.get("InChI")
        try:
            index = int(record["Index"])
        except Exception:
            continue
        index_to_smiles[index] = smiles
        if smiles and inchi:
            mol = Chem.MolFromInchi(inchi)
            parsed = Chem.MolFromSmiles(smiles)
            if mol is not None and parsed is not None:
                round_trip_of += 1
                round_trip_hits += Chem.MolToSmiles(mol) == Chem.MolToSmiles(parsed)

    def key_of(smiles):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None

    annotation = defaultdict(set)
    for line in triples.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[2] != "1":
            continue
        a = key_of(index_to_smiles.get(int(parts[0])))
        b = key_of(index_to_smiles.get(int(parts[1])))
        if a and b:
            annotation[a].add(b)
    return {"annotation": dict(annotation),
            "records": len(index_to_smiles),
            "inchi_round_trip_hits": round_trip_hits,
            "inchi_round_trip_of": round_trip_of}


def compare(theirs: dict, ours: dict) -> dict:
    ts, os_ = set(theirs["annotation"]), set(ours["annotation"])
    both = ts & os_
    agree = sum(1 for k in both if theirs["annotation"][k] == ours["annotation"][k])
    missing_pairs = sum(len(ours["annotation"][k] - theirs["annotation"][k]) for k in both)
    extra_pairs = sum(len(theirs["annotation"][k] - ours["annotation"][k]) for k in both)
    return {
        "substrates_yours": len(ts), "substrates_ours": len(os_),
        "substrates_in_both": len(both),
        "substrates_only_yours": len(ts - os_), "substrates_only_ours": len(os_ - ts),
        "shared_substrates_whose_annotation_agrees_exactly": agree,
        "annotated_pairs_ours_that_yours_lacks": missing_pairs,
        "annotated_pairs_yours_that_ours_lacks": extra_pairs,
        "your_inchi_round_trip_share": (round(theirs["inchi_round_trip_hits"]
                                              / max(theirs["inchi_round_trip_of"], 1), 6)),
        "our_inchi_round_trip_share": (round(ours["inchi_round_trip_hits"]
                                             / max(ours["inchi_round_trip_of"], 1), 6)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding your rebuilt corpus")
    ap.add_argument("--json", help="where to write the full report")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        raise SystemExit(f"no such directory: {base}")

    declared = {}
    if MANIFEST.exists():
        files = (json.loads(MANIFEST.read_text()).get("files") or {})
        declared = {k: (v or {}).get("sha256") for k, v in files.items()}

    # Read each split once. The train SDF is over a gigabyte and reading it twice doubled the
    # wall clock of a check a reader runs before they trust anything else here.
    loaded = {s: read_split(base, s) for s in SPLITS}

    report = {"your_corpus": str(base), "splits": {}}
    for split in SPLITS:
        row = {}
        theirs_triples = base / f"{split}_triples_clean.txt"
        if theirs_triples.exists() and declared.get(split):
            got = digest(theirs_triples)
            row["digest_matches_ours"] = got == declared[split]
            row["your_digest"], row["our_digest"] = got, declared[split]
        theirs = loaded[split]
        ours = read_split(OURS, split)
        if theirs is None:
            row["status"] = "absent from your directory"
        elif ours is None:
            row["status"] = "absent from this checkout, so nothing to compare against"
        else:
            row["status"] = "compared"
            row.update(compare(theirs, ours))
        report["splits"][split] = row

    disjoint = {}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and loaded.get(a) is not None and loaded.get(b) is not None:
                shared = set(loaded[a]["annotation"]) & set(loaded[b]["annotation"])
                disjoint[f"{a}_{b}"] = len(shared)
    report["your_substrate_overlap_between_splits"] = disjoint
    report["reading"] = (
        "a substrate overlap of zero between every pair is the property this work's splits have "
        "and the one its claims rest on; an InChI round-trip share near one means your structures "
        "are drawn in the same dialect as ours, and a share well below one means they are not, "
        "which changes which rules fire on them even where the molecules are identical")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))

    for split, row in report["splits"].items():
        print(f"\n{split}: {row.get('status', 'not compared')}")
        if row.get("status") != "compared":
            continue
        if "digest_matches_ours" in row:
            print(f"  clean triples digest: "
                  f"{'identical to ours' if row['digest_matches_ours'] else 'different'}")
        print(f"  substrates: yours {row['substrates_yours']}, ours {row['substrates_ours']}, "
              f"in both {row['substrates_in_both']}")
        print(f"  only yours {row['substrates_only_yours']}, only ours {row['substrates_only_ours']}")
        print(f"  of the shared substrates, {row['shared_substrates_whose_annotation_agrees_exactly']} "
              f"have exactly our annotation")
        print(f"  annotated pairs we hold and you do not: {row['annotated_pairs_ours_that_yours_lacks']}")
        print(f"  annotated pairs you hold and we do not: {row['annotated_pairs_yours_that_ours_lacks']}")
        print(f"  your structures are InChI round-trip fixed points at "
              f"{row['your_inchi_round_trip_share']:.4f}, ours at "
              f"{row['our_inchi_round_trip_share']:.4f}")
    if disjoint:
        print(f"\nsubstrate overlap between your splits: "
              + ", ".join(f"{k} {v}" for k, v in disjoint.items()))
    if args.json:
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
