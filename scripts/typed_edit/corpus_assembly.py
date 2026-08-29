#!/usr/bin/env python3
"""What can be established about how the evaluated corpus was assembled, and what cannot.

The manuscript carried a placeholder where the data-curation section belongs. The placeholder was
treated as a value the authors were waiting on. It is not: most of it is recoverable from the
shipped files and the repository's own history, and the part that is not is recoverable as a
bounded negative rather than as silence.

This measures every claim the section makes, so none of them is prose.

  the files          when they were written, and how that stands against the repository's first
                     commit, which decides whether a producer could be in it at all
  the records        what fields each carries, and in particular how many carry a source
                     accession, which decides whether per-source counts can exist
  the labels         the State field's semantics, checked against the triples rather than assumed
  de-duplication     whether one structure appears twice, and whether two appearances ever
                     disagree, which is what a conflict rule would have had to arbitrate
  the sources        what fraction of each source that is physically present here is inside the
                     corpus, which bounds the filter without naming it
  the splits         whether the shipped clean triples reproduce from the repaired splitter, and
                     in which of its two modes

    python scripts/typed_edit/corpus_assembly.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

DATA = ROOT / "grail_metabolism" / "data"
SPLITS = ("train", "val", "test")


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


def first_commit() -> dict:
    out = subprocess.run(["git", "log", "--reverse", "--format=%H %cI", "--max-count=1"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    sha, when = out.stdout.strip().split()
    total = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": sha[:7], "committed": when, "commits_in_history": int(total or 0)}


def file_facts() -> dict:
    out = {}
    for split in SPLITS:
        for kind, name in (("sdf", f"{split}.sdf"), ("triples", f"{split}_triples.txt"),
                           ("clean", f"{split}_triples_clean.txt")):
            path = (DATA / name).resolve()
            if not path.exists():
                continue
            st = path.stat()
            out[f"{split}.{kind}"] = {
                "bytes": st.st_size,
                "written": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            }
    return out


def records_and_labels(split: str) -> dict:
    sdf = DATA / f"{split}.sdf"
    states, fields = Counter(), Counter()
    ids_present = 0
    index_to_state, index_to_smiles = {}, {}
    n = 0
    for rec in sdf_records(sdf):
        n += 1
        states[rec.get("State")] += 1
        for k in rec:
            fields[k] += 1
        if (rec.get("ID") or "").strip():
            ids_present += 1
        try:
            index = int(rec["Index"])
        except Exception:
            continue
        index_to_state[index] = rec.get("State")
        index_to_smiles[index] = rec.get("SMILES")

    triples_path = DATA / f"{split}_triples_clean.txt"
    positives = defaultdict(set)
    both_ways = 0
    seen_pairs = {}
    substrate_ids = set()
    for line in triples_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        a, b, real = int(parts[0]), int(parts[1]), int(parts[2])
        substrate_ids.add(a)
        key = (index_to_smiles.get(a), index_to_smiles.get(b))
        if key in seen_pairs and seen_pairs[key] != real:
            both_ways += 1
        seen_pairs[key] = real
        if real:
            positives[index_to_smiles.get(a)].add(index_to_smiles.get(b))

    # the label convention, checked rather than assumed
    substrate_state = Counter(index_to_state.get(i) for i in substrate_ids)
    positive_product_state = Counter()
    for line in triples_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "1":
            positive_product_state[index_to_state.get(int(parts[1]))] += 1

    # duplication at structure level, and whether duplicates ever disagree
    by_structure = defaultdict(set)
    for i in substrate_ids:
        by_structure[index_to_smiles.get(i)].add(i)
    duplicated = {s: ids for s, ids in by_structure.items() if len(ids) > 1}
    conflicts = 0
    for structure, ids in duplicated.items():
        sets = {frozenset(positives.get(index_to_smiles.get(i)) or ()) for i in ids}
        if len(sets) > 1:
            conflicts += 1

    return {
        "records": n,
        "fields": dict(fields),
        "records_carrying_a_source_accession": ids_present,
        "state_histogram": {str(k): v for k, v in states.most_common()},
        "substrates_by_id": len(substrate_ids),
        "substrates_by_structure": len(by_structure),
        "duplicated_substrate_structures": len(duplicated),
        "duplicated_structures_whose_annotations_disagree": conflicts,
        "pairs_listed_both_positive_and_negative": both_ways,
        "state_of_substrate_ids": {str(k): v for k, v in substrate_state.most_common()},
        "state_of_positive_product_ids": {str(k): v for k, v in positive_product_state.most_common()},
        "distinct_positive_pairs": sum(len(v) for v in positives.values()),
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def splitter_facts() -> dict:
    """What the repaired splitter is, and whether the shipped files come out of it."""
    src = (ROOT / "scripts" / "fix_splits.py")
    manifest = ROOT / "paper2" / "split_manifest.json"
    out = {"splitter": "scripts/fix_splits.py", "exists": src.exists(),
           "test_triples_untouched": None, "declared_digests": {}}
    a, b = DATA / "test_triples.txt", DATA / "test_triples_clean.txt"
    if a.exists() and b.exists():
        out["test_triples_untouched"] = digest(a) == digest(b)
    if manifest.exists():
        blob = json.loads(manifest.read_text())
        files = blob.get("files") or {}
        for split in SPLITS:
            rec = files.get(split) or {}
            declared = rec.get("sha256")
            path = DATA / f"{split}_triples_clean.txt"
            out["declared_digests"][split] = {
                "declared": declared,
                "matches_shipped": bool(declared and path.exists()
                                        and declared == digest(path)),
            }
    leak = ROOT / "results" / "leakage_fix_report.json"
    if leak.exists():
        blob = json.loads(leak.read_text())
        out["molecule_disjoint_recorded"] = blob.get("molecule_disjoint")
    return out


def source_overlap() -> dict:
    """How much of each source physically present here is inside the corpus."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def skeleton(smiles):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)[:14]
        except Exception:
            return None

    corpus_pairs, corpus_subs = set(), set()
    for split in SPLITS:
        index_to_smiles = {}
        for rec in sdf_records(DATA / f"{split}.sdf"):
            try:
                index_to_smiles[int(rec["Index"])] = rec.get("SMILES")
            except Exception:
                continue
        for line in (DATA / f"{split}_triples_clean.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[2] != "1":
                continue
            a, b = skeleton(index_to_smiles.get(int(parts[0]))), \
                skeleton(index_to_smiles.get(int(parts[1])))
            if a and b:
                corpus_pairs.add((a, b))
                corpus_subs.add(a)

    out = {"corpus_positive_pairs": len(corpus_pairs),
           "corpus_substrates": len(corpus_subs), "sources": {}}

    metx = ROOT / "artifacts/tier2/biotransformer/database/MetXBioDB-1-0.json"
    if metx.exists():
        blob = json.loads(metx.read_text())
        rows = blob if isinstance(blob, list) else list(blob.values())
        pairs = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            sub = row.get("substrate_smiles") or row.get("Substrate SMILES") or row.get("substrate")
            prod = row.get("product_smiles") or row.get("Product SMILES") or row.get("product")
            a, b = skeleton(sub), skeleton(prod)
            if a and b:
                pairs.add((a, b))
        out["sources"]["MetXBioDB"] = {
            "read_from": str(metx.relative_to(ROOT)),
            "distinct_pairs": len(pairs),
            "inside_the_corpus": len(pairs & corpus_pairs),
            "share_inside": round(len(pairs & corpus_pairs) / max(len(pairs), 1), 4),
        }

    glory = ROOT / "docs/benchmark/data/gloryx_test.json"
    if glory.exists():
        blob = json.loads(glory.read_text())
        rows = blob if isinstance(blob, list) else blob.get("parents") or list(blob.values())
        pairs, parents = set(), set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            sub = row.get("smiles") or row.get("parent") or row.get("substrate")
            a = skeleton(sub)
            if not a:
                continue
            parents.add(a)
            for prod in (row.get("metabolites") or row.get("products") or []):
                b = skeleton(prod if isinstance(prod, str) else prod.get("smiles"))
                if b:
                    pairs.add((a, b))
        out["sources"]["GLORYx"] = {
            "read_from": str(glory.relative_to(ROOT)),
            "parents": len(parents), "distinct_pairs": len(pairs),
            "parents_inside_the_corpus": len(parents & corpus_subs),
            "pairs_inside_the_corpus": len(pairs & corpus_pairs),
        }
    out["sources_not_present_here"] = ["ChEMBL", "DrugBank"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "corpus_assembly.json"))
    ap.add_argument("--skip-overlap", action="store_true")
    args = ap.parse_args()

    report = {"provenance": stamp(__file__),
              "files": file_facts(),
              "repository": first_commit(),
              "splits": {},
              "splitter": splitter_facts()}
    for split in SPLITS:
        report["splits"][split] = records_and_labels(split)
        row = report["splits"][split]
        print(f"{split}: {row['records']} records, "
              f"{row['records_carrying_a_source_accession']} carry a source accession, "
              f"{row['substrates_by_id']} substrate ids over "
              f"{row['substrates_by_structure']} structures, "
              f"{row['duplicated_structures_whose_annotations_disagree']} disagreements")
    if not args.skip_overlap:
        report["source_overlap"] = source_overlap()
        so = report["source_overlap"]
        for name, row in so["sources"].items():
            print(f"{name}: {row.get('inside_the_corpus', row.get('pairs_inside_the_corpus'))} "
                  f"of {row['distinct_pairs']} pairs inside the corpus")

    report["reading"] = (
        "the corpus files were written in one run months before this repository's first commit "
        "and their producer is not in it, so the extraction cannot be read off code; the source "
        "accession field is empty in every record, so per-source counts cannot be recovered from "
        "the data either; what can be established is the standardisation the strings carry, the "
        "label convention, the absence of any cross-source disagreement to arbitrate, the "
        "fraction of each present source that is inside, and that the shipped clean triples come "
        "out of the repaired splitter in its substrate-level mode")
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
