#!/usr/bin/env python3
"""Package what a Kaggle run needs, and nothing it does not.

Training the full split needs the corpus, the rule bank and the package. It does not need the
artifacts, the results, the manuscripts or the third-party distributions, and putting those on a
platform this project has no arrangement with would be careless in a way the paper spends a section
warning others about.

Two archives are written, because they are two decisions:

    grail-code.tar.gz   the package, its config and the rule bank. No licence question: this is
                        ours and GPL-3.
    grail-corpus.tar.gz the three splits and their clean triples. This IS a licence question. The
                        corpus is assembled from ChEMBL under ShareAlike and DrugBank under
                        NonCommercial, the manuscript's availability section says a derivative
                        drawing on both cannot satisfy either, and it is not redistributed. A
                        private dataset on a third party's platform is not publication, but it is a
                        transfer to a third party under that party's hosting terms, and it is the
                        author's decision rather than this script's. The script writes the archive
                        and says so; uploading it is a separate act.

    python scripts/pack_for_kaggle.py --out ~/kaggle_upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything the training entry point imports or reads, and nothing else.
CODE = ["grail_metabolism", "configs/paper_full_converged.yaml", "requirements.txt",
        "pyproject.toml", "README.md", "LICENSE", "NOTICE.md"]
CORPUS = ["train.sdf", "val.sdf", "test.sdf",
          "train_triples_clean.txt", "val_triples_clean.txt", "test_triples_clean.txt"]
DATA = ROOT / "grail_metabolism" / "data"
# The bank the paper measures on, which is the one a retraining must use to be comparable. It is
# gitignored here for the licence reason NOTICE.md gives and travels inside the code archive.
BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the archives into")
    ap.add_argument("--skip-corpus", action="store_true",
                    help="write only the code archive, leaving the licence decision unmade")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    missing = [p for p in CODE if not (ROOT / p).exists()]
    if missing:
        print(f"REFUSING: not in this checkout: {', '.join(missing)}", file=sys.stderr)
        return 1
    if not BANK.exists():
        print(f"REFUSING: {BANK.relative_to(ROOT)} is absent, and a retraining on the released "
              f"bank would not be comparable with anything this paper reports", file=sys.stderr)
        return 1

    manifest = {"written_by": "scripts/pack_for_kaggle.py",
                "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
                "code": {}, "corpus": {}}

    # grail_metabolism/data holds 633 MB of corpus, most of it symlinks into another checkout.
    # The package's own featurisation files live there too and are needed, so the directory is
    # filtered rather than excluded: what travels is what the loader reads and is not the corpus.
    KEEP_IN_DATA = {"pca_ats.pkl", "pca_ats_single.pkl", "pca_bonds.pkl", "pca_bonds_single.pkl",
                    "smirks.txt", "merged_smirks.txt", "reactions.txt", "xtracted.txt"}

    def _code_filter(t):
        if "__pycache__" in t.name or t.name.endswith(".pyc"):
            return None
        parts = Path(t.name).parts
        if len(parts) > 2 and parts[:2] == ("grail_metabolism", "data"):
            return t if parts[-1] in KEEP_IN_DATA else None
        return t

    code_tar = out / "grail-code.tar.gz"
    with tarfile.open(code_tar, "w:gz") as tf:
        for rel in CODE:
            tf.add(ROOT / rel, arcname=rel, filter=_code_filter)
        tf.add(BANK, arcname="grail_metabolism/resources/extended_smirks.txt")
    manifest["code"] = {"archive": code_tar.name, "sha256": _sha(code_tar),
                        "bytes": code_tar.stat().st_size}
    print(f"  {code_tar.name}  {code_tar.stat().st_size / 1e6:.1f} MB")

    if not args.skip_corpus:
        absent = [f for f in CORPUS if not (DATA / f).exists()]
        if absent:
            print(f"REFUSING: the corpus is not in this checkout: {', '.join(absent)}",
                  file=sys.stderr)
            return 1
        corpus_tar = out / "grail-corpus.tar.gz"
        # dereference: these are symlinks into the main checkout, and an archive of symlinks
        # arrives on another machine as six broken names and no corpus.
        with tarfile.open(corpus_tar, "w:gz", dereference=True) as tf:
            for f in CORPUS:
                tf.add(DATA / f, arcname=f)
        manifest["corpus"] = {"archive": corpus_tar.name, "sha256": _sha(corpus_tar),
                              "bytes": corpus_tar.stat().st_size,
                              "licence_note": (
                                  "assembled from ChEMBL (ShareAlike) and DrugBank "
                                  "(NonCommercial); not redistributed by this project. Uploading "
                                  "this archive transfers it to a third party under that party's "
                                  "hosting terms and is the author's decision.")}
        print(f"  {corpus_tar.name}  {corpus_tar.stat().st_size / 1e6:.1f} MB")
        print("\n  The corpus archive carries third-party terms. Read its note in the manifest "
              "before uploading it anywhere.")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out}/manifest.json with the digests, so what arrives can be checked against "
          f"what left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
