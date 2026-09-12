#!/usr/bin/env python3
"""Package a self-contained deploy bundle for way2drug (or anyone), and nothing it should not carry.

What a recipient needs to run `grail-metabolites in.smi out.tsv`: the package, the released rule
bank, the released checkpoints, and the install metadata. What they must NOT get: the corpus (a
licence question the manuscript spends a section on), the full rule bank `extended_smirks.txt` or any
other file that carries the 611 BioTransformer templates, and the full-bank checkpoint (unused by the
deploy and carrying weight rows for those templates).

The bundle extracts to a single directory the recipient installs with `pip install -e .`.

    python scripts/pack_for_deploy.py --out ~/grail_deploy_dist
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
TOP = "grail-metabolites-deploy"  # the single directory the archive unpacks into

# Install metadata and licences, copied verbatim.
TOP_FILES = ["pyproject.toml", "poetry.lock", "README.md", "LICENSE", "NOTICE.md"]
# The released pair, the only model files the deploy loads.
CKPTS = ROOT / "artifacts" / "full5000_released" / "checkpoints"
CKPT_ARC = "artifacts/full5000_released/checkpoints"
RELEASED_BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks_released.txt"

# The one bank that ships. Every other bank file is dropped, because three of them
# (extended_smirks.txt, resources/notebooks_rules.txt, data/smirks.txt) carry the withheld
# BioTransformer templates verbatim, and the deploy needs only the released bank.
KEEP_RESOURCES = {"extended_smirks_released.txt"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_filter(t: tarfile.TarInfo):
    """Filter grail_metabolism/: drop the corpus, every bank but the released one, caches, pickles."""
    name = t.name
    parts = Path(name).parts
    base = Path(name).name
    if "__pycache__" in parts or base.endswith(".pyc") or base.endswith(".pkl"):
        return None
    # parts[0] is TOP, parts[1] == "grail_metabolism"
    sub = parts[2:] if len(parts) > 2 else ()
    if sub and sub[0] == "data":
        # the corpus and training-only featurisation; the deploy loads none of it
        return None
    if len(sub) >= 2 and sub[0] == "resources":
        return t if sub[-1] in KEEP_RESOURCES else None
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the archive into")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    gen, filt = CKPTS / "generator.pt", CKPTS / "filter.pt"
    for required in (RELEASED_BANK, gen, filt, *[ROOT / f for f in TOP_FILES if f in ("pyproject.toml",)]):
        if not required.exists():
            print(f"REFUSING: required deploy input is missing: {required.relative_to(ROOT)}",
                  file=sys.stderr)
            return 1
    # A full bank present in the tree is fine (it is filtered out); assert it never enters the archive.
    full_bank_arc = f"{TOP}/grail_metabolism/resources/extended_smirks.txt"

    archive = out / f"{TOP}.tar.gz"
    written = []
    with tarfile.open(archive, "w:gz", dereference=True) as tf:
        def record(t):
            t = _package_filter(t)
            if t is not None:
                written.append(t.name)
            return t

        tf.add(ROOT / "grail_metabolism", arcname=f"{TOP}/grail_metabolism", filter=record)
        tf.add(ROOT / "deploy", arcname=f"{TOP}/deploy")
        tf.add(gen, arcname=f"{TOP}/{CKPT_ARC}/generator.pt")
        tf.add(filt, arcname=f"{TOP}/{CKPT_ARC}/filter.pt")
        for f in TOP_FILES:
            p = ROOT / f
            if p.exists():
                tf.add(p, arcname=f"{TOP}/{f}")

    # Safety gate: the withheld-template carriers must not be in the archive.
    forbidden = [n for n in written if Path(n).name in
                 {"extended_smirks.txt", "notebooks_rules.txt", "smirks.txt", "merged_smirks.txt"}]
    if forbidden or full_bank_arc in written:
        archive.unlink()
        print(f"REFUSING: a withheld-template bank leaked into the archive: {forbidden}",
              file=sys.stderr)
        return 1

    with tarfile.open(archive) as tf:
        members = tf.getnames()
    bank_in = f"{TOP}/grail_metabolism/resources/extended_smirks_released.txt" in members
    ckpt_in = all(f"{TOP}/{CKPT_ARC}/{n}" in members for n in ("generator.pt", "filter.pt"))
    corpus_leak = [m for m in members if "/grail_metabolism/data/" in m]

    manifest = {
        "written_by": "scripts/pack_for_deploy.py",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "archive": archive.name, "sha256": _sha(archive), "bytes": archive.stat().st_size,
        "members": len(members),
        "contains": {"released_bank": bank_in, "released_checkpoints": ckpt_in},
        "excludes": {"corpus_files_in_archive": len(corpus_leak),
                     "full_bank_extended_smirks_txt": full_bank_arc in members,
                     "biotransformer_carriers": forbidden},
        "released_bank_rules": sum(1 for _ in open(RELEASED_BANK)),
        "install": "extract, then from the extracted dir run `pip install -e .`; then "
                   "`grail-metabolites in.smi out.tsv`. See deploy/README.md.",
        "notice": ("excludes the BioTransformer templates (released bank only) and the corpus; "
                   "the generator checkpoint is the deployed generator subset to the released bank."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"  {archive.name}  {archive.stat().st_size / 1e6:.1f} MB  ({len(members)} members)")
    print(f"  released bank: {bank_in} ({manifest['released_bank_rules']} rules) | "
          f"checkpoints: {ckpt_in} | corpus files: {len(corpus_leak)} | full bank present: "
          f"{full_bank_arc in members}")
    print(f"  wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
