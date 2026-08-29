#!/usr/bin/env python3
"""The candidate pools, packaged for Zenodo with digests the paper can be checked against.

Two artifacts the paper depends on are too large for the repository: the train and validation
candidate pools, about 48 MB each, holding every candidate the generator produced for every
substrate of their population together with both component scores. Everything else the paper
pins is committed; these two are deposited instead.

A deposit that a reader cannot check against the paper is a file somewhere, not an artifact. So
this writes a manifest carrying, for each file, the digest of the raw JSON the pipeline reads,
the digest of the gzip actually uploaded, and the record counts -- substrates, candidates,
references -- so a download can be verified without trusting the size alone. The gzip is written
with a zero mtime and a fixed compression level, because a gzip that embeds the clock has a
different digest on every build and cannot be pinned.

    python scripts/zenodo_deposit.py --build            # bundle + manifest
    python scripts/zenodo_deposit.py --verify           # recheck a bundle against the manifest
    python scripts/zenodo_deposit.py --upload           # needs ZENODO_TOKEN in the environment

The upload step is deliberately separate and never runs as part of --build: it publishes to an
external service under the author's account, and that is the author's action.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from _provenance import stamp  # noqa: E402

FILES = ("results/train_pools.json", "results/val_pools.json")
BUNDLE = ROOT / "paper2" / "zenodo_bundle"
MANIFEST = ROOT / "paper2" / "zenodo_manifest.json"
GZIP_LEVEL = 9

TITLE = "GRAIL candidate pools: per-substrate rule-application candidates with generator and filter scores"
# ChEMBL 34 is CC BY-SA 3.0 and DrugBank 5.1.10 is CC BY-NC 4.0, and those do not combine: no
# single licence can be correct for a derivative holding records from both. The question is
# closed by not raising it. Every corpus structure -- substrates as well as annotated
# metabolites -- is replaced by its tautomer-canonical InChIKey before the deposit is built, so
# what is uploaded holds no source record. What remains is this work's own output, the candidate
# structures and the two scores, and that carries CC BY 4.0.
LICENSE = "cc-by-4.0"
DESCRIPTION = """<p>Candidate pools for the train and validation populations of GRAIL, a
rule-grounded predictor of xenobiotic metabolite structures. Each file records, for every
substrate of its population, every candidate structure produced by applying the rule bank,
together with the generator's probability for the rule that produced it and the filter's score
for the (substrate, candidate) pair.</p>

<p>These two files are the only artifacts the accompanying manuscript pins that are too large to
commit; the remaining pinned artifacts, the evaluation harness, the split manifest, the frozen
predictions of every comparator and the preregistration are in the repository at
<a href="https://github.com/doctawho42/GRAIL">github.com/doctawho42/GRAIL</a>.</p>

<p>Each file carries its own provenance stamp: the producing script, the SHA-256 of that
script's source, and the commit it ran at. <code>paper2/zenodo_manifest.json</code> in the
repository pins the SHA-256 of both the raw JSON and the uploaded gzip, so a download can be
verified against the version the paper used.</p>"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gzip_deterministic(src: Path, dst: Path) -> None:
    """Compress with a zero mtime, so the digest depends on the content and nothing else."""
    with src.open("rb") as fin, dst.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=GZIP_LEVEL, mtime=0) as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)


def counts(path: Path) -> dict:
    d = json.loads(path.read_text())
    pools, refs = d.get("pools", {}), d.get("references", {})
    return {
        "split": d.get("split"),
        "match": d.get("match"),
        "substrates": len(pools),
        "candidates": sum(len(v) for v in pools.values()),
        "references": sum(len(v) for v in refs.values()),
        "producer": d["provenance"]["script"],
        "producer_sha256": d["provenance"]["source_sha256"],
        "producer_commit": d["provenance"]["git_commit"],
    }


def hash_keyed(src: Path, dst: Path) -> dict:
    """Replace every corpus structure with its tautomer-canonical InChIKey.

    The pools are keyed by the substrate's SMILES, and the reference lists beside them are
    keyed the same way. A SMILES is the structure; an InChIKey is a hash of it. Keying by the
    hash is what makes the deposit hold no source record, and it costs a reader nothing they
    cannot recover: holding their own licences for the four sources, they compute the same key
    from their own copy of the structure and join on it.

    The candidate structures are left as they are. They are produced by applying this work's
    rule bank and are not corpus records.
    """
    from grail_metabolism.metrics import _tautomer_inchikey as key_of

    blob = json.loads(src.read_text())
    pools, references = blob.get("pools") or {}, blob.get("references") or {}
    keyed_pools, keyed_refs, collisions, unkeyable = {}, {}, 0, 0
    for substrate, candidates in pools.items():
        k = key_of(substrate)
        if not k:
            unkeyable += 1
            continue
        if k in keyed_pools:
            collisions += 1
            continue
        keyed_pools[k] = candidates
        if substrate in references:
            keyed_refs[k] = references[substrate]
    out = dict(blob)
    out["pools"], out["references"] = keyed_pools, keyed_refs
    out["substrate_keying"] = {
        "key": "tautomer-canonical InChIKey of the substrate",
        "why": ("the corpus records are drawn from sources whose licences do not combine, so no "
                "source structure is redistributed; the key is a hash and the candidates beside "
                "it are this work's output"),
        "substrates_in": len(pools), "substrates_out": len(keyed_pools),
        "unkeyable": unkeyable, "collisions_dropped": collisions,
    }
    dst.write_text(json.dumps(out, indent=1))
    return out["substrate_keying"]


def build() -> dict:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel in FILES:
        raw = ROOT / rel
        if not raw.exists():
            raise SystemExit(f"missing: {rel}. The pools are gitignored; build them first.")
        src = raw.with_name(raw.stem + "_keyed.json")
        keying = hash_keyed(raw, src)
        gz = BUNDLE / (src.name + ".gz")
        gzip_deterministic(src, gz)
        entries.append({
            "name": gz.name,
            "source": rel,
            "derived": str(src.relative_to(ROOT)),
            "substrate_keying": keying,
            "bytes_raw": src.stat().st_size,
            "bytes_gz": gz.stat().st_size,
            "sha256_raw": sha256(src),
            "sha256_gz": sha256(gz),
            **counts(src),
        })
    rep = {
        "provenance": stamp(__file__),
        "deposit": {"title": TITLE, "license": LICENSE, "doi": None,
                    "note": "doi is filled by --upload, or by hand from the Zenodo record"},
        "gzip": {"level": GZIP_LEVEL, "mtime": 0,
                 "note": "fixed so the compressed digest is a function of the content alone"},
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(rep, indent=1))
    return rep


def verify(where=None) -> int:
    """Check a bundle against the manifest. `where` is the directory a reader downloaded into.

    A missing gzip is a failed download and fails. A missing raw JSON is the ordinary state of a
    fresh clone, where the pools are gitignored and not yet unpacked, and is reported without
    failing -- otherwise the check every reader runs first would always be red.
    """
    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST}; run --build")
    rep = json.loads(MANIFEST.read_text())
    where = Path(where) if where else BUNDLE
    bad = 0
    for e in rep["files"]:
        gz = where / e["name"]
        # the digest is of the file that was compressed, which is the hash-keyed derivative and
        # not the raw pool it came from; checking the raw pool here reported a mismatch on a
        # bundle that was correct
        raw = ROOT / (e.get("derived") or e["source"])
        for label, path, want in (("gz", gz, e["sha256_gz"]), ("raw", raw, e["sha256_raw"])):
            if not path.exists():
                print(f"  {e['name']:<28} {label:<4} "
                      + ("absent (not unpacked)" if label == "raw" else "ABSENT"))
                bad += label == "gz"
                continue
            got = sha256(path)
            ok = got == want
            print(f"  {e['name']:<28} {label:<4} {'ok' if ok else 'MOVED ' + got[:12]}")
            bad += not ok
    print(f"\n{'all files match the manifest' if not bad else f'{bad} mismatches'}")
    return 1 if bad else 0


def upload() -> int:
    # A published Zenodo record cannot be withdrawn, so the last gate before one is created is
    # that the deposit holds no source structure. The manifest records the keying; if a file in
    # it was built without one, this refuses regardless of the licence.
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        unkeyed = [e["name"] for e in manifest.get("files", []) if not e.get("substrate_keying")]
        if unkeyed:
            raise SystemExit(
                "These bundle files were built before substrate hash-keying and would "
                f"redistribute corpus structures: {unkeyed}\nRebuild with --build first.")
    if LICENSE is None:
        raise SystemExit(
            "The deposit has no settled licence and will not be uploaded.\n"
            "  ChEMBL 34 is CC BY-SA 3.0: a derivative must carry the same or a compatible\n"
            "    ShareAlike licence and may not add restrictions.\n"
            "  DrugBank 5.1.10 is CC BY-NC 4.0: commercial use is forbidden.\n"
            "  MetXBioDB 1.0 ships with BioTransformer, whose terms require explicit permission\n"
            "    for redistribution.\n"
            "These do not combine. The deposit holds corpus substrate structures, hashes of the\n"
            "annotated metabolites, and candidates produced by this work; the metabolites\n"
            "themselves are not redistributed in structural form.\n"
            "Set LICENSE in this file once the position is settled. Publishing a Zenodo record\n"
            "cannot be undone.")
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set. Create a personal access token at\n"
            "  https://zenodo.org/account/settings/applications/tokens/new/\n"
            "with the deposit:write and deposit:actions scopes, then\n"
            "  ZENODO_TOKEN=... python scripts/zenodo_deposit.py --upload")
    import requests

    rep = json.loads(MANIFEST.read_text())
    api, auth = "https://zenodo.org/api", {"access_token": token}

    r = requests.post(f"{api}/deposit/depositions", params=auth, json={})
    r.raise_for_status()
    dep = r.json()
    bucket, dep_id = dep["links"]["bucket"], dep["id"]
    print(f"deposition {dep_id} created")

    for e in rep["files"]:
        gz = BUNDLE / e["name"]
        with gz.open("rb") as fh:
            requests.put(f"{bucket}/{e['name']}", data=fh, params=auth).raise_for_status()
        print(f"  uploaded {e['name']} ({e['bytes_gz'] / 1e6:.1f} MB)")

    meta = {"metadata": {
        "title": rep["deposit"]["title"], "upload_type": "dataset",
        "description": DESCRIPTION, "license": rep["deposit"]["license"],
        "related_identifiers": [{"identifier": "https://github.com/doctawho42/GRAIL",
                                 "relation": "isSupplementTo", "scheme": "url"}]}}
    requests.put(f"{api}/deposit/depositions/{dep_id}", params=auth, json=meta).raise_for_status()

    print(f"\nDraft ready, not published: https://zenodo.org/deposit/{dep_id}\n"
          "Add the author list there and press Publish. Publishing mints the DOI and is\n"
          "irreversible: a Zenodo record cannot be withdrawn, only superseded by a new version.\n"
          "Then put the DOI into paper2/zenodo_manifest.json and the manuscript.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--dir", default=None,
                    help="directory holding a downloaded bundle "
                         "(default: paper2/zenodo_bundle)")
    a = ap.parse_args()
    if a.upload:
        return upload()
    if a.verify:
        return verify(a.dir)
    if a.build or not any((a.build, a.verify, a.upload)):
        rep = build()
        for e in rep["files"]:
            print(f"  {e['name']:<28} {e['bytes_raw'] / 1e6:6.1f} -> {e['bytes_gz'] / 1e6:5.1f} MB"
                  f"   {e['substrates']} substrates, {e['candidates']:,} candidates")
        print(f"\nmanifest: {MANIFEST.relative_to(ROOT)}")
        print(f"bundle:   {BUNDLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
