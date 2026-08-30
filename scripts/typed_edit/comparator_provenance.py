#!/usr/bin/env python3
"""What this repository can prove about each comparator: version, configuration, and when it ran.

A comparison is only as reproducible as its comparators, and this paper says so about other
people's work. Its own record was thinner than it should be: SyGMa carried a version and a
scenario, BioTransformer a jar digest and a checkout date, and MetaTox and MetaPredictor carried
neither a version nor a date. Two of the four are not installable packages -- one is a web service
and one is a source checkout -- so "the version" is not always a thing that exists. What does exist
in every case is the frozen prediction file, its digest, and the date it entered the repository,
which bounds when the run happened and is recoverable by anyone holding the history.

This records all of it in one place, marks each field as known or not available and says why, and
refuses to invent a version string for a service that does not publish one.

    python scripts/typed_edit/comparator_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def first_seen(rel: str) -> str | None:
    """The date the file entered the repository: an upper bound on when the run produced it."""
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ad", "--date=short",
                              "--", rel], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    dates = [line for line in out.stdout.splitlines() if line.strip()]
    return dates[-1] if dates else None


def digest(rel: str) -> str | None:
    path = ROOT / rel
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


COMPARATORS = {
    "SyGMa": {
        "kind": "installed Python package",
        "version": None,          # filled from the installed module below
        "configuration": "scenario [[phase1, 1], [phase2, 1]], one cycle of each ruleset",
        "predictions": "results/sygma_fulltest_predictions.json",
        "version_note": "read from the installed module at run time",
    },
    "MetaTox": {
        "kind": "web service",
        "version": None,
        "configuration": "layer 1 only, without the SMIRKS-rule variant, per the supplier's note",
        "predictions": "results/metatox_preds.json",
        "version_note": ("the service publishes no version string to a user, so none can be "
                         "recorded; the submission set and the returned predictions are pinned "
                         "instead, and the service is the one running at way2drug.ru on the date "
                         "below"),
    },
    "MetaPredictor": {
        "kind": "source checkout, run locally",
        "version": None,
        "configuration": "two-stage translation on CPU, the repository's own predict script",
        "predictions": "artifacts/tier2/metapredictor_preds.json",
        "version_note": ("the upstream release carries no version string; the checkout used is in "
                         "the repository under artifacts/tier2/metapredictor_src"),
    },
    "BioTransformer": {
        "kind": "jar",
        "version": None,          # read from the jar name the split manifest pins
        "build": None,
        "configuration": "the shipped metabolic reaction database, unmodified",
        "predictions": "artifacts/tier2/biotransformer_preds.json",
        "version_note": "the jar's own build stamp, with its digest in paper2/split_manifest.json",
    },
}


def main() -> int:
    try:
        import sygma
        COMPARATORS["SyGMa"]["version"] = getattr(sygma, "__version__", None)
    except Exception:
        COMPARATORS["SyGMa"]["version_note"] += "; the module was not importable in this run"

    # BioTransformer's version lives in the name of the jar the split manifest already pins, so it
    # is read from there rather than typed a second time and left to drift.
    import re

    manifest = json.loads((ROOT / "paper2/split_manifest.json").read_text())
    jar = (((manifest.get("third_party") or {}).get("biotransformer") or {}).get("jar") or {})
    match = re.search(r"BioTransformer([0-9.]+?)_(\d{8})\.jar", jar.get("name") or "")
    if match:
        COMPARATORS["BioTransformer"]["version"] = match.group(1)
        COMPARATORS["BioTransformer"]["build"] = match.group(2)

    rows = {}
    for name, row in COMPARATORS.items():
        rel = row["predictions"]
        rows[name] = {**row,
                      "predictions_sha256": digest(rel) if rel else None,
                      "predictions_first_in_the_repository": first_seen(rel) if rel else None}
    known = sum(1 for r in rows.values() if r["version"])
    report = {"provenance": stamp(__file__),
              "comparators": rows,
              "n_comparators": len(rows),
              "n_carrying_a_version_string": known,
              "reading": (
                  f"{known} of {len(rows)} comparators expose a version string. For the others the "
                  "record is the frozen predictions, their digest, and the date they entered this "
                  "repository, which bounds the run. A service that publishes no version cannot be "
                  "pinned by one, and reporting the bound is the honest alternative to inventing "
                  "it.")}
    (ROOT / "results/comparator_provenance.json").write_text(json.dumps(report, indent=1))
    for name, row in rows.items():
        print(f"{name:15s} version {str(row['version'] or '--'):18s} "
              f"predictions first seen {row['predictions_first_in_the_repository'] or '--'}")
    print(f"\n{known} of {len(rows)} carry a version string")
    print("wrote results/comparator_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
