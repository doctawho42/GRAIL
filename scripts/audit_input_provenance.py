#!/usr/bin/env python3
"""Provenance checking that does not stop one level from the number.

`_provenance.stamp` records which code wrote an artifact and `record_inputs` records the digests of
the files it read. `audit_artifact_provenance.py` then verifies, for a pinned list, that the
producer has not moved since it ran. That apparatus is one level deep. It asks whether an
artifact's own producer is current; it never asks whether the artifacts that artifact READ have
producers at all.

The difference is not hypothetical here. `results/deployment_table.json` is stamped, pinned and
current, and every recall figure in the released column comes out of it. Its candidate pools come
from `results/widepools_k30/all.json`, which records the checkpoints it used, by digest, and
records nothing about what turned those checkpoints into pools. `results/retraining_spread.json` is
in the same position: the spread its numbers are quoted in units of is computed from three
`results/seedpools/interactive_seed*.json` files that name no producer and that no script in this
repository writes. Verifying the consumer tells a reader nothing about that.

So this walks the edge instead of the node. For every tracked artifact under `results/` that
records its inputs, it follows each JSON input and asks the same question of it, and reports the
ones that cannot answer.

Three answers are distinguished, because they are three different situations and lumping them
together would make the count useless:

    stamped        the input carries a producer, and the existing audit already covers it
    half           the input records what it READ, by digest, but not what wrote it. A reader can
                   check the inputs were the intended ones and cannot check the step between them
    bare           the input is a mapping from substrate to predictions with no header at all, so
                   there is nowhere for provenance to live without changing the format

Files this project does not produce are declared, with the reason, rather than being silently
absent from the count: a third party's database has no producer of ours by definition and its
provenance is a citation. A census that called those defects would be a detector nobody could
trust, which is the failure mode this file exists to avoid rather than to repeat.

    python scripts/audit_input_provenance.py            # census, exit 0
    python scripts/audit_input_provenance.py --strict   # exit 1 if any consumer got worse
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

# Inputs this project does not write. Their provenance is a citation, not a stamp, and counting
# them as unprovenanced would inflate the census with things no change here could fix.
NOT_OURS = {
    "grail_metabolism/resources/external/":
        "third-party rule and reaction databases, redistributed under their own terms or read in "
        "place; NOTICE.md carries the attribution and the producer records each digest",
    "artifacts/tier2/biotransformer/":
        "BioTransformer's own distribution, obtainable from its project",
    "artifacts/tier2/metapredictor_src/":
        "MetaPredictor's own source checkout",
    "artifacts/external/gao2026/":
        "the prediction files Gao et al. deposited on Zenodo (10.5281/zenodo.17878495, CC BY 4.0)",
    "docs/benchmark/data/":
        "published benchmark test sets, read as distributed",
}

# The count the tree is expected to be at. A census that only ever reports is a census nobody
# reads; --strict fails when a consumer acquires an unstamped input it did not have, which is the
# direction that matters. Raising this number is a decision, not a fix.
EXPECTED_CONSUMERS = 24


def _tracked() -> set:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return set(out.stdout.split())


def _classify(path: Path) -> str:
    """stamped, half, bare, or unreadable -- what this file records about where it came from."""
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return "unreadable"
    if not isinstance(blob, dict):
        return "bare"
    if isinstance(blob.get("provenance"), dict) and blob["provenance"].get("script_path"):
        return "stamped"
    # Records the files it read, by digest, but not the step that read them. A reader can check the
    # inputs and not the transformation.
    for field in ("checkpoints", "inputs", "config", "source"):
        if field in blob:
            return "half"
    return "bare"


def _why_not_ours(rel: str):
    for prefix, reason in NOT_OURS.items():
        if rel.startswith(prefix):
            return reason
    return None


def build() -> dict:
    tracked = _tracked()
    consumers, tally = [], {"stamped": 0, "half": 0, "bare": 0, "unreadable": 0, "not_ours": 0}
    seen_inputs = {}

    for rel in sorted(p for p in tracked if p.startswith("results/") and p.endswith(".json")):
        path = ROOT / rel
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict) or not isinstance(blob.get("inputs"), list):
            continue
        weak = []
        for item in blob["inputs"]:
            ip = item.get("path") if isinstance(item, dict) else None
            if not ip or not ip.endswith(".json"):
                continue
            target = ROOT / ip
            if not target.exists():
                continue
            reason = _why_not_ours(ip)
            if reason:
                if ip not in seen_inputs:
                    tally["not_ours"] += 1
                seen_inputs[ip] = "not_ours"
                continue
            kind = _classify(target)
            if ip not in seen_inputs:
                tally[kind] += 1
            seen_inputs[ip] = kind
            if kind in ("half", "bare", "unreadable"):
                weak.append({"input": ip, "records": kind, "tracked": ip in tracked})
        if weak:
            consumers.append({"artifact": rel, "inputs_recorded": len(blob["inputs"]),
                              "inputs_without_a_producer": len(weak), "which": weak})

    consumers.sort(key=lambda c: -c["inputs_without_a_producer"])
    return {
        "provenance": stamp(__file__),
        "question": ("whether the artifacts a manuscript number is computed from can themselves "
                     "say what produced them, which the existing audit never asks"),
        "how": ("every tracked results/*.json that records its inputs is walked, each JSON input "
                "is classified by what it records about its own origin, and inputs this project "
                "does not write are declared rather than counted"),
        "classes": {
            "stamped": "carries a producer; the existing audit covers it",
            "half": "records what it read, by digest, but not what wrote it",
            "bare": "no header at all, so provenance has nowhere to live without a format change",
            "not_ours": "a third party's file; its provenance is a citation",
        },
        "distinct_inputs_by_class": tally,
        "consumers_reading_an_input_without_a_producer": len(consumers),
        "expected": EXPECTED_CONSUMERS,
        "consumers": consumers,
        "not_ours": NOT_OURS,
        "reading": ("A consumer on this list is not wrong. It is unverifiable one step back: its "
                    "own producer can be checked and the step that made its inputs cannot. The "
                    "ones that matter most are the ones whose numbers reach the manuscript."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when more consumers read an unprovenanced input than expected")
    args = ap.parse_args()

    report = build()
    out = ROOT / "results" / "input_provenance.json"
    out.write_text(json.dumps(report, indent=1))

    t = report["distinct_inputs_by_class"]
    print(f"\n  distinct JSON inputs read by tracked artifacts")
    for k in ("stamped", "half", "bare", "unreadable", "not_ours"):
        print(f"    {k:12s} {t[k]}")
    n = report["consumers_reading_an_input_without_a_producer"]
    print(f"\n  {n} tracked artifacts read at least one input that cannot say what produced it")
    for c in report["consumers"][:8]:
        print(f"    {c['artifact']:52s} {c['inputs_without_a_producer']} of "
              f"{c['inputs_recorded']}")
    if len(report["consumers"]) > 8:
        print(f"    ... and {len(report['consumers']) - 8} more, all in the artifact")
    print(f"\nwrote {out.relative_to(ROOT)}")

    if args.strict and n > EXPECTED_CONSUMERS:
        print(f"\nREFUSING: {n} consumers read an input with no producer, against the "
              f"{EXPECTED_CONSUMERS} this tree is known to have. Something acquired one. Either "
              f"stamp the new input's producer or raise EXPECTED_CONSUMERS deliberately.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
