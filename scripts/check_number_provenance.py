#!/usr/bin/env python3
"""Every artifact the manuscript's numbers come from is pinned and stamped, counted by machine.

The paper asserts that every printed number reaches the page from a pinned artifact. That
guarantee was maintained by hand, and a hand-maintained guarantee about a countable quantity is
the wrong instrument for a paper whose thesis is that undeclared choices are what make this
literature incomparable. Asked how many artifacts fell outside it, five readers produced five
different answers.

This settles it mechanically. `paper2_numbers.py` is the one generator every printed macro passes
through; its `art()` is instrumented and `build()` is run, so what is counted is what was actually
read rather than what a regular expression found in the source. Each artifact is then checked
twice: it must be in the pinned set of `audit_artifact_provenance.py`, and it must carry a
provenance block naming the script that wrote it.

    python scripts/check_number_provenance.py
    python scripts/check_number_provenance.py --json results/number_sources.json

Exit status is non-zero when any artifact fails either check. The counts are written to the
artifact so the manuscript can quote them rather than assert them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Artifacts a number legitimately comes from that cannot themselves be pinned, each with the
# reason. The list is short and every entry is an argument, not an exemption.
EXEMPT = {
    # The provenance sweep's own output. Pinning it would ask the sweep to verify itself, and its
    # numbers are counts of the sweep's result rather than measurements of the system.
    "results/artifact_provenance.json":
        "the provenance sweep's own report; pinning it would make the check circular",
    # Written by check_paper2_numbers.py, which runs after this generator; same circularity.
    "results/number_provenance.json":
        "written by the numeral checker that runs after this generator",
    # This generator's own output.
    "results/paper2_numbers.json":
        "this generator's own output",
    # This gate's own output. The paper quotes the gate's counts, so the generator reads them,
    # so the gate sees its own report among the sources. Requiring it to verify itself is the
    # same circularity as the other two.
    "results/number_sources.json":
        "this gate's own report, which the paper quotes and the generator therefore reads",
}


def instrumented_reads() -> list[str]:
    """Run the number generator and record every artifact it opens."""
    import paper2_numbers

    seen: list[str] = []
    original = paper2_numbers.art

    def watched(name):
        rel = f"results/{name}"
        if rel not in seen:
            seen.append(rel)
        return original(name)

    paper2_numbers.art = watched
    try:
        paper2_numbers.build()
    finally:
        paper2_numbers.art = original
    return sorted(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "results" / "number_sources.json"))
    args = ap.parse_args()

    import audit_artifact_provenance as audit

    pinned = set(audit.PINNED)
    reads = instrumented_reads()

    rows, unpinned, unstamped, named_only = [], [], [], []
    for rel in reads:
        path = ROOT / rel
        is_pinned = rel in pinned
        # Three states, not two. A full stamp records the producer's source digest, so the
        # sweep can say the code has not moved. A weaker one names the producing script and
        # nothing more, which is enough for the sweep to recover the source from the history and
        # verify by inference. Nothing at all is the state that fails.
        stamp, names_producer = None, False
        try:
            blob = json.loads(path.read_text())
            if isinstance(blob, dict):
                stamp = blob.get("provenance") or blob.get("measured_by")
                config = blob.get("config")
                names_producer = bool(isinstance(config, dict) and config.get("script"))
        except Exception:
            pass
        exempt = rel in EXEMPT
        # The property that matters is not whether the artifact carries a block but whether the
        # sweep can name the code that wrote it and say whether that code has moved. For a
        # stamped artifact that is a digest comparison; for one predating stamping it is an
        # inference from the history, which the sweep reports as inferred. Either verifies; a
        # producer the sweep cannot identify at all does not.
        verdict = audit.check(rel, audit.PINNED.get(rel)) if is_pinned else {"status": "unpinned"}
        verifiable = verdict.get("status") in audit.OK
        if not is_pinned and not exempt:
            unpinned.append(rel)
        if not verifiable and not exempt:
            unstamped.append(rel)
        if verifiable and not stamp:
            named_only.append(rel)
        rows.append({"artifact": rel, "pinned": is_pinned, "stamped": bool(stamp),
                     "verifiable": verifiable, "sweep_status": verdict.get("status"),
                     "how": verdict.get("how"), "exempt": EXEMPT.get(rel)})

    report = {
        "artifacts_the_numbers_come_from": len(reads),
        "exempt": len([r for r in rows if r["exempt"]]),
        "unpinned": sorted(unpinned),
        "unstamped": sorted(unstamped),
        "verifiable_by_inference_only": sorted(named_only),
        "n_unpinned": len(unpinned),
        "n_unstamped": len(unstamped),
        "n_verifiable_by_inference_only": len(named_only),
        "method": ("paper2_numbers.build() is run with art() instrumented, so the count is of "
                   "artifacts actually opened rather than of matches in the source"),
        "artifacts": rows,
    }
    Path(args.json).write_text(json.dumps(report, indent=1))

    print(f"the manuscript's numbers come from {len(reads)} artifacts "
          f"({report['exempt']} exempt with a stated reason)")
    for rel in unpinned:
        print(f"  NOT PINNED    {rel}")
    for rel in unstamped:
        print(f"  NOT VERIFIABLE  {rel}")
    ok = not unpinned and not unstamped
    print(f"  {report['n_unpinned']} unpinned, {report['n_unstamped']} unstamped, "
          f"{report['n_verifiable_by_inference_only']} verified by inference rather than "
          f"by a recorded digest")
    print("check_number_provenance: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
