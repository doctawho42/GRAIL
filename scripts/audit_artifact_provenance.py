#!/usr/bin/env python3
"""Every number the registration leans on, checked against the code that wrote it.

Artifacts here are gated on write and read blind, and this project has three of its own
defects from that asymmetry. The sharpest is `bank_without_selection.json`, which carried
`ceiling_on_this_subset: 0.7284` through a correction that took the same quantity to 0.8007:
the value went on looking committed because nothing that read it asked which version of the
code had produced it. That artifact is the first thing this sweep flags.

Two strengths of evidence, and they are labelled apart.

  recorded    the artifact carries the digest of its producer's source, written at the same
              moment as the numbers. Nothing is assumed.
  inferred    the artifact predates stamping. The commit that ADDED it is recovered from the
              log and the producer as it stood there is hashed. That is the producer at write
              time only if the script did not change between the run and the commit -- an
              assumption about how the work was done, not a fact the artifact records.

PINNED lists the artifacts the preregistration and the freeze depend on. Those must be current;
everything else is counted and reported. A changed producer does not prove a number wrong, it
proves nobody has checked, which is the state all three of this project's defects shared.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import CURRENT, infer, verify  # noqa: E402

TE = "scripts/typed_edit"
PINNED = {
    # the type vocabulary the H1 stratum and the appendix's counts are keyed to
    "results/typed_edit_type_curve.json": f"{TE}/type_curve.py",
    "results/typed_edit_type_carriers.json": f"{TE}/type_carrier_provenance.py",
    # the strata H1 and H6 are registered on
    "results/h1_stratum.json": f"{TE}/build_h1_stratum.py",
    "results/h1_join_sensitivity.json": f"{TE}/h1_join_sensitivity.py",
    "results/h6_stratum.json": f"{TE}/build_h6_stratum.py",
    # the measurements H5 and the step-0 verdict rest on
    "results/typed_edit_known_pairs.json": f"{TE}/known_type_recovery.py",
    "results/typed_edit_known_type_recovery.json": f"{TE}/known_type_recovery.py",
    "results/typed_edit_step0.json": f"{TE}/run_bank.py",
    "results/typed_edit_relaxation.json": f"{TE}/relaxation_ladder.py",
    # the emission comparison, and the freeze itself
    "results/emission_leaderboard.json": f"{TE}/emission_leaderboard.py",
    "paper2/split_manifest.json": f"{TE}/freeze_split.py",
}


def check(rel: str, producer: str | None) -> dict:
    path = ROOT / rel
    if not path.exists():
        return {"artifact": rel, "status": "absent", "detail": "not in this checkout"}
    v = verify(path)
    if v["status"] == CURRENT or producer is None:
        return v
    if v["status"] == "unstamped":
        return infer(path, producer)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also sweep every other artifact")
    ap.add_argument("--out", default=str(ROOT / "results" / "artifact_provenance.json"))
    args = ap.parse_args()

    pinned = [check(rel, prod) for rel, prod in sorted(PINNED.items())]
    bad = [r for r in pinned if r["status"] != CURRENT]

    print(f"{'artifact':<46}{'status':<18}evidence")
    for r in pinned:
        how = "recorded" if r.get("how") == "recorded source digest" else (
            "inferred" if r.get("inferred") else r.get("detail", "")[:34])
        print(f"  {r['artifact']:<44}{r['status']:<18}{how}")

    sweep = Counter()
    others = []
    if args.all:
        for p in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
            rel = str(Path(p).relative_to(ROOT))
            if rel in PINNED:
                continue
            v = verify(p)
            sweep[v["status"]] += 1
            if v["status"] != CURRENT:
                others.append(v)
        print(f"\nthe other {sum(sweep.values())} artifacts: " +
              ", ".join(f"{k} {v}" for k, v in sweep.most_common()))
        changed = [o for o in others if o["status"] == "producer_changed"]
        if changed:
            print(f"  {len(changed)} name a producer that has changed since they were written:")
            for o in changed[:10]:
                print(f"    {o['artifact']:<46}{o.get('detail','')[:44]}")

    rep = {"pinned": pinned, "n_pinned": len(pinned), "n_pinned_stale": len(bad),
           "sweep": dict(sweep), "sweep_not_current": others}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    if bad:
        print(f"\nFAIL: {len(bad)} pinned artifacts are not current")
        for r in bad:
            print(f"  {r['artifact']}: {r['status']} -- {r.get('detail','')}")
        return 1
    print(f"\nall {len(pinned)} pinned artifacts trace to the code that wrote them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
