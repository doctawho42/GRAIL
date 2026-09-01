#!/usr/bin/env python3
"""The comparison pools, taken from the whole-test build rather than computed a second time.

The pools the comparison table is read from were scored by a filter the repository no longer
releases. The whole-test build covers the same population as a superset, at the same rule budget,
under the same substrate presentation, and was scored by the released pair. Restricting it to the
comparison set is therefore the same computation the comparison needs, already done, and running
the builder again over those substrates would only reproduce it at the cost of an afternoon.

What this refuses. It refuses unless every comparison substrate is present in the whole-test build,
so a partial build cannot silently narrow the population. It refuses unless the candidate sets
match the published pools substrate for substrate: the generator is the same in both builds and its
scores are byte-identical, so the enumeration must agree, and if it does not then the two builds
differ in something other than the filter and neither can be substituted for the other. What it
expects to differ, and reports, is the filter score on every candidate.

    python scripts/typed_edit/comparison_pools_from_fulltest.py
    python scripts/typed_edit/comparison_pools_from_fulltest.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

PUBLISHED = ROOT / "results" / "widepools_implicit"
FULLTEST = ROOT / "results" / "widepools_fulltest"
SHARDS = 6


def load(directory: Path):
    pools, refs, meta = {}, {}, {}
    for f in sorted(glob.glob(str(directory / "w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs.update(blob["references"])
        for k in ("top_k", "present", "population", "checkpoints"):
            if k in blob:
                meta.setdefault(k, blob[k])
    return pools, refs, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--keep", default="",
                    help="directory to move the existing pools to before replacing them")
    args = ap.parse_args()

    old, old_refs, _ = load(PUBLISHED)
    new, new_refs, meta = load(FULLTEST)
    if not old:
        raise SystemExit(f"no published pools under {PUBLISHED}")
    missing = sorted(set(old) - set(new))
    if missing:
        raise SystemExit(
            f"the whole-test build covers {len(set(old) & set(new))} of the {len(old)} comparison "
            f"substrates; {len(missing)} are absent, so restricting it would narrow the "
            f"population rather than rescore it. Let the build finish.")

    # The enumeration must agree, or the two builds differ in more than the filter.
    # Compared as a mapping, not as a list. A pool is stored in rank order and the rank is
    # filter x generator, so replacing the filter reorders it: comparing the stored sequences
    # calls every substrate a mismatch precisely when the only thing that changed is the thing
    # being changed. What has to agree is which candidates were enumerated and what the
    # generator said about each.
    # What must agree is what the measurement is made of: the matching keys, since recall is
    # computed on those, and the generator's score on every candidate both builds carry. What is
    # allowed to differ is the rank order, which is filter x generator, and which candidate
    # represents a key where two share one, since the survivor of that tie is chosen by the same
    # product. Comparing stored SMILES sequences instead calls all 291 substrates a mismatch and
    # comparing SMILES sets calls 12 of them one, in both cases for the reason the substitution
    # exists.
    enum_mismatch, filter_moved, moved_max = [], 0, 0.0
    reordered = representative_swaps = 0
    for s in sorted(old):
        if sorted(c.get("key") for c in old[s]) != sorted(c.get("key") for c in new[s]):
            enum_mismatch.append(s)
            continue
        ga = {c["smiles"]: round(float(c["generator"]), 9) for c in old[s]}
        gb = {c["smiles"]: round(float(c["generator"]), 9) for c in new[s]}
        shared = set(ga) & set(gb)
        if any(ga[k] != gb[k] for k in shared):
            enum_mismatch.append(s)
            continue
        if set(ga) != set(gb):
            representative_swaps += 1
        if [c["smiles"] for c in old[s]] != [c["smiles"] for c in new[s]]:
            reordered += 1
        fa = {c["smiles"]: float(c["filter"]) for c in old[s]}
        for c in new[s]:
            if c["smiles"] in fa:
                d = abs(float(c["filter"]) - fa[c["smiles"]])
                if d > 1e-9:
                    filter_moved += 1
                    moved_max = max(moved_max, d)
    if enum_mismatch:
        raise SystemExit(
            f"{len(enum_mismatch)} substrates differ in their matching keys or in a generator "
            f"score, so the two builds are not the same configuration and one cannot stand for "
            f"the other; first: {enum_mismatch[0][:60]}")

    total = sum(len(v) for v in old.values())
    print(f"{len(old)} comparison substrates, {total} candidates")
    print(f"  matching keys and generator scores identical on every substrate")
    print(f"  substrates the new filter reorders: {reordered} of {len(old)}")
    print(f"  substrates where it picks a different representative of one key: "
          f"{representative_swaps}")
    print(f"  filter scores that move: {filter_moved} of {total}, largest {moved_max:.4f}")
    print(f"  released pair: {json.dumps(meta.get('checkpoints', {}))[:120]}")
    if args.dry_run:
        print("dry run; nothing written")
        return 0

    if args.keep:
        dest = Path(args.keep)
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(str(PUBLISHED / "w*.json"))):
            shutil.copy2(f, dest / Path(f).name)
        print(f"  the pools being replaced are copied to {dest}")

    # Written in the shard layout the twenty-odd readers of this directory already glob for.
    subs = sorted(old)
    size = (len(subs) + SHARDS - 1) // SHARDS
    for f in sorted(glob.glob(str(PUBLISHED / "w*.json"))):
        Path(f).unlink()
    for i in range(SHARDS):
        part = subs[i * size:(i + 1) * size]
        if not part:
            continue
        (PUBLISHED / f"w{i}.json").write_text(json.dumps({
            "provenance": stamp(__file__),
            "inputs": record_inputs(sorted(glob.glob(str(FULLTEST / "w*.json")))),
            "derived_from": "results/widepools_fulltest, restricted to the comparison set",
            "why": ("the whole-test build covers this population as a superset at the same rule "
                    "budget and presentation and was scored by the released pair; rebuilding "
                    "these substrates separately would repeat that computation"),
            "slice": [i * size, i * size + len(part)],
            "top_k": meta.get("top_k"), "present": meta.get("present"),
            "population": "comparison",
            "checkpoints": meta.get("checkpoints"),
            "pools": {s: new[s] for s in part},
            "references": {s: new_refs.get(s, old_refs.get(s, [])) for s in part}}, indent=1))
    print(f"wrote {SHARDS} shards to {PUBLISHED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
