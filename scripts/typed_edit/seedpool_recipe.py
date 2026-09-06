#!/usr/bin/env python3
"""How the seed pools were made, recorded because for a while nothing recorded it.

`results/retraining_spread.json` computes the spread that every registered effect in this work is
reported in units of. It reads six files. The three exhaustive ones name their producer. The three
interactive ones named nothing, no script in the tree wrote to their path, and the files are
gitignored and absent from the deposit, so a reader who wanted to know how the ruler was made had
nowhere to look.

The producer existed the whole time. `build_wide_pools.py` writes exactly that shape when it runs
as a single unmerged shard, and its `--top-k` is the rule budget: the whole bank is the
selector-free arm, and the value the checkpoint records is the trained one. The gap was an
asymmetry inside that one file, where the merge path stamped itself and the shard path did not, so
a merged pool carried a producer and a single-shard pool did not. Single-shard runs are precisely
the ones that cover a whole population at once, which is what these are.

This settles it by reproduction rather than by argument. Rebuilding seed 0 from the recorded
checkpoints under the recorded budget reproduced `interactive_seed0.json` byte for byte, sha256
prefix 883567a9866d1135, which is the digest `retraining_spread.json` records for its own input.
Identical key sets, identical rank order and identical component scores on all 291 substrates.

    python scripts/typed_edit/seedpool_recipe.py             # record the recipe and the digests
    python scripts/typed_edit/seedpool_recipe.py --verify 0  # rebuild seed 0 and compare

One thing about a later rebuild. The shard path now stamps itself, so a pool built today carries a
provenance block the archived ones do not, and its bytes therefore differ from theirs by that
header. `--verify` compares the payload -- the pools, the references and the checkpoints -- rather
than the file, and says which comparison it made, so a difference in the header is not read as a
difference in the pool and a difference in the pool is not hidden by the header.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

BUILDER = "scripts/typed_edit/build_wide_pools.py"
CKPT = "artifacts/multiseed_full5000_implicit_seed{seed}/checkpoints"
POOL = "results/seedpools/interactive_seed{seed}.json"
SEEDS = (0, 1, 2)

# What the archived pools were built with, and what a new seed must be built with to belong in the
# same spread. Every value here appears in the archived files' own header, so the recipe is checked
# against them rather than asserted.
ARGS = {"top_k": 30, "population": "comparison", "present": "stored", "slice": [0, 291]}

# The reproduction this recipe rests on: seed 0, rebuilt from its recorded checkpoints, compared
# against the archived file as bytes. Recorded as a fact with its date rather than re-run on every
# invocation, because the rebuild costs about four minutes and the claim it supports does not
# change unless the builder does.
REPRODUCED = {
    "seed": 0,
    "compared": "the whole file, as bytes",
    "sha256_16": "883567a9866d1135",
    "identical": True,
    "substrates": 291,
    "identical_key_sets": 291,
    "identical_rank_order": 291,
    "identical_component_scores": 291,
    "builder_at_the_time": "before the shard path stamped itself, so the formats matched exactly",
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _argv(seed: int, out: Path) -> list:
    base = ROOT / CKPT.format(seed=seed)
    return [sys.executable, str(ROOT / BUILDER),
            "--start", str(ARGS["slice"][0]), "--end", str(ARGS["slice"][1]),
            "--top-k", str(ARGS["top_k"]),
            "--population", ARGS["population"], "--present", ARGS["present"],
            "--gen-ckpt", str(base / "generator.pt"),
            "--filter-ckpt", str(base / "filter.pt"),
            "--out", str(out)]


def _check_header(seed: int) -> dict:
    """The archived pool's own header against the recipe, so the recipe cannot drift from it."""
    p = ROOT / POOL.format(seed=seed)
    if not p.exists():
        return {"present": False}
    blob = json.loads(p.read_text())
    mismatched = {k: {"recipe": v, "file": blob.get(k)}
                  for k, v in ARGS.items() if blob.get(k) != v}
    return {"present": True, "sha256_16": _digest(p), "bytes": p.stat().st_size,
            "checkpoints": blob.get("checkpoints"),
            "header_matches_the_recipe": not mismatched,
            "mismatched": mismatched or None,
            "records_its_own_producer":
                bool((blob.get("provenance") or {}).get("script_path"))}


def verify(seed: int) -> int:
    """Rebuild one seed and compare the payload, reporting which comparison was made."""
    archived = ROOT / POOL.format(seed=seed)
    if not archived.exists():
        print(f"REFUSING: {archived.relative_to(ROOT)} is not here to compare against",
              file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rebuilt.json"
        rc = subprocess.call(_argv(seed, out))
        if rc != 0 or not out.exists():
            print(f"REFUSING: the rebuild exited {rc}", file=sys.stderr)
            return 1
        old, new = json.loads(archived.read_text()), json.loads(out.read_text())
        byte_identical = _digest(archived) == _digest(out)
        payload = all(old.get(k) == new.get(k)
                      for k in ("pools", "references", "checkpoints", *ARGS))
        print(f"\n  seed {seed}")
        print(f"    bytes identical    {byte_identical}")
        print(f"    payload identical  {payload}  (pools, references, checkpoints and the recipe "
              f"fields)")
        if not payload:
            for k in ("pools", "references", "checkpoints", *ARGS):
                if old.get(k) != new.get(k):
                    print(f"      differs: {k}")
            return 1
        if not byte_identical:
            print("    the files differ only outside the payload, which is the provenance block "
                  "the shard path now writes and the archived pools predate")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, metavar="SEED",
                    help="rebuild this seed's pool and compare it against the archived one")
    args = ap.parse_args()

    if args.verify is not None:
        return verify(args.verify)

    per_seed = {str(s): _check_header(s) for s in SEEDS}
    bad = [s for s, v in per_seed.items()
           if v.get("present") and not v["header_matches_the_recipe"]]
    if bad:
        print(f"REFUSING: the archived pool for seed {', '.join(bad)} does not match the recipe "
              f"recorded here; one of the two is wrong and guessing which would be worse than "
              f"stopping", file=sys.stderr)
        return 1

    report = {
        "provenance": stamp(__file__),
        "question": ("how results/seedpools/interactive_seed*.json were produced, which nothing "
                     "recorded and which retraining_spread.json depends on"),
        "builder": BUILDER,
        "command": " ".join(["python", BUILDER,
                             "--start 0 --end 291 --top-k 30",
                             "--population comparison --present stored",
                             f"--gen-ckpt {CKPT.format(seed='<N>')}/generator.pt",
                             f"--filter-ckpt {CKPT.format(seed='<N>')}/filter.pt",
                             f"--out {POOL.format(seed='<N>')}"]),
        "arguments": ARGS,
        "seeds": list(SEEDS),
        "archived": per_seed,
        "reproduced": REPRODUCED,
        "why_this_file_exists": (
            "the producer existed but did not name itself: build_wide_pools.py stamped its merge "
            "path and not its shard path, and these pools are single-shard runs. The shard path "
            "stamps now; this records the recipe for the pools written before it did"),
        "adding_a_seed": (
            "train a run at the deployed configuration, then run the command above with its "
            "checkpoints and --out results/seedpools/interactive_seed<N>.json. "
            "retraining_spread.py globs the directory, so it picks the new seed up with no change. "
            "A seed built any other way does not belong in the same spread"),
    }
    out = ROOT / "results" / "seedpool_recipe.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"\n  {report['builder']}, one unmerged shard, {ARGS['top_k']} rules, "
          f"{ARGS['population']} population")
    for s, v in per_seed.items():
        if not v.get("present"):
            print(f"    seed {s}: absent")
            continue
        print(f"    seed {s}: {v['sha256_16']}  header matches the recipe: "
              f"{v['header_matches_the_recipe']}  stamped: {v['records_its_own_producer']}")
    r = REPRODUCED
    print(f"\n  seed {r['seed']} rebuilds byte-identically ({r['sha256_16']}), "
          f"{r['identical_rank_order']} of {r['substrates']} substrates in identical rank order")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
