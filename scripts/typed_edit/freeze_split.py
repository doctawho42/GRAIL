#!/usr/bin/env python3
"""The split manifest: what was frozen, recorded so a later run can prove it is the same split.

The dataset is external and gitignored, so the repository cannot hold the split itself. What it
can hold is a fingerprint of it. This writes one: the content digest of each triples file, the
digest of each split's substrate set, the digest of the evaluated test set and of every stratum
membership file, and the digest of the rule bank, since a bank that moves moves every ceiling.

The point is not secrecy. It is that a preregistration whose data can be swapped underneath it
registers nothing, and `the split is frozen` is a claim a reader has no way to check unless the
fingerprint is published with it.

The cross-split audit is read from `results/leakage_fix_report.json` rather than recomputed, so
the manifest and the audit cannot disagree; that file is produced by `scripts/audit_leakage.py`,
which derives the sets from the committed clean triples without touching any data file.

    python scripts/typed_edit/freeze_split.py
    python scripts/typed_edit/freeze_split.py --verify     # a later run, against the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = ROOT / "grail_metabolism" / "data"
BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
AUDIT = ROOT / "results" / "leakage_fix_report.json"
OUT = ROOT / "paper2" / "split_manifest.json"
STRATA = ROOT / "strata"


def digest_file(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "present": False}
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "present": True, "bytes": len(data),
            "lines": data.count(b"\n"), "sha256": hashlib.sha256(data).hexdigest()}


def digest_set(items) -> dict:
    """Order-independent digest of a set of strings, so a reshuffled file still matches.

    The argument is materialised first: passed a generator, joining it exhausts it and the
    count that follows reads zero beside a perfectly good digest.
    """
    items = sorted(set(items))
    return {"n": len(items), "sha256": hashlib.sha256("\n".join(items).encode()).hexdigest()}


def substrates_of(triples: Path) -> list:
    if not triples.exists():
        return []
    out = []
    for line in triples.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append(parts[0])
    return out


def find_biotransformer() -> Path | None:
    """Where the BioTransformer checkout is, if it is anywhere.

    `bt_predict.py` resolves it as `ROOT.parent / GRAIL_baselines / biotransformer`, which is
    correct from the main checkout and wrong from a worktree, whose parent is the worktrees
    directory. The candidates are tried in order and the one that exists is recorded, so the
    manifest pins what was actually read rather than where it was expected.
    """
    env = os.environ.get("BIOTRANSFORMER_DIR")
    cands = [Path(env)] if env else []
    cands += [ROOT.parent / "GRAIL_baselines" / "biotransformer"]
    cands += [a / "GRAIL_baselines" / "biotransformer" for a in ROOT.parents]
    for c in cands:
        if (c / "BioTransformer3.0_20230525.jar").exists():
            return c
    return None


def third_party() -> dict:
    """The comparators the registration closes its list on, pinned by digest where possible."""
    out = {}
    bt_dir = find_biotransformer()
    bt = {"upstream": "https://bitbucket.org/wishartlab/biotransformer3.0jar.git",
          "found_at": str(bt_dir) if bt_dir else None}
    if bt_dir:
        jar = bt_dir / "BioTransformer3.0_20230525.jar"
        data = jar.read_bytes()
        bt["jar"] = {"name": jar.name, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()}
        try:
            bt["checkout_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=bt_dir, capture_output=True,
                text=True).stdout.strip() or None
            bt["checkout_date"] = subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=short"], cwd=bt_dir,
                capture_output=True, text=True).stdout.strip() or None
        except Exception:  # noqa: BLE001
            bt["checkout_commit"] = bt["checkout_date"] = None
    # the template files the reach figure reads, which are in this repository and are what the
    # 994 in the appendix is counted from
    db = [ROOT / "artifacts/tier2/biotransformer/database/metabolicReactions.json",
          ROOT / "artifacts/tier2/biotransformer/database/ENVMICRO/metabolicReactions.json",
          ROOT / "artifacts/tier2/biotransformer/database/standardizationReactions.json"]
    bt["template_files"] = [digest_file(f) for f in db]
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from decompose_biotransformer import load_bt_templates
        bt["templates_loaded"] = len(load_bt_templates(db))
    except Exception as e:  # noqa: BLE001
        bt["templates_loaded"] = None
        bt["templates_error"] = str(e)
    out["biotransformer"] = bt

    try:
        import sygma
        out["sygma"] = {"version": getattr(sygma, "__version__", None),
                        "module": str(Path(sygma.__file__).parent)}
    except Exception as e:  # noqa: BLE001
        out["sygma"] = {"error": str(e)}
    return out


def build() -> dict:
    man = {"what_this_pins": "the split, the evaluated test set, the strata, the rule bank "
                             "and the third-party comparators that can be pinned",
           "splits": {}, "files": {}, "strata": {}, "bank": digest_file(BANK),
           "third_party": third_party()}

    for split in ("train", "val", "test"):
        f = DATA / f"{split}_triples_clean.txt"
        man["files"][split] = digest_file(f)
        subs = substrates_of(f)
        if subs:
            man["splits"][split] = {"substrates": digest_set(subs), "triples": len(subs)}

    # the set the paper actually evaluates on, which is not the same as the triples file: it is
    # what load_test_map returns after parsing and after references are attached
    try:
        from scripts.run_benchmark import load_test_map
        tm = load_test_map(None, 42)
        man["evaluated_test_set"] = {
            "substrates": digest_set(tm),
            "references": sum(len(v) for v in tm.values()),
            "pairs": digest_set(f"{s}>>{p}" for s, ps in tm.items() for p in ps),
        }
    except Exception as e:  # noqa: BLE001
        man["evaluated_test_set"] = {"error": str(e)}

    for name in sorted(p.name for p in STRATA.glob("*.txt")) if STRATA.exists() else []:
        man["strata"][name] = digest_file(STRATA / name)

    if AUDIT.exists():
        a = json.loads(AUDIT.read_text())
        overlaps = a.get("clean_overlap")
        # An audit read through the wrong key yields an empty dict, and `no failing pair` over
        # an empty dict is vacuously true: the manifest would certify a split it never read.
        # The keys are required, and their absence is a failure and not a pass.
        required = {"train_val", "train_test", "val_test"}
        missing = required - set(overlaps or {})
        bad = [k for k, v in (overlaps or {}).items()
               if v.get("substrate_overlap") or v.get("positive_pair_overlap")]
        man["cross_split_audit"] = {
            "source": str(AUDIT.relative_to(ROOT)),
            "overlaps": overlaps,
            "sizes": {k: {"substrates": v.get("remaining_substrates"),
                          "positive_pairs": v.get("remaining_positive_pairs")}
                      for k, v in (a.get("clean_split_stats") or {}).items()},
            "substrate_disjoint": a.get("substrate_disjoint"),
            "positive_pair_disjoint": a.get("positive_pair_disjoint"),
            "structure_leak": a.get("structure_leak"),
            "pairs_checked": sorted(set(overlaps or {})),
            "clean": not bad and not missing and bool(a.get("substrate_disjoint"))
            and bool(a.get("positive_pair_disjoint")),
            "note": "molecule overlap is expected and reported: a product annotated in one "
                    "split can be a substrate in another. Substrate and positive-pair overlap "
                    "are the ones that must be zero.",
        }
        if missing:
            man["cross_split_audit"]["missing_pairs"] = sorted(missing)
        if bad:
            man["cross_split_audit"]["failing"] = bad

    try:
        man["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        man["git_commit"] = None
    return man


def compare(old: dict, new: dict) -> list:
    """Every leaf that moved, named."""
    diffs = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in ("git_commit", "found_at", "module"):
                    continue
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif a != b:
            diffs.append(f"{path}: manifest {a!r} now {b!r}")

    walk(old, new, "")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against the committed manifest")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    new = build()
    path = Path(args.out)
    if args.verify:
        if not path.exists():
            print(f"no manifest at {path}", file=sys.stderr)
            return 1
        diffs = compare(json.loads(path.read_text()), new)
        for d in diffs:
            print(f"  MOVED: {d}")
        print("the split matches the manifest" if not diffs
              else f"{len(diffs)} leaves moved since the freeze")
        return 0 if not diffs else 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, indent=1))
    ev = new.get("evaluated_test_set", {})
    print(json.dumps({k: v for k, v in new.items() if k not in ("strata",)}, indent=1)[:1800])
    print(f"\nstrata pinned: {len(new['strata'])}")
    print(f"evaluated test set: {ev.get('substrates', {}).get('n')} substrates, "
          f"{ev.get('references')} references")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
