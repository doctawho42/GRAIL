#!/usr/bin/env python3
"""Recover the cap and seed behind artifacts that recorded neither.

Twenty-one result files were measured on a subsample of a split and record only its size. The
substrate set is not recoverable from a size: `_sample_triples` draws without replacement, so caps
are not nested and a wrong cap yields a different set of the same size rather than a subset. The
audit in paper/SELF_CLAIMS.md carries this as its one open item, and the reason given for leaving it
open -- that the invocations were not recorded -- is only half true. The draw is deterministic in
(cap, seed) over a fixed pool, so the cap can be searched for rather than remembered.

The search needs an exact yield function, because a cap does not give the number of substrates
directly. `MolFrame.from_file` keeps a triple only when both of its ids resolve in the SDF, and then
keys the map by SMILES rather than by id, so ids sharing a structure collapse. Replicating that
gives the map size for any cap in milliseconds once the SDF is parsed, against roughly a hundred
seconds per cap through the real loader.

The replica is validated before it is trusted, against caps whose yield was measured through the
loader itself. A recovered cap is written into the artifact as `config_reconstructed` and never as
`config`: it is an inference from a size, corroborated where the artifact's numbers could be
recomputed, and the distinction between that and a recorded invocation is exactly what this audit
exists to keep.

Ambiguity is reported rather than resolved. Where several caps yield the same size the artifact gets
all of them, since picking one would manufacture the certainty the exercise is about.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.utils.preparation import MolFrame

RDLogger.DisableLog("rdApp.*")

# yields measured through load_dataset_bundle itself, used to validate the replica before use
KNOWN = {("test", 245): 238, ("test", 248): 237, ("test", 250): 245,
         ("test", 252): 246, ("test", 255): 246}
# Which population each artifact is on, declared from evidence rather than inferred from its size.
# This table is the safeguard, and it exists because the search without it manufactures provenance:
# run unguarded it proposed a metabolism val-split cap for xdomain_retro_protocol.json and
# retro_transfer.json, which are USPTO-50k retrosynthesis with no metabolism split behind them at
# all. A size that happens to be reachable is not evidence of how a set was drawn.
PINNED = {
    "artifacts/tier2/substrates.json": ["budget_matched_frontier", "match_sensitivity",
                                        "match_sensitivity_3method", "match_sensitivity_4method",
                                        "match_sensitivity_5method", "rank_flip_ci",
                                        "rank_flip_ci_metatrans_sygma"],
    "the whole GLORYx set": ["gloryx_criterion_ladder", "gloryx_rank_flip_ci"],
    "the whole clean val split": ["prune_and_rerank_val", "ablate_id_embedding_val"],
    "USPTO-50k retrosynthesis, not a metabolism split": ["retro_transfer", "xdomain_retro_protocol"],
}
# Only these are seeded draws over a metabolism split, so only these admit a cap search.
SUBSAMPLED = {"prior_vs_learned", "prior_vs_learned_propensity", "selection_ablation",
              "selection_ablation_prior300", "selection_ablation_ranksignal",
              "ablate_id_embedding", "hybrid_rerank", "metatox_complementarity"}

SEEDS = {"train": 0, "val": 1, "test": 2}   # sampling_seed + this offset, per workflows/data.py


def _rel(p) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


class Split:
    """Everything needed to compute the map size for any cap, parsed once."""

    def __init__(self, name: str):
        self.name = name
        tri_path = ROOT / f"grail_metabolism/data/{name}_triples_clean.txt"
        sdf_path = ROOT / f"grail_metabolism/data/{name}.sdf"
        self.triples = MolFrame.read_triples(str(tri_path))
        need = {a for a, _, _ in self.triples} | {b for _, b, _ in self.triples}
        self.id2smi: dict[int, str] = {}
        for fallback, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False), start=1):
            if mol is None:
                continue
            try:
                idx = int(mol.GetProp("Index")) if mol.HasProp("Index") else fallback
            except Exception:
                idx = fallback
            if idx not in need:
                continue
            self.id2smi[idx] = (mol.GetProp("SMILES") if mol.HasProp("SMILES")
                                else Chem.MolToSmiles(mol, isomericSmiles=False))
            if len(self.id2smi) == len(need):
                break
        self.unique_ids = sorted({a for a, _, _ in self.triples})

    def yield_at(self, cap: int | None, sampling_seed: int = 42) -> int:
        """Map size for this cap, replicating _sample_triples then MolFrame.from_file."""
        if cap and 0 < cap < len(self.unique_ids):
            rng = np.random.default_rng(sampling_seed + SEEDS[self.name])
            keep = set(rng.choice(np.array(self.unique_ids), size=cap, replace=False).tolist())
            triples = [t for t in self.triples if t[0] in keep]
        else:
            triples = self.triples
        subs = set()
        for a, b, real in triples:
            if real != 1:
                continue
            sa, sb = self.id2smi.get(a), self.id2smi.get(b)
            if sa is not None and sb is not None:
                subs.add(sa)
        return len(subs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-max", type=int, default=1400)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--apply", action="store_true",
                    help="write config_reconstructed into the artifacts (default: report only)")
    ap.add_argument("--out", default=str(ROOT / "results" / "subsample_config_recovery.json"))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    splits = {}
    for name in ("test", "val"):
        if (ROOT / f"grail_metabolism/data/{name}.sdf").exists():
            splits[name] = Split(name)
            print(f"{name}: {len(splits[name].unique_ids)} sampled ids, "
                  f"full map {splits[name].yield_at(None)}", flush=True)

    bad = [(c, KNOWN[("test", c)], splits["test"].yield_at(c)) for c in
           sorted(k[1] for k in KNOWN if k[0] == "test")]
    print("\nreplica against yields measured through the loader:")
    for cap, want, got in bad:
        print(f"  cap {cap}: loader {want}, replica {got}  {'OK' if want == got else 'MISMATCH'}")
    if any(w != g for _, w, g in bad):
        raise SystemExit("the replica does not reproduce known yields -- it cannot be used to search")

    targets = {}
    for p in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or any(k in d for k in ("config", "max_substrates", "sampling_seed")):
            continue
        n = d.get("n") or d.get("n_substrates")
        if isinstance(n, int) and 0 < n < 1000:
            targets[p.name] = n

    pinned_by = {a: src for src, names in PINNED.items() for a in names}
    print(f"\nartifacts recording a size and no configuration: {len(targets)}")
    rep = {"config": _code_version(), "validated_against": {f"{k[0]}:{k[1]}": v for k, v in KNOWN.items()},
           "artifacts": {}}
    for name, n in sorted(targets.items()):
        stem = name[:-5]
        if stem in pinned_by:
            rep["artifacts"][name] = {"n": n, "status": "not a seeded subsample",
                                      "population": pinned_by[stem], "candidates": []}
            print(f"  {name:38} n={n:<5} {'pinned':12} {pinned_by[stem]}")
            continue
        if stem not in SUBSAMPLED:
            rep["artifacts"][name] = {"n": n, "status": "population undeclared", "candidates": [],
                                      "note": "no cap searched: a size alone is not evidence of a draw"}
            print(f"  {name:38} n={n:<5} {'undeclared':12} no search")
            continue
        hits = []
        for sname, sp in splits.items():
            for seed in seeds:
                for cap in range(2, min(args.cap_max, len(sp.unique_ids)) + 1):
                    if sp.yield_at(cap, seed) == n:
                        hits.append({"split": sname, "sampling_seed": seed, "max_substrates": cap})
        status = ("unique" if len(hits) == 1 else
                  "ambiguous" if hits else
                  "not a seeded subsample of either split")
        rep["artifacts"][name] = {"n": n, "status": status, "candidates": hits[:12],
                                  "n_candidates": len(hits)}
        shown = hits[0] if len(hits) == 1 else f"{len(hits)} candidates"
        print(f"  {name:38} n={n:<5} {status:12} {shown}")

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")

    if args.apply:
        # Three statuses, three fields, and the names differ on purpose. `population` records a fact
        # about where the set came from; `config_reconstructed` records an inference from its size;
        # `config_candidates` records that the inference did not resolve. None of them is `config`,
        # which is reserved for what a run actually recorded about itself.
        wrote = {"population": 0, "reconstructed": 0, "candidates": 0}
        CORROBORATED = {"selection_ablation_ranksignal.json": {
            "split": "test", "sampling_seed": 42, "max_substrates": 250,
            "corroboration": "the sibling prior_vs_learned.py defaults to 250, and re-running "
                             "selection_ablation.py at that cap reproduces this artifact's "
                             "recall@15 of 0.413 and pool size of 107.6 exactly"}}
        for name, info in rep["artifacts"].items():
            path = ROOT / "results" / name
            d = json.loads(path.read_text())
            if info["status"] == "not a seeded subsample":
                d["population"] = {"set": info["population"],
                                   "note": "not drawn with a cap and seed; the substrate set is "
                                           "fixed by the source named here"}
                wrote["population"] += 1
            elif name in CORROBORATED:
                d["config_reconstructed"] = {**CORROBORATED[name],
                                             "recovered_by": "scripts/recover_subsample_config.py"}
                wrote["reconstructed"] += 1
            elif info["status"] == "unique":
                d["config_reconstructed"] = {**info["candidates"][0],
                                             "note": "inferred from the recorded substrate count and "
                                                     "unique over caps 2..%d; not recorded by the run"
                                                     % args.cap_max,
                                             "recovered_by": "scripts/recover_subsample_config.py"}
                wrote["reconstructed"] += 1
            elif info["status"] == "ambiguous":
                d["config_candidates"] = {"candidates": info["candidates"],
                                          "note": "several caps yield this substrate count; the run "
                                                  "recorded none of them and this does not resolve it"}
                wrote["candidates"] += 1
            else:
                continue
            path.write_text(json.dumps(d, indent=1))
        print(f"wrote population on {wrote['population']}, config_reconstructed on "
              f"{wrote['reconstructed']}, config_candidates on {wrote['candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
