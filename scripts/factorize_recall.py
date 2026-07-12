#!/usr/bin/env python
"""Per-substrate coverage x selection x ranking factorization of GRAIL recall@15,
tautomer-InChIKey, on the clean test split. Emits results/recall_factorization.json.

This is a DECOMPOSITION (an accounting identity), not a theorem. Deployed recall@15
factors, per the telescoping identity, into

    micro_recall = coverage_bank x selection_retention x ranking_conversion
                 = (SigmaCfull/SigmaU) x (SigmaCbud/SigmaCfull) x (SigmaH/SigmaCbud)

where, per substrate s (tautomer-InChIKey):
  U      = |T_s|, the tautomer-distinct true metabolites;
  Cfull  = trues recovered by the FULL-BANK depth-1 products (rule-bank ceiling);
  Cbud   = trues recovered by the DEPLOYED GENERATOR POOL (generate_scored, top_k=MAX_POOL);
  H      = trues recovered by the DEPLOYED gen x filter top-15 output.

H is computed by running the ACTUAL deployed pipeline (generator.pt + filter.pt from
artifacts/full5000_single/checkpoints) on EVERY test-map substrate -- NOT read from the
291-substrate export CSV, which is a non-representative subset (an ensemble.py export cap).

Reuses grail_metabolism.stats.factor_bootstrap_ci (cluster/substrate bootstrap) and
run_benchmark._tautomer_recovered (the full-bank tautomer ceiling helper).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.wrapper import ModelWrapper
from grail_metabolism.stats import factor_bootstrap_ci
from grail_metabolism.utils.preparation import (
    apply_rules_to_molecule,
    load_default_rules,
    resolve_default_rule_bank,
)
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator
from rdkit import Chem, RDLogger

from run_benchmark import _tautomer_recovered  # per-substrate (denominator, recovered) tautomer ceiling

RDLogger.DisableLog("rdApp.*")

K = 15
MAX_POOL = 200
# Deployed evaluation operating point (artifacts/full5000_single/config.yaml, evaluation:).
CANDIDATE_TOP_K = 30       # generator rules fed to the ensemble output path
FILTER_CANDIDATE_CAP = 32  # generator top-N the filter scores
DEPLOYED_DIR = ROOT / "artifacts" / "full5000_single" / "checkpoints"
GEN_CKPT = DEPLOYED_DIR / "generator.pt"
FILTER_CKPT = DEPLOYED_DIR / "filter.pt"
# The shipped full5000_single generator.pt predates the persistent rule_prior_logits buffer, so
# reloading it leaves the empirical rule prior at its zero init -- which does NOT reproduce the
# deployed recall (the deployed eval had the prior populated in memory). The byte-identical-weights
# full5000_priors/generator.pt carries the SAME trained prior as a persistent buffer; copying it
# restores the deployed operating point. (Decided empirically; see task-2-report.md.)
PRIORS_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"


def _taut_key_set(smiles_iter):
    out = set()
    for s in smiles_iter:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            out.add(s)
    return out


def tautomer_hits(preds, trues) -> int:
    """# of tautomer-distinct true SMILES matched by any predicted SMILES."""
    pk = _taut_key_set(preds)
    tk = _taut_key_set(trues)
    return len(tk & pk)


# --- parallel full-bank ceiling pass (torch-free; RDKit only) --------------------------
# The per-substrate C_full computation applies all ~7581 SMIRKS (~4.7 s/substrate, pure
# RDKit). Substrates are independent, so this pass fans out across CPU cores via a `spawn`
# Pool. The worker uses ONLY RDKit + run_benchmark._tautomer_recovered -- no torch objects
# cross the pool boundary, no CUDA, and the loaded model stays in the parent's serial pass.
_WORKER_RULES = None


def _ceiling_init():
    """Pool initializer: silence RDKit, load the rule bank once into a module global."""
    from rdkit import RDLogger as _RDLogger

    _RDLogger.DisableLog("rdApp.*")
    # Keep each worker single-threaded so N workers don't oversubscribe the cores.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    global _WORKER_RULES
    _WORKER_RULES = load_default_rules()
    if _WORKER_RULES is None:
        raise RuntimeError("rule bank not found (load_default_rules() returned None)")


def _ceiling_worker(item):
    """Full-bank depth-1 tautomer ceiling for ONE substrate (pure RDKit).

    item = (sub_smiles, list_of_true_smiles) -> (sub_smiles, U_i, Cfull_i), where
    U_i = |tautomer-distinct trues| (denominator) and Cfull_i = trues recovered by the
    full-bank depth-1 products. Returns (sub, 0, 0) for unparseable / empty substrates.
    """
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return (sub, 0, 0)
    full_products = list(
        apply_rules_to_molecule(mol, _WORKER_RULES, normalization_mode="canonical").keys()
    )
    u_i, cfull_i, _ = _tautomer_recovered(trues, full_products, audit=False)
    return (sub, int(u_i), int(cfull_i))


def run_ceiling_pass(items, workers):
    """Full-bank ceiling over ALL substrates -> {sub: (U_i, Cfull_i)}.

    `workers <= 1` forces the serial path (reproducibility/debug); otherwise fans out via a
    `spawn`-context Pool. The ceiling is deterministic, so the parallel result is identical
    to the serial one. Returns (ceiling_dict, mean_seconds_per_substrate).
    """
    ceil_items = [(sub, list(prods)) for sub, prods in items]
    n = len(ceil_items)
    ceiling = {}
    t0 = time.perf_counter()
    if workers <= 1:
        _ceiling_init()
        for i, item in enumerate(ceil_items, 1):
            sub, u_i, cfull_i = _ceiling_worker(item)
            ceiling[sub] = (u_i, cfull_i)
            if i == 1 or i % 25 == 0 or i == n:
                print(f"  [ceiling serial] {i}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        pool = multiprocessing.get_context("spawn").Pool(workers, initializer=_ceiling_init)
        try:
            for i, (sub, u_i, cfull_i) in enumerate(
                pool.imap_unordered(_ceiling_worker, ceil_items, chunksize=4), 1
            ):
                ceiling[sub] = (u_i, cfull_i)
                if i == 1 or i % 25 == 0 or i == n:
                    print(f"  [ceiling x{workers}] {i}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
        finally:
            pool.close()
            pool.join()
    elapsed = time.perf_counter() - t0
    mean_s = elapsed / max(1, n)
    print(f"  ceiling pass: {n} subs in {elapsed:.0f}s ({mean_s:.2f}s/sub, workers={workers})", flush=True)
    return ceiling, mean_s


def _load(path, build_fn):
    """Established deployed-checkpoint load (mirrors scripts/reeval_ranking.py:_load)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    print(f"  loaded {Path(path).name}: missing={len(missing)} unexpected={len(unexpected)} "
          f"threshold={model.calibrated_threshold}", flush=True)
    return model


def build_deployed_model(copy_prior: bool) -> ModelWrapper:
    gen = _load(GEN_CKPT, lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.gen_normalization = "canonical"  # deployed run had dataset.standardize=False
    if copy_prior:
        priors_state = torch.load(PRIORS_CKPT, map_location="cpu", weights_only=False)["state_dict"]
        trained_prior = priors_state["rule_prior_logits"].to(gen.rule_prior_logits.dtype)
        assert trained_prior.shape == gen.rule_prior_logits.shape, "rule count / bank mismatch"
        with torch.no_grad():
            gen.rule_prior_logits.copy_(trained_prior)
        rp = gen.rule_prior_logits
        assert float(rp.abs().max()) > 0 and float(rp.std()) > 0, "rule_prior_logits is degenerate"
        print(f"  restored trained prior: std={float(rp.std()):.3f} "
              f"range[{float(rp.min()):.2f},{float(rp.max()):.2f}] prior_strength={gen.prior_strength}", flush=True)
    filt = _load(FILTER_CKPT, lambda a, r: build_filter(FilterConfig(**a)))
    return ModelWrapper(filt, gen)


def build_dataset_config(max_test) -> DatasetConfig:
    return DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=max_test,
        sampling_seed=42,
    )


def compute_records(model, items, ceiling, log_every=25):
    """Serial torch pass over (substrate -> true products): deployed generator pool (Cbud)
    and deployed gen x filter top-15 (H). Reuses (U_i, Cfull_i) from the parallel ceiling
    pass (`ceiling`), skipping substrates with U_i == 0. Returns per-substrate records with
    the deployed top-15 SMILES persisted for Task 4."""
    gen = model.generator
    gen_threshold = getattr(gen, "calibrated_threshold", None)
    records = []
    t0 = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % log_every == 0 or i == len(items):
            print(f"  [torch] {i}/{len(items)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        # C_full / U: reuse the parallel full-bank ceiling (denominator U, recovered Cfull)
        u_i, cfull_i = ceiling.get(sub, (0, 0))
        if u_i == 0:
            continue
        true_prods = list(prods)
        # C_bud: deployed generator pool (the candidate set the filter reranks)
        scored = gen.generate_scored(sub, top_k=MAX_POOL, threshold=gen_threshold)
        cbud_i = tautomer_hits([s for s, _ in scored], true_prods)
        # H: deployed gen x filter top-15, EXACT deployed operating point (rank policy, no gate)
        deployed_top15 = model.generate(
            sub,
            top_k=CANDIDATE_TOP_K,
            threshold=gen_threshold,
            max_output=K,
            gate_by_filter=False,
            filter_candidate_cap=FILTER_CANDIDATE_CAP,
        )
        h_i = tautomer_hits(deployed_top15, true_prods)
        # Monotonicity clamp: nested pools guarantee H <= Cbud <= Cfull; the clamp only defends
        # against cross-path SMILES canonicalization drift (does not change the telescoping identity).
        cbud_i = min(cbud_i, cfull_i)
        h_i = min(h_i, cbud_i)
        records.append({
            "sub": sub, "U": u_i, "Cfull": cfull_i, "Cbud": cbud_i, "H": h_i,
            "deployed_top15": list(deployed_top15),
        })
    return records


def _macro(records, num_field: str, den_field: str) -> float:
    """Mean of per-substrate ratios (the metrics.py / deployed-headline convention). The
    decomposition itself is MICRO (only ratio-of-sums telescopes into the identity, and micro
    coverage_bank == the published rule-bank ceiling); macro_* are emitted only for reconciliation
    with the quoted deployed recall."""
    vals = [r[num_field] / r[den_field] for r in records if r[den_field] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=100000, help="cap test substrates (default: all ~1170)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes for the parallel full-bank ceiling pass (--workers 1 = serial)")
    ap.add_argument("--copy-prior", dest="copy_prior", action="store_true", default=True,
                    help="restore the trained empirical rule prior (reproduces the deployed operating point)")
    ap.add_argument("--no-copy-prior", dest="copy_prior", action="store_false")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "recall_factorization.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rules = load_default_rules()  # the ~7581 SMIRKS strings (resolve_default_rule_bank returns the Path)
    print(f"rule bank: {len(rules)} rules from {resolve_default_rule_bank()}", flush=True)
    model = build_deployed_model(copy_prior=args.copy_prior)

    ds = build_dataset_config(args.max_test)
    print("loading clean test split...", flush=True)
    bundle = load_dataset_bundle(ds)
    items = list(bundle.test.map.items())
    print(f"test substrates: {len(items)}", flush=True)

    # Parallel full-bank ceiling FIRST (torch-free RDKit fan-out), then the serial torch pass.
    print(f"ceiling pass over {len(items)} substrates (workers={args.workers})...", flush=True)
    ceiling, _ceil_mean_s = run_ceiling_pass(items, args.workers)
    records = compute_records(model, items, ceiling)

    specs = {
        "coverage_bank": ("Cfull", "U"),
        "selection_retention": ("Cbud", "Cfull"),
        "ranking_conversion": ("H", "Cbud"),
    }
    factors = factor_bootstrap_ci(records, specs, n_boot=10000, seed=0)
    U = sum(r["U"] for r in records)
    micro_recall = sum(r["H"] for r in records) / U
    oracle_recall = sum(r["Cbud"] for r in records) / U
    report = {
        "match": "inchikey_tautomer",
        "k": K,
        "n_substrates": len(records),
        "factors": factors,
        "micro_recall": micro_recall,
        "oracle_recall": oracle_recall,
        "macro_recall": _macro(records, "H", "U"),
        "macro_coverage_bank": _macro(records, "Cfull", "U"),
        "macro_oracle_recall": _macro(records, "Cbud", "U"),
        "aggregation_note": (
            "Factors + micro_recall are pooled ratio-of-sums (the telescoping-identity frame; "
            "micro coverage_bank == the published rule-bank ceiling 0.735). macro_* are "
            "per-substrate means (metrics.py convention); macro_recall reproduces the deployed "
            "headline recall@15 0.330 (the earlier ~0.334 was a 291-substrate eval, now corrected)."
        ),
        "provenance": {
            "split": "clean test (test_triples_clean.txt + test.sdf)",
            "ceiling_pool": f"full extended_smirks bank, depth-1 (n_rules={len(rules)})",
            "deployed_pool": f"generator.pt generate_scored top_k={MAX_POOL}, threshold=calibrated",
            "deployed_output": (
                f"generator.pt x filter.pt: generate(top_k={CANDIDATE_TOP_K}, max_output={K}, "
                f"ranking_policy=rank, filter_candidate_cap={FILTER_CANDIDATE_CAP})"
            ),
            "match": "inchikey_tautomer",
            "resampling_unit": "substrate",
            "n_boot": 10000,
            "seed": 0,
            "workers": args.workers,
            "source_checkpoints": {
                "generator": str(GEN_CKPT.relative_to(ROOT)),
                "filter": str(FILTER_CKPT.relative_to(ROOT)),
                "prior_restored_from": (str(PRIORS_CKPT.relative_to(ROOT)) if args.copy_prior else None),
            },
        },
        "per_substrate": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("factors", "micro_recall", "oracle_recall", "n_substrates")}, indent=2), flush=True)

    # sanity: the decomposition is an identity -> product of factors equals micro_recall to rounding
    prod = (factors["coverage_bank"]["point"] * factors["selection_retention"]["point"]
            * factors["ranking_conversion"]["point"])
    assert abs(prod - micro_recall) < 1e-6, (prod, micro_recall)
    print(f"\nidentity closes: {prod:.6f} == micro_recall {micro_recall:.6f}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
