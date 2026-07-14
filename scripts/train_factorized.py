#!/usr/bin/env python3
"""Train the FactorizedGenerator (type + site heads) by MLE and run the pivotal Task-4 val gate.

The whole factorized redesign only escapes GRAIL's PU degeneracy (Prop 2 — the old 7,581-way
rule selector loses to a frequency prior) if the DENSE type head, trained by real maximum
likelihood (BCEWithLogits, not PU), actually beats the type-frequency prior on held-out
substrates. This script trains the model on TRAIN and reports that comparison on VAL.

Ground truth for the val gate:
  - Train labels: `factorized_data.build_factorized_dataset` reuses the mining catalog's
    per-rule `source_pairs` provenance (Task 2) -- fast, no re-apply.
  - Val labels: catalog `source_pairs` are recorded ONLY for TRAIN pairs, so they cannot supply
    ground truth for (train-disjoint) val substrates. Instead, for each val substrate we
    directly RE-APPLY the ~1,892 catalog rules that map to a real (non-"other") type (a
    from-scratch version of `utils.preparation.apply_rules_to_molecule`'s AddHs/RunReactants
    pipeline, with a shorter per-rule timeout/product-cap and a hard per-substrate wall-clock
    budget -- see `_val_true_types`), and keep the type(s) whose applied product matches an
    annotated val metabolite by tautomer-InChIKey (`metrics._tautomer_inchikey`). This is the
    same "which rule reproduces the true product" provenance definition Task 1/2 use for train,
    just computed live instead of read from a cache -- feasible because val is capped
    (`max_val_substrates`) and the search is restricted to the 1,892 typed rules, not the full
    ~7,581-rule bank.
  - Site labels (both splits): `model.som.derive_som_labels`, independent of catalog/rule_to_type.

See docs/superpowers/plans/2026-07-14-grail-factorized-generator.md (Task 4) and
.superpowers/sdd/task-4-brief.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.config import DatasetConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.factorized import FactorizedGenerator
from grail_metabolism.model.factorized_data import build_factorized_dataset
from grail_metabolism.model.som import derive_som_labels
from grail_metabolism.utils.preparation import (
    _clean_product_smiles,
    _compile_rule_pattern,
    _compile_rule_reaction,
    safe_run_reactants,
)
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.workflows.data import load_dataset_bundle

CATALOG_PATH = ROOT / "results" / "mined_rule_catalog_v2.json"
VOCAB_PATH = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"
GATE_C_SITE_HIT_AT_3 = 0.807  # committed baseline (results/redesign_gate_c.json)


def _load_vocab() -> Dict[str, int]:
    vocab = json.loads(VOCAB_PATH.read_text())
    return vocab["rule_to_type"]


def _load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def _typed_rules(rule_to_type: Dict[str, int]) -> List[str]:
    """SMIRKS whose rule maps to a real (non-'other') dense type id."""
    return [smirks for smirks, type_id in rule_to_type.items() if type_id >= 0]


def _prepare_typed_rules(typed_rules: Sequence[str]) -> List[Tuple[str, "Chem.Mol", object]]:
    """Pre-compile (smirks, reactant-pattern, reaction) triples, skipping unparseable SMIRKS.

    `_compile_rule_pattern`/`_compile_rule_reaction` are `lru_cache`d in `preparation.py`, so
    this also warms that cache once instead of paying compile cost per val substrate.
    """
    prepared: List[Tuple[str, "Chem.Mol", object]] = []
    for smirks in typed_rules:
        pattern = _compile_rule_pattern(smirks)
        rxn = _compile_rule_reaction(smirks)
        if pattern is not None and rxn is not None:
            prepared.append((smirks, pattern, rxn))
    return prepared


def _val_true_types(
    val_map,
    prepared_rules: Sequence[Tuple[str, "Chem.Mol", object]],
    rule_to_type: Dict[str, int],
    log_every: int = 25,
    rule_timeout: float = 1.0,
    max_products: int = 20,
    per_substrate_budget: float = 15.0,
) -> Dict[str, Set[int]]:
    """{substrate_smiles: {type_id, ...}} for val, by live rule re-application (see module doc).

    Uses a SHORT per-rule RDKit timeout/product-cap (vs. `apply_rules_to_molecule`'s default
    5s/500) plus a hard per-substrate wall-clock budget: a small fraction of substrates hit
    combinatorial matches (large/symmetric rings) that would otherwise cost minutes each on a
    ~1,892-rule scan. Once a type is confirmed true for a substrate, further rules mapping to
    that SAME type are skipped (the label is multi-hot presence, not provenance count).
    """
    out: Dict[str, Set[int]] = {}
    n = len(val_map)
    n_truncated = 0
    t0 = time.perf_counter()
    for i, (sub, mets) in enumerate(val_map.items(), 1):
        if log_every and i % log_every == 0:
            print(f"  val truth {i}/{n}  ({time.perf_counter() - t0:.0f}s, truncated={n_truncated})", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            out[sub] = set()
            continue
        met_list = [mets] if isinstance(mets, str) else list(mets)
        true_keys: Set[str] = set()
        for met in met_list:
            try:
                true_keys.add(_tautomer_inchikey(met))
            except Exception:
                continue
        if not true_keys:
            out[sub] = set()
            continue

        substrate_h = Chem.AddHs(Chem.Mol(mol))
        true_types: Set[int] = set()
        sub_t0 = time.perf_counter()
        for smirks, pattern, rxn in prepared_rules:
            if time.perf_counter() - sub_t0 > per_substrate_budget:
                n_truncated += 1
                break
            type_id = rule_to_type.get(smirks, -1)
            if type_id < 0 or type_id in true_types:
                continue  # already confirmed, or not a dense type -- skip re-testing
            try:
                if not substrate_h.HasSubstructMatch(pattern):
                    continue
            except Exception:
                continue
            outcomes = safe_run_reactants(rxn, substrate_h, timeout=rule_timeout, max_products=max_products)
            matched = False
            for product_tuple in outcomes:
                for prod in product_tuple:
                    try:
                        smi = Chem.MolToSmiles(prod)
                    except Exception:
                        continue
                    for fragment in _clean_product_smiles(smi):
                        try:
                            key = _tautomer_inchikey(fragment)
                        except Exception:
                            continue
                        if key in true_keys:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                true_types.add(type_id)
        out[sub] = true_types
    print(f"  val truth done: {n_truncated}/{n} substrates hit the per-substrate time budget", flush=True)
    return out


def _recall_at_k(ranked_ids: Sequence[int], true_ids: Set[int], k: int) -> float:
    if not true_ids:
        return float("nan")
    topk = set(ranked_ids[:k])
    return len(topk & true_ids) / len(true_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-val-substrates", type=int, default=300)
    ap.add_argument("--max-train-substrates", type=int, default=0, help="0 = all available")
    ap.add_argument("--min-pairs", type=int, default=5, help="must match coarse_type_vocab.json build")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rule-timeout", type=float, default=1.0, help="per-rule RDKit RunReactants timeout (s)")
    ap.add_argument("--rule-max-products", type=int, default=20, help="per-rule RDKit maxProducts cap")
    ap.add_argument("--substrate-time-budget", type=float, default=15.0,
                     help="hard wall-clock cap (s) on the val ground-truth rule scan per substrate")
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "factorized_v1" / "checkpoints" / "factorized.pt"))
    ap.add_argument("--report", type=str, default=str(ROOT / "results" / "factorized_val.json"))
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    rule_to_type = _load_vocab()
    num_types = 1 + max(rule_to_type.values())
    catalog = _load_catalog()
    typed_rules = _typed_rules(rule_to_type)
    prepared_typed_rules = _prepare_typed_rules(typed_rules)
    print(f"vocab: num_types={num_types}  typed_rules={len(typed_rules)}/{len(rule_to_type)}  "
          f"parsed={len(prepared_typed_rules)}", flush=True)

    dataset_cfg = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=(args.max_train_substrates or None),
        max_val_substrates=args.max_val_substrates,
        sampling_seed=args.seed,
    )
    print("loading dataset bundle...", flush=True)
    t0 = time.perf_counter()
    bundle = load_dataset_bundle(dataset_cfg)
    print(f"  train substrates={len(bundle.train.map)}  val substrates={len(bundle.val.map)}  "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print("building TRAIN factorized dataset (catalog source_pairs)...", flush=True)
    train_dataset = build_factorized_dataset(bundle.train, rule_to_type, catalog=catalog)
    n_pos_type = int(sum(float(d.y_type.sum()) for d in train_dataset))
    n_pos_site = int(sum(float(d.y_site.sum()) for d in train_dataset))
    print(f"  train graphs={len(train_dataset)}  positive type labels={n_pos_type}  "
          f"positive site atoms={n_pos_site}", flush=True)
    if not train_dataset or n_pos_type == 0:
        print("ERROR: no dense type training signal (empty dataset or no positive types)", flush=True)
        return 1

    print("training FactorizedGenerator (MLE: BCEWithLogits on both heads)...", flush=True)
    model = FactorizedGenerator(num_types=num_types)
    t0 = time.perf_counter()
    history = model.fit(train_dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    for i, loss in enumerate(history, 1):
        if i == 1 or i % 5 == 0 or i == len(history):
            print(f"  epoch {i:>3}/{len(history)}  loss={loss:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"saved checkpoint -> {out_path}", flush=True)

    # ---- VAL GATE ----------------------------------------------------------------
    print("computing val ground-truth types (live rule re-application)...", flush=True)
    val_true_types = _val_true_types(
        bundle.val.map, prepared_typed_rules, rule_to_type,
        rule_timeout=args.rule_timeout, max_products=args.rule_max_products,
        per_substrate_budget=args.substrate_time_budget,
    )
    n_localizable_type = sum(1 for s in val_true_types.values() if s)
    print(f"  val substrates with >=1 true type: {n_localizable_type}/{len(bundle.val.map)}", flush=True)

    # global type-frequency prior: counts over TRAIN's y_type (marginal, substrate-independent)
    prior_counts = torch.zeros(num_types)
    for d in train_dataset:
        prior_counts += d.y_type
    prior_ranked = [int(i) for i in torch.argsort(-prior_counts).tolist()]

    model.eval()
    ks = (5, 10)
    model_recalls: Dict[int, List[float]] = {k: [] for k in ks}
    prior_recalls: Dict[int, List[float]] = {k: [] for k in ks}

    site_hits3 = 0
    site_localizable = 0

    with torch.no_grad():
        for sub, mets in bundle.val.map.items():
            mol = Chem.MolFromSmiles(sub)
            data = from_rdmol(mol) if mol is not None else None
            if data is None or data.x.size(0) == 0:
                continue

            true_types = val_true_types.get(sub, set())
            if true_types:
                type_probs = torch.sigmoid(model.type_logits(data)).view(-1)
                model_ranked = [int(i) for i in torch.argsort(-type_probs).tolist()]
                for k in ks:
                    model_recalls[k].append(_recall_at_k(model_ranked, true_types, k))
                    prior_recalls[k].append(_recall_at_k(prior_ranked, true_types, k))

            true_sites = derive_som_labels(sub, mets)
            if true_sites:
                site_probs = torch.sigmoid(model.site_logits(data)).view(-1).cpu().numpy()
                if site_probs.size:
                    site_localizable += 1
                    ranked_atoms = list(np.argsort(-site_probs))
                    top3 = set(int(a) for a in ranked_atoms[:3])
                    if top3 & set(int(a) for a in true_sites if 0 <= a < site_probs.size):
                        site_hits3 += 1

    def _macro(vals: List[float]) -> float:
        vals = [v for v in vals if not np.isnan(v)]
        return round(float(np.mean(vals)), 4) if vals else None

    report = {
        "config": {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "max_val_substrates": args.max_val_substrates,
            "max_train_substrates": args.max_train_substrates or None,
            "seed": args.seed, "num_types": num_types,
        },
        "n_train_graphs": len(train_dataset),
        "n_val_substrates": len(bundle.val.map),
        "n_val_with_true_type": n_localizable_type,
        "n_val_site_localizable": site_localizable,
        "train_loss_history": [round(h, 4) for h in history],
        "type_head_recall@5": _macro(model_recalls[5]),
        "type_head_recall@10": _macro(model_recalls[10]),
        "type_frequency_prior_recall@5": _macro(prior_recalls[5]),
        "type_frequency_prior_recall@10": _macro(prior_recalls[10]),
        "site_hit@3": round(site_hits3 / site_localizable, 4) if site_localizable else None,
        "gate_c_site_hit@3_baseline": GATE_C_SITE_HIT_AT_3,
        "checkpoint": str(out_path.relative_to(ROOT)) if ROOT in out_path.resolve().parents else str(out_path),
    }

    model_r10 = report["type_head_recall@10"] or 0.0
    prior_r10 = report["type_frequency_prior_recall@10"] or 0.0
    gate_pass = model_r10 > prior_r10
    report["gate_verdict"] = "PASS" if gate_pass else "FAIL"

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(
        f"\nTASK-4 VAL GATE: type_head_recall@10={model_r10} vs "
        f"type_frequency_prior_recall@10={prior_r10} "
        f"=> {'PASS (redesign escapes PU degeneracy)' if gate_pass else 'FAIL (degeneracy persists at type level)'}",
        flush=True,
    )
    print(f"wrote {report_path}", flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
