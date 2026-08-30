#!/usr/bin/env python3
"""Every setting the released checkpoints were trained under, read from the runs themselves.

The Supporting Information described the two models in prose and gave one line of optimisation
detail. That is not enough to retrain either of them, and a paper whose argument is that
undeclared choices make this literature incomparable should not leave its own training
undeclared. Nothing here is typed: the values are read from the `config.yaml` each run wrote
beside its checkpoints and from the training reports written when it finished, so a table built
from this cannot drift from the weights that produced the numbers.

Two checkpoints are released and they come from different runs -- the generator from the run whose
labels were built in the firing convention, the filter from the earlier one -- so both are read and
the fields that differ between them are marked rather than averaged away.

    python scripts/typed_edit/hyperparameters.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

RUNS = {"generator": "artifacts/full5000_implicit", "filter": "artifacts/full5000_priors"}


def load(run: str) -> dict:
    import yaml

    base = ROOT / run
    cfg = yaml.safe_load((base / "config.yaml").read_text())
    reports = {}
    for name in ("generator_training", "filter_training"):
        path = base / "reports" / f"{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text())
    return {"config": cfg, "reports": reports}


def main() -> int:
    runs = {which: load(path) for which, path in RUNS.items()}
    gen_cfg = runs["generator"]["config"]
    fil_cfg = runs["filter"]["config"]

    # Which run each released checkpoint comes from, so a reader knows which column to trust for
    # which component. The filter's own settings are identical across the two runs; the
    # generator's are not, which is why the generator is read from its own run.
    shared_differs = sorted(k for k in gen_cfg if gen_cfg[k] != fil_cfg.get(k))

    def block(cfg, *path):
        node = cfg
        for key in path:
            node = (node or {}).get(key)
        return node

    rows = {
        "generator": {
            "run": RUNS["generator"],
            "graph convolution": block(gen_cfg, "generator", "conv_kind"),
            "encoder hidden dims": block(gen_cfg, "generator", "hidden_dims"),
            "rule encoder hidden dims": block(gen_cfg, "generator", "rule_hidden_dims"),
            "projection dim": block(gen_cfg, "generator", "projection_dim"),
            "scoring": block(gen_cfg, "generator", "scoring"),
            "rule budget at training": block(gen_cfg, "generator", "top_k"),
            "unlabelled weight": block(gen_cfg, "generator", "unlabeled_weight"),
            "ranking auxiliary weight": block(gen_cfg, "generator", "rank_weight"),
            "ranking margin": block(gen_cfg, "generator", "ranking_margin"),
            "frequency prior weight": block(gen_cfg, "generator", "prior_strength"),
            "inapplicable-rule penalty": block(gen_cfg, "generator", "applicability_penalty"),
            "candidate aggregation": block(gen_cfg, "generator", "candidate_aggregation"),
            "optimiser": "Adam",
            "learning rate": block(gen_cfg, "generator_optim", "lr"),
            "batch size": block(gen_cfg, "generator_optim", "batch_size"),
            "weight decay": block(gen_cfg, "generator_optim", "weight_decay"),
            "epochs requested": block(gen_cfg, "generator_optim", "epochs"),
            "epochs run": block(runs["generator"]["reports"], "generator_training",
                                "epochs_trained"),
            "early stopping patience": block(gen_cfg, "generator_optim", "patience"),
            "non-negative PU risk": block(gen_cfg, "generator_optim", "nnpu"),
            "class prior": block(gen_cfg, "generator_optim", "prior"),
            "decision threshold": block(runs["generator"]["reports"], "generator_training",
                                        "calibrated_threshold"),
            "seed": gen_cfg.get("seed"),
        },
        "filter": {
            "run": RUNS["filter"],
            "graph convolution": block(fil_cfg, "filter", "conv_kind"),
            "encoder hidden dims": block(fil_cfg, "filter", "hidden_dims"),
            "dropout": block(fil_cfg, "filter", "dropout"),
            "pairing mode": block(fil_cfg, "filter", "mode"),
            "fingerprints": block(fil_cfg, "filter", "use_fingerprint"),
            "optimiser": "Adam",
            "learning rate": block(fil_cfg, "filter_optim", "lr"),
            "batch size": block(fil_cfg, "filter_optim", "batch_size"),
            "weight decay": block(fil_cfg, "filter_optim", "weight_decay"),
            "epochs requested": block(fil_cfg, "filter_optim", "epochs"),
            "epochs run": block(runs["filter"]["reports"], "filter_training", "epochs_trained"),
            "early stopping patience": block(fil_cfg, "filter_optim", "patience"),
            "non-negative PU risk": block(fil_cfg, "filter_optim", "nnpu"),
            "class prior": block(fil_cfg, "filter_optim", "prior"),
            "decision threshold": block(runs["filter"]["reports"], "filter_training",
                                        "calibrated_threshold"),
            "seed": fil_cfg.get("seed"),
        },
    }

    data = {
        "training substrates": block(gen_cfg, "dataset", "max_train_substrates"),
        "validation substrates": block(gen_cfg, "dataset", "max_val_substrates"),
        "sampling seed": block(gen_cfg, "dataset", "sampling_seed"),
        "clean splits": block(gen_cfg, "dataset", "use_clean_splits"),
        "rule bank": block(gen_cfg, "dataset", "rules_path"),
        "pretraining": block(gen_cfg, "pretrain", "enabled"),
    }

    report = {"provenance": stamp(__file__),
              "source": "the config.yaml and training reports each run wrote beside its weights",
              "runs": RUNS,
              "components": rows,
              "data": data,
              "config_blocks_that_differ_between_the_two_runs": shared_differs,
              "reading": (
                  "The generator's row is read from the run its released checkpoint comes from and "
                  "the filter's from the run its own comes from. The filter is trained under "
                  "cross-entropy and the generator under a non-negative positive-unlabelled risk; "
                  "that difference is a setting here rather than a claim in prose.")}
    (ROOT / "results/hyperparameters.json").write_text(json.dumps(report, indent=1))
    for name, row in rows.items():
        print(f"\n{name} ({row['run']})")
        for key, value in row.items():
            if key != "run":
                print(f"  {key:28s} {value}")
    print("\nwrote results/hyperparameters.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
