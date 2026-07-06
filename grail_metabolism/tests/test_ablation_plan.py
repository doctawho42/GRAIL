"""Dataset-free guard tests for grail_metabolism/ablation_plan.py -- the shared
planning/selection/verdict module used by BOTH scripts/run_ablation_local.py (local
sequential fallback) and scripts/modal_ablation.py (parallel Modal orchestrator).

No dataset, no subprocess execution of run_gflownet.py, no Modal import anywhere in
this file or in ablation_plan.py itself -- these are pure-function/synthetic-dict
tests, mirroring the project's existing test_run_ablation_local.py convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import grail_metabolism.ablation_plan as ablation_plan  # noqa: E402


def test_no_modal_import_in_source():
    """ablation_plan.py must never import modal -- it is shared by the LOCAL runner
    too, and must stay importable without a Modal account (mirrors run_ablation_local's
    own no-Modal guard test)."""
    src = (ROOT / "grail_metabolism" / "ablation_plan.py").read_text()
    assert "import modal" not in src
    assert "from modal" not in src


def test_plan_configs_produces_independent_config_set():
    plan = ablation_plan.plan_configs(
        beta_prime_grid=[2.0, 4.0, 6.0], seeds=[0, 1, 2], m_ensemble=3,
    )
    assert set(plan.keys()) == {"sweep", "val", "test"}

    # Sweep: one config per grid point, all at seed 0 (first seed), mode="single".
    assert len(plan["sweep"]) == 3
    assert {c["beta_prime"] for c in plan["sweep"]} == {2.0, 4.0, 6.0}
    assert all(c["mode"] == "single" for c in plan["sweep"])
    assert all(c["seed"] == 0 for c in plan["sweep"])
    assert all(c["eval_split"] == "val" for c in plan["sweep"])

    # VAL: one single + one ensemble config per seed -> 2 * 3 = 6 configs.
    assert len(plan["val"]) == 6
    assert sum(1 for c in plan["val"] if c["mode"] == "single") == 3
    assert sum(1 for c in plan["val"] if c["mode"] == "ensemble") == 3
    assert {c["seed"] for c in plan["val"]} == {0, 1, 2}
    assert all(c["beta_prime"] is None for c in plan["val"])  # filled in later

    # Test: exactly one single + one ensemble config, at seed 0, eval_split="test".
    assert len(plan["test"]) == 2
    modes = {c["mode"] for c in plan["test"]}
    assert modes == {"single", "ensemble"}
    assert all(c["seed"] == 0 for c in plan["test"])
    assert all(c["eval_split"] == "test" for c in plan["test"])

    # Every config is a plain, JSON-serializable dict with a unique tag.
    all_configs = plan["sweep"] + plan["val"] + plan["test"]
    tags = [c["tag"] for c in all_configs]
    assert len(tags) == len(set(tags)), "every config must have a UNIQUE tag"
    import json
    for cfg in all_configs:
        json.dumps(cfg)  # must not raise -- JSON-serializable contract


def test_plan_configs_independent_of_wave_ordering_within_a_wave():
    """Configs within the SAME wave differ only in seed/mode/beta_prime -- none of
    them reference another config's tag/output, confirming they are safe to run in
    parallel (no config depends on another config's result within a wave)."""
    plan = ablation_plan.plan_configs(beta_prime_grid=[2.0, 6.0], seeds=[0, 1], m_ensemble=2)
    for wave_name in ("sweep", "val", "test"):
        for cfg in plan[wave_name]:
            # A config never carries a reference to another config's tag.
            assert not any(
                other["tag"] in str(cfg.get(k, "")) for other in plan[wave_name] for k in cfg
                if other["tag"] != cfg["tag"]
            )


def test_fill_beta_prime_returns_new_list_without_mutating_input():
    plan = ablation_plan.plan_configs(beta_prime_grid=[6.0], seeds=[0], m_ensemble=2)
    original = plan["val"]
    filled = ablation_plan.fill_beta_prime(original, 8.0)
    assert all(c["beta_prime"] is None for c in original), "input configs must not be mutated"
    assert all(c["beta_prime"] == 8.0 for c in filled)
    assert filled is not original


def test_sweep_scores_from_results_reads_ablation01_auc():
    plan = ablation_plan.plan_configs(beta_prime_grid=[2.0, 4.0, 6.0], seeds=[0])
    results_by_tag = {
        plan["sweep"][0]["tag"]: {"metrics": {"ablation01_union_at_k_auc": 0.10}},
        plan["sweep"][1]["tag"]: {"metrics": {"ablation01_union_at_k_auc": 0.30}},
        # third config missing (e.g. failed / not yet run) -> None
    }
    scores = ablation_plan.sweep_scores_from_results(plan["sweep"], results_by_tag)
    assert scores == {2.0: 0.10, 4.0: 0.30, 6.0: None}


def test_select_beta_prime_picks_max_val_score():
    scores = {2.0: 0.10, 4.0: 0.25, 6.0: 0.30, 8.0: 0.22, 10.0: 0.05}
    assert ablation_plan.select_beta_prime(scores) == 6.0


def test_select_beta_prime_warns_on_endpoint_optimum(capsys):
    scores = {6.0: 0.10, 10.0: 0.30}
    best = ablation_plan.select_beta_prime(scores)
    assert best == 10.0
    captured = capsys.readouterr()
    assert "endpoint-widen" in captured.out or "GRID ENDPOINT" in captured.out


def test_select_beta_prime_raises_on_all_none():
    try:
        ablation_plan.select_beta_prime({2.0: None, 6.0: None})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_pick_beta_prime_alias_matches_select_beta_prime():
    """Backward-compat alias used by run_ablation_local.py."""
    assert ablation_plan.pick_beta_prime is ablation_plan.select_beta_prime


def test_degeneracy_guarded_margin_fixed_fallback_below_floor():
    assert ablation_plan.degeneracy_guarded_margin(std=0.001, mean_auc=0.30) == 0.02


def test_degeneracy_guarded_margin_fixed_fallback_above_cv_bound():
    assert ablation_plan.degeneracy_guarded_margin(std=0.10, mean_auc=0.05) == 0.02


def test_degeneracy_guarded_margin_one_x_std_when_healthy():
    assert ablation_plan.degeneracy_guarded_margin(std=0.02, mean_auc=0.30) == 0.02


def test_compute_delta_sensitivity_grid_reports_all_four_thresholds():
    grid = ablation_plan.compute_delta_sensitivity_grid(
        gflownet_auc=0.30, abl01_auc=0.20, abl02_auc=0.22, std=0.02,
    )
    assert set(grid.keys()) == {"0.5x_std", "1.0x_std", "1.5x_std", "fixed_0.02"}
    for outcome in grid.values():
        assert outcome in ("confirmed", "null", "partial")


def test_mean_std_handles_empty_and_single_element():
    assert ablation_plan.mean_std([]) == (0.0, 0.0)
    mean, std = ablation_plan.mean_std([5.0])
    assert mean == 5.0
    assert std == 0.0


def test_per_substrate_aucs_reads_union_curve_from_dict():
    eval_ckpt = {
        "rows": {
            "CCO": {"gflownet_union_curve": {"5": 0.4, "10": 0.5, "15": 0.6, "20": 0.6, "30": 0.7, "50": 0.8}},
            "CCCO": {"gflownet_union_curve": {"5": 0.5, "10": 0.5, "15": 0.5, "20": 0.5, "30": 0.5, "50": 0.5}},
        }
    }
    aucs = ablation_plan.per_substrate_aucs(eval_ckpt, "gflownet")
    assert set(aucs) == {"CCO", "CCCO"}
    assert aucs["CCO"] > aucs["CCCO"]  # higher curve -> higher AUC


def test_paired_arrays_from_ckpts_intersects_by_shared_root():
    gflownet_ckpt = {
        "rows": {
            "CCO": {"gflownet_union_curve": {"5": 0.4, "10": 0.5, "15": 0.6, "20": 0.6, "30": 0.7, "50": 0.8}},
            "CCCO": {"gflownet_union_curve": {"5": 0.5, "10": 0.5, "15": 0.5, "20": 0.5, "30": 0.5, "50": 0.5}},
            "CCCCO": {"gflownet_union_curve": {"5": 0.9, "10": 0.9, "15": 0.9, "20": 0.9, "30": 0.9, "50": 0.9}},
        }
    }
    abl_ckpt = {
        "rows": {
            "CCO": {"ablation01_union_curve": {"5": 0.2, "10": 0.3, "15": 0.3, "20": 0.4, "30": 0.4, "50": 0.5}},
            "CCCO": {"ablation01_union_curve": {"5": 0.1, "10": 0.1, "15": 0.1, "20": 0.1, "30": 0.1, "50": 0.1}},
            # CCCCO missing here (e.g. under-produced, skipped for this arm)
        }
    }
    gflownet_arr, abl_arr = ablation_plan.paired_arrays_from_ckpts(gflownet_ckpt, abl_ckpt, "ablation01")
    assert len(gflownet_arr) == 2  # CCCCO excluded (no ablation01 curve)
    assert len(abl_arr) == 2
    assert all(g > a for g, a in zip(gflownet_arr, abl_arr))


def test_aggregate_and_verdict_produces_full_report_shape():
    """Synthetic end-to-end aggregate_and_verdict call -- byte-shape-identical to
    run_ablation_local.py's `report` dict (the exact keys/nesting the SUMMARY/CLI
    output depend on)."""
    config = {
        "train_substrates_requested": 5, "eval_split": "test", "beta": 6.0,
        "max_size": 6, "top_k": 20, "epochs": 1,
    }

    def _result(gflownet_auc, abl01_auc=None, abl02_auc=None):
        metrics = {"gflownet_union_at_k_auc": gflownet_auc}
        if abl01_auc is not None:
            metrics["ablation01_union_at_k_auc"] = abl01_auc
        if abl02_auc is not None:
            metrics["ablation02_union_at_k_auc"] = abl02_auc
        return {"config": dict(config), "metrics": metrics}

    val_single_results = [
        _result(0.30, abl01_auc=0.20) for _ in range(3)
    ]
    val_ensemble_results = [
        _result(0.30, abl02_auc=0.22) for _ in range(3)
    ]
    test_single_result = _result(0.32, abl01_auc=0.18)
    test_ensemble_result = _result(0.32, abl02_auc=0.20)

    def _eval_ckpt(series, values):
        rows = {}
        for i, v in enumerate(values):
            root = f"root{i}"
            rows.setdefault(root, {})["gflownet_union_curve"] = {
                "5": v, "10": v, "15": v, "20": v, "30": v, "50": v
            }
            rows[root][f"{series}_union_curve"] = {
                "5": v * 0.5, "10": v * 0.5, "15": v * 0.5, "20": v * 0.5, "30": v * 0.5, "50": v * 0.5
            }
        return {"rows": rows}

    test_single_eval_ckpt = _eval_ckpt("ablation01", [0.3, 0.4, 0.5])
    test_ensemble_eval_ckpt = _eval_ckpt("ablation02", [0.3, 0.4, 0.5])

    report = ablation_plan.aggregate_and_verdict(
        val_single_results=val_single_results,
        val_ensemble_results=val_ensemble_results,
        test_single_result=test_single_result,
        test_ensemble_result=test_ensemble_result,
        test_single_eval_ckpt=test_single_eval_ckpt,
        test_ensemble_eval_ckpt=test_ensemble_eval_ckpt,
        chosen_beta_prime=6.0,
        sweep_scores={2.0: 0.10, 6.0: 0.20},
        m_ensemble=3,
    )

    expected_keys = {
        "beta_prime_sweep", "chosen_beta_prime", "val_seed_aucs", "test_table",
        "primary_paired_bootstrap", "secondary_seed_level_verdict",
        "secondary_margin_used", "sensitivity_grid", "m_ensemble", "note_fix_e",
    }
    assert set(report.keys()) == expected_keys
    assert report["chosen_beta_prime"] == 6.0
    assert report["test_table"]["gflownet_union_at_k_auc"] == 0.32
    assert set(report["val_seed_aucs"].keys()) == {"gflownet", "ablation01", "ablation02"}
    assert set(report["primary_paired_bootstrap"].keys()) == {"vs_ablation01", "vs_ablation02"}
    assert report["secondary_seed_level_verdict"] in ("confirmed", "null", "partial")
    assert report["m_ensemble"] == 3


def test_aggregate_and_verdict_raises_on_config_mismatch():
    """FIX C gate must fire when the three test-touch arms' configs disagree outside
    the allowed-drift fields (e.g. a silently different train_substrates)."""
    good_config = {"train_substrates_requested": 5, "eval_split": "test", "top_k": 20}
    bad_config = {"train_substrates_requested": 999, "eval_split": "test", "top_k": 20}

    test_single_result = {
        "config": good_config,
        "metrics": {"gflownet_union_at_k_auc": 0.3, "ablation01_union_at_k_auc": 0.2},
    }
    test_ensemble_result = {
        "config": bad_config,  # drifted train_substrates -- must trigger ValueError
        "metrics": {"gflownet_union_at_k_auc": 0.3, "ablation02_union_at_k_auc": 0.2},
    }
    empty_ckpt = {"rows": {}}

    try:
        ablation_plan.aggregate_and_verdict(
            val_single_results=[{"metrics": {"gflownet_union_at_k_auc": 0.3, "ablation01_union_at_k_auc": 0.2}}],
            val_ensemble_results=[{"metrics": {"gflownet_union_at_k_auc": 0.3, "ablation02_union_at_k_auc": 0.2}}],
            test_single_result=test_single_result,
            test_ensemble_result=test_ensemble_result,
            test_single_eval_ckpt=empty_ckpt,
            test_ensemble_eval_ckpt=empty_ckpt,
            chosen_beta_prime=6.0,
            sweep_scores={},
            m_ensemble=3,
        )
        assert False, "expected ValueError from assert_config_match"
    except ValueError as exc:
        assert "train_substrates_requested" in str(exc)
