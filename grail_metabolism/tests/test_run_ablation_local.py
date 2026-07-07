"""Dataset-free guard tests for scripts/run_ablation_local.py -- the LOCAL sequential
ablation runner that replaces the (unavailable) Modal orchestration for Phase 3's
scoped M2-scale ablation (03-03-PLAN.md, LOCAL_OVERRIDE).

Mirrors the project's existing source-inspection + pure-function testing convention
(test_eval_diversity.py's ``RUN_GFLOWNET_PATH`` pattern) -- no dataset, no real
subprocess execution of run_gflownet.py (that is the smoke-run's job, done separately
on real data), no Modal import anywhere.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "run_ablation_local.py"

sys.path.insert(0, str(ROOT))

import scripts.run_ablation_local as runner  # noqa: E402


def _code_only(src: str) -> str:
    """Strip the module docstring (everything up to the first blank-line-terminated
    triple-quoted block starting the file) so source-inspection guards below only see
    ACTUAL CODE, not prose that legitimately discusses (and disclaims) Modal/.map() in
    the docstring's LOCAL_OVERRIDE explanation. Mirrors the comment-stripping convention
    used by the project's other modal_m2.py-style guard tests."""
    lines = src.splitlines()
    if not lines or not lines[0].startswith("#!"):
        return src
    # Find the module docstring bounds (first `"""` opens it, next `"""` closes it).
    joined = "\n".join(lines)
    first = joined.find('"""')
    if first == -1:
        return joined
    second = joined.find('"""', first + 3)
    if second == -1:
        return joined
    return joined[second + 3:]


def test_no_modal_import_or_fanout_in_source():
    """The LOCAL_OVERRIDE contract: this script must NEVER import modal or use any
    Modal fan-out primitive. Filters out comment/docstring lines mentioning "Modal" in
    prose (the module docstring explicitly discusses why Modal is NOT used)."""
    src = _code_only(RUNNER_PATH.read_text())
    assert "import modal" not in src
    assert "from modal" not in src
    assert ".map(" not in src
    assert ".starmap(" not in src
    assert "@app.function" not in src
    assert "modal.App" not in src


def test_seeds_run_sequentially_not_via_fanout():
    """Source-inspection: the per-seed loops in main() are plain Python `for seed in
    args.seeds:` loops (subprocess.run sequentially), mirroring modal_m2.py::run_m2's
    sequential-in-one-container shape, generalized to a local process instead of a
    Modal container."""
    src = RUNNER_PATH.read_text()
    assert "for seed in args.seeds:" in src
    assert re.search(r"subprocess\.run\(cmd, check=True", src)


def test_idempotent_skip_mirrors_modal_m2_pattern():
    """The runner's _invoke() must skip already-complete out paths (idempotent resume),
    the exact contract modal_m2.py::run_m2 implements via `if os.path.exists(out): skip`."""
    src = RUNNER_PATH.read_text()
    assert "out_path.exists()" in src
    assert "already complete" in src


def test_subprocess_cmd_is_a_fixed_list_no_shell_interpolation():
    """No shell=True anywhere, and no f-string/format building of a single command
    STRING that would then need shell parsing -- every subprocess.run call in this
    module is called with a Python list (the safe, non-shell-interpolated form)."""
    src = _code_only(RUNNER_PATH.read_text())
    assert "shell=True" not in src
    assert "os.system(" not in src
    assert "subprocess.run(cmd, check=True" in src


def test_per_seed_out_and_checkpoint_paths_are_seed_keyed():
    """Preemption/interruption resilience (RESEARCH Landmine #5, mirrored locally): every
    ablation-mode invocation gets a seed-keyed --out/--resume-ckpt/--resume-eval-ckpt
    triple, generalizing modal_m2.py's `ablation02_seed{seed}` convention to both
    ablation modes."""
    ckpt, eval_ckpt = runner._ckpt_paths(Path("/tmp/x"), "ablation_single_val_seed2")
    assert "seed2" in str(ckpt)
    assert "seed2" in str(eval_ckpt)
    assert ckpt.name.endswith(".ckpt.pt")
    assert eval_ckpt.name.endswith(".eval_ckpt.json")

    src = RUNNER_PATH.read_text()
    assert '"--resume-ckpt"' in src
    assert '"--resume-eval-ckpt"' in src
    assert "seed={seed}" in src or "seed{seed}" in src


def test_verdict_delegated_to_shared_aggregate_and_verdict():
    """FIX 1 wiring: main() must compute its final verdict by calling the SHARED
    ``grail_metabolism.ablation_plan.aggregate_and_verdict`` -- the same function
    ``scripts/modal_ablation.py`` calls -- instead of duplicating the verdict/
    config-match-gate logic inline (which had drifted to reference
    ``compute_ablation_verdict`` without importing it, an unguarded NameError).
    ``aggregate_and_verdict`` applies the FIX C ``assert_config_match`` gate
    internally before reading any test-table value, so this runner no longer needs
    (and no longer has) its own inline gate call."""
    src = RUNNER_PATH.read_text()
    code = _code_only(src)
    assert "aggregate_and_verdict(" in code, "main() must call the shared aggregate_and_verdict"
    assert "compute_ablation_verdict" not in code, (
        "compute_ablation_verdict must not be referenced directly in this file's CODE -- it "
        "is only used inside the shared ablation_plan.aggregate_and_verdict (a bare, "
        "unimported reference here is exactly the NameError bug this fix addresses)"
    )
    # No duplicate/divergent verdict logic: this file must not import diversity's
    # verdict primitives directly anymore -- it delegates to ablation_plan instead.
    assert "from grail_metabolism.eval.diversity import" not in code


def test_no_unimported_name_references_in_source():
    """Regression guard for the exact class of bug this fix addresses: every bare
    name used in the module must resolve to something imported or defined in this
    file (catches a future NameError-by-refactor before it reaches a multi-hour
    Modal-adjacent run). Cheap proxy: compile the file (catches syntax errors) and
    confirm the historically-broken symbol is gone from this file's namespace."""
    import ast

    src = RUNNER_PATH.read_text()
    compile(src, str(RUNNER_PATH), "exec")  # must not raise
    tree = ast.parse(src)
    assert isinstance(tree, ast.Module)


def test_pick_beta_prime_selects_max_val_score():
    scores = {2.0: 0.10, 4.0: 0.25, 6.0: 0.30, 8.0: 0.22, 10.0: 0.05}
    assert runner.pick_beta_prime(scores) == 6.0


def test_pick_beta_prime_warns_on_endpoint_optimum(capsys):
    # Best-of-{6,10} at an endpoint (10) should trigger the D-10 endpoint-widen WARNING.
    scores = {6.0: 0.10, 10.0: 0.30}
    best = runner.pick_beta_prime(scores)
    assert best == 10.0
    captured = capsys.readouterr()
    assert "endpoint-widen" in captured.out or "GRID ENDPOINT" in captured.out


def test_pick_beta_prime_no_warning_for_interior_optimum(capsys):
    scores = {2.0: 0.10, 6.0: 0.30, 10.0: 0.05}
    best = runner.pick_beta_prime(scores)
    assert best == 6.0
    captured = capsys.readouterr()
    assert "endpoint-widen" not in captured.out


def test_pick_beta_prime_raises_on_all_none():
    scores = {2.0: None, 6.0: None}
    try:
        runner.pick_beta_prime(scores)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_degeneracy_guarded_margin_uses_fixed_fallback_below_floor():
    # std below the 0.005 absolute floor -> fixed fallback, regardless of mean.
    assert runner.degeneracy_guarded_margin(std=0.001, mean_auc=0.30) == 0.02


def test_degeneracy_guarded_margin_uses_fixed_fallback_above_cv_bound():
    # std/mean > 1.0 (high relative noise) -> fixed fallback even if std clears the floor.
    assert runner.degeneracy_guarded_margin(std=0.10, mean_auc=0.05) == 0.02


def test_degeneracy_guarded_margin_uses_one_x_std_when_healthy():
    assert runner.degeneracy_guarded_margin(std=0.02, mean_auc=0.30) == 0.02 * 1.0


def test_compute_delta_sensitivity_grid_reports_all_four_thresholds():
    grid = runner.compute_delta_sensitivity_grid(
        gflownet_auc=0.30, abl01_auc=0.20, abl02_auc=0.22, std=0.02,
    )
    assert set(grid.keys()) == {"0.5x_std", "1.0x_std", "1.5x_std", "fixed_0.02"}
    for outcome in grid.values():
        assert outcome in ("confirmed", "null", "partial")


def test_paired_arrays_intersects_by_shared_root_and_reads_union_curves(tmp_path):
    """Build a synthetic --resume-eval-ckpt JSON (the shape run_gflownet.py's
    _save_eval_ckpt writes) and confirm paired_arrays extracts matched per-substrate AUC
    arrays over the shared root intersection, keyed off {series}_union_curve."""
    ckpt_path = tmp_path / "fake.eval_ckpt.json"
    rows = {
        "CCO": {
            "gflownet_union_curve": {"5": 0.4, "10": 0.5, "15": 0.6, "20": 0.6, "30": 0.7, "50": 0.8},
            "ablation01_union_curve": {"5": 0.2, "10": 0.3, "15": 0.3, "20": 0.4, "30": 0.4, "50": 0.5},
        },
        "CCCO": {
            "gflownet_union_curve": {"5": 0.5, "10": 0.5, "15": 0.5, "20": 0.5, "30": 0.5, "50": 0.5},
            "ablation01_union_curve": {"5": 0.1, "10": 0.1, "15": 0.1, "20": 0.1, "30": 0.1, "50": 0.1},
        },
        "CCCCO": {
            # Only gflownet ran for this root (e.g. ablation01 under-produced and was
            # skipped) -- must be excluded from the shared intersection.
            "gflownet_union_curve": {"5": 0.9, "10": 0.9, "15": 0.9, "20": 0.9, "30": 0.9, "50": 0.9},
        },
    }
    with open(ckpt_path, "w") as fh:
        json.dump({"config_fingerprint": "x", "rows": rows, "next_idx": 3}, fh)

    gflownet_arr, abl01_arr = runner.paired_arrays(ckpt_path, ckpt_path, "ablation01")
    assert len(gflownet_arr) == 2  # CCCCO excluded (no ablation01 curve)
    assert len(abl01_arr) == 2
    # gflownet's own AUC on "CCO" should be strictly greater than on the flat "CCCO" curve's
    # ablation01 AUC (sanity that real values, not placeholders, were read).
    assert all(g > a for g, a in zip(gflownet_arr, abl01_arr))


def test_fixed_args_builds_a_plain_list_not_a_string():
    args = runner._fixed_args(
        train_substrates=5, test_substrates=3, epochs=1, prewarm_waves=1,
        eval_beam=False, top_k=20, max_size=6, max_depth=2, workers=4,
        logz_lr=0.1, n_samples=2,
    )
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)
    assert "--no-eval-beam" in args
    assert "--train-substrates" in args and "5" in args
