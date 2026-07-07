"""Dataset-free guard tests for scripts/modal_ablation.py -- the PARALLEL Modal
orchestrator that un-defers 03-03's Modal task.

Unlike test_run_ablation_local.py (which guards that Modal is NEVER imported),
modal_ablation.py legitimately imports modal (it IS the Modal orchestrator) -- so
these tests are SKIPPED if the ``modal`` package is not installed in the current
environment (matching the plan's "Modal imports guarded so make test doesn't
require Modal" requirement: the test suite must stay green in an environment
without Modal, it just skips this file's module-import-dependent checks in that
case). Source-inspection checks that only read the .py file as TEXT (no import)
always run, regardless of whether ``modal`` is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODAL_ABLATION_PATH = ROOT / "scripts" / "modal_ablation.py"

sys.path.insert(0, str(ROOT))

modal_available = True
try:
    import modal  # noqa: F401
except ImportError:
    modal_available = False


def _code_only(src: str) -> str:
    """Strip the module docstring so source-inspection guards only see ACTUAL CODE,
    not prose that legitimately discusses .map()/fan-out in the module docstring's
    design explanation. Generalizes test_run_ablation_local.py's helper of the same
    name to modules without a `#!` shebang line (modal_ablation.py has none)."""
    first = src.find('"""')
    if first == -1:
        return src
    second = src.find('"""', first + 3)
    if second == -1:
        return src
    return src[second + 3:]


def test_source_never_maps_across_substrates_only_across_configs():
    """D-08: .map()/.starmap() may fan out across INDEPENDENT CONFIGS (the whole
    point of this file) but must never appear inside a per-substrate eval loop --
    that loop lives unchanged inside run_gflownet.py/evaluate_matrix, never here.
    Confirms this file's only ACTUAL fan-out call sites are `run_one_config.map(...)`
    (a real Python call, not prose mentioning ``.map()`` inside docstrings)."""
    import re

    src = _code_only(MODAL_ABLATION_PATH.read_text())
    # Match an actual call expression: `<identifier>.map(` or `<identifier>.starmap(`,
    # not backtick-quoted/double-backtick prose like `` `.map()` `` or ``.map()``.
    call_pattern = re.compile(r"(\w+)\.(map|starmap)\(")
    map_calls = [
        (line, m.group(1)) for line in src.splitlines()
        for m in [call_pattern.search(line)] if m
    ]
    assert map_calls, "expected at least one .map()/.starmap() call (the config fan-out)"
    for line, receiver in map_calls:
        assert receiver == "run_one_config", (
            f"unexpected .map()/.starmap() call site (must only fan out over "
            f"run_one_config, got receiver={receiver!r}): {line}"
        )


def test_source_reuses_shared_ablation_plan_module_not_reimplemented():
    """The orchestrator must import its planning/selection/verdict logic from
    grail_metabolism.ablation_plan, never reimplement plan_configs/select_beta_prime/
    aggregate_and_verdict inline."""
    src = MODAL_ABLATION_PATH.read_text()
    assert "from grail_metabolism.ablation_plan import" in src
    assert "plan_configs" in src
    assert "select_beta_prime" in src
    assert "aggregate_and_verdict" in src
    # Must NOT redefine these as local functions (would silently fork the science).
    assert "def plan_configs(" not in src
    assert "def select_beta_prime(" not in src
    assert "def aggregate_and_verdict(" not in src


def test_source_reuses_modal_m2_image_and_volumes_not_reinvented():
    """Reuse contract: the image/Volumes/data-symlink constants come from
    scripts/modal_m2.py, not a re-declared image build."""
    src = MODAL_ABLATION_PATH.read_text()
    assert "from scripts.modal_m2 import" in src
    assert "modal.Image.debian_slim" not in src, "must not re-declare the base image"


def test_source_uses_config_keyed_idempotent_out_paths():
    """Mirrors modal_m2.py::run_m2's `if os.path.exists(out): skip` contract, keyed
    by config tag rather than seed."""
    src = MODAL_ABLATION_PATH.read_text()
    assert "os.path.exists(out)" in src
    assert "already complete" in src
    assert "art_vol.commit()" in src


def test_source_uses_run_gflownet_ablation_mode_cli_unchanged():
    """The per-config unit must invoke run_gflownet.py's existing --ablation-mode
    CLI (never reimplement the training/eval science inline)."""
    src = MODAL_ABLATION_PATH.read_text()
    assert '"scripts/run_gflownet.py"' in src
    assert '"--ablation-mode"' in src
    assert '"--beta-prime"' in src
    assert '"--resume-ckpt"' in src
    assert '"--resume-eval-ckpt"' in src


def test_source_has_a_prewarm_barrier_before_any_parallel_wave():
    """The env cache must be built once (prewarm) BEFORE the first .map() fan-out,
    not concurrently with it -- guards against the cache-build race the module
    docstring's safety argument depends on."""
    src = MODAL_ABLATION_PATH.read_text()
    prewarm_pos = src.find("prewarm.remote()")
    first_map_pos = src.find("_run_wave(plan[\"sweep\"]")
    assert prewarm_pos != -1, "expected an explicit prewarm barrier call"
    assert first_map_pos != -1
    assert prewarm_pos < first_map_pos, "prewarm must run BEFORE the first parallel wave"


@pytest.mark.skipif(not modal_available, reason="modal package not installed in this environment")
def test_module_imports_without_live_modal_auth():
    """App/Volume construction (modal.App(...), modal.Volume.from_name(...)) must not
    require a live authenticated session at import time -- this is what lets
    `make test` stay green in CI without Modal credentials configured, matching the
    already-shipped modal_m2.py's same property."""
    import importlib

    mod = importlib.import_module("scripts.modal_ablation")
    assert hasattr(mod, "app")
    assert hasattr(mod, "run_one_config")
    assert hasattr(mod, "run_ablation")
    assert hasattr(mod, "orchestrate_ablation")


def test_orchestration_moved_cloud_side_entrypoint_only_spawns():
    """Regression guard for the detach bug: a `modal run --detach` of a
    `@app.local_entrypoint` dies when the CLI disconnects, because .remote()/.map()
    calls issued FROM the local entrypoint run in the local caller's process. The fix
    moves the whole orchestration (prewarm -> sweep -> select beta_prime -> val runs ->
    test touch -> aggregate_and_verdict -> write verdict) into a cloud-side
    `@app.function` (`orchestrate_ablation`), and shrinks `run_ablation` down to a
    `.spawn()` + immediate return -- so the entrypoint survives only long enough to
    hand off, and the actual multi-hour work runs server-side, immune to CLI exit."""
    src = _code_only(MODAL_ABLATION_PATH.read_text())

    # `orchestrate_ablation` must be a real Modal function (cloud-side), not a plain
    # helper -- the whole point is that it executes remotely, not in the local CLI.
    orchestrate_pos = src.find("def orchestrate_ablation(")
    assert orchestrate_pos != -1, "expected a cloud-side orchestrate_ablation function"
    preceding = src[:orchestrate_pos]
    last_decorator_start = preceding.rfind("@app.function(")
    assert last_decorator_start != -1
    # No blank-line-free code between the last @app.function( and orchestrate_ablation's
    # def would be overly strict given kwargs span multiple lines; instead confirm the
    # decorator block immediately precedes the def with no other `def ` in between.
    between = preceding[last_decorator_start:]
    assert "def " not in between, "orchestrate_ablation must be directly decorated by @app.function"

    # The heavy orchestration body (prewarm barrier + wave fan-outs + verdict) must
    # live INSIDE orchestrate_ablation, not inside run_ablation.
    entrypoint_pos = src.find("def run_ablation(")
    assert entrypoint_pos != -1
    smoke_pos = src.find("def smoke(")
    assert smoke_pos != -1 and smoke_pos > entrypoint_pos
    entrypoint_body = src[entrypoint_pos:smoke_pos]

    assert "prewarm.remote()" not in entrypoint_body, "prewarm must run inside orchestrate_ablation, not the entrypoint"
    assert "_run_wave(" not in entrypoint_body, "wave fan-out must run inside orchestrate_ablation, not the entrypoint"
    assert "aggregate_and_verdict(" not in entrypoint_body, "verdict aggregation must run inside orchestrate_ablation"

    # The entrypoint's ONLY interaction with the heavy work must be a .spawn() call
    # (fire-and-forget) -- never .remote()/.get() (which would block the CLI process
    # on the full multi-hour run, reintroducing the exact detach bug being fixed).
    # Strip the function's own docstring first: it legitimately discusses ``.get()``
    # in prose (contrasting .spawn() with the blocking alternative it replaces).
    doc_start = entrypoint_body.find('"""')
    doc_end = entrypoint_body.find('"""', doc_start + 3) + 3 if doc_start != -1 else -1
    entrypoint_code_only = entrypoint_body[doc_end:] if doc_end != -1 else entrypoint_body

    assert "orchestrate_ablation.spawn(" in entrypoint_body
    assert "orchestrate_ablation.remote(" not in entrypoint_body
    assert ".get()" not in entrypoint_code_only


def test_smoke_also_spawns_cloud_side_not_blocking():
    """The smoke entrypoint must exercise the SAME cloud-side detach path as the real
    run (spawn + return), so a detached smoke run actually proves CLI-independence
    rather than blocking on a local .remote()/.map() call."""
    src = _code_only(MODAL_ABLATION_PATH.read_text())
    smoke_pos = src.find("def smoke(")
    assert smoke_pos != -1
    smoke_body = src[smoke_pos:]
    assert "orchestrate_ablation.spawn(" in smoke_body
    assert "orchestrate_ablation.remote(" not in smoke_body


def test_smoke_top_k_reaches_k_max_so_ablation_aucs_are_populated():
    """Regression guard: a first detached smoke run crashed inside
    aggregate_and_verdict with `KeyError: 'ablation01_union_at_k_auc'`.
    run_gflownet.py only populates that metrics key when at least one eval
    substrate's (weaker) ablation01/02 arm reaches k_max=50 DISTINCT candidates; at
    top_k=20 the reranker pool is capped below k_max, so the weaker single-terminal/
    ensemble arms can structurally never get there. The smoke's top_k must be >= the
    hardcoded k_max=50 in run_gflownet.py's eval ks=(5,10,15,20,30,50) so the
    aggregate_and_verdict call (which reads this key unconditionally, unlike
    run_gflownet.py's own defensive `in metrics` guard) doesn't KeyError."""
    src = _code_only(MODAL_ABLATION_PATH.read_text())
    smoke_pos = src.find("def smoke(")
    assert smoke_pos != -1
    smoke_body = src[smoke_pos:]
    spawn_pos = smoke_body.find("orchestrate_ablation.spawn(")
    assert spawn_pos != -1
    call_end = smoke_body.find(")", spawn_pos)
    spawn_call = smoke_body[spawn_pos:call_end]
    assert "top_k=50" in spawn_call, (
        "smoke's top_k must be >= run_gflownet.py's hardcoded k_max=50 so the "
        "weaker ablation01/02 arms can reach 50 distinct candidates -- otherwise "
        "aggregate_and_verdict KeyErrors on a missing *_union_at_k_auc key"
    )


def test_smoke_and_full_run_verdict_tags_are_isolated_by_parameter():
    """The verdict report filename must be parameterized (verdict_tag) so a smoke run
    can write its own tiny verdict without colliding with (or false-skipping) the full
    run's verdict_report.json."""
    src = MODAL_ABLATION_PATH.read_text()
    assert "verdict_tag" in src
    assert 'verdict_tag: str = "verdict_report"' in src
    assert 'verdict_tag: str = "smoke_verdict_report"' in src
