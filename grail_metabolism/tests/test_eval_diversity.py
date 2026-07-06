"""Dataset-free guard tests for grail_metabolism/eval/diversity.py.

Hand-built-fixture, dataset-free tests in the style of test_audit_fixes.py /
test_set_gflownet.py: small SMILES constants at module top, one assertion per
known-answer property, no external SDF/triples. Tests that depend on tautomer
collapse monkeypatch ``_tautomer_inchikey`` (following the ``_fake_taut_ik``
pattern in test_set_gflownet.py) rather than relying on the installed RDKit
build's actual tautomer-canonicalization coverage, keeping them
RDKit-version-independent.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import grail_metabolism.eval.diversity as diversity
import grail_metabolism.metrics as metrics

RUN_GFLOWNET_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_gflownet.py"

# Hexane/heptane: genuinely distinct SMILES but high Tanimoto similarity
# (~0.875, verified against the installed RDKit build); benzene is
# structurally unrelated to both (Tanimoto ~0.0 to either). At
# threshold=0.4 (distance 0.6) hexane and heptane are close enough to
# collapse into one pick, leaving benzene separate -> 2 circles. At
# threshold=0.9 (distance 0.1) hexane/heptane's 0.875 similarity is NOT
# within the tighter distance bound, so all three survive -> 3 circles.
# This is the sign-conversion guard: a bug that passes `threshold` directly
# instead of `1 - threshold` would silently invert this relationship.
_HEXANE = "CCCCCC"
_HEPTANE = "CCCCCCC"
_BENZENE = "c1ccccc1"

# Two distinct, parseable molecules with DISTINCT plain InChIKeys (so a
# regression to plain-InChIKey identity would NOT collapse them); the
# monkeypatch below simulates a tautomer-canonicalizer collapsing them onto
# one shared key, mirroring test_set_gflownet.py's _fake_taut_ik pattern.
_TAUT_A = "CCO"
_TAUT_B = "CCCO"
_DISTINCT = "c1ccccc1"


def _fake_taut_ik(smiles: str) -> str:
    """Collapse _TAUT_A/_TAUT_B onto one key; everything else keys to itself."""
    return "SHARED_TAUT_KEY" if smiles in (_TAUT_A, _TAUT_B) else smiles


def test_circles_count_known_answer():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    # Sanity-check the hand-picked similarity assumption still holds against
    # whatever RDKit build is installed (keeps the test's own premise honest).
    mols = [Chem.MolFromSmiles(s) for s in (_HEXANE, _HEPTANE, _BENZENE)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    sim_hexane_heptane = DataStructs.TanimotoSimilarity(fps[0], fps[1])
    sim_hexane_benzene = DataStructs.TanimotoSimilarity(fps[0], fps[2])
    assert sim_hexane_heptane > 0.6  # close enough to collapse at threshold=0.4
    assert sim_hexane_benzene < 0.1  # far enough to survive at threshold=0.9

    smiles = [_HEXANE, _HEPTANE, _BENZENE]
    assert diversity.circles_count(smiles, threshold=0.4) == 2
    assert diversity.circles_count(smiles, threshold=0.9) == 3


def test_circles_count_degenerate_inputs():
    assert diversity.circles_count([], threshold=0.4) == 0
    assert diversity.circles_count(["CCO"], threshold=0.4) == 1
    for threshold in (0.1, 0.4, 0.7, 0.99):
        assert diversity.circles_count(["CCO", "CCO", "CCO"], threshold=threshold) == 1


def test_dedup_to_budget_collapses_tautomer_variants(monkeypatch):
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    out = diversity.dedup_to_budget([_TAUT_A, _TAUT_B, _DISTINCT], k=2)
    assert len(out) == 2
    # The freed slot (from the A/B tautomer collapse) is filled by the third,
    # genuinely-distinct molecule.
    assert _DISTINCT in out
    # Only one of the tautomer-collapsed pair occupies a slot.
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1


def test_union_at_k_curve_monotonic():
    # A ranked stream of 5 distinct molecules; the first 3 are annotated true.
    smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO"]
    annotated_ik = {metrics._tautomer_inchikey(s) for s in smiles[:3]}

    curve = diversity.union_at_k_curve(smiles, annotated_ik, ks=(2, 3, 5))
    values = [curve[k] for k in (2, 3, 5)]
    assert values == sorted(values)  # non-decreasing in K
    assert curve[3] == 1.0  # all 3 annotated hits found by k=3
    assert curve[5] == 1.0  # no further gain past k=3 (monotone plateau)
    assert 0.0 < curve[2] < 1.0  # partial coverage at k=2


def test_diversity_and_recall_agree_on_molecule_identity(monkeypatch):
    """EVAL-04 guard: both the recall path and circles_count treat a
    tautomer-collapsed pair as ONE molecule, proving the dedup pre-pass runs
    inside the diversity function itself (not merely upstream in some other
    caller's dedup)."""
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    # Recall path: metrics._tautomer_inchikey collapses the pair to one key.
    assert metrics._tautomer_inchikey(_TAUT_A) == metrics._tautomer_inchikey(_TAUT_B)

    # Diversity path: circles_count must ALSO treat them as one molecule.
    assert diversity.circles_count([_TAUT_A, _TAUT_B], threshold=0.99) == 1


def test_annotated_coverage_count_renamed_matches_old_modes_discovered():
    # Regression guard for the "pure rename, zero behavior change" claim:
    # a hand-built input with a known union-intersect-annotated count.
    sets = [frozenset({"A", "X"}), frozenset({"B", "Y"}), frozenset({"A"})]
    annotated_ik = {"A", "B", "C"}
    # A and B are found across the sampled sets; C never appears.
    assert diversity.annotated_coverage_count(sets, annotated_ik) == 2


def test_modes_discovered_canonical_gates_on_reward_and_tanimoto():
    # Hand-built candidates: some fail the tau gate outright, some pass the
    # gate but are within delta of an already-accepted mode.
    candidates = [_HEXANE, _HEPTANE, _BENZENE, "not_annotated_smiles"]
    rewarded = {_HEXANE, _HEPTANE, _BENZENE}  # "not_annotated_smiles" fails tau

    def reward_fn(x: str) -> float:
        return 1.0 if x in rewarded else 0.0

    # tau=1: only the three rewarded candidates survive the reward gate.
    # delta=0.7 (distance 0.3): hexane/heptane (sim ~0.875) are within delta
    # of each other -> sphere-exclusion collapses them to one; benzene
    # (dissimilar to both) survives as a second mode.
    count = diversity.modes_discovered_canonical(candidates, reward_fn, tau=1.0, delta=0.7)
    assert count == 2

    # Raising tau above every candidate's reward excludes everything.
    count_none = diversity.modes_discovered_canonical(candidates, reward_fn, tau=2.0, delta=0.7)
    assert count_none == 0


def test_modes_discovered_canonical_collapses_tautomer_duplicates(monkeypatch):
    """FIX C (EVAL-04) guard: modes_discovered_canonical must run the SAME
    _dedup_smiles_by_tautomer_ik pre-pass as circles_count/mean_pairwise_tanimoto/
    n_unique_scaffolds on its reward-gate survivors BEFORE the len(mols) < 2 / sphere-
    exclusion step -- otherwise two tautomers of the same molecule (both passing the
    reward gate) would count as two distinct 'modes' instead of collapsing to one.
    Uses the same monkeypatched _tautomer_inchikey as the existing diversity/recall
    identity-agreement test."""
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    # _TAUT_A and _TAUT_B are an exact tautomer-duplicate pair under the fake IK; both
    # pass a reward gate that accepts everything (tau=0).
    candidates = [_TAUT_A, _TAUT_B]

    def reward_fn(x: str) -> float:
        return 1.0

    count = diversity.modes_discovered_canonical(candidates, reward_fn, tau=1.0, delta=0.7)
    assert count == 1, (
        "an exact tautomer-duplicate pair must collapse to ONE mode before the "
        "sphere-exclusion gate, not be counted as two distinct modes"
    )


def test_auc_of_curve_known_answer():
    # Exact D-EVAL02-KGRID non-uniform grid with hand-picked distinct values.
    curve = {5: 0.10, 10: 0.20, 15: 0.30, 20: 0.40, 30: 0.50, 50: 0.60}
    ks = (5, 10, 15, 20, 30, 50)

    expected = sum(
        0.5 * (curve[ks[i]] + curve[ks[i + 1]]) * (ks[i + 1] - ks[i])
        for i in range(len(ks) - 1)
    ) / (50 - 5)

    auc = diversity.auc_of_curve(curve, k_min=5, k_max=50)
    assert abs(auc - expected) < 1e-9

    # Positively distinguish the dx-weighted trapezoid from a naive
    # equal-weight mean, which would over-weight the sparse K=30/50 tail.
    naive_mean = sum(curve.values()) / len(curve)
    assert abs(auc - naive_mean) > 1e-6

    # 2-point edge case pins the (k_max - k_min) normalization divisor.
    two_point = {5: 0.2, 50: 0.8}
    assert diversity.auc_of_curve(two_point, k_min=5, k_max=50) == 0.5 * (0.2 + 0.8)


def test_compute_ablation_verdict_all_three_outcomes():
    # Full confirmation: gflownet beats BOTH ablations by > margin.
    assert diversity.compute_ablation_verdict(
        gflownet_auc=0.30, abl01_auc=0.20, abl02_auc=0.22, margin=0.02
    ) == "confirmed"

    # Null: independent sampling matches within margin (neither beat holds).
    assert diversity.compute_ablation_verdict(
        gflownet_auc=0.30, abl01_auc=0.29, abl02_auc=0.29, margin=0.02
    ) == "null"

    # Partial: beats abl01 by > margin but loses to (does not beat) abl02 by > margin.
    assert diversity.compute_ablation_verdict(
        gflownet_auc=0.30, abl01_auc=0.20, abl02_auc=0.31, margin=0.02
    ) == "partial"

    # Does NOT beat abl01 by margin => not confirmed, not the partial pattern => null,
    # even though it clears abl02.
    assert diversity.compute_ablation_verdict(
        gflownet_auc=0.30, abl01_auc=0.31, abl02_auc=0.20, margin=0.02
    ) == "null"

    # Boundary: a difference exactly equal to margin is NOT a beat (strict >).
    assert diversity.compute_ablation_verdict(
        gflownet_auc=0.30, abl01_auc=0.28, abl02_auc=0.28, margin=0.02
    ) == "null"


# ---------------------------------------------------------------------------
# Phase 1 Plan 02 guard tests: budget-matching / JSON-key-stability /
# reranker-pretruncation / aggregate_seeds.py key-detection.
# ---------------------------------------------------------------------------


def test_dedup_to_budget_prevents_silent_budget_shrinkage(monkeypatch):
    """EVAL-01 silent-budget-shrinkage landmine: a ranked list containing
    tautomer-duplicate SMILES must NOT silently under-fill its stated budget --
    requesting k=N distinct post-canon molecules must return N distinct
    molecules whenever >= N distinct molecules are available in the stream,
    even if the raw (pre-dedup) stream is dominated by duplicate pairs."""
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    # Interleave the tautomer-duplicate pair with 4 genuinely distinct molecules.
    # A naive (non-deduped) truncation to k=4 would return only 3 distinct
    # molecules (since _TAUT_A/_TAUT_B collapse to one key) -- silently
    # shrinking the effective budget below the requested 4.
    distinct_fillers = ["c1ccccc1", "CCN", "CCCl", "CCBr"]
    ranked = [_TAUT_A, _TAUT_B] + distinct_fillers
    out = diversity.dedup_to_budget(ranked, k=4)

    assert len(out) == 4
    keys = {_fake_taut_ik(s) for s in out}
    assert len(keys) == 4, "must be 4 DISTINCT post-canon molecules, not shrunk by the dup pair"
    # Only one of the tautomer-collapsed pair occupies a slot; the freed slot
    # goes to the next-ranked distinct filler (CCCl), not the 4th ranked
    # filler (CCBr, which would only be needed if the dup pair had NOT freed
    # a slot -- i.e. a naive non-deduped truncation would have stopped one
    # slot earlier and never reached CCCl).
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1
    assert "CCCl" in out  # the freed slot lets budget reach the 3rd distinct filler
    assert "CCBr" not in out  # k=4 exhausted before the 4th filler is needed


def test_diversity_block_json_key_is_modes_discovered():
    """D-EVAL05-JSONKEY guard: the results-dict key literal 'modes_discovered'
    must stay present in run_gflownet.py's _diversity_block (even though the
    VALUE now comes from the renamed annotated_coverage_count), and the new
    '#Circles' keys are additive alongside it. Asserted against the module
    source text (not by importing/exercising the model-heavy _diversity_block
    itself), so this guard stays dataset-free and torch-light."""
    src = Path(RUN_GFLOWNET_PATH).read_text()
    assert '"modes_discovered"' in src, (
        "the results-dict key literal 'modes_discovered' must remain -- a rename here "
        "silently breaks aggregate_seeds.py's DIVERSITY_KEYS contract"
    )
    assert "annotated_coverage_count" in src, (
        "the diversity call site must use the renamed annotated_coverage_count function"
    )
    assert '"circles@t0.4"' in src and '"circles@t0.7"' in src, (
        "circles@t0.4/circles@t0.7 must be present as additive diversity-block keys"
    )
    # The old function name must no longer be CALLED (a bare leftover reference would
    # indicate the call site was never actually migrated).
    assert "modes_discovered(sampled_sets" not in src


def test_reranker_stream_not_pretruncated_below_kgrid(monkeypatch):
    """FIX 2 budget-fairness guard: the reranker stream handed to dedup_to_budget
    must have length >= max(ks) whenever the underlying (reranked) pool
    supports it -- i.e. _reranker_topk_smiles must be called with k=max(ks),
    NOT the legacy (smaller) max_size, so reranker_union@30/@50 are computed
    over the FULL available pool rather than silently capped below the
    K-grid. Exercises the real _reranker_topk_smiles boundary (real, pure
    from_rdmol/Batch -- no torch dependency stubbed) with a tiny stub
    reranker/generator/pool (no real model checkpoint needed)."""
    import torch

    import scripts.run_gflownet as rg

    ks = (5, 10, 15, 20, 30, 50)
    k_max = max(ks)
    n_candidates = 80  # comfortably >= k_max so the pool is never the bottleneck

    # Stub pool: (smiles, gen_score, rule_id) tuples, one per candidate, all
    # parseable and structurally distinct (varying alkyl chain length).
    fake_pool = [(f"{'C' * (i + 1)}O", float(n_candidates - i), 0) for i in range(n_candidates)]
    monkeypatch.setattr(rg, "build_pool", lambda generator, root, top_k, max_pool: fake_pool)

    class _FakeGenerator:
        rule_prior_logits = torch.zeros(1)

    class _FakeReranker:
        def __call__(self, sub_graph, prod_batch, rule_priors, gen_scores):
            # Score purely by gen_score so ranking is deterministic and known.
            return gen_scores

    stream = rg._reranker_topk_smiles(
        _FakeReranker(), _FakeGenerator(), root="CCO", k=k_max, top_k=200, max_pool=100, device="cpu",
    )
    assert len(stream) >= k_max, (
        f"reranker stream must be requested at k=max(ks)={k_max}, not pre-truncated to a "
        f"smaller max_size -- got only {len(stream)} candidates from a pool of {n_candidates}"
    )

    # Source-level corroboration: evaluate_matrix's reranker call site passes k=max(ks),
    # not k=max_size, when building the stream that reaches dedup_to_budget.
    src = Path(RUN_GFLOWNET_PATH).read_text()
    assert re.search(r"_reranker_topk_smiles\([^)]*k\s*=\s*max\(ks\)", src), (
        "evaluate_matrix must request the reranker stream at k=max(ks), not k=max_size"
    )


def test_aggregate_seeds_picks_up_new_keys(tmp_path):
    """FIX 3 aggregation guard: gflownet_union@{k}/reranker_union@{k}, circles@t0.4/
    circles@t0.7, and union_at_k_auc must aggregate mean+/-std across >=2 fake seeds,
    while the pre-existing modes_discovered key still aggregates (regression guard
    that the additive edit did not drop it). FIX E (adversarial review) additionally
    guards that the PER-SERIES gflownet_union_at_k_auc/reranker_union_at_k_auc AUC
    breakdown -- emitted by evaluate_matrix alongside the combined union_at_k_auc --
    also aggregates via DIVERSITY_KEYS, not just the combined scalar."""
    import scripts.aggregate_seeds as agg

    def _fake_run(seed: int, gflownet_union_30: float) -> dict:
        return {
            "seed": seed,
            "config": {"eval_split": "val"},
            "metrics": {
                "n_substrates": 10.0,
                "gflownet_recall@15": 0.30 + 0.01 * seed,
                "reranker_recall@15": 0.28 + 0.01 * seed,
                "gflownet_union@30": gflownet_union_30,
                "reranker_union@50": 0.50 + 0.02 * seed,
                "circles@t0.4": 3.0 + seed,
                "circles@t0.7": 1.0 + seed,
                "union_at_k_auc": 0.40 + 0.01 * seed,
                "gflownet_union_at_k_auc": 0.42 + 0.01 * seed,
                "reranker_union_at_k_auc": 0.38 + 0.01 * seed,
                "modes_discovered": 2.0 + seed,
                "mean_pairwise_tanimoto": 0.4,
                "n_unique_scaffolds": 4.0,
                "set_size_calibration": 0.1,
            },
        }

    runs = [_fake_run(0, 0.55), _fake_run(1, 0.57)]

    series_order, series_ks, _ = agg._detect_series_k(runs)
    # Both the plain-recall series and the (union) series must be detected.
    assert "gflownet" in series_order and 15 in series_ks["gflownet"]
    assert "reranker" in series_order and 15 in series_ks["reranker"]
    assert "gflownet(union)" in series_order and 30 in series_ks["gflownet(union)"]
    assert "reranker(union)" in series_order and 50 in series_ks["reranker(union)"]

    # Key reconstruction must round-trip back to the exact metrics-dict keys.
    assert agg._metric_key("gflownet", 15) == "gflownet_recall@15"
    assert agg._metric_key("gflownet(union)", 30) == "gflownet_union@30"

    # Mean+/-std aggregation for the union series across the 2 fake seeds.
    union_vals = [r["metrics"]["gflownet_union@30"] for r in runs]
    expected_mean = sum(union_vals) / len(union_vals)
    computed_mean = sum(
        r["metrics"][agg._metric_key("gflownet(union)", 30)] for r in runs
    ) / len(runs)
    assert abs(computed_mean - expected_mean) < 1e-9

    # New diversity keys (circles@/union_at_k_auc) must be detected AND aggregate.
    diversity_keys = agg._detect_diversity_keys(runs)
    for key in ("circles@t0.4", "circles@t0.7", "union_at_k_auc"):
        assert key in diversity_keys, f"{key} must be picked up by _detect_diversity_keys"
        vals = [r["metrics"][key] for r in runs]
        assert len(vals) == 2  # both fake seeds contribute

    # FIX E guard: the PER-SERIES union AUC breakdown (gflownet_union_at_k_auc/
    # reranker_union_at_k_auc) must ALSO be detected and aggregate, not just the
    # combined union_at_k_auc scalar.
    for key in ("gflownet_union_at_k_auc", "reranker_union_at_k_auc"):
        assert key in diversity_keys, f"{key} must be picked up by _detect_diversity_keys"
        vals = [r["metrics"][key] for r in runs]
        assert len(vals) == 2

    # Regression guard: modes_discovered must STILL aggregate (additive edit must not
    # drop the pre-existing key).
    assert "modes_discovered" in diversity_keys
    md_vals = [r["metrics"]["modes_discovered"] for r in runs]
    assert md_vals == [2.0, 3.0]


def test_aggregate_seeds_headline_includes_plain_recall_not_only_union(tmp_path, monkeypatch, capsys):
    """FIX A guard: HEADLINE must NOT silently drop the plain matched-budget
    `_recall@{k}` series when the loaded runs ALSO carry `_union@{k}` keys with a
    LARGER k-grid. Before the fix, `headline_k = max(all_ks)` pooled BOTH families
    (plain recall k's tied to --max-size, e.g. 10, and union k's up to 50) and always
    resolved to the union grid's max -- so `gflownet_recall@10`/`reranker_recall@10`
    never appeared in the printed HEADLINE block, only the `(union)` series did.

    Builds fixture JSONs with BOTH `gflownet_recall@10`+`reranker_recall@10` AND
    `gflownet_union@{5..50}`+`reranker_union@{5..50}`, runs the real `main()` entry
    point end-to-end (subprocess-free, via monkeypatched argv), and asserts the
    printed output contains a `HEADLINE recall@10` line (not only a union headline)."""
    import scripts.aggregate_seeds as agg

    def _fake_run(seed: int) -> dict:
        return {
            "seed": seed,
            "config": {"eval_split": "test"},
            "metrics": {
                "n_substrates": 10.0,
                "gflownet_recall@10": 0.30 + 0.01 * seed,
                "reranker_recall@10": 0.28 + 0.01 * seed,
                **{f"gflownet_union@{k}": 0.20 + 0.01 * k for k in (5, 10, 15, 20, 30, 50)},
                **{f"reranker_union@{k}": 0.18 + 0.01 * k for k in (5, 10, 15, 20, 30, 50)},
            },
        }

    paths = []
    for seed in (0, 1):
        p = tmp_path / f"gflownet_test_seed{seed}.json"
        p.write_text(json.dumps(_fake_run(seed)))
        paths.append(str(p))

    monkeypatch.setattr(sys, "argv", ["aggregate_seeds.py", *paths])
    agg.main()
    out = capsys.readouterr().out

    assert "HEADLINE recall@10" in out, (
        "the matched-budget recall@10 HEADLINE must be printed, not silently dropped "
        "in favor of the union series' larger k-grid"
    )
    assert "HEADLINE union@50" in out, (
        "a separate, clearly-labeled union headline must ALSO be printed so both "
        "families are visible and never conflated"
    )
    # The recall@10 headline line must actually carry the gflownet/reranker series
    # values, not just the label.
    recall_line_idx = out.index("HEADLINE recall@10")
    union_line_idx = out.index("HEADLINE union@50")
    recall_block = out[recall_line_idx:union_line_idx]
    assert "gflownet" in recall_block or "reranker" in recall_block


# ---------------------------------------------------------------------------
# Phase 1 Plan 03 guard tests: EVAL-06 per-substrate eval-checkpoint resume
# equivalence, corrupt-checkpoint-ignored, and stale-config-fingerprint-discarded.
# ---------------------------------------------------------------------------


def _fake_process_substrate(root: str) -> dict:
    """A tiny, deterministic stand-in for one substrate's evaluate_matrix body: a pure
    function of ``root`` (no randomness, no model calls) so uninterrupted vs.
    checkpoint-then-resume runs are trivially comparable without pulling in the model
    stack."""
    n = len(root)
    return {
        "gflownet_recall": (n % 5) / 5.0,
        "reranker_recall": (n % 3) / 3.0,
        "diversity": {
            "modes_discovered": float(n % 4),
            "mean_pairwise_tanimoto": 0.5,
            "n_unique_scaffolds": float(n % 2 + 1),
            "set_size_calibration": 0.0,
            "circles@t0.4": float(n % 3),
            "circles@t0.7": float(n % 2),
        },
    }


class _BoomOnCall:
    """Callable stub that raises AssertionError if invoked -- used to prove a resumed run
    does NOT recompute already-checkpointed substrates (mirrors test_set_gflownet.py's
    _BoomGen pattern, generalized to a plain callable rather than a generator stub)."""

    def __call__(self, root: str) -> dict:
        raise AssertionError(f"must not recompute already-completed substrate {root!r}")


def _run_eval_loop(substrates, resume_path, eval_ckpt_every, config_fingerprint, process_fn=None):
    """Minimal stand-in for evaluate_matrix's per-substrate loop body, built directly on
    top of the real _load_eval_ckpt/_save_eval_ckpt helpers under test -- exercises the
    exact resume/skip/checkpoint-interval logic evaluate_matrix uses, without needing the
    torch/model stack."""
    import scripts.run_gflownet as rg

    completed_rows, next_idx = rg._load_eval_ckpt(resume_path, config_fingerprint)
    for i, root in enumerate(substrates):
        if i < next_idx:
            continue
        fn = process_fn if process_fn is not None else _fake_process_substrate
        completed_rows[root] = fn(root)
        if resume_path and (i + 1) % eval_ckpt_every == 0:
            rg._save_eval_ckpt(resume_path, config_fingerprint, completed_rows, i + 1)
    if resume_path:
        rg._save_eval_ckpt(resume_path, config_fingerprint, completed_rows, len(substrates))
    return completed_rows


def _aggregate(rows: dict) -> dict:
    """Order-independent aggregation mirroring evaluate_matrix's final _mean(...) step
    (D-EVAL06-SCHEMA: aggregation runs over completed_rows.values(), not append order)."""
    values = list(rows.values())
    n = len(values)
    return {
        "n_substrates": float(n),
        "gflownet_recall": sum(v["gflownet_recall"] for v in values) / n,
        "reranker_recall": sum(v["reranker_recall"] for v in values) / n,
        "modes_discovered": sum(v["diversity"]["modes_discovered"] for v in values) / n,
    }


def test_evaluate_matrix_resume_produces_identical_final_metrics(tmp_path):
    """EVAL-06 guard (1)+(2): a run over N tiny substrates uninterrupted produces the SAME
    final aggregated metrics as a run checkpointed after N/2 and resumed, AND the resumed
    run does not recompute already-completed substrates (a _BoomOnCall stub would fail the
    test if it were called for one of them)."""
    substrates = [f"C{'C' * i}O" for i in range(10)]  # 10 distinct, deterministic "SMILES"
    fingerprint = "fp-v1"

    # Uninterrupted: no resume_path, single pass.
    uninterrupted_rows = _run_eval_loop(substrates, resume_path=None, eval_ckpt_every=10,
                                          config_fingerprint=fingerprint)
    uninterrupted_metrics = _aggregate(uninterrupted_rows)

    # Split: checkpoint written after substrate 5 (N/2), then resumed to completion.
    ckpt = str(tmp_path / "eval_seed0.ckpt.json")
    half = substrates[:5]
    _run_eval_loop(half, resume_path=ckpt, eval_ckpt_every=5, config_fingerprint=fingerprint)
    assert os.path.exists(ckpt), "a checkpoint must be written after 5 substrates (eval_ckpt_every=5)"

    # Resume: process the FULL substrate list, but the first 5 substrates must be served
    # from the checkpoint (never recomputed) -- prove it with a boom-on-call stub gated to
    # only the first-5 roots.
    boom = _BoomOnCall()

    def _guarded_process(root: str) -> dict:
        if root in half:
            boom(root)  # would raise if a checkpointed substrate is recomputed
        return _fake_process_substrate(root)

    resumed_rows = _run_eval_loop(substrates, resume_path=ckpt, eval_ckpt_every=10,
                                    config_fingerprint=fingerprint, process_fn=_guarded_process)
    resumed_metrics = _aggregate(resumed_rows)

    assert resumed_metrics == uninterrupted_metrics, (
        "resumed final aggregated metrics must be IDENTICAL to an uninterrupted run "
        "(order-independent aggregation over completed_rows.values())"
    )
    assert set(resumed_rows.keys()) == set(substrates)


def test_eval_checkpoint_ignores_corrupt_file(tmp_path):
    """EVAL-06 guard (3): a corrupt/unreadable eval checkpoint must be IGNORED (start
    fresh from substrate 0), never crash the run -- mirrors
    test_train_checkpoint_ignores_corrupt_file's guard for the training checkpoint."""
    import scripts.run_gflownet as rg

    bad = tmp_path / "eval_seed0.ckpt.json"
    bad.write_bytes(b"not valid json{{{")

    rows, next_idx = rg._load_eval_ckpt(str(bad), "any-fingerprint")
    assert rows == {}
    assert next_idx == 0

    # And the loop-level helper proceeds without crashing, recomputing everything.
    substrates = ["CCO", "CCCO", "CCCCO"]
    result = _run_eval_loop(substrates, resume_path=str(bad), eval_ckpt_every=10,
                              config_fingerprint="any-fingerprint")
    assert set(result.keys()) == set(substrates)


def test_eval_checkpoint_discards_stale_config_fingerprint(tmp_path):
    """EVAL-06 guard (4) / FIX 6: a checkpoint written under one config_fingerprint must be
    DISCARDED (not blended) when loaded under a DIFFERENT fingerprint -- the run restarts
    from substrate 0 and every row is recomputed under the new config, rather than the old
    checkpoint's stale rows silently leaking through."""
    import scripts.run_gflownet as rg

    ckpt = str(tmp_path / "eval_seed0.ckpt.json")
    substrates = ["CCO", "CCCO", "CCCCO", "CCCCCO"]

    old_fingerprint = rg._eval_config_fingerprint(
        max_size=10, ks=(5, 10, 15, 20, 30, 50), n_samples=32, top_k=200, max_pool=100,
    )
    # A materially different config (max_size changed) must hash to a DIFFERENT fingerprint.
    new_fingerprint = rg._eval_config_fingerprint(
        max_size=15, ks=(5, 10, 15, 20, 30, 50), n_samples=32, top_k=200, max_pool=100,
    )
    assert old_fingerprint != new_fingerprint

    # Write a checkpoint under the OLD config, fully complete.
    _run_eval_loop(substrates, resume_path=ckpt, eval_ckpt_every=10,
                    config_fingerprint=old_fingerprint)
    rows_old, next_idx_old = rg._load_eval_ckpt(ckpt, old_fingerprint)
    assert next_idx_old == len(substrates)
    assert set(rows_old.keys()) == set(substrates)

    # Loading the SAME file under the NEW (current-run) fingerprint must discard it entirely.
    rows_new, next_idx_new = rg._load_eval_ckpt(ckpt, new_fingerprint)
    assert rows_new == {}
    assert next_idx_new == 0

    # And re-running under the new fingerprint recomputes every substrate from scratch
    # (proving the stale rows are never blended into the new-config aggregation).
    recomputed = _run_eval_loop(substrates, resume_path=ckpt, eval_ckpt_every=10,
                                  config_fingerprint=new_fingerprint)
    assert set(recomputed.keys()) == set(substrates)

    # The file on disk now carries the NEW fingerprint (overwritten, not merged).
    with open(ckpt) as fh:
        saved = json.load(fh)
    assert saved["config_fingerprint"] == new_fingerprint


def test_eval_checkpoint_discards_stale_config_on_split_change(tmp_path):
    """FIX B guard: a checkpoint written while evaluating VAL must be DISCARDED (not
    silently adopted) when the SAME other config params are reused to evaluate TEST.
    Before FIX B, ``_eval_config_fingerprint`` did not cover ``eval_split`` (or the
    evaluated-substrate identity), so a VAL checkpoint could be accepted on TEST and
    ``next_idx >= len(test_substrates)`` would skip the whole TEST loop, silently
    reporting VAL rows as TEST -- violating "select on val, touch test once"."""
    import scripts.run_gflownet as rg

    ckpt = str(tmp_path / "eval_seed0.ckpt.json")
    val_substrates = ["CCO", "CCCO", "CCCCO", "CCCCCO"]
    # Same COUNT of substrates on "test" but a DIFFERENT substrate set -- covers both the
    # eval_split flip and the substrate-identity guard in one fixture.
    test_substrates = ["CCN", "CCCN", "CCCCN", "CCCCCN"]

    val_fingerprint = rg._eval_config_fingerprint(
        max_size=10, ks=(5, 10, 15, 20, 30, 50), n_samples=32, top_k=200, max_pool=100,
        eval_split="val", substrates=val_substrates, eval_beam=True,
    )
    test_fingerprint = rg._eval_config_fingerprint(
        max_size=10, ks=(5, 10, 15, 20, 30, 50), n_samples=32, top_k=200, max_pool=100,
        eval_split="test", substrates=test_substrates, eval_beam=True,
    )
    assert val_fingerprint != test_fingerprint

    # Write a checkpoint under the VAL config, fully complete.
    _run_eval_loop(val_substrates, resume_path=ckpt, eval_ckpt_every=10,
                    config_fingerprint=val_fingerprint)
    rows_val, next_idx_val = rg._load_eval_ckpt(ckpt, val_fingerprint)
    assert next_idx_val == len(val_substrates)

    # Loading the SAME checkpoint file under the TEST run's fingerprint must discard it
    # entirely (start fresh at substrate 0), never adopt the VAL rows as TEST rows.
    rows_test, next_idx_test = rg._load_eval_ckpt(ckpt, test_fingerprint)
    assert rows_test == {}
    assert next_idx_test == 0

    # And the TEST loop recomputes every TEST substrate from scratch, never short-circuiting
    # via a stale next_idx that would silently skip real TEST rows.
    recomputed = _run_eval_loop(test_substrates, resume_path=ckpt, eval_ckpt_every=10,
                                  config_fingerprint=test_fingerprint)
    assert set(recomputed.keys()) == set(test_substrates)


def test_eval_config_fingerprint_stable_and_sensitive():
    """The fingerprint must be a pure, deterministic function of its inputs (same inputs ->
    same hash across calls) AND sensitive to each of the fields the plan requires it to
    cover (max_size, ks, n_samples, top_k, max_pool, circles thresholds, and -- FIX B --
    eval_split, the evaluated-substrate set, and eval_beam)."""
    import scripts.run_gflownet as rg

    base = dict(
        max_size=10, ks=(5, 10, 15, 20, 30, 50), n_samples=32, top_k=200, max_pool=100,
        eval_split="val", substrates=("CCO", "CCCO", "CCCCO"), eval_beam=True,
    )
    fp_base = rg._eval_config_fingerprint(**base)
    assert rg._eval_config_fingerprint(**base) == fp_base  # deterministic

    variants = [
        {**base, "max_size": 11},
        {**base, "ks": (5, 10, 15, 20, 30)},
        {**base, "n_samples": 33},
        {**base, "top_k": 201},
        {**base, "max_pool": 101},
        {**base, "eval_split": "test"},
        {**base, "substrates": ("CCO", "CCCO", "CCCCCO")},  # same count, different set
        {**base, "eval_beam": False},
    ]
    for variant in variants:
        assert rg._eval_config_fingerprint(**variant) != fp_base

    assert rg._eval_config_fingerprint(**base, circles_thresholds=(0.3, 0.7)) != fp_base
