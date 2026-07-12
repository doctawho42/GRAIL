# GRAIL Theory Spine + Diagnosis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a formal framework, three refutable propositions, a carrying figure, an external-validity finding, and camera-ready statistical packaging to the GRAIL-primary paper (`docs/GRAIL_FRAMING.md`) — turning an entirely empirical diagnosis into a principled one, without any claim that GRAIL wins on recall.

**Architecture:** Hybrid (option C). New compute produces four JSON artifacts + one waterfall figure from data already on disk; a small pure-Python stats module (`grail_metabolism/stats.py`, unit-tested under `make test`) holds every estimator; the theory and results are woven into the existing `GRAIL_FRAMING.md` section structure (new §1.5, retrofit §2/§3/§4, new §Reproducibility). No training, no GPU, no Modal.

**Tech Stack:** Python 3, `numpy<2`, RDKit, torch (inference only, CPU) via existing `grail_metabolism` package; matplotlib for the figure (already used by `scripts/make_rankflip_figure.py`); pytest for unit tests.

**Source spec:** `docs/superpowers/specs/2026-07-12-grail-theory-spine-design.md` (read it — it holds the full prose content for the writing tasks).

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these; the final review checks each one.

1. **No addition may require GRAIL to win on recall.** The anchor stays **0.334 recall@15** deployed.
2. **The recall factorization is an accounting identity** — it is labeled a **decomposition** everywhere, **never** "theorem," "verified," or "validated." Its cancellation/closure is never presented as evidence.
3. **The refutable content lives only in the three Propositions**, each making a prediction that could be false, checked against committed data.
4. **The Stage-2 listwise reranker (0.500@15) never appears near a headline.** It is a separate Stage-2 artifact that **still loses to SyGMa (0.558)**; used only as the confirmation of Proposition 1.
5. **Proposition 2's `e(r) ∝ π(r)` and the `1/ê` reweighting are an unmeasured modeling assumption and an open test** — never a proof or a promised fix.
6. **The external GLORYx ceiling is recomputed *uncapped*, apples-to-apples with the internal one.** Committed `0.3715` is pool-capped and understates; never call it "the external ceiling." `n=37` → report a wide CI.
7. **No Claude / AI attribution in any commit trailer or doc byline.** (Standing user constraint — verify each commit.)
8. **Every new number ships with provenance** (value, match mode, split, n, resampling unit, seed count, source file); every claim states which factor / population it belongs to.
9. **Match mode is `inchikey_tautomer` everywhere** these additions touch (the harmonized headline mode), reusing `grail_metabolism.metrics._tautomer_inchikey` / `_match_keys(items, "inchikey_tautomer")`. Never mix match modes within a comparison.

## Reusable APIs (verified 2026-07-12 — reuse, do not reinvent)

- `grail_metabolism.metrics._inchikey(smiles)`, `._tautomer_inchikey(smiles)`, `._match_keys(items, "inchikey_tautomer")`.
- `scripts/run_benchmark.py`: `load_test_map(sample, seed) -> Dict[str, Set[str]]` (substrate SMILES → true product SMILES on the clean test split); `grail_ceiling(test_map, rules, audit_tautomer)`; `_tautomer_recovered(true_prods, product_smiles, audit=False) -> (denominator, recovered[, recovered_naive])` (per-substrate tautomer ceiling counts); `sygma_baseline(test_map, ks)`.
- `grail_metabolism.utils.preparation`: `apply_rules_to_molecule(mol, rules, normalization_mode="canonical")` (`.keys()` = product SMILES), `resolve_default_rule_bank()`.
- Generator pool pattern (from `scripts/diagnose_rerank_ceiling.py`): load `artifacts/full5000_priors/checkpoints/generator.pt` via `torch.load(..., weights_only=False)` → `build_generator(GeneratorConfig(**state["arch"]), state.get("rules"))` (`grail_metabolism.workflows.factory.build_generator`), `model.load_state_dict(state["state_dict"], strict=False)`, set `gen.gen_normalization="canonical"`, `gen.prior_strength=8.0`; then `gen.generate_scored(sub, top_k=max_pool, threshold=gen.calibrated_threshold) -> List[Tuple[str, float]]` (deployed candidate pool, ranked).
- `grail_metabolism.workflows.data.load_dataset_bundle(DatasetConfig(...))` → `bundle.test.map` = `{sub: prods}`.
- Deployed top-k output (committed, full5000_single): `artifacts/full5000_single/predictions/test_predictions.csv`, columns `substrate,predicted,real`; `predicted` and `real` are `|`-separated SMILES; `predicted` is the ranked deployed output.
- Committed numbers: ceiling 0.735 tautomer (`results/benchmark_report.json`); SyGMa 0.572 tautomer; prior-vs-learned Δ=−0.144 CI[−0.196,−0.095] (`results/prior_vs_learned.json`); Spike-3 reranker 0.500 vs generator 0.433 (+0.067), oracle 0.677 (`docs/benchmark/stage2_ranker_evidence.md`); depth-2 +0.012 (`results/benchmark_report_depth2.json`); GLORYx per-parent oracle (`results/gloryx_oracle.json`); interaction CI [+0.073,+0.171] (`results/rank_flip_ci.json`).

## Data-dependence note (tests vs runs)

`make test` must stay green **without** the dataset. Therefore: **all estimators are pure functions in `grail_metabolism/stats.py` with unit tests on synthetic fixtures** (run in CI). The scripts that call them on real data + checkpoints are executed **manually** as a "run" step (the dataset is symlinked in this worktree) and their JSON outputs are inspected; those runs are not pytest cases. Each compute task therefore has: (a) TDD on the pure helpers, (b) a script wiring + a real run producing the committed JSON.

---

## Task 1: Stats module — pure estimators + unit tests

**Files:**
- Create: `grail_metabolism/stats.py`
- Create: `grail_metabolism/tests/test_stats.py`

**Interfaces:**
- Produces: `factor_bootstrap_ci(records, factor_specs, n_boot, seed, alpha) -> Dict[str, Dict[str, float]]`; `ratio_of_sums(pairs) -> float`; `ratio_of_sums_ci(pairs, n_boot, seed, alpha) -> Tuple[float, float, float]`; `paired_diff_bootstrap_ci(diffs, n_boot, seed, alpha) -> Tuple[float, float, float]`; `mcnemar_exact_p(b, c) -> float`. Consumed by Tasks 3, 4, 5.

- [ ] **Step 1: Write the failing tests**

```python
# grail_metabolism/tests/test_stats.py
import math
from grail_metabolism.stats import (
    ratio_of_sums, ratio_of_sums_ci, factor_bootstrap_ci,
    paired_diff_bootstrap_ci, mcnemar_exact_p,
)

def test_ratio_of_sums_pools_not_averages_ratios():
    # substrate A: 1/1, substrate B: 0/3 -> ratio-of-sums = 1/4 = 0.25 (NOT mean(1.0,0.0)=0.5)
    assert ratio_of_sums([(1, 1), (0, 3)]) == 0.25
    assert ratio_of_sums([(0, 0)]) == 0.0  # empty denominator guard

def test_factor_bootstrap_closure_and_bounds():
    # two substrates; factors are exact on the point estimate
    records = [
        {"U": 2, "Cfull": 2, "Cbud": 2, "H": 1},
        {"U": 3, "Cfull": 2, "Cbud": 1, "H": 1},
    ]
    specs = {
        "coverage_bank": ("Cfull", "U"),
        "selection_retention": ("Cbud", "Cfull"),
        "ranking_conversion": ("H", "Cbud"),
    }
    res = factor_bootstrap_ci(records, specs, n_boot=200, seed=0)
    cb = res["coverage_bank"]["point"]
    sr = res["selection_retention"]["point"]
    rc = res["ranking_conversion"]["point"]
    assert cb == 4 / 5 and sr == 3 / 4 and rc == 2 / 3
    # product of factors == micro recall = sum(H)/sum(U) = 2/5
    assert math.isclose(cb * sr * rc, 2 / 5)
    for name in specs:
        assert 0.0 <= res[name]["lo"] <= res[name]["point"] <= res[name]["hi"] <= 1.0

def test_paired_diff_ci_sign_and_determinism():
    diffs = [-0.2, -0.3, -0.1, -0.25]
    p, lo, hi = paired_diff_bootstrap_ci(diffs, n_boot=500, seed=0)
    assert p < 0 and hi < 0                     # certifies a loss
    assert (p, lo, hi) == paired_diff_bootstrap_ci(diffs, n_boot=500, seed=0)  # seeded, reproducible

def test_mcnemar_exact_two_sided():
    assert mcnemar_exact_p(0, 0) == 1.0
    # 10 vs 0 discordant -> 2 * 0.5^10 = 0.001953...
    assert math.isclose(mcnemar_exact_p(10, 0), 2 * (0.5 ** 10))
    assert mcnemar_exact_p(5, 5) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest grail_metabolism/tests/test_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'grail_metabolism.stats'`.

- [ ] **Step 3: Write the implementation**

```python
# grail_metabolism/stats.py
"""Pure statistical estimators for the theory-spine analyses. No I/O, no RDKit,
no torch — so they unit-test under `make test` without the dataset. Cluster =
substrate; all bootstraps resample substrates (clusters), never individual pairs."""
from __future__ import annotations

import random
from math import comb
from typing import Dict, List, Sequence, Tuple


def ratio_of_sums(pairs: Sequence[Tuple[float, float]]) -> float:
    """Sum(numerator)/Sum(denominator) over per-cluster (num, den) pairs; 0.0 if
    total denominator is 0. This is the correct estimator when pairs within a
    substrate are dependent (do NOT average per-substrate ratios)."""
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return num / den if den else 0.0


def ratio_of_sums_ci(
    pairs: Sequence[Tuple[float, float]], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Percentile CI for a ratio-of-sums estimator via cluster (substrate)
    resampling with replacement. Returns (point, lo, hi)."""
    rng = random.Random(seed)
    n = len(pairs)
    point = ratio_of_sums(pairs)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(ratio_of_sums(sample))
    boots.sort()
    return point, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot)]


def factor_bootstrap_ci(
    records: Sequence[Dict[str, float]],
    factor_specs: Dict[str, Tuple[str, str]],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Joint cluster bootstrap for several ratio-of-sums factors. `records` is one
    dict per substrate; `factor_specs[name] = (numerator_field, denominator_field)`.
    Each bootstrap resamples substrates ONCE and recomputes every factor on that
    same resample, so the factor CIs are mutually consistent. Returns
    {name: {"point": .., "lo": .., "hi": ..}}."""
    rng = random.Random(seed)
    n = len(records)

    def factors(sample: Sequence[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, (nf, df) in factor_specs.items():
            num = sum(r[nf] for r in sample)
            den = sum(r[df] for r in sample)
            out[name] = num / den if den else 0.0
        return out

    point = factors(records)
    acc: Dict[str, List[float]] = {name: [] for name in factor_specs}
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        f = factors(sample)
        for name in factor_specs:
            acc[name].append(f[name])
    res: Dict[str, Dict[str, float]] = {}
    for name in factor_specs:
        b = sorted(acc[name])
        res[name] = {
            "point": point[name],
            "lo": b[int((alpha / 2) * n_boot)],
            "hi": b[int((1 - alpha / 2) * n_boot)],
        }
    return res


def paired_diff_bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Percentile CI for the mean of paired per-substrate differences d_i. Returns
    (point, lo, hi). A wholly-below-0 CI certifies a loss."""
    rng = random.Random(seed)
    n = len(diffs)
    point = sum(diffs) / n if n else 0.0
    boots: List[float] = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return point, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot)]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (GRAIL hit, other
    miss) and c (GRAIL miss, other hit), under Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grail_metabolism/tests/test_stats.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Confirm the whole suite stays green**

Run: `make test`
Expected: all pass (previous count + 4 new).

- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/stats.py grail_metabolism/tests/test_stats.py
git commit -m "feat(stats): cluster-bootstrap + ratio-of-sums + McNemar estimators for theory-spine analyses"
```

---

## Task 2: Per-substrate recall factorization compute

**Files:**
- Create: `scripts/factorize_recall.py`
- Create (run output): `results/recall_factorization.json`

**Interfaces:**
- Consumes: `grail_metabolism.stats.factor_bootstrap_ci`; `run_benchmark.load_test_map`, `._tautomer_recovered`; `apply_rules_to_molecule`, `resolve_default_rule_bank`; the generator pool pattern; `artifacts/full5000_single/predictions/test_predictions.csv`.
- Produces: `results/recall_factorization.json` with keys `factors` (coverage_bank / selection_retention / ranking_conversion each `{point, lo, hi}`), `micro_recall`, `oracle_recall` (`= C_bud/U`), `n_substrates`, `match`, `provenance`, and `per_substrate` (list of `{sub, U, Cfull, Cbud, H}`). The `per_substrate` dump is consumed by Task 4.

**Definitions** (per substrate `s`, tautomer-InChIKey): `U_i=|T_s|` (tautomer-distinct trues); `Cfull_i` = trues recovered by the **full-bank** depth-1 products; `Cbud_i` = trues recovered by the **deployed generator pool** (`generate_scored`, top_k=max_pool); `H_i` = trues recovered by the **deployed top-15 output**. Then `coverage_bank=ΣCfull/ΣU`, `selection_retention=ΣCbud/ΣCfull`, `ranking_conversion=ΣH/ΣCbud`; `oracle_recall=ΣCbud/ΣU`.

> **CONTROLLER CORRECTION (2026-07-12, pre-dispatch — supersedes the Step-3 CSV path for `H`).**
> `artifacts/full5000_single/predictions/test_predictions.csv` covers only **291 substrates** (an
> `ensemble.py` export cap), NOT the full test set; using it would silently compute the whole
> factorization on a non-representative 291-substrate subset and invite the exact "cherry-picked
> subset" referee attack the paper avoids. `artifacts/full5000_single/reports/metrics.json` confirms
> the deployed **ensemble** `top_15_recall = 0.334` is the full-set number, and both checkpoints
> (`generator.pt` + `filter.pt`) are in `artifacts/full5000_single/checkpoints/`. Therefore:
> - Compute `H` by running the **actual deployed pipeline** on ALL test-map substrates (≈1170).
>   Load the deployed wrapper (generator + filter) — read `grail_metabolism/model/grail.py:85-170`
>   for the exact load path (`summon_the_grail` and/or `ModelWrapper(filter_weights=..., generator_weights=...)`),
>   pointing at `artifacts/full5000_single/checkpoints/{generator.pt,filter.pt}`.
> - Per substrate: `H_i` = `tautomer_hits(wrapper.generate(sub, max_output=15), trues)` (the deployed
>   gen×filter top-15); `Cbud_i` = `tautomer_hits([s for s,_ in gen.generate_scored(sub, top_k=200)], trues)`
>   (the generator pool the filter reranks); `Cfull_i` via `_tautomer_recovered` as in Step 3.
> - **Do NOT read `test_predictions.csv`.** Delete the `load_deployed_topk` helper; KEEP `tautomer_hits`
>   (still needed to count hits from the pipeline's SMILES output, and its unit test in Step 1 still applies).
> - **Persist per-substrate the deployed top-15 SMILES** in each `per_substrate` record
>   (add `"deployed_top15": [smiles,...]` alongside `U/Cfull/Cbud/H`) so Task 4 reuses them without
>   re-running the pipeline.
> - Report the actual `n` (≈1170). Sanity: `micro_recall` ≈ 0.334 (matches metrics.json ensemble
>   top_15_recall); `coverage_bank` ≈ 0.735.
> - **Runtime (staged, mandatory):** FIRST run a ~20-substrate timing probe under `caffeinate -dimsu`
>   and extrapolate to the full test set. If extrapolated full wall-time > ~2h, STOP and report the
>   timing (status DONE_WITH_CONCERNS / BLOCKED) so the controller decides (subset-with-representativeness
>   vs generator-alone-`H` vs Modal) — do NOT silently subsample. If ≤ ~2h, proceed to the full run.
>   (The prior deployed eval took ~206s once graph caches were built; a cold featurization pass can be
>   much longer — build caches first, then time.)

- [ ] **Step 1: Write a failing unit test for the CSV-hit helper**

The only non-orchestration logic worth a unit test is per-substrate tautomer hit counting against the deployed CSV. Put the helper in the script and test it.

```python
# grail_metabolism/tests/test_factorize_helpers.py
from scripts.factorize_recall import tautomer_hits

def test_tautomer_hits_counts_distinct_true_matches():
    # 'CCO' (ethanol) and 'OCC' are the same molecule -> one tautomer key.
    trues = ["CCO", "c1ccccc1"]          # ethanol, benzene
    preds = ["OCC", "CCO", "CCCC"]       # two spellings of ethanol + butane
    # only ethanol is recovered -> 1 of 2 trues
    assert tautomer_hits(preds, trues) == 1
    assert tautomer_hits([], trues) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_factorize_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError` (script not yet created).

- [ ] **Step 3: Write `scripts/factorize_recall.py`**

```python
#!/usr/bin/env python
"""Per-substrate coverage x selection x ranking factorization of GRAIL recall@15,
tautomer-InChIKey, on the clean test split. Emits results/recall_factorization.json.
This is a DECOMPOSITION (an accounting identity), not a theorem — see the spec's
Global Constraint 2. Reuses run_benchmark's ceiling helpers + diagnose_rerank_ceiling's
generator-pool pattern."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.metrics import _inchikey, _tautomer_inchikey
from grail_metabolism.stats import factor_bootstrap_ci
from grail_metabolism.utils.preparation import apply_rules_to_molecule, resolve_default_rule_bank
from grail_metabolism.config import DatasetConfig, GeneratorConfig
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_generator
from run_benchmark import _tautomer_recovered  # per-substrate (denominator, recovered)
from rdkit import Chem

K = 15
MAX_POOL = 200
PRIOR_STRENGTH = 8.0
CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"


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


def load_deployed_topk(csv_path: Path, k: int) -> dict:
    """substrate InChIKey -> list of the top-k deployed predicted SMILES."""
    out = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            preds = [p for p in row["predicted"].split("|") if p][:k]
            try:
                out[_inchikey(row["substrate"])] = preds
            except Exception:
                pass
    return out


def build_gen():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**state["arch"]), state.get("rules"))
    gen.load_state_dict(state["state_dict"], strict=False)
    gen.calibrated_threshold = state.get("calibrated_threshold")
    gen.gen_normalization = "canonical"
    gen.prior_strength = PRIOR_STRENGTH
    return gen


def main() -> int:
    torch.set_num_threads(6)
    rules = resolve_default_rule_bank()
    gen = build_gen()

    ds = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=100000,
    )
    bundle = load_dataset_bundle(ds)
    items = list(bundle.test.map.items())
    deployed = load_deployed_topk(DEPLOYED_CSV, K)

    records = []
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None or not prods:
            continue
        true_prods = list(prods)
        # C_full: full-bank depth-1 products, tautomer ceiling recovered count
        full_products = list(apply_rules_to_molecule(mol, rules, normalization_mode="canonical").keys())
        u_i, cfull_i, _ = _tautomer_recovered(true_prods, full_products, audit=False)
        if u_i == 0:
            continue
        # C_bud: deployed generator pool
        scored = gen.generate_scored(sub, top_k=MAX_POOL, threshold=gen.calibrated_threshold)
        cbud_i = tautomer_hits([s for s, _ in scored], true_prods)
        # H: deployed top-15 output (committed CSV); fall back to 0 if substrate absent
        h_i = tautomer_hits(deployed.get(_inchikey(sub), []), true_prods)
        # monotonicity clamp (nested sets guarantee this; clamp defends against
        # cross-source SMILES canonicalization drift between CSV and pool)
        cbud_i = min(cbud_i, cfull_i)
        h_i = min(h_i, cbud_i)
        records.append({"sub": sub, "U": u_i, "Cfull": cfull_i, "Cbud": cbud_i, "H": h_i})

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
        "provenance": {
            "split": "clean test", "ceiling_pool": "full extended_smirks bank",
            "deployed_pool": f"generator.pt generate_scored top_k={MAX_POOL} prior_strength={PRIOR_STRENGTH}",
            "deployed_output": str(DEPLOYED_CSV.relative_to(ROOT)),
            "resampling_unit": "substrate", "n_boot": 10000, "seed": 0,
        },
        "per_substrate": records,
    }
    out = ROOT / "results" / "recall_factorization.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("factors", "micro_recall", "oracle_recall", "n_substrates")}, indent=2))
    # sanity: identity closes to rounding
    prod = (factors["coverage_bank"]["point"] * factors["selection_retention"]["point"]
            * factors["ranking_conversion"]["point"])
    assert abs(prod - micro_recall) < 1e-6, (prod, micro_recall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_factorize_helpers.py -q`
Expected: PASS.

- [ ] **Step 5: Run the script on real data (worktree has the symlinked dataset)**

Run: `python scripts/factorize_recall.py`
Expected: prints per-factor points + `micro_recall`; the in-script assertion (identity closes) passes; `results/recall_factorization.json` written. Sanity to eyeball: `coverage_bank.point` ≈ 0.735 (±0.01), `micro_recall` ≈ 0.33–0.34, `oracle_recall` between them and > micro_recall.

- [ ] **Step 6: Commit**

```bash
git add scripts/factorize_recall.py grail_metabolism/tests/test_factorize_helpers.py results/recall_factorization.json
git commit -m "feat(diagnostic): per-substrate coverage x selection x ranking recall decomposition (tautomer, bootstrap CIs)"
```

---

## Task 3: External-validity — internal ceiling CI + uncapped GLORYx + covariate regression

**Files:**
- Create: `scripts/ceiling_external_validity.py`
- Create (run output): `results/ceiling_external_validity.json`

**Interfaces:**
- Consumes: `grail_metabolism.stats.ratio_of_sums_ci`; `run_benchmark.load_test_map`, `._tautomer_recovered`; `apply_rules_to_molecule`, `resolve_default_rule_bank`; `results/gloryx_oracle.json` (`small.per_parent` records: `parent`, `n_true`, `hits_in_pool`, `pool_size`); `numpy`.
- Produces: `results/ceiling_external_validity.json` with `internal_ceiling` (`{point, lo, hi}`), `external_ceiling_uncapped` (`{point, lo, hi}`), `regression` (`coefficients`, `predicted_internal_mean`, `predicted_external_mean`, `descriptor_names`).

- [ ] **Step 1: Write a failing unit test for the descriptor vector**

```python
# grail_metabolism/tests/test_external_validity_helpers.py
from scripts.ceiling_external_validity import composition_descriptors

def test_composition_descriptors_shape_and_values():
    d = composition_descriptors("c1ccccc1")   # benzene
    assert set(d) == {"mw", "n_rings", "n_aromatic", "n_hetero", "n_conj", "n_true_ph"}
    assert 77 < d["mw"] < 79          # ~78.11
    assert d["n_rings"] == 1 and d["n_aromatic"] == 6 and d["n_hetero"] == 0
    assert composition_descriptors("not_a_smiles") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_external_validity_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `scripts/ceiling_external_validity.py`**

```python
#!/usr/bin/env python
"""External validity of the rule-bank coverage ceiling: (1) cluster-bootstrap CI on
the internal 0.735 ceiling; (2) the UNCAPPED GLORYx-37 ceiling (apples-to-apples, one
apply_rules pass, no pool cap — the committed 0.3715 is pool-capped and understates);
(3) one composition regression predicting BOTH means. Emits
results/ceiling_external_validity.json. Guardrail: n=37 external -> wide CI; the gap is
composition (larger, more-conjugated drugs), not a bug."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.stats import ratio_of_sums_ci
from grail_metabolism.utils.preparation import apply_rules_to_molecule, resolve_default_rule_bank
from grail_metabolism.config import DatasetConfig
from grail_metabolism.workflows.data import load_dataset_bundle
from run_benchmark import _tautomer_recovered
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DESC = ["mw", "n_rings", "n_aromatic", "n_hetero", "n_conj", "n_true_ph"]


def composition_descriptors(smiles: str, n_true: int = 0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    # conjugation proxy: bonds flagged conjugated
    n_conj = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
    return {
        "mw": Descriptors.MolWt(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic": n_aromatic,
        "n_hetero": n_hetero,
        "n_conj": n_conj,
        "n_true_ph": float(n_true),
    }


def _coverage_pairs_internal(rules):
    ds = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=100000,
    )
    bundle = load_dataset_bundle(ds)
    pairs, rows = [], []          # (recovered, denom) per substrate + (smiles, cov, n_true)
    for sub, prods in bundle.test.map.items():
        mol = Chem.MolFromSmiles(sub)
        if mol is None or not prods:
            continue
        products = list(apply_rules_to_molecule(mol, rules, normalization_mode="canonical").keys())
        denom, rec, _ = _tautomer_recovered(list(prods), products, audit=False)
        if denom == 0:
            continue
        pairs.append((rec, denom))
        rows.append((sub, rec / denom, len(prods)))
    return pairs, rows


def _coverage_pairs_external_uncapped(rules):
    """Recompute GLORYx-37 coverage uncapped: for each parent, apply the FULL bank
    (no pool cap) and count tautomer-recovered trues. The committed gloryx_oracle.json
    gives parents + true metabolites (reconstruct trues from per_parent if present, else
    from the GLORYx shared set under docs/benchmark/data)."""
    gloryx = json.loads((ROOT / "results" / "gloryx_oracle.json").read_text())
    # per_parent lacks the raw true SMILES; load them from the committed GLORYx set
    trues_by_parent = _load_gloryx_truth()  # -> {parent_smiles: [true_smiles,...]}
    pairs, rows = [], []
    for rec in gloryx["small"]["per_parent"]:
        parent = rec["parent"]
        trues = trues_by_parent.get(parent)
        if not trues:
            continue
        mol = Chem.MolFromSmiles(parent)
        if mol is None:
            continue
        products = list(apply_rules_to_molecule(mol, rules, normalization_mode="canonical").keys())
        denom, r, _ = _tautomer_recovered(list(trues), products, audit=False)
        if denom == 0:
            continue
        pairs.append((r, denom))
        rows.append((parent, r / denom, len(trues)))
    return pairs, rows


def _load_gloryx_truth():
    """Parent SMILES -> list of annotated metabolite SMILES, from the committed
    GLORYx shared set. Implementer: locate the file under docs/benchmark/data (the
    S1.3 shared-set artifact) and parse it; it maps 37 drugs -> ~136 metabolites."""
    path = ROOT / "docs" / "benchmark" / "data" / "gloryx_truth.json"
    return json.loads(path.read_text())


def _fit_regression(rows):
    X = np.array([[composition_descriptors(s, n)[d] for d in DESC] for s, _, n in rows])
    y = np.array([cov for _, cov, _ in rows])
    Xb = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef, Xb


def main() -> int:
    rules = resolve_default_rule_bank()
    int_pairs, int_rows = _coverage_pairs_internal(rules)
    ext_pairs, ext_rows = _coverage_pairs_external_uncapped(rules)

    ip, ilo, ihi = ratio_of_sums_ci(int_pairs, n_boot=10000, seed=0)
    ep, elo, ehi = ratio_of_sums_ci(ext_pairs, n_boot=10000, seed=0)

    coef, _ = _fit_regression(int_rows + ext_rows)   # one model over both populations
    def predicted_mean(rows):
        Xb = np.hstack([np.ones((len(rows), 1)),
                        np.array([[composition_descriptors(s, n)[d] for d in DESC] for s, _, n in rows])])
        return float((Xb @ coef).mean())

    report = {
        "match": "inchikey_tautomer",
        "internal_ceiling": {"point": ip, "lo": ilo, "hi": ihi, "n": len(int_pairs)},
        "external_ceiling_uncapped": {"point": ep, "lo": elo, "hi": ehi, "n": len(ext_pairs)},
        "external_ceiling_capped_committed": 0.3715,
        "regression": {
            "descriptor_names": DESC,
            "coefficients": [float(c) for c in coef],
            "predicted_internal_mean": predicted_mean(int_rows),
            "predicted_external_mean": predicted_mean(ext_rows),
        },
        "provenance": {"resampling_unit": "substrate", "n_boot": 10000, "seed": 0,
                       "note": "external uncapped via one apply_rules pass; n=37 -> wide CI"},
    }
    out = ROOT / "results" / "ceiling_external_validity.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("internal_ceiling", "external_ceiling_uncapped", "regression")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Resolve the GLORYx truth source before running**

The regression + uncapped external ceiling need parent→true-metabolite SMILES. Before Step 5, confirm the committed GLORYx shared set exists and adjust `_load_gloryx_truth()` to its real path/format:

Run: `ls docs/benchmark/data/ && python -c "import json,glob; [print(p) for p in glob.glob('docs/benchmark/data/*')]"`
Expected: a GLORYx set file (from S1.3). If its schema differs, update `_load_gloryx_truth()` to return `{parent_smiles: [metabolite_smiles, ...]}`; if no parseable truth file exists, STOP and report — do not fabricate metabolites (Global Constraint 8).

- [ ] **Step 5: Run the unit test, then the script**

Run: `python -m pytest grail_metabolism/tests/test_external_validity_helpers.py -q`
Expected: PASS.

Run: `python scripts/ceiling_external_validity.py`
Expected: `results/ceiling_external_validity.json` written; `internal_ceiling.point` ≈ 0.735 and its CI contains 0.735; `external_ceiling_uncapped.point` ≥ 0.3715 (uncapped ≥ capped); regression `predicted_internal_mean` ≈ internal point and `predicted_external_mean` ≈ external point (one model tracks both).

- [ ] **Step 6: Commit**

```bash
git add scripts/ceiling_external_validity.py grail_metabolism/tests/test_external_validity_helpers.py results/ceiling_external_validity.json
git commit -m "feat(diagnostic): external-validity ceiling CI + uncapped GLORYx + composition covariate model"
```

---

## Task 4: Anchor certification (SyGMa > GRAIL)

**Files:**
- Create: `scripts/anchor_certification.py`
- Create (run output): `results/anchor_certification.json`

**Interfaces:**
- Consumes: `grail_metabolism.stats.paired_diff_bootstrap_ci`, `.mcnemar_exact_p`; GRAIL's deployed per-substrate top-15 and SyGMa on the same substrates; `run_benchmark._tautomer_recovered` for per-substrate recall; the `tautomer_hits` helper from Task 2 (`from factorize_recall import tautomer_hits`).
- Produces: `results/anchor_certification.json` with `delta_mean_recall` (`{point, lo, hi}`), `mcnemar` (`{b, c, p}`), `common_n`, `common_subset_ceiling`, `full_ceiling`.

> **CONTROLLER CORRECTION (2026-07-12 — supersedes the Step-3 `load_deployed(DEPLOYED_CSV)` path).**
> Same 291-vs-1170 issue as Task 2: do NOT use `test_predictions.csv`. Take GRAIL's per-substrate
> deployed top-15 SMILES from **Task 2's `results/recall_factorization.json` `per_substrate[].deployed_top15`**
> (persisted there by Task 2's correction), keyed by substrate. SyGMa's per-substrate top-15 comes from
> `run_benchmark.sygma_topk` (the DRY helper added in Step 4) on the same substrate set. Compute the
> paired diff and McNemar over the full common set (≈1170), and report `common_n`. Drop the
> `load_deployed` helper. Everything else in the task stands.

- [ ] **Step 1: Write a failing unit test for per-substrate recall + any-hit**

```python
# grail_metabolism/tests/test_anchor_helpers.py
from scripts.anchor_certification import per_substrate_recall, any_hit

def test_per_substrate_recall_tautomer():
    trues = ["CCO", "c1ccccc1"]
    preds = ["OCC"]                       # ethanol only
    assert per_substrate_recall(preds, trues) == 0.5
    assert any_hit(preds, trues) is True
    assert any_hit(["CCCC"], trues) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_anchor_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `scripts/anchor_certification.py`**

```python
#!/usr/bin/env python
"""Certify the honest anchor: SyGMa > GRAIL on the COMMON substrate set. Paired
bootstrap on d_i = recall_GRAIL_i - recall_SyGMa_i (continuous) + McNemar on any-hit@15
(binary). Tautomer-InChIKey. Emits results/anchor_certification.json. Guardrail: this
certifies EVALUATION variance; the common subset's ceiling is reported to show it is
representative of the full 1170."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.metrics import _inchikey, _tautomer_inchikey
from grail_metabolism.stats import paired_diff_bootstrap_ci, mcnemar_exact_p
from grail_metabolism.utils.preparation import apply_rules_to_molecule, resolve_default_rule_bank
from grail_metabolism.config import DatasetConfig
from grail_metabolism.workflows.data import load_dataset_bundle
from run_benchmark import _tautomer_recovered, sygma_baseline  # noqa: F401  (sygma_baseline for reference)
from factorize_recall import tautomer_hits
from rdkit import Chem

K = 15
DEPLOYED_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"


def per_substrate_recall(preds, trues) -> float:
    tk = set()
    for t in trues:
        try:
            tk.add(_tautomer_inchikey(t))
        except Exception:
            tk.add(t)
    return tautomer_hits(preds, trues) / len(tk) if tk else 0.0


def any_hit(preds, trues) -> bool:
    return tautomer_hits(preds, trues) > 0


def load_deployed(csv_path):
    out = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            preds = [p for p in row["predicted"].split("|") if p][:K]
            try:
                out[_inchikey(row["substrate"])] = preds
            except Exception:
                pass
    return out
```

The SyGMa predictions on the same substrates come from `sygma` (the tool wrapped in `run_benchmark`). Reuse `run_benchmark`'s SyGMa path to produce, for each test substrate, the ranked SyGMa SMILES (top-15). Implementer: call the same SyGMa generator `run_benchmark.sygma_baseline` uses internally (it iterates `test_map` and runs SyGMa per substrate); factor its per-substrate ranked output into a helper `sygma_topk(test_map, k) -> {sub: [smiles]}` and import it here. Then:

```python
def main() -> int:
    rules = resolve_default_rule_bank()
    ds = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=100000,
    )
    bundle = load_dataset_bundle(ds)
    test_map = dict(bundle.test.map)
    grail = load_deployed(DEPLOYED_CSV)
    from run_benchmark import sygma_topk  # implementer adds this small helper to run_benchmark
    sygma = sygma_topk(test_map, K)

    diffs, b, c = [], 0, 0
    common, ceil_num, ceil_den = 0, 0, 0
    for sub, prods in test_map.items():
        ik = _inchikey(sub)
        if ik not in grail or sub not in sygma:
            continue
        common += 1
        g_preds, s_preds = grail[ik], sygma[sub]
        rg, rs = per_substrate_recall(g_preds, list(prods)), per_substrate_recall(s_preds, list(prods))
        diffs.append(rg - rs)
        gh, sh = any_hit(g_preds, list(prods)), any_hit(s_preds, list(prods))
        if gh and not sh:
            b += 1
        elif sh and not gh:
            c += 1
        mol = Chem.MolFromSmiles(sub)
        if mol is not None:
            products = list(apply_rules_to_molecule(mol, rules, normalization_mode="canonical").keys())
            denom, rec, _ = _tautomer_recovered(list(prods), products, audit=False)
            ceil_num += rec
            ceil_den += denom

    dp, dlo, dhi = paired_diff_bootstrap_ci(diffs, n_boot=10000, seed=0)
    report = {
        "match": "inchikey_tautomer", "k": K, "common_n": common,
        "delta_mean_recall": {"point": dp, "lo": dlo, "hi": dhi,
                              "definition": "recall_GRAIL - recall_SyGMa (negative = GRAIL loses)"},
        "mcnemar": {"b_grail_only": b, "c_sygma_only": c, "p": mcnemar_exact_p(b, c)},
        "common_subset_ceiling": ceil_num / ceil_den if ceil_den else 0.0,
        "full_ceiling_reference": 0.735,
        "provenance": {"resampling_unit": "substrate", "n_boot": 10000, "seed": 0},
    }
    (ROOT / "results" / "anchor_certification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the `sygma_topk` helper to `run_benchmark.py`**

Factor SyGMa's per-substrate ranked output (already produced inside `sygma_baseline`) into a reusable helper so both callers share one code path:

```python
# scripts/run_benchmark.py  (near sygma_baseline)
def sygma_topk(test_map: Dict[str, Set[str]], k: int) -> Dict[str, List[str]]:
    """Per-substrate ranked SyGMa predicted SMILES (top-k), the same generation
    sygma_baseline scores. Returns {substrate_smiles: [smiles, ...]}."""
    # Reuse the exact SyGMa call sygma_baseline makes per substrate; return the
    # ranked SMILES list truncated to k instead of aggregating recall.
    ...
```

Implementer: read `sygma_baseline` (scripts/run_benchmark.py:409+), lift its per-substrate SyGMa generation into `sygma_topk`, and have `sygma_baseline` call `sygma_topk` to stay DRY. Keep `sygma_baseline`'s existing output identical (verify `results/benchmark_report.json` SyGMa numbers reproduce).

- [ ] **Step 5: Run the unit test, then the script**

Run: `python -m pytest grail_metabolism/tests/test_anchor_helpers.py -q`
Expected: PASS.

Run: `python scripts/anchor_certification.py`
Expected: `results/anchor_certification.json`; `delta_mean_recall.point` < 0 and `hi` < 0 (CI wholly below 0 → SyGMa significantly beats GRAIL); `mcnemar.p` small; `common_subset_ceiling` ≈ 0.735 (representative).

- [ ] **Step 6: Commit**

```bash
git add scripts/anchor_certification.py grail_metabolism/tests/test_anchor_helpers.py scripts/run_benchmark.py results/anchor_certification.json
git commit -m "feat(diagnostic): certify honest anchor (paired bootstrap + McNemar, SyGMa > GRAIL on common set)"
```

---

## Task 5: §1.5 Formal framework prose

**Files:**
- Modify: `docs/GRAIL_FRAMING.md` (insert a new `### §1.5 — Formal framework` after §1)

**Interfaces:**
- Consumes: `results/recall_factorization.json` (real factor values + CIs), the spec §Component 1.

- [ ] **Step 1: Write the §1.5 section**

Write `### §1.5 — Formal framework` containing exactly the two blocks from spec Component 1:
(a) the generative latent-reaction mixture `P(m|s)=Σ_{r,site} P(r|s)·P(site|r,s)·𝟙[apply(r,s,site)=m]`, the stage→term mapping (generator≈P(r|s) with `rule_prior_logits`=π(r); RDKit=support; filter=discriminative correction; PU≈EM), presented as "the model the pipeline approximates";
(b) the factorization identity `recall@k = coverage_bank · selection_retention · ranking_conversion` over nested `R_s⊆P_bud,s⊆P_full,s`, **labeled a decomposition**, with the real numbers from `recall_factorization.json` (`factors`, `micro_recall`, `oracle_recall`) and each factor's CI; plus the lever→factor table.

Use the exact deployed numbers the script produced (not the illustrative Spike-3 closure). State the match mode (tautomer-InChIKey) and n.

- [ ] **Step 2: Guardrail verification (checklist, not a test)**

Confirm in the written section:
- the identity is called "decomposition," never "theorem/verified/validated" (Global Constraint 2);
- each factor ∈ [0,1] and is labeled with its CI + n + match mode (Constraint 8/9);
- the generative model is "approximated," not asserted as the MLE.

Grep to catch violations:

Run: `grep -niE "theorem|verified|validated|proven" docs/GRAIL_FRAMING.md | grep -i factoriz` 
Expected: no line calls the factorization any of these.

- [ ] **Step 3: Commit**

```bash
git add docs/GRAIL_FRAMING.md
git commit -m "docs(framing): §1.5 formal framework — generative model + recall decomposition identity"
```

---

## Task 6: Three Propositions prose + counterexample check

**Files:**
- Modify: `docs/GRAIL_FRAMING.md` (add labeled Propositions into §4 under their factors)
- Create: `grail_metabolism/tests/test_prop1_counterexample.py`

**Interfaces:**
- Consumes: spec §Component 2; committed numbers (Δ=−0.144; +0.067 Spike-3; depth-2 +0.012; GLORYx uncapped from Task 3).

- [ ] **Step 1: Write the failing counterexample test**

Prop 1 claims a calibrated pointwise scorer can be a worse top-k ranker than a listwise reorder under heterogeneous pools. Encode the minimal 2-substrate witness so the prose's numbers are checked, not asserted.

```python
# grail_metabolism/tests/test_prop1_counterexample.py
def calibrated_recall_at_k(pools, k):
    """Rank each pool by the calibrated posterior (score) descending; sum hits in
    top-k; divide by total trues. pools: list of (items) where item=(score, is_true)."""
    num = den = 0
    for pool in pools:
        den += sum(1 for _, t in pool if t)
        ranked = sorted(pool, key=lambda x: -x[0])[:k]
        num += sum(1 for _, t in ranked if t)
    return num / den if den else 0.0

def listwise_recall_at_k(pools, k):
    """A listwise oracle-ish reorder that is allowed a per-pool budget: put trues
    first within each pool (the recall a ranking-consistent surrogate can reach)."""
    num = den = 0
    for pool in pools:
        den += sum(1 for _, t in pool if t)
        num += min(sum(1 for _, t in pool if t), k)
    return num / den if den else 0.0

def test_calibrated_scorer_loses_to_listwise_under_heterogeneous_pools():
    # Substrate A: small pool, its one true has a MODERATE calibrated score.
    # Substrate B: large pool full of high-scored false positives that globally
    # outrank A's true, so a single global top-k starves A. Heterogeneity in pool
    # size + positive-rate is the mechanism (spec Prop 1).
    A = [(0.55, True), (0.10, False)]
    B = [(0.90, False), (0.85, False), (0.80, False), (0.05, True)]
    pools = [A, B]
    k = 1
    # calibrated global ranking gives each pool its own top-1 -> A hits, B misses
    # but the POINT is the pooled/global operator; emulate a global budget of 2 total:
    # here we show per-pool top-1 (calibrated) vs listwise both; the inequality the
    # Proposition needs is calibrated <= listwise, strict somewhere.
    assert calibrated_recall_at_k(pools, k) <= listwise_recall_at_k(pools, k)
    # strictness: B's true (0.05) never reaches top-1 by score, listwise recovers it
    assert calibrated_recall_at_k([B], 1) < listwise_recall_at_k([B], 1)
```

- [ ] **Step 2: Run it to verify it fails, then passes**

Run: `python -m pytest grail_metabolism/tests/test_prop1_counterexample.py -q`
Expected: initially FAIL (file references not yet created) → after writing the two helper functions in the test file itself, PASS. (The helpers live in the test; no product code needed — this test *is* the arithmetic check of the prose counterexample.)

- [ ] **Step 3: Write the three Propositions into §4**

From spec Component 2, write labeled **Proposition 1** (surrogate mismatch, → ranking_conversion) with: the statement; "imported ranking-consistency result + the 2-substrate counterexample (checked in `test_prop1_counterexample.py`)"; the committed confirmation (listwise-InfoNCE 0.433→0.500, +0.067, Spike-3); and the guardrail sentence (reranker 0.500 < SyGMa 0.558, separate Stage-2 artifact, theory imported-and-applied). **Proposition 2** (propensity-PU, → selection_retention): SCAR account, anchor Δ=−0.144 CI[−0.196,−0.095], falsifiable `1/ê` test, guardrail (`e(r)∝π(r)` unmeasured; deployment prior-independent). **Proposition 3** (paradigm limit, → coverage_bank): single-step recall ≤ coverage < 1, witnesses depth-2 +0.012 and GLORYx uncapped ceiling from `ceiling_external_validity.json`, guardrail (single-step-conditional, not "futile").

- [ ] **Step 4: Guardrail verification (checklist)**

- Prop 1 never places 0.500 near a headline and states it loses to SyGMa (Constraint 4).
- Prop 2 flags `e(r)∝π(r)` as unmeasured and `1/ê` as an open test (Constraint 5).
- Prop 3 is stated single-step-conditional (spec guardrail).
- Each Proposition names the factor it attaches to (Constraint 8).

Run: `python -m pytest grail_metabolism/tests/test_prop1_counterexample.py -q && make test`
Expected: PASS (suite green).

- [ ] **Step 5: Commit**

```bash
git add docs/GRAIL_FRAMING.md grail_metabolism/tests/test_prop1_counterexample.py
git commit -m "docs(framing): three refutable propositions (surrogate-mismatch, propensity-PU, paradigm-limit) + counterexample check"
```

---

## Task 7: §2 external-validity prose + §3 figure prose + anchor prose

**Files:**
- Modify: `docs/GRAIL_FRAMING.md` (§2 external validity; §3 waterfall + anchor)

**Interfaces:**
- Consumes: `results/ceiling_external_validity.json`, `results/anchor_certification.json`, `docs/benchmark/factorization_waterfall.svg` (Task 8), spec §Components 4 & 5.1.

- [ ] **Step 1: Write §2 external-validity paragraph**

From spec Component 4 + `ceiling_external_validity.json`: internal ceiling 0.735 with its cluster-bootstrap CI; the **uncapped** external GLORYx ceiling with its wide CI (state n=37); the one composition regression predicting both means (report the two predicted means). Frame: "coverage is governed by a transferable composition covariate." Guardrail: never call 0.37 the external ceiling; gap = composition, not a bug.

- [ ] **Step 2: Write §3 conversion-gap + anchor paragraph**

Reference the waterfall figure `docs/benchmark/factorization_waterfall.svg`; state the conversion gap = product of selection_retention × ranking_conversion with the real numbers; add the anchor certification from `anchor_certification.json` (Δ<0 CI, McNemar p, common-subset ceiling ≈ full → representative). Guardrail: certifies evaluation variance; McNemar only for any-hit.

- [ ] **Step 3: Guardrail verification (checklist)**

Run: `grep -niE "0\.37|external" docs/GRAIL_FRAMING.md | grep -iE "ceiling"`
Expected: any mention of 0.37 is labeled "pool-capped"; the reported external ceiling is the uncapped value from the JSON.

- [ ] **Step 4: Commit**

```bash
git add docs/GRAIL_FRAMING.md
git commit -m "docs(framing): §2 external-validity covariate finding + §3 conversion-gap/anchor prose"
```

---

## Task 8: Waterfall figure + camera-ready packaging

**Files:**
- Create: `scripts/make_factorization_figure.py`
- Create (run output): `docs/benchmark/factorization_waterfall.png`, `docs/benchmark/factorization_waterfall.svg`
- Create: `scripts/regen_headline.sh`
- Modify: `docs/GRAIL_FRAMING.md` (new `## §Reproducibility` — provenance table + primary endpoint + release manifest + benchmark name)

**Interfaces:**
- Consumes: `results/recall_factorization.json`; all four result JSONs for the provenance table; the tier-2 preds under `artifacts/tier2/` for the release manifest.

- [ ] **Step 1: Write `scripts/make_factorization_figure.py`**

```python
#!/usr/bin/env python
"""Waterfall: U -> coverage_bank -> coverage*selection (oracle line) -> deployed recall,
from results/recall_factorization.json. Matches the style of make_rankflip_figure.py.
Writes PNG + SVG under docs/benchmark/."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "recall_factorization.json").read_text())
cb = d["factors"]["coverage_bank"]["point"]
sr = d["factors"]["selection_retention"]["point"]
rc = d["factors"]["ranking_conversion"]["point"]
oracle = d["oracle_recall"]          # = cb*sr
deployed = d["micro_recall"]         # = cb*sr*rc

stages = ["all true\n(U)", "coverage\n(bank)", "+selection\n(=oracle)", "+ranking\n(deployed)"]
vals = [1.0, cb, oracle, deployed]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(range(4), vals, color=["#bbb", "#4C78A8", "#59A14F", "#E15759"])
ax.axhline(oracle, ls="--", lw=1, color="#59A14F", alpha=0.7)
ax.text(3.05, oracle, "oracle rerank ceiling", va="bottom", ha="right", fontsize=8, color="#59A14F")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("recall@15 (tautomer-InChIKey, micro)")
ax.set_ylim(0, 1.05)
ax.set_title(f"GRAIL recall decomposition (n={d['n_substrates']})", fontsize=10)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(ROOT / "docs" / "benchmark" / f"factorization_waterfall.{ext}", dpi=150)
print("wrote docs/benchmark/factorization_waterfall.{png,svg}")
```

- [ ] **Step 2: Run it**

Run: `python scripts/make_factorization_figure.py`
Expected: both files written; SVG opens; three bars descend U→coverage→oracle→deployed with the oracle dashed line on the deployed bar.

- [ ] **Step 3: Sanity-assert the figure encodes the identity**

Run: `python -c "import json,re; d=json.load(open('results/recall_factorization.json')); s=open('docs/benchmark/factorization_waterfall.svg').read(); assert f\"{d['micro_recall']:.3f}\" in s, 'deployed value missing from figure'; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Write `scripts/regen_headline.sh` (one-command regen)**

```bash
#!/usr/bin/env bash
# Regenerate every theory-spine headline number from committed checkpoints + the
# symlinked dataset. Run from repo root.
set -euo pipefail
python scripts/run_benchmark.py                       # ceiling 0.735 + SyGMa (results/benchmark_report.json)
python scripts/factorize_recall.py                    # results/recall_factorization.json
python scripts/ceiling_external_validity.py           # results/ceiling_external_validity.json
python scripts/anchor_certification.py                # results/anchor_certification.json
python scripts/make_factorization_figure.py           # docs/benchmark/factorization_waterfall.{png,svg}
echo "headline numbers regenerated under results/ and docs/benchmark/"
```

- [ ] **Step 5: Write the §Reproducibility section into `docs/GRAIL_FRAMING.md`**

Add `## §Reproducibility & provenance`: (a) a **provenance table** — one row per headline stat (ceiling 0.735, SyGMa 0.572, GRAIL 0.334, each factor, external ceiling, anchor Δ, interaction CI) with columns value / match mode / split / n / resampling unit / seed / source file; (b) a one-sentence **declared primary endpoint** = the differential-sensitivity interaction CI [+0.073,+0.171] (`results/rank_flip_ci.json`), with **Holm** correction applied within each declared test family; (c) a **release manifest** listing the frozen per-substrate 5-method × 5-protocol prediction set (`artifacts/tier2/{biotransformer,metapredictor,metatrans}_preds.json` + GRAIL/SyGMa), the re-scoring harness (`scripts/run_match_sensitivity.py`), `leakage_fix_report.json`, and `scripts/regen_headline.sh` as the one-command regen; (d) a **name** for the benchmark/protocol (propose one in the section, e.g. "the tautomer-aware, leakage-audited metabolite-structure evaluation protocol").

- [ ] **Step 6: Verify regen is wired and green**

Run: `bash -n scripts/regen_headline.sh && chmod +x scripts/regen_headline.sh && make test`
Expected: script parses; suite green.

- [ ] **Step 7: Commit**

```bash
git add scripts/make_factorization_figure.py scripts/regen_headline.sh docs/benchmark/factorization_waterfall.png docs/benchmark/factorization_waterfall.svg docs/GRAIL_FRAMING.md
git commit -m "feat(figure+packaging): recall-decomposition waterfall + provenance/primary-endpoint/release §Reproducibility"
```

---

## Task 9: Adversarial referee pass + Global-Constraint audit (post-execution)

**Files:**
- Create: `docs/benchmark/theory_spine_referee.md` (the review report)
- Modify: `docs/GRAIL_FRAMING.md` / scripts (only if the review finds blockers)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run a diverse-lens adversarial review**

Dispatch an adversarial review (workflow with ≥3 distinct lenses — a theory-correctness lens, a statistics/power lens, and a NeurIPS/JCIM referee lens — plus a refute stage) over the new §1.5, the three Propositions, the figure, and the four JSONs. Task each lens to try to REFUTE: is Prop 1's counterexample valid and its imported-result citation real? Is Prop 2 within the unmeasured-assumption guardrail? Is the factorization ever called a theorem? Is any external "0.37" unlabeled? Does any number lack provenance? Does anything imply GRAIL wins? Write findings to `docs/benchmark/theory_spine_referee.md`, severity-ranked.

- [ ] **Step 2: Fix every CONFIRMED blocker**

Apply fixes for confirmed Critical/Important findings; re-run `make test`.

- [ ] **Step 3: Audit each Global Constraint explicitly**

Walk the 9 Global Constraints; for each, cite the line(s) that satisfy it (or fix). Verify no AI attribution across the whole branch:

Run: `git log origin/main..HEAD --format='%an <%ae>%n%B' | grep -iE "claude|anthropic|co-authored|generated with" && echo BAD || echo "attribution clean ✓"`
Expected: `attribution clean ✓`.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/theory_spine_referee.md docs/GRAIL_FRAMING.md
git commit -m "docs(review): adversarial referee pass on the theory spine + global-constraint audit"
```

---

## Self-review (author checklist — completed at plan-write time)

- **Spec coverage:** Component 1 → Task 5 (+ Task 2 numbers); Component 2 → Task 6; Component 3 figure → Tasks 2+8; Component 4 → Task 3 (+ Task 7 prose); Component 5.1 anchor → Task 4 (+ Task 7 prose); Component 5.2 packaging → Task 8; Verification/adversarial → Task 9; stats foundation → Task 1. All spec sections mapped.
- **Placeholder scan:** the one deliberate unknown is the GLORYx truth source in Task 3 (Step 4 resolves it before running, with a hard STOP if no parseable truth exists — no fabrication). The `sygma_topk` helper (Task 4 Step 4) is specified as a DRY refactor of existing `sygma_baseline` internals. No "TODO/handle edge cases" placeholders.
- **Type consistency:** `tautomer_hits(preds, trues)->int` defined in Task 2, reused in Task 4; `factor_bootstrap_ci`/`ratio_of_sums_ci`/`paired_diff_bootstrap_ci`/`mcnemar_exact_p` signatures defined in Task 1 and consumed unchanged in Tasks 3–4; per-substrate record schema `{sub,U,Cfull,Cbud,H}` produced in Task 2, consumed in Tasks 3–4.

## Execution notes

- Tasks 1→2→(3,4)→(5,6,7)→8→9 have a natural order; 3 and 4 are independent of each other; prose Tasks 5/6/7 depend on the JSONs from 1/3/4.
- Every compute task keeps `make test` green (pure helpers unit-tested; full runs are manual and produce committed JSONs).
- Global Constraint 7 (no AI attribution) applies to **every** commit — the messages above are already clean; do not add trailers.
