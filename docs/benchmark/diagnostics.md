# Benchmark diagnostics — what actually limits metabolite structure prediction

Stage-1 analysis section (assembled from existing measurements; no new runs). All GRAIL
numbers are recall@k with tautomer-canonical InChIKey matching, rank-only, on the
molecule-disjoint clean split (val ≈ test).

## 1. Coverage vs. ranking decomposition (the key figure)

A rule-based method loses true metabolites two ways: **coverage** (no rule reaches the
structure, single-step) and **ranking** (a rule reaches it, but it isn't in the top-k
output). Decomposing GRAIL's recall@15 ≈ 0.334:

| component | fraction of true metabolites | what it is |
|---|---|---|
| **captured** (recall@15) | **0.334** | reachable *and* surfaced in top-15 |
| **ranking loss** | ~0.384 | reachable single-step, but ranked below top-15 |
| **coverage loss** | ~0.282 | unreachable by any single-step rule |

Derivation: rule-bank single-step **recall ceiling = 0.718** (measured, 7581 rules,
InChIKey; 1865/2597 test metabolites reachable, 85.5% of substrates with ≥1 reachable).
GRAIL converts only **0.334 / 0.718 ≈ 47%** of the *reachable* set into its top-15 — so the
larger loss is **ranking (~0.384), not coverage (~0.282)**.

> Caveat: the 0.718 ceiling was measured on the full 1170-substrate test; the 0.334 on a
> 300-substrate clean-test subset. The exact same-set decomposition is a pending (deferred)
> run; the qualitative conclusion (ranking ≳ coverage as the dominant gap) is robust to it.

**Implication & contrast with SyGMa.** SyGMa has a *smaller* rule bank (lower ceiling) yet
reaches recall@15 = 0.558 — it converts a *higher* fraction of its reachable set into top-k.
So the gap between GRAIL (0.33) and SyGMa (0.56) is **ranking/selection quality**, not
coverage. This is the paper's central diagnostic claim and motivates Stage 2 (a better
generative ranker).

## 2. Data-scaling curve (ranking loss does not close with data)

GRAIL ensemble recall@15 (tautomer), single-mode filter, vs train-set size:

| train substrates | recall@15 | vs SyGMa 0.558 | note |
|---|---|---|---|
| 400 | ~0.10 | 18% | severely undertrained |
| 2418 | 0.330 | 59% | val 0.327 (≈ test) |
| 4787 | 0.334 | 60% | +0.004 over 2418 = noise |

400→2418 (6× data) tripled recall; 2418→4787 (2× more) was flat. **The ranking loss is a
model/method ceiling, not a data ceiling** — more data is exhausted as a lever. (Samples are
not nested — `np.choice` is not prefix-stable — but identical recall at 2× size is an
unambiguous plateau.)

## 3. Component analysis: the learned filter is a good classifier but a neutral ranker

Across all three scales, the learned pair/single filter reaches ROC-AUC ≈ 0.80 / MCC ≈ 0.30,
yet the ensemble (filter × generator) ≈ generator-only at every k. The generator's own
ranking already subsumes what the filter would re-rank. (Earlier apparent filter/SoM lifts
were artifacts of a stochastic-dropout inference regime; they vanish under deterministic
eval — a methodology lesson in itself.)

## 4. Decomposition of the unreachable (coverage-loss) tail

From `run_benchmark.py --gap-analysis` (300-substrate sample) on the metabolites no single
rule reaches: **phase-I transformations dominate** (hydroxylation, reduction, di-oxidation,
methylation, dihydrodiol, desaturation); **phase-II conjugations are only ~6%**; and **~50%
is a long tail of unique mass-shifts** whose inspected causes are (a) **regioselectivity**
(right reaction, wrong site → InChIKey miss), (b) genuinely **multi-step** metabolites
(e.g. oxidation→conjugation), and (c) N-oxides / unusual tautomers. Hand-adding phase-II
rules gives **zero** ceiling lift on this benchmark (the mined bank already subsumes them).

## 5. What this says for the benchmark and the method

- The headline number is dominated by **ranking**, and **how you match** changes it (§
  match-sensitivity) — both are protocol/evaluation effects the field handles
  inconsistently. That is the benchmark's contribution.
- The next *method* lever is a better ranker, not more data or more rules. Two concrete
  bets: (a) **empirical rule priors** (SyGMa's strength; GRAIL has them but under-weights /
  recently un-persisted them — pending experiment) and (b) **multi-step / set-level
  generation** (Stage 2 GFlowNet) for the reachable-but-mis-ranked and the genuinely
  multi-step tail.

## Sources (existing artifacts, no new compute)

- Ceiling & gap: `results/` benchmark reports (`run_benchmark.py`), `docs/FINDINGS.md`.
- Scaling: `artifacts/{subset_train_v, full2500_single, full5000_single}/reports/metrics.json`.
- Match-sensitivity engine + first table: `scripts/run_match_sensitivity.py`.
