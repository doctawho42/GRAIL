# GRAIL — consolidated findings

Status as of 2026-06-28. Honest summary of where GRAIL (rule-based + learned metabolite
structure predictor) stands against the SOTA goal, the methodology used, and what was
learned. Numbers are reproducible from the committed code; artifacts are gitignored.

## Goal & bar

Beat the rule-based SOTA for drug/xenobiotic metabolite **structure** prediction. Primary
comparator: **SyGMa** on the same clean test split — recall@5/10/12/15 = **0.470 / 0.531 /
0.543 / 0.558**, precision@5 ≈ 0.175, ~81 products/substrate (InChIKey matching).

## Methodology (how we measure — non-negotiable)

- **Match by tautomer-canonical InChIKey** (`EvaluationConfig.match="inchikey_tautomer"`):
  the rule engine emits a different tautomer of the reference than standard InChI
  normalizes, so plain InChIKey under-counts true hits.
- **Lead with recall@k + mean_output_size**; precision is a pessimistic lower bound under
  incomplete annotation.
- **Molecule-disjoint clean splits**; select on val, touch test once. Reported numbers have
  **val ≈ test** (e.g. 0.327 vs 0.330 @15) → trustworthy, not overfit.
- **Rule-bank recall ceiling = 0.718** (measured, InChIKey, 7581 rules). A rule-based method
  cannot exceed it. SyGMa (0.558) and GRAIL (0.33–0.36) are both well under it.

## What was shipped (correctness + speed)

- **Ranking:** rank-only ensemble policy (rank by `filter×generator`, take top-k — the hard
  filter gate hurt recall@k); tautomer-invariant matching; output dedup by the metric's
  InChIKey key so `max_output` holds structure-distinct molecules.
- **SoM regioselectivity prior** (`model/som.py`): per-atom site model (val AUC 0.74),
  per-product reweight `filter×generator×som^β`. Opt-in; β=0 is a no-op.
- **Performance:** rule-bank encoding cache (`score_rules` 1.06s→0.05s, eval-mode
  deterministic); **filter candidate cap** (the pair-filter's per-candidate MCS was the real
  eval bottleneck — up to 32s/substrate); **single-mode filter** (no MCS: filter_train
  611s→20s). Net: a 245-substrate eval dropped ~80min → ~12min.
- **Bug fixes:** canonical-for-labeling tautomer bottleneck; empirical rule priors
  (`rule_prior_logits`) were `persistent=False` and silently dropped on checkpoint save
  (reloaded models lost ~0.03 recall@15) — now persisted.

## Headline results — recall@15 (tautomer InChIKey, rank-only)

| Train substrates | gen-only | ensemble | val | vs SyGMa 0.558 |
|---|---|---|---|---|
| 400 | ~0.10 | ~0.10 | — | ~18% |
| 2418 | 0.331 | 0.330 | 0.327 | 59% |
| 4787 | — | **0.334** | — | 60% |

Filter: MCC 0.30→0.32, ROC-AUC 0.80→0.81 across scales.

## Findings

1. **Data was the dominant gap up to ~2400 substrates, then it SATURATED.** 400→2418 (6×)
   tripled recall (0.10→0.33); 2418→4787 (2× more) gave +0.004 = noise. The remaining gap to
   SyGMa is **not** a data problem. (Caveat: samples not nested — `np.choice` — but identical
   recall at 2× size is an unambiguous plateau.)

2. **Not a coverage problem either.** Rule-bank ceiling 0.718 ≫ both methods. GRAIL converts
   only ~0.33/0.718 of reachable metabolites into top-15; SyGMa converts 0.558 of its
   coverage. **SyGMa's ranking/selection beats GRAIL's learned generator** — the gap is
   *ranking quality*, a method ceiling.

3. **The learned filter is a good CLASSIFIER but a neutral RANKER** — robust across all
   scales (ensemble ≈ generator-only despite ROC-AUC 0.81). The generator's own ranking
   already subsumes it.

4. **SoM regioselectivity prior is ~neutral** on these models (helped a noisy +7% only in the
   earlier stochastic-dropout regime; vanished under deterministic eval).

5. **Process lessons:** long CPU runs on a laptop stall on idle-sleep (~98% of wall-clock) —
   `caffeinate -i -s` + lid-open + plugged is required. Reaction labeling and single-graph
   build are the serial bottlenecks for fresh runs (cached/resumable). `np.choice(replace=
   False)` is not prefix-nested, so a larger subsample can't reuse a smaller one's cache.

## Honest assessment

**SOTA (beating SyGMa) was NOT achieved.** GRAIL plateaus at **~0.33–0.36 recall@15 ≈ 60–65%
of SyGMa's 0.558**, data-saturated, against a ranking/method ceiling. The result is clean and
reproducible (clean splits, val≈test), but the headline claim is not met on this comparator.

## Open levers (methodological, not data)

- **Better ranking toward SyGMa's empirical probabilities** — the generator now persists
  empirical rule priors (`rule_prior_logits`); test whether weighting them more
  (`prior_strength`) on a freshly-trained model closes part of the gap.
- **Multi-step generation** (built, currently inert) for single-step-unreachable metabolites
  — part of the 0.33→0.72 headroom; earlier ceiling probe was ambiguous.
- **Generator capacity/loss redesign** — investigate why it caps at 0.33/0.718.
