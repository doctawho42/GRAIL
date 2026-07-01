# Stage 2b design: a multi-step Set-GFlowNet over the rule forest

**Status:** design (brainstorming output) · **Date:** 2026-07-01 · **Author:** GRAIL team
**Supersedes:** §4 of `2026-06-29-rule-env-reranker-setgflownet-design.md` (updated for Stage-2a findings).

## 1. Goal and evidence

Stage 2a landed a rule-preserving, no-MCS listwise reranker: clean-test recall@15 **0.500 ± 0.015**
(3 seeds, full test n=1170; val 0.507 ≈ test 0.500, no overfit), = 90% of SyGMa (0.558), +15.7% over
the generator; external GLORYx-37 0.351 (+44%). Three Stage-2a findings reshape the GFlowNet design:

- **M0 decomposition:** 96% of the rerank headroom is **cross-rule** (which rule fires), only 4% is
  **within-rule** (site). ⇒ actions are at the **rule/product level**, *not* `(rule×site)`-factored
  (the old §4 premise); site factoring would blow up the action space for a 4% slice.
- **The reranker works and `rule_prior[rule_id]` is the load-bearing feature** (ablation −0.017 vs
  neutral gen-score). ⇒ the reranker is the natural **forward policy** `P_F` ("reranker = policy").
- **The residual gap is coverage (multi-step).** Single-step oracle caps at 0.677 (clean) / 0.499
  (GLORYx); GLORYx references include multi-generation metabolites a single-step pool cannot reach.
  A depth-2 *ceiling* probe on the clean test gave **0 lift** — so any recall win from depth must be
  shown on an **external / multi-generation** set, not the clean test.

**Positioning (decided): hybrid.** Build a multi-step **forest** with a **set-coverage** reward.
Lead with the **method claim** (diversity/coverage advantage at matched budget over reranker-topk /
temperature-sampling / beam — robust to whether recall rises), and **add** a multi-step **recall**
story where it applies (external multi-generation). One architecture, two claims.

## 2. Success criteria

- **Method claim (primary, main-track):** at a matched output budget K, the Set-GFlowNet's sampled
  sets achieve **recall/coverage parity** with the reranker top-K *and* a measurable **diversity /
  modes-discovered advantage** over reranker-topk, temperature-sampling, and deterministic beam.
- **Applied claim (secondary):** on an external multi-generation set, forest depth ≥2 reaches
  annotated metabolites the single-step pool cannot — recall **> the single-step oracle (0.499 on
  GLORYx)**. Scoped to external; **not** claimed on the clean test (depth-2 probe = 0 lift there).
- **Discipline:** select `β`, `λ`, K, D on **val**; touch test once; mean±std over ≥2 seeds
  (`aggregate_seeds.py`); tautomer-InChIKey matching; rule environment preserved (every emitted node
  is a valid RDKit rule product).
- **Non-goals:** beating every method on every protocol; leaving the rule env; a recall lift on the
  clean test. If M0 (below) shows ~0 multi-step chains everywhere, the design **falls back to
  diversity-only** — still a valid method contribution.

## 3. Formulation — the Set-GFlowNet

### 3.1 Environment (state / action / terminal)
- **State `F_t`** — a partial **forest**: the substrate as root plus already-emitted metabolite
  nodes, each with a parent pointer to the molecule it was derived from. The **frontier** = forest
  molecules still expandable (depth < D, set size < K). Reuse `model/multistep.py:MetabolicTree` for
  the forest env (it already builds root→child rule expansions and a deterministic beam baseline).
- **Action** at `F_t`: either
  - `ADD(p, c)` — pick a frontier molecule `p` and one of its rule-children `c` (generator + RDKit
    enumerate children, InChIKey-deduped, budget-capped), append `c` as a new node with parent `p`;
  - `STOP` — terminate.
- **Terminal object** = the InChIKey set `S` = all non-root nodes. The reward depends **only** on `S`
  (not on ordering or parent structure).
- State identity = the forest's node-InChIKeys + parent edges; InChIKey-canonical states make "same
  molecule / same forest" well-defined for the DAG.

### 3.2 Forward policy `P_F` (the reranker, local scoring)
`P_F(action | F_t)` = softmax over `[ reranker(p, c) for every candidate (p, c) ] ∪ [ stop_logit ]`.
The reranker scores the **immediate transformation** `(parent → child)`: at depth-1 this is exactly
`(substrate, product)`, its training distribution; global coverage is handled by the reward, not the
policy. `stop_logit` is a **new STOP head** — a small MLP over a pooled representation of the current
forest/frontier (stop when the set is "good enough"). The reranker's `rule_prior[rule_id]` scalar
feature (load-bearing per Stage-2a) is retained. `logZ` is a learned scalar (reuse `gflownet.log_z`).

### 3.3 Reward (PU-aware set-coverage)
`R(S) = exp(β · (TP(S) − λ·|S|))`, `TP(S) = |{ c ∈ S : InChIKey(c) ∈ annotations(root) }|`.
**PU-correct:** non-annotated members contribute 0 to `TP` and cost only `λ` (the size penalty) —
they are *unlabeled*, never scored as false negatives. `β` (reward temperature) and `λ` (size
penalty) are val-selected; hard caps `|S| ≤ K` and `depth ≤ D` live in the environment so the reward
need not solely bound size.

### 3.4 Backward policy + Trajectory-Balance loss (leaf-analytic)
To reach a forest `F_t`, the last-added node must have been a **leaf** of `F_t`; so the analytic
set-DAG backward is `P_B(remove leaf) = 1 / (#leaves of F_t)`. For a single-step set (all nodes are
leaves) this reduces to `1/|S_t|` (the old §4 form) — a special case. Loss:
`L = ( logZ + Σ_t log P_F(a_t|F_t) − Σ_t log P_B(·|F_{t+1}) − log R(S) )²`.
Variance-reduction fallback if plain TB is noisy on the set-DAG: **sub-trajectory balance (SubTB)**.

### 3.5 Intermediate-node bootstrap (the key risk mitigation)
The reranker is trained on `(substrate, product)`; interior forest nodes are `(intermediate,
product)` it has not seen. Mitigation: build **depth-1 ∪ depth-2** training pairs — for annotated
metabolites unreachable in one step but reachable as `root → m1 → m2` in the rule env, add `(m1, m2)`
positives with `m1`'s other rule-children as negatives — and fine-tune the reranker so
`reranker(intermediate, ·)` is calibrated. The count of such chains is measured first (M0).

## 4. Evaluation (dual matrix)

All baselines evaluated at a **matched output budget K**; tautomer-InChIKey matching.

**Method claim (clean test + external).** Baselines: reranker top-K, temperature-sampling
(reranker / generator), deterministic beam (`multistep.py`). For **recall@K** the GFlowNet output is
its **single highest-reward sampled set** truncated to K (matched to reranker top-K); **diversity /
modes** are measured over **M independent samples**. Metrics:
- recall@K / set-coverage — **parity** target (GFlowNet no worse);
- diversity — #distinct InChIKey modes across sampled sets, mean pairwise Tanimoto, #unique
  scaffolds;
- **modes discovered** — over M sampled sets, #distinct high-reward metabolites found (the GFlowNet
  signature advantage vs argmax/beam);
- set-size calibration — does `|S|` track the true annotation count.

**Applied claim (external multi-generation).** Recall@k where the reference includes multi-generation
metabolites; show forest depth ≥2 exceeds the single-step oracle (0.499 on GLORYx).

## 5. Milestones / compute (validate-small → scale-up, Colab Pro+)
- **M0 (de-risk, first, CPU, cheap):** the **depth-2 reachability census** — how many annotated
  metabolites are `root→m1→m2`-reachable in the rule env but not depth-1, on clean test vs GLORYx.
  **Go/no-go for the recall claim**: ~0 everywhere ⇒ diversity-only fallback. `scripts/` census.
- **M1 (small GPU):** Set-GFlowNet on ~200–500 substrates, depth ≤2, small K — TB converges,
  leaf-`P_B` correct, sampled-set reward > random, diversity > beam. Machinery sanity.
- **M2 (scale-up, Colab GPU):** full run + the eval matrix; headline diversity/coverage advantage +
  recall parity (clean) + recall lift (external multi-gen). Seeds, val-selection.
RDKit rule application is CPU-bound → precompute + cache candidate pools per state (as in Stage 2a).

## 6. Components: reused vs new
- **Reused:** `gflownet.py:GFlowNetTrainer` (TB loop, `logZ`, `sample_trajectory` — generalize
  single-terminal → forest/set), `model/multistep.py:MetabolicTree` (forest env + beam baseline),
  the Stage-2a reranker (as `P_F`), generator + RDKit child enumeration (the env), `from_rdmol` /
  reranker featurization, tautomer-InChIKey metrics, PU / `MolFrame.negs`, `EnsembleWorkflow.
  run_bundle` chokepoint, `aggregate_seeds.py`.
- **New:** forest state + set terminal + leaf-`P_B` (extend `gflownet.py`); set-coverage reward
  (extend `annotation_reward_fn`); STOP head on the reranker; intermediate-bootstrap pairs + reranker
  fine-tune; diversity / coverage / modes eval harness; `scripts/run_gflownet.py`; the depth-2 census
  script; Colab notebook; guard tests (leaf-`P_B` correctness on a toy forest, set-coverage reward,
  PU treatment, depth-2 census correctness, TB gradient flow through the reranker `P_F`).

## 7. Risks and mitigations
1. **Intermediate-node generalization** — reranker unseen on `(intermediate, product)`. Mitigate: the
   depth-1∪2 bootstrap (§3.5); M0 census tells us whether it matters at all before we build.
2. **Multi-parent over-counting (v1 simplification)** — a metabolite reachable from several
   intermediates: v1 takes state = forest (with edges), reward = f(InChIKey-set), so sets reachable
   by many forests are sampled slightly more often (a known set-GFlowNet bias). Accept in v1;
   correct later by dividing by the #forests per set.
3. **TB variance on the set-DAG** — start with small K/D; SubTB fallback; leaf-`P_B` is analytic (no
   estimation).
4. **No multi-step recall lift on the clean test** — expected (depth-2 probe = 0 there); the recall
   claim is scoped to external multi-gen. If M0 shows ~0 chains even externally, drop the recall
   claim and keep the diversity-only method contribution (still main-track).
5. **Compute** — GFlowNet training is sample-heavy; cache pools per state as in Stage 2a; validate
   small (M0/M1) before the M2 scale-up.

## 8. Honest framing
Stage 1 (benchmark/protocol) is the standalone A*/Q1 result; Stage 2a is the diagnosis-plus-fix
(rank-by-rule is provably tied, a reranker recovers curated-probability recall from 7,581 mined
rules). Stage 2b is the **novel method**: a Set-GFlowNet whose terminal object is a *set/forest* of
metabolites with a *set-level* reward — beyond single-terminal GFlowNets and beyond argmax/beam —
demonstrating a diversity/coverage advantage pointwise rerankers cannot match, positioned vs
RGFN / SynFlowNet / RxnFlow / RetroGFN / Deleu-2022. Recall claims are bounded by the measured oracle
ceilings and scoped (parity in-distribution, lift only on external multi-generation). The design has
no failure branch: worst case (no multi-step chains) still yields the diversity method contribution.
