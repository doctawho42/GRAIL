# Set-GFlowNet parallel cache pre-warm — design

**Date:** 2026-07-03
**Status:** approved (brainstorming)
**Related:** Stage-2b Set-GFlowNet (`grail_metabolism/model/set_gflownet.py`), task `task_283fb1c1`

## Motivation

The Stage-2b Set-GFlowNet's cold "epoch 1" is the dominant cost of every M2/headline run.
`SetGFlowNetTrainer.fit()` runs its forest-rollout loop **serially** (one substrate at a
time, single process); each never-seen state calls `candidate_children`, which does RDKit
rule application (top_k SMIRKS) + `from_rdmol` featurization + tautomer-InChIKey
canonicalization. On the M2-scale run (1200 substrates, top_k=50, depth-2) this cold epoch
took **>4h and never completed**, on both GPU and CPU — because the wall is serial
CPU-bound RDKit, not torch (GPU util ~0%).

Meanwhile the **reranker pool-gen** (`grail_metabolism/workflows/reranker.py`) already
parallelizes the *identical* `generator.generate_scored_with_details` work across an
8-worker spawn pool, doing 1200 substrates in ~62 min. That asymmetry is the ~8× left on
the table — and the concurrency gotchas (spawn-not-fork, /dev/shm, plain-data returns,
worker cap) are **already solved** in reranker.py. This is a mirror job, not a from-scratch
parallelization.

## Goal

Populate the Set-GFlowNet's deterministic `(state,top_k)→children` and
`SMILES→tautomer-InChIKey` caches **in parallel** (~8×) before the serial `fit()` and
before eval, so both the training epoch-1 grind AND eval's forest rollouts run
cache-warm (torch-only → fast). **The learned result must be byte-identical to the
serial path** — this is a pure speedup of a deterministic cache build.

## Approach: two-wave parallel pre-warm

The forest env has `max_depth=2`, so `candidate_children` is only ever called on states at
depth 0 (roots) and depth 1 (their children); depth-2 nodes are terminal. So the full set
of states the fit/eval will expand = `{roots} ∪ {depth-1 children of roots}`. Warm them in
two waves:

- **Wave 1** — fan all substrate **roots** out to the spawn pool; each worker runs
  `generate_scored_with_details(root, top_k, compute_sites=False)` → returns
  `(root_smiles, [(child_smiles, gscore, rid), …])`. Collect the union of depth-1 child
  SMILES.
- **Wave 2** — fan those **depth-1 children** out the same way → depth-2 grandchildren.
- Also warm `_ik_cache` for every SMILES touched (root + all children), via the same
  workers (tautomer-IK is deterministic and independently parallelizable).
- Merge all worker results into `self._child_cache` / `self._ik_cache`, then
  `self.save_caches()`.

After pre-warm, the existing serial `fit()`/`evaluate_matrix` run unchanged — every
`candidate_children`/`ik` call is a cache hit.

Rejected alternatives: (2) on-demand parallel batching *inside* fit — dynamic, hard to keep
deterministic, more coordination, no extra payoff; (3) a standalone offline prewarm script —
adds a manual step and doesn't ride run_gflownet's flow.

## Components

### 1. `SetGFlowNetTrainer.prewarm_caches(root_smiles, workers, gen_ckpt)`
New method on the trainer. Mirrors reranker.py's parallel pool builder (~`reranker.py:582`,
`_bi_worker_init`/`_bi_pool_worker`, `_MAX_SPAWN_WORKERS=8`):
- Warm the generator's rule-bank embeddings ONCE in the parent
  (`generator._rule_embeddings(cpu).detach().cpu().contiguous()`), inject into workers so
  each spawn worker SKIPS the ~80s 7581-rule GNN encoding.
- `mp.get_context("spawn").Pool(processes=min(workers, cpu_count, 8), initializer=…)`.
- Worker initializer loads the generator ONCE from `gen_ckpt`, pins torch to 1 thread,
  silences RDKit, sets `generator._rule_embedding_cache = rule_emb`.
- Worker fn takes a state SMILES, returns a **plain tuple** `(state_smiles, children_list,
  {smiles: ik})` where `children_list = [(smiles, float(gscore), int(rid))]` — matching
  `candidate_children`'s stored shape EXACTLY. **No tensors cross the process boundary**
  (avoids the torch shared-mem mmap ENOMEM).
- `pool.imap_unordered(worker, states, chunksize=…)`; merge into the caches.
- Guard: `workers<=1` or no `gen_ckpt` → no-op (serial path handles warming lazily).

### 2. Wiring in `scripts/run_gflownet.py`
Reuse the existing `--workers` flag (already threaded through for the reranker pool-gen and
already has `GEN_CKPT`):
- Before `trainer.fit(...)`: `if args.workers > 1: trainer.prewarm_caches(train_roots,
  args.workers, str(GEN_CKPT))`.
- Before `evaluate_matrix(...)`: same on the eval-test roots (the `eval_bundle.map` keys,
  capped to the eval count).
- No new flag. `--workers 1` = today's exact serial behaviour (safe default/fallback).

### 3. Correctness guard — `grail_metabolism/tests/test_set_gflownet.py`
New test: on a tiny substrate set (a handful, top_k small), build the caches two ways —
(a) serial (repeated `candidate_children` calls), (b) `prewarm_caches(workers=2)` — and
assert:
- `_child_cache` and `_ik_cache` are **equal as dicts** (same keys, same value lists/order
  where the serial path is order-deterministic).
- A fixed-seed `sample_forest(root)` returns the **identical** forest under both.
Runs locally (macOS spawn Pool works) → no Modal cost. Keeps `make test` green.

## Data flow

```
train_roots ─┐
             ├─► prewarm_caches(workers=8, gen_ckpt)
eval_roots ──┘        │  wave1: roots      → spawn pool → {root: children}, {smiles: ik}
                      │  wave2: depth-1     → spawn pool → {child: grandchildren}, …
                      ▼
        merge → self._child_cache / self._ik_cache → save_caches() (Volume pickle)
                      ▼
        serial fit() / evaluate_matrix()  — all candidate_children/ik = cache HIT
```

## Invariants / constraints (do not regress)

- **Determinism:** `(state,top_k)→children` and `SMILES→tautomer-IK` are deterministic. The
  parallel build MUST produce caches identical to serial. `candidate_children` stores
  DETACHED `(smiles, float(gscore), int(rid))` — keep that (no grad tensors in the cache).
- **Spawn, not fork** (fork+torch deadlocks on Linux); worker cap 8 (`_MAX_SPAWN_WORKERS`,
  /dev/shm mmap); workers return plain python data, never tensors; `imap_unordered` (no
  head-of-line stall on big-drug substrates). Mirror reranker.py — do not reinvent.
- **RNG:** pre-warm is pure cache population (no sampling), so it must not touch the RNG
  stream that drives the serial rollout sampling. `seed_everything` is applied in
  `run_bundle`/run_gflownet as today.
- **Persistence:** merged caches persist via the existing `child_cache_path`/`ik_cache_path`
  pickles — the whole point is cross-run reuse.

## Non-goals (v1)

- NOT the learned NN reaction-surrogate (separate, riskier Stage-3 direction — see memory
  `grail-stage3-surrogate-idea`). This is the exact, deterministic parallelization only.
- NOT parallelizing eval's `_reranker_topk_smiles` pool or `beam_search` baselines — those
  are separate, smaller per-substrate cold paths. Parallelize later only if they prove to
  matter after the forest path is warm.
- NOT parallelizing the `fit()` torch/autograd loop — torch isn't the wall.

## Verification

- `make test` green; the serial-vs-parallel cache-equality + identical-forest guard passes
  locally (Mac spawn).
- A tiny Modal validation run (≈20 substrates, `--workers 8`) confirms the spawn pool works
  at Modal/Linux scale without shm/deadlock issues (~cents).
- Then a full 1200-scale M2 run reuses the machinery: report the measured epoch-1 wall time
  before (serial, >4h) vs after (parallel pre-warm) as the speedup evidence.

## Cost / sequencing

Bulk of the work is local (Mac) and free: the method + worker fns + the correctness test.
Only the tiny Modal validation + the eventual full run touch credits (~$2-4 total),
fitting the remaining ~$15. The cheap 300-scale CPU run stays running as a safety-net
result in the meantime.
