# Set-GFlowNet Parallel Cache Pre-warm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the Set-GFlowNet's deterministic `(state,top_k)→children` and `SMILES→tautomer-InChIKey` caches in parallel (~8×) before the serial `fit()`/eval, so the cold epoch-1 grind and eval forest rollouts run cache-warm.

**Architecture:** A new `SetGFlowNetTrainer.prewarm_caches(roots, workers, gen_ckpt)` method fans `candidate_children` expansion across an 8-worker `spawn` pool in two waves (roots → depth-1 children), mirroring the proven pattern in `grail_metabolism/workflows/reranker.py`. The serial `fit()`/`evaluate_matrix` are unchanged and just run warm. `--workers 1` keeps today's exact serial behaviour.

**Tech Stack:** Python, `multiprocessing` (spawn context), RDKit, torch/torch-geometric, pytest.

## Global Constraints

- `numpy<2` (RDKit/torch-geometric stack).
- Parallelism over PROCESSES with the **spawn** start method (fork+torch deadlocks on Linux); worker count capped at `_MAX_SPAWN_WORKERS = 8` (/dev/shm mmap).
- Workers return **plain python data** (str/float/int/dict), NEVER torch tensors (shared-mem mmap ENOMEM).
- **Determinism:** the parallel-built caches MUST equal the serial build. `_child_cache` values stay detached `(smiles, float(gscore), int(rid))` tuples; no grad tensors in caches.
- Pre-warm must NOT consume RNG (it is pure cache population; `seed_everything` drives sampling as today).
- `make test` stays green. Do NOT add `Co-Authored-By` trailers to commits.

---

### Task 1: Extract shared `_expand_state` helper (pure refactor)

Factor the per-state expansion out of `candidate_children` into a module-level pure function so the parallel workers and `candidate_children` share ONE code path (guarantees parallel == serial by construction).

**Files:**
- Modify: `grail_metabolism/model/set_gflownet.py` (lines ~140-163, `candidate_children`)
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Produces: `_expand_state(generator, state_smiles: str, top_k: int) -> list[tuple[str, float, int]]` (module-level). `candidate_children` now delegates to it.

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_set_gflownet.py
def test_expand_state_matches_candidate_children(monkeypatch):
    from grail_metabolism.model import set_gflownet as sg

    class _StubGen:
        # returns fixed (smiles, gscore, rid, ...) rows; dupes + a bad smiles to exercise filtering
        def generate_scored_with_details(self, s, top_k, compute_sites=False):
            return [("CCO", 0.9, 3), ("CCO", 0.4, 3), ("bad_smiles", 0.5, 7), ("CCCO", 0.2, 9)]

    out = sg._expand_state(_StubGen(), "c1ccccc1", top_k=5)
    assert out == [("CCO", 0.9, 3), ("CCCO", 0.2, 9)]   # dedup + drop unparseable, order preserved
    assert all(isinstance(g, float) and isinstance(r, int) for _, g, r in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_expand_state_matches_candidate_children -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_expand_state'`

- [ ] **Step 3: Add the module-level helper and delegate from `candidate_children`**

```python
# module level in set_gflownet.py (near the top-level helpers)
def _expand_state(generator, state_smiles, top_k):
    """Deterministic top_k child expansion of one state: dedup by SMILES, drop unparseable,
    keep detached (smiles, float gscore, int rid). Shared by candidate_children AND the
    parallel pre-warm workers so the two paths are identical by construction."""
    seen, out = set(), []
    for smiles, gscore, rid, *_ in generator.generate_scored_with_details(
        state_smiles, top_k=top_k, compute_sites=False
    ):
        if smiles in seen:
            continue
        seen.add(smiles)
        if Chem.MolFromSmiles(smiles) is None:
            continue
        out.append((smiles, float(gscore), int(rid)))
    return out
```

Replace the body of `candidate_children`:

```python
    def candidate_children(self, state_smiles):
        if state_smiles not in self._child_cache:
            self._child_cache[state_smiles] = _expand_state(
                self.generator, state_smiles, self.config.top_k
            )
        return self._child_cache[state_smiles]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -q`
Expected: PASS (new test + all existing set_gflownet tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/tests/test_set_gflownet.py
git commit -m "refactor(gflownet): extract _expand_state shared by candidate_children + parallel prewarm"
```

---

### Task 2: `prewarm_caches` — two-wave parallel cache build

Add the pre-warm method with a serial path (`workers<=1`) and a spawn-parallel path (`workers>1`) mirroring reranker.py. The pytest validates the **orchestration + serial-equivalence** with a stub generator (dataset-free); the real spawn path is code-reviewed against reranker and validated on Modal in Task 4.

**Files:**
- Modify: `grail_metabolism/model/set_gflownet.py` (add `_gfn_worker_init`, `_gfn_pool_worker`, `SetGFlowNetTrainer._expand_many`, `SetGFlowNetTrainer.prewarm_caches`)
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Consumes: `_expand_state` (Task 1); `self._child_cache`, `self._ik_cache`, `self.save_caches()`, `self._ik_cache_path`, `self._child_cache_path`, `self.config.top_k`, `self.config.max_depth`, `self.generator`.
- Produces:
  - `SetGFlowNetTrainer._expand_many(states: list[str], workers: int, gen_ckpt: str | None) -> dict[str, list]` — expands the given states (skipping already-cached), merges children into `_child_cache` and iks into `_ik_cache`, returns `{state: children}` for the newly expanded states.
  - `SetGFlowNetTrainer.prewarm_caches(root_smiles: Iterable[str], workers: int, gen_ckpt: str | None = None, verbose: bool = False) -> None`.
  - `_gfn_worker_init(gen_ckpt: str, top_k: int, rule_emb=None) -> None`, `_gfn_pool_worker(state_smiles: str) -> tuple[str, list, dict]` (module-level; return = `(state, children, {smiles: ik})`).

- [ ] **Step 1: Write the failing test (orchestration + serial-equivalence, stub gen)**

```python
def test_prewarm_matches_serial(monkeypatch):
    from grail_metabolism.model import set_gflownet as sg

    # deterministic 1-level expansion: root -> two children, each child -> one grandchild
    KIDS = {
        "ROOT":  [("CCO", 0.9, 1), ("CCCO", 0.5, 2)],
        "CCO":   [("CCOC", 0.7, 3)],
        "CCCO":  [("CCCOC", 0.4, 4)],
    }
    class _StubGen:
        def generate_scored_with_details(self, s, top_k, compute_sites=False):
            return [(c, g, r) for (c, g, r) in KIDS.get(s, [])]
    # plain-identity tautomer-IK so no RDKit dependency in the test
    monkeypatch.setattr(sg, "_tautomer_inchikey", lambda s: f"IK::{s}")

    from grail_metabolism.config import GFlowNetConfig
    cfg = GFlowNetConfig(max_depth=2, top_k=5)
    def _mk():
        t = sg.SetGFlowNetTrainer(_StubGen(), reranker=_StubReranker(), config=cfg,
                                  annotated_ik_fn=lambda s: set(), device="cpu")
        return t

    # serial reference: lazily expand roots + their depth-1 children
    ser = _mk()
    for root in ("ROOT",):
        for c, _, _ in ser.candidate_children(root):
            ser.candidate_children(c)          # depth-1 children expanded (depth-2 are terminal)

    # prewarm path (workers=1 => in-method serial map over the same _expand_state)
    par = _mk()
    par.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None)

    assert par._child_cache == ser._child_cache      # identical (dict eq ignores insertion order)
    # ik cache covers every touched smiles (root + all children)
    for s in ("ROOT", "CCO", "CCCO", "CCOC", "CCCOC"):
        assert par._ik_cache[s] == f"IK::{s}"
```

(`_StubReranker` is a minimal `torch.nn.Module` stub already used by existing tests, or add a 1-line stub with a `.parameters()` returning an empty iterator and `.to()/.eval()` no-ops. Reuse whatever the current test module uses to build a trainer.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_prewarm_matches_serial -q`
Expected: FAIL — `AttributeError: 'SetGFlowNetTrainer' object has no attribute 'prewarm_caches'`

- [ ] **Step 3: Implement the workers + `_expand_many` + `prewarm_caches`**

```python
# module level, mirroring reranker.py's _bi_worker_init / _bi_pool_worker
import multiprocessing as _mp
_MAX_GFN_WORKERS = 8
_GFN_WORKER = {"gen": None, "top_k": None}

def _gfn_worker_init(gen_ckpt, top_k, rule_emb=None):
    import torch as _t
    _t.set_num_threads(1)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.model.grail import _read_checkpoint
    from grail_metabolism.workflows.factory import build_generator
    from grail_metabolism.config import GeneratorConfig
    state = _read_checkpoint(gen_ckpt)
    gen = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    gen.load_state_dict(state["state_dict"], strict=False)
    if rule_emb is not None:
        gen._rule_embedding_cache = rule_emb   # skip the ~80s 7581-rule GNN encoding
    _GFN_WORKER["gen"], _GFN_WORKER["top_k"] = gen, top_k

def _gfn_pool_worker(state_smiles):
    gen, top_k = _GFN_WORKER["gen"], _GFN_WORKER["top_k"]
    children = _expand_state(gen, state_smiles, top_k)               # [(smiles, float, int)]
    iks = {state_smiles: _tautomer_inchikey(state_smiles)}
    for c, _g, _rid in children:
        iks[c] = _tautomer_inchikey(c)
    return state_smiles, children, iks                              # PLAIN data only
```

```python
    # methods on SetGFlowNetTrainer
    def _expand_many(self, states, workers, gen_ckpt):
        todo = [s for s in dict.fromkeys(states) if s and s not in self._child_cache]
        if not todo:
            return {}
        workers = max(1, min(int(workers), os.cpu_count() or 1, _MAX_GFN_WORKERS))
        results = {}
        if workers <= 1 or not gen_ckpt:
            for s in todo:
                children = _expand_state(self.generator, s, self.config.top_k)
                iks = {s: _tautomer_inchikey(s)}
                for c, _g, _rid in children:
                    iks[c] = _tautomer_inchikey(c)
                results[s] = (children, iks)
        else:
            rule_emb = self.generator._rule_embeddings(torch.device("cpu")).detach().cpu().contiguous()
            pool = _mp.get_context("spawn").Pool(
                processes=workers, initializer=_gfn_worker_init,
                initargs=(gen_ckpt, self.config.top_k, rule_emb),
            )
            try:
                for s, children, iks in pool.imap_unordered(_gfn_pool_worker, todo, chunksize=2):
                    results[s] = (children, iks)
            finally:
                pool.close(); pool.join()
        for s, (children, iks) in results.items():
            self._child_cache[s] = children
            for k, v in iks.items():
                self._ik_cache.setdefault(k, v)
        return {s: children for s, (children, iks) in results.items()}

    def prewarm_caches(self, root_smiles, workers, gen_ckpt=None, verbose=False):
        """Populate _child_cache/_ik_cache for every state the depth-<=max_depth fit/eval will
        expand (roots + their depth-1 children), in parallel. Deterministic; identical to lazy
        serial candidate_children. Safe to call repeatedly (only expands uncached states)."""
        roots = list(dict.fromkeys(s for s in root_smiles if s))
        wave1 = self._expand_many(roots, workers, gen_ckpt)
        if verbose:
            print(f"[gflownet] prewarm wave1: {len(wave1)} roots expanded", flush=True)
        if int(getattr(self.config, "max_depth", 2)) >= 2:
            depth1 = list(dict.fromkeys(
                c for children in wave1.values() for c, _g, _rid in children
            ))
            wave2 = self._expand_many(depth1, workers, gen_ckpt)
            if verbose:
                print(f"[gflownet] prewarm wave2: {len(wave2)} depth-1 states expanded", flush=True)
        if self._child_cache_path or self._ik_cache_path:
            self.save_caches()
```

Add `import os` and `import multiprocessing as _mp` at the top if not present; `_tautomer_inchikey` is already imported in this module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -q`
Expected: PASS (all set_gflownet tests, including the new prewarm test).

- [ ] **Step 5: Full test suite green**

Run: `make test`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/tests/test_set_gflownet.py
git commit -m "feat(gflownet): parallel two-wave prewarm_caches (spawn pool, mirrors reranker)"
```

---

### Task 3: Wire pre-warm into `run_gflownet.py`

Call `prewarm_caches` before `fit()` (train roots) and before `evaluate_matrix` (eval-test roots) when `--workers > 1`, reusing the existing flag and `GEN_CKPT`.

**Files:**
- Modify: `scripts/run_gflownet.py` (around the trainer construction ~line 437 and the eval call ~line 445)

**Interfaces:**
- Consumes: `trainer.prewarm_caches` (Task 2), `args.workers`, `GEN_CKPT`, `train_substrates_list`, `eval_bundle`, `eval_count`.

- [ ] **Step 1: Add the pre-warm calls**

After `trainer = SetGFlowNetTrainer(...)` and before `trainer.fit(...)`:

```python
    if args.workers > 1:
        print(f"[gflownet] parallel pre-warming train caches ({args.workers} workers) ...", flush=True)
        t0 = time.time()
        trainer.prewarm_caches(train_substrates_list, args.workers, gen_ckpt=str(GEN_CKPT), verbose=True)
        print(f"[gflownet] train prewarm done in {time.time()-t0:.1f}s", flush=True)
```

Before `evaluate_matrix(...)` (after `eval_bundle`/`eval_count` are known):

```python
    if args.workers > 1:
        eval_roots = list(eval_bundle.map.keys())[:eval_count]
        print(f"[gflownet] parallel pre-warming eval caches ({len(eval_roots)} roots) ...", flush=True)
        t0 = time.time()
        trainer.prewarm_caches(eval_roots, args.workers, gen_ckpt=str(GEN_CKPT), verbose=True)
        print(f"[gflownet] eval prewarm done in {time.time()-t0:.1f}s", flush=True)
```

- [ ] **Step 2: Verify the script imports and the flag is wired**

Run: `python scripts/run_gflownet.py --help`
Expected: prints usage including `--workers`; no import error.

Run: `python -c "import ast; ast.parse(open('scripts/run_gflownet.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Full suite green**

Run: `make test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_gflownet.py
git commit -m "feat(gflownet): call prewarm_caches before fit + eval when --workers>1"
```

---

### Task 4: Modal validation + full-scale run (compute step)

The pytest covers orchestration + serial-equivalence; this task validates the real spawn path at Modal/Linux scale and measures the speedup. Requires Modal credits (~$2-4 of the remaining budget).

- [ ] **Step 1: Rebuild the Modal image with the new code**

The image git-clones `metabench-reranker`; push first, then bust the clone-cache layer so it re-clones. In `scripts/modal_m2.py`, keep the branch clone; a fresh `modal run` after push re-pulls only if the layer key changed — force it by appending the current short SHA to the clone `run_commands` string (a comment `&& echo <sha>`), or run `modal run ... ::warmup` which rebuilds. Confirm `warmup` prints the new `set_gflownet` has `prewarm_caches` (add `assert hasattr(SetGFlowNetTrainer, "prewarm_caches")` to warmup, optional).

- [ ] **Step 2: Tiny Modal validation (~20 substrates, ~cents)**

Temporarily set `M2_ARGS` train/test to ~20 / ~10, `--workers 8`, `--epochs 2`. `modal run --detach scripts/modal_m2.py`. Confirm in logs: `prewarm wave1`/`wave2` lines, no spawn deadlock / shm ENOMEM, then `setgfn epoch=` lines flow, then a `seed0.json`. This proves the parallel path works on Linux.

- [ ] **Step 3: Full-scale run + measure**

Restore `M2_ARGS` to the full headline scale (e.g. `--train-substrates 1200 --test-substrates 400 --n-samples 8 --epochs 15 --logz-lr 0.04 --workers 8`) on CPU-only. Compare the pre-warm wall time + total run time against the serial baseline (epoch 1 was >4h serial). Record the measured speedup in memory `grail-sota-goal` and the spec's Verification section.

- [ ] **Step 4: Aggregate + ingest headline**

`modal volume get` the 3 seed JSONs → `scripts/aggregate_seeds.py` → ingest mean±std into `docs/benchmark/{gloryx_results,stage2_ranker_evidence}.md` (touch-once).

---

## Notes for the implementer

- The spawn path is a faithful mirror of `grail_metabolism/workflows/reranker.py` (`_bi_worker_init`, `_bi_pool_worker`, `_MAX_SPAWN_WORKERS`, the `spawn` Pool + `imap_unordered` + parent-side rule-embedding warm). Read that first; do not reinvent the concurrency handling.
- Determinism check to keep in mind: each state is fully expanded by exactly one worker via `_expand_state`, so per-state child order matches serial; dict equality ignores key insertion order, so `_child_cache` (parallel) `==` `_child_cache` (serial).
- Keep `_child_cache` values as plain `(str, float, int)` tuples — never store tensors (grad retention + shm mmap).
