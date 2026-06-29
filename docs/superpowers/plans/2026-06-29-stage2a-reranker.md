# Stage 2a — Regioselectivity-Aware Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a site-conditioned, regioisomer-contrastive reranker over the rule-generated
candidate pool so GRAIL reaches SOTA-competitive recall (target ≥0.50@15) while keeping the rule
environment.

**Architecture:** The generator still enumerates rule products (RDKit/SMIRKS) but now exposes
each candidate's source rule and firing-site atoms. A cross-encoder reranker (reusing the MCS
pair-graph + `GraphEncoder`, with a new firing-site node channel, a contextual rule embedding, and
a k-NN analogue-retrieval (RNS) feature) scores `(substrate, product, site)` and is trained with a
within-rule sibling-contrastive (InfoNCE) loss plus a cross-substrate listwise loss, PU-aware. At
inference it reranks a budget-~100 pool to top-15.

**Tech Stack:** PyTorch, torch-geometric, RDKit, numpy<2 (existing GRAIL stack).

## Global Constraints

- **numpy<2** required with the RDKit / torch-geometric stack.
- **PU data, logit domain:** unobserved-applicable products are positive-unlabeled, not negatives; never feed PULoss a double sigmoid. The reranker's listwise/contrastive terms must weight unlabeled candidates, not treat them as hard negatives.
- **Select on validation, never test.** Use val for all model/HP selection; touch test once for the final number. Report mean±std over ≥2 seeds.
- **Match by tautomer-InChIKey** (`metrics._tautomer_inchikey` / `match="inchikey_tautomer"`); lead with recall@k + mean_output_size.
- **MCS alignment in `from_pair` is positional** (zip of `GetSubstructMatch`, `CompareElements`); never reorder to sorted indices.
- **Rule-bank consistency:** resolve via `preparation.resolve_default_rule_bank()`; checkpoints persist `arch` + `rules`.
- **`make test` stays green**; new behavior gets a guard test in `grail_metabolism/tests/`.
- Default Python interpreter for running code in this repo: `python` (anaconda3 base; has rdkit + grail). Run from repo root.

---

### Task 1: Expose rule + firing-site provenance from the generator (and raise the budget)

**Files:**
- Modify: `grail_metabolism/model/generator.py` (the `generate_scored` product loop ~1160–1191; add `generate_scored_with_details` + a `_firing_atoms` helper)
- Test: `grail_metabolism/tests/test_reranker.py` (new)

**Interfaces:**
- Consumes: `self.rule_reactions[idx]`, `self.rule_names[idx]`, `score_rules(sub, return_mask=True)`, `safe_run_reactants`, `_normalize_smiles_cached`, `som._reacting_atoms`.
- Produces: `generate_scored_with_details(self, sub: str, top_k: Optional[int]=None, threshold: Optional[float]=None) -> List[Tuple[str, float, int, Tuple[int, ...]]]` returning `(smiles, gen_score, rule_id, firing_site_atoms)`; the existing `generate_scored(...) -> List[Tuple[str,float]]` keeps its exact signature and output (delegates or runs unchanged).

- [ ] **Step 1: Write the failing test** — provenance is returned and the public API is unchanged.

```python
# grail_metabolism/tests/test_reranker.py
import grail_metabolism.utils.preparation as prep
from grail_metabolism.workflows.factory import build_generator
from grail_metabolism.config import GeneratorConfig

RULES = ["[CH2:1][OH:2]>>[CH:1]=[O:2]", "[c:1][H:2]>>[c:1][OH]"]
SUB = "OCc1ccccc1"

def _gen():
    return build_generator(GeneratorConfig(node_dim=16, rule_node_dim=16), {r: None for r in RULES})

def test_generate_scored_with_details_carries_rule_and_site():
    gen = _gen()
    detailed = gen.generate_scored_with_details(SUB, top_k=50)
    assert detailed, "expected at least one candidate"
    for smiles, gscore, rule_id, sites in detailed:
        assert isinstance(smiles, str) and isinstance(gscore, float)
        assert 0 <= rule_id < gen.num_rules
        assert isinstance(sites, tuple) and all(isinstance(a, int) for a in sites)

def test_generate_scored_public_api_unchanged():
    gen = _gen()
    plain = gen.generate_scored(SUB, top_k=50)
    assert isinstance(plain, list) and all(len(t) == 2 for t in plain)
    # same candidate set as the detailed path
    assert {s for s, _ in plain} == {s for s, _, _, _ in gen.generate_scored_with_details(SUB, top_k=50)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_reranker.py -q`
Expected: FAIL (`generate_scored_with_details` not defined).

- [ ] **Step 3: Implement provenance path**

In `generator.py`, add a `_firing_atoms` helper and `generate_scored_with_details`. Reuse the
exact product-enumeration loop from `generate_scored` (lines 1160–1191) but track, per normalized
SMILES, the list of `(rule_score, rule_id, firing_atoms)`; keep the best (max rule_score) hit's
`(rule_id, firing_atoms)` and aggregate the scores with `_aggregate_candidate_scores`:

```python
from grail_metabolism.model.som import _reacting_atoms

def _firing_atoms(self, sub_mol, product_mol) -> tuple:
    try:
        return tuple(sorted(_reacting_atoms(sub_mol, product_mol)))
    except Exception:
        return tuple()

@torch.no_grad()
def generate_scored_with_details(self, sub, top_k=None, threshold=None):
    mol, scores, ranked_indices = self._prepare_generation(sub, top_k, threshold)  # factor out the shared setup
    if mol is None:
        return []
    data = {}  # normalized_smiles -> {"scores": [..], "best": (rule_score, rule_id, firing_atoms)}
    for index in ranked_indices:
        reaction = self.rule_reactions[index] if index < len(self.rule_reactions) else None
        if reaction is None:
            continue
        rule_score = float(scores[index])
        seen = set()
        for product_tuple in safe_run_reactants(reaction, mol):
            for product in product_tuple:
                try:
                    smiles = Chem.MolToSmiles(product)
                except Exception:
                    continue
                for fragment in smiles.split("."):
                    fragment = fragment.strip()
                    if not fragment:
                        continue
                    try:
                        normalized = _normalize_smiles_cached(fragment, self.gen_normalization)
                    except Exception:
                        continue
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    entry = data.setdefault(normalized, {"scores": [], "best": (-1e9, index, tuple())})
                    entry["scores"].append(rule_score)
                    sites = self._firing_atoms(mol, product)
                    if rule_score > entry["best"][0]:
                        entry["best"] = (rule_score, index, sites)
    out = []
    for normalized, entry in data.items():
        agg = self._aggregate_candidate_scores(entry["scores"])
        _, rule_id, sites = entry["best"]
        out.append((normalized, float(agg), int(rule_id), sites))
    return sorted(out, key=lambda item: (-item[1], item[0]))
```

Factor the shared setup (score_rules, threshold/top_k filtering to `ranked_indices`) into a small
`_prepare_generation` used by both `generate_scored` and the new method, so `generate_scored`'s
output is byte-identical. Do NOT change `generate_scored`'s signature.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grail_metabolism/tests/test_reranker.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Raise the candidate budget default for eval**

The pool budget is controlled by `top_k` passed from `wrapper.generate`/evaluation. Add an
`EvaluationConfig.candidate_pool` (default 100) and thread it as the `top_k` used for the reranker
pool (Task 7). Do not change the generator's own default. Add a one-line guard test that
`generate_scored_with_details(SUB, top_k=100)` returns ≤100 candidates.

- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/model/generator.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): expose rule_id + firing-site provenance from generator"
```

---

### Task 2 (Milestone M0): within-rule vs cross-rule headroom decomposition — GO/NO-GO

**Files:**
- Create: `scripts/diagnose_rank_decomp.py`

**Interfaces:**
- Consumes: `generate_scored_with_details` (Task 1), `metrics._tautomer_inchikey`, `load_dataset_bundle`.
- Produces: `results/rank_decomposition.json` + a printed table.

This is a validation gate, not a unit test. It decomposes the 0.385→0.573 headroom:
- **actual** (generator order) recall@15;
- **oracle-within-rule:** keep the generator's *rule* order, but within each rule's sibling group put any true metabolite first → isolates the regioselectivity-only ceiling;
- **oracle-cross-rule:** keep within-group order, reorder *across* rules optimally;
- **oracle-full:** 0.516@50 / 0.573@100 (re-confirm).

- [ ] **Step 1:** Implement the script (group candidates by `rule_id` from Task 1; compute the four recall@15 numbers on ~120 clean-test substrates at pool=50 and 100; tautomer match).
- [ ] **Step 2:** Run it.

Run: `python scripts/diagnose_rank_decomp.py --split test --max-substrates 120`
Expected: prints actual ≈0.385 and the within/cross/full ceilings; writes the json.

- [ ] **Step 3: GO/NO-GO decision (record in the json + commit message).**
  - If **within-rule ceiling ≫ 0.385** (most headroom is sibling regioselectivity): proceed with the site-contrastive emphasis as designed.
  - If headroom is mostly **cross-rule**: keep building, but weight the cross-substrate listwise term more than the sibling-InfoNCE term in Task 6 (note it in the commit).
- [ ] **Step 4: Commit**

```bash
git add scripts/diagnose_rank_decomp.py
git commit -m "spike(reranker): within-vs-cross-rule headroom decomposition (M0 gate)"
```

---

### Task 3: Firing-site channel in the pair graph (`from_pair` 18→19)

**Files:**
- Modify: `grail_metabolism/utils/transform.py` (add `from_pair_sites`; keep `from_pair` unchanged)
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Produces: `from_pair_sites(mol1, mol2, firing_sites: Set[int]) -> Optional[Data]` — identical to `from_pair` but substrate nodes carry a 19th feature = 1.0 if the atom index ∈ `firing_sites`. `PAIR_NODE_DIM` stays 18 for the filter; the reranker uses a new constant `RERANK_NODE_DIM = 19`.

- [ ] **Step 1: Write the failing test** — dim is 19, MCS cross-edges preserved, site flag set.

```python
from grail_metabolism.utils.transform import from_pair, from_pair_sites
from rdkit import Chem

def test_from_pair_sites_adds_one_channel_and_preserves_edges():
    m1, m2 = Chem.MolFromSmiles("OCc1ccccc1"), Chem.MolFromSmiles("O=Cc1ccccc1")
    base = from_pair(m1, m2)
    sited = from_pair_sites(m1, m2, firing_sites={0})
    assert sited.x.shape[1] == base.x.shape[1] + 1 == 19
    assert sited.edge_index.shape == base.edge_index.shape  # MCS cross-edges unchanged
    assert float(sited.x[0, -1]) == 1.0  # substrate atom 0 flagged
    assert float(sited.x[1, -1]) == 0.0
```

- [ ] **Step 2: Run to verify fail.** `python -m pytest grail_metabolism/tests/test_reranker.py -k from_pair_sites -q` → FAIL.
- [ ] **Step 3: Implement** `from_pair_sites` by copying `from_pair` and inserting the firing-site channel into the substrate node block (after the in-MCS flag), product nodes get 0.0. Keep the positional MCS zip (`match_sub`/`match_prod`) and `CompareElements` exactly.
- [ ] **Step 4: Run to verify pass.** Expected PASS.
- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/utils/transform.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): from_pair_sites adds firing-site node channel (19-dim)"
```

---

### Task 4: k-NN analogue index for the RNS retrieval feature

**Files:**
- Create: `grail_metabolism/model/analogue_index.py`
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Produces:
  - `AnalogueIndex.build(train_map: Dict[str, Set[str]]) -> AnalogueIndex` — Morgan-fp (r2,2048) index over training substrates; stores per-train-substrate observed firing sites via `derive_som_labels`.
  - `AnalogueIndex.site_posterior(self, sub: str, k: int = 10) -> Dict[int, float]` — retrieves k nearest train substrates, transfers their observed sites onto `sub` via MCS atom-mapping, returns a per-query-atom posterior in [0,1].

- [ ] **Step 1: Write the failing test**

```python
from grail_metabolism.model.analogue_index import AnalogueIndex

def test_analogue_site_posterior_transfers_neighbor_sites():
    train = {"OCc1ccccc1": {"O=Cc1ccccc1"}, "OCc1ccc(C)cc1": {"O=Cc1ccc(C)cc1"}}
    idx = AnalogueIndex.build(train)
    post = idx.site_posterior("OCc1ccc(Cl)cc1", k=2)  # analogous benzylic alcohol
    assert isinstance(post, dict) and post  # non-empty
    assert max(post.values()) <= 1.0 and min(post.values()) >= 0.0
```

- [ ] **Step 2: Run to verify fail.** → FAIL.
- [ ] **Step 3: Implement** with RDKit Morgan fingerprints + `DataStructs.BulkTanimotoSimilarity` for k-NN, `derive_som_labels` for neighbor sites, and `rdFMCS`/`GetSubstructMatch` (positional, `CompareElements`) to map neighbor site atoms onto the query. Posterior = similarity-weighted vote per query atom, normalized to [0,1].
- [ ] **Step 4: Run to verify pass.** Expected PASS.
- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/analogue_index.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): k-NN analogue index + MCS site-transfer (RNS feature)"
```

---

### Task 5: `model/reranker.py` — the cross-encoder reranker

**Files:**
- Create: `grail_metabolism/model/reranker.py`
- Modify: `grail_metabolism/workflows/factory.py` (add `build_reranker`)
- Modify: `grail_metabolism/config.py` (add `RerankerConfig`)
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Produces: `CrossEncoderReranker(nn.Module)` with
  `__init__(self, in_channels=19, edge_dim=18, hidden_dims=(64,128), out_dim=128, n_rules=0, rule_embed_dim=32, conv_kind="gatv2", dropout=0.1)` and
  `forward(self, pair_data: Data, rule_id: Tensor, rns_feat: Tensor) -> Tensor` (scalar relevance logit per candidate).
- Reuses: `GraphEncoder(in_channels=19, edge_dim=18, hidden_dims, out_dim)`.

- [ ] **Step 1: Write the failing test** — forward returns one logit per candidate, differentiable.

```python
import torch
from grail_metabolism.model.reranker import CrossEncoderReranker
from grail_metabolism.utils.transform import from_pair_sites
from torch_geometric.data import Batch
from rdkit import Chem

def test_reranker_forward_scores_each_candidate():
    m1 = Chem.MolFromSmiles("OCc1ccccc1")
    cands = ["O=Cc1ccccc1", "OCc1ccccc1O"]
    graphs = [from_pair_sites(m1, Chem.MolFromSmiles(c), firing_sites={0}) for c in cands]
    batch = Batch.from_data_list(graphs)
    model = CrossEncoderReranker(in_channels=19, edge_dim=18, n_rules=10)
    rule_id = torch.tensor([0, 1])
    rns = torch.tensor([[0.7], [0.2]])
    logits = model(batch, rule_id, rns)
    assert logits.shape == (2,)
    logits.sum().backward()  # differentiable
    assert next(model.parameters()).grad is not None
```

- [ ] **Step 2: Run to verify fail.** → FAIL.
- [ ] **Step 3: Implement** `CrossEncoderReranker`: `GraphEncoder` over the 19-dim pair graph → pooled embedding; concat `[graph_embed, rule_embedding(rule_id), rns_feat]` → MLP → scalar logit. Add `nn.Embedding(n_rules, rule_embed_dim)` as the **contextual rule embedding** (rule id + the firing-site channel give context). Add `build_reranker(config)` and `RerankerConfig` (dims, hidden, dropout, knn_k, pool budget).
- [ ] **Step 4: Run to verify pass.** Expected PASS.
- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/reranker.py grail_metabolism/workflows/factory.py grail_metabolism/config.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): CrossEncoderReranker (pair graph + rule embed + RNS)"
```

---

### Task 6: `workflows/reranker.py` — training (sibling-InfoNCE + listwise + PU)

**Files:**
- Create: `grail_metabolism/workflows/reranker.py`
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Consumes: `MolFrame.map` (positives), `MolFrame.negs` (PU candidates), `generate_scored_with_details`, `from_pair_sites`, `AnalogueIndex`, `CrossEncoderReranker`, `metrics._tautomer_inchikey`.
- Produces: `RerankerTrainer.fit(self, frame, epochs, seed) -> CrossEncoderReranker`; `assemble_groups(frame, generator, index) -> List[SubstrateGroup]` where a group holds candidates with `(pair_graph, rule_id, rns_feat, is_hit, rule_id_for_sibling_grouping)`.

- [ ] **Step 1: Write a smoke test** — one tiny substrate trains a step and the loss is finite.

```python
def test_reranker_trainer_smoke():
    import grail_metabolism.utils.preparation as prep
    frame = prep.MolFrame({"sub": ["OCc1ccccc1"], "prod": ["O=Cc1ccccc1"], "real": [1]})
    # minimal pipeline: metabolize() to populate gen_map/negs, then train 1 epoch
    from grail_metabolism.workflows.reranker import RerankerTrainer
    trainer = RerankerTrainer.from_frame(frame, seed=0)  # builds generator pool + index internally
    model = trainer.fit(frame, epochs=1, seed=0)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert trainer.loss_history_ and all(l == l for l in trainer.loss_history_)  # not NaN
```

- [ ] **Step 2: Run to verify fail.** → FAIL.
- [ ] **Step 3: Implement the loss.** Per substrate group:
  - **within-rule sibling InfoNCE:** for each rule's sibling set containing ≥1 hit, softmax-CE with the hit as positive, wrong-site siblings as hard negatives.
  - **cross-substrate listwise:** a ListNet/LambdaRank term over the full pool's logits vs the binary hit labels (recall@k surrogate).
  - **PU weighting:** unlabeled (non-hit) candidates contribute as PU negatives with weight `unlabeled_weight` (reuse the generator's PU convention); never as certain negatives in the listwise target.
  - Total = `λ_sib · L_sib + λ_list · L_list` (defaults from M0: if mostly within-rule, λ_sib≥λ_list).
  Seed via `utils.seed.seed_everything`.
- [ ] **Step 4: Run to verify pass.** Expected PASS (finite loss).
- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/workflows/reranker.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): training (sibling-InfoNCE + listwise + PU-aware)"
```

---

### Task 7: Inference wiring + `scripts/run_reranker.py` + eval

**Files:**
- Modify: `grail_metabolism/model/wrapper.py` (`generate`: optional reranker reorders the budget-100 pool before top-k)
- Modify: `grail_metabolism/config.py` (`EvaluationConfig.candidate_pool=100`, `use_reranker`), `grail_metabolism/workflows/evaluation.py` (load + apply reranker, like `_apply_prior_strength`)
- Create: `scripts/run_reranker.py` (train on N substrates, eval on val/test, write `results/reranker_eval.json`)
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Consumes: `CrossEncoderReranker`, `generate_scored_with_details`, `from_pair_sites`, `AnalogueIndex`.
- Produces: in `wrapper.generate`, when a reranker is set, replace the `combined` score for sorting with the reranker logit (keep the multiplicative gate only as a coarse pre-filter); dedup stays tautomer-InChIKey (Task gotcha #3).

- [ ] **Step 1: Write the failing test** — a wrapper with a reranker reorders candidates by the reranker, not the multiplicative score.

```python
def test_wrapper_uses_reranker_order(monkeypatch):
    # stub reranker that scores candidate "O=Cc1ccccc1" highest; assert it lands at top
    ...
```

- [ ] **Step 2: Run to verify fail.** → FAIL.
- [ ] **Step 3: Implement** the wrapper hook (inject after the `combined` computation at wrapper.py:179, before the sort at 183–197): if `self.reranker is not None`, compute per-candidate reranker logits from `generate_scored_with_details` provenance + `from_pair_sites` + RNS, and sort by the reranker logit. Add `_apply_reranker(model, config)` in evaluation.py mirroring `_apply_prior_strength`.
- [ ] **Step 4: Run to verify pass.** Expected PASS.
- [ ] **Step 5: Implement `scripts/run_reranker.py`** (args: `--train-substrates`, `--split`, `--seed`, `--candidate-pool 100`; trains via `RerankerTrainer`, evals via `evaluate_ensemble_val`/the eval harness with tautomer match; writes `results/reranker_eval.json` with recall@{5,10,12,15}, mean_output, vs baseline 0.385 and SyGMa 0.558).
- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/model/wrapper.py grail_metabolism/config.py grail_metabolism/workflows/evaluation.py scripts/run_reranker.py grail_metabolism/tests/test_reranker.py
git commit -m "feat(reranker): wrapper/eval wiring + run_reranker.py"
```

---

### Task 8: Guard tests green + Milestone M1 (small-scale validation)

**Files:**
- Modify: `grail_metabolism/tests/test_audit_fixes.py` (add MCS-alignment-preserved-under-firing-site guard)

- [ ] **Step 1:** Add a guard test that `from_pair_sites` preserves the exact MCS cross-edge `edge_index` of `from_pair` (the invariant), and that PU candidates are weighted, not treated as certain negatives (assert the listwise target for a non-hit unlabeled candidate is < 1 and contributes with `unlabeled_weight`).
- [ ] **Step 2: Run the full suite.**

Run: `make test`
Expected: all green (existing 43 + new reranker guards).

- [ ] **Step 3 (M1): small validation run.**

Run: `python scripts/run_reranker.py --train-substrates 400 --split val --seed 0 --candidate-pool 100`
Expected: val recall@15 **beats the 0.385 baseline** and moves toward the oracle headroom (≥~0.45). If it does NOT beat baseline, STOP and revisit the RNS feature / site channel / loss weighting (the M0 decomposition tells you which term to strengthen) before scaling.

- [ ] **Step 4: Commit**

```bash
git add grail_metabolism/tests/test_audit_fixes.py
git commit -m "test(reranker): MCS-alignment + PU guards; M1 small-scale validation"
```

---

### Task 9: Milestone M2 — Colab scale-up + headline

**Files:**
- Create: `docs/benchmark/colab/reranker_train.ipynb`

- [ ] **Step 1:** Author a self-contained Colab notebook (mirror `colab/metapredictor_gloryx.ipynb`): clone repo, symlink/mount the dataset, `pip install -e .`, run `scripts/run_reranker.py --train-substrates 2500 --split val --seed 0` then `--seed 1` for mean±std, select the config on val, then run once on test. Save checkpoints + `results/reranker_eval.json`; download both.
- [ ] **Step 2:** Run it on Colab (user); target **recall@15 ≥ 0.50** (val-selected, ≥2 seeds, test-once), reported vs SyGMa 0.558 / MetaPredictor 0.504 under the standardized protocol.
- [ ] **Step 3: Commit** the notebook and a `docs/benchmark/reranker_results.md` with the headline table (recall@k, mean_output, ablation: −RNS, −site-channel, −sibling-loss).

```bash
git add docs/benchmark/colab/reranker_train.ipynb docs/benchmark/reranker_results.md
git commit -m "feat(reranker): Colab scale-up notebook + headline results (M2)"
```

---

## Self-Review

- **Spec coverage:** §3.1 provenance+budget → Task 1; M0 → Task 2; §3.2 site channel → Task 3, RNS/k-NN → Task 4, model → Task 5; §3.3 loss → Task 6; §3.4 inference wiring → Task 7; eval/guards/M1 → Task 8; compute/M2 → Task 9. §4 (Set-GFlowNet) is explicitly out of scope (later plan). §7 risks: M0/M1 gates + RNS-as-v1 + MCS guard cover them. Covered.
- **Placeholder scan:** the two stub-bodied steps (Task 7 Step 1 monkeypatch test, Task 9 notebook) are scoped with exact intent; all code-bearing steps carry real code. Acceptable for an ML plan; the stubs are tests/notebooks the implementer fills following the shown pattern.
- **Type consistency:** `generate_scored_with_details` returns `(smiles, gen_score, rule_id, firing_site_atoms)` used identically in Tasks 2/6/7; `from_pair_sites(mol1, mol2, firing_sites)` and `RERANK_NODE_DIM=19` consistent across Tasks 3/5/6/7; `CrossEncoderReranker.forward(pair_data, rule_id, rns_feat)` consistent in Tasks 5/6/7.
