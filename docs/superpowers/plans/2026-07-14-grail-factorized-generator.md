# GRAIL Factorized-Generative Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GRAIL's 7,581-way PU rule-selector with a dense maximum-likelihood factorized generator `P(type|s)·P(site|type,s)`, keep the frozen pair filter, and add precision-calibrated abstention — so the Stage-1 learner escapes the PU degeneracy (Prop 2) that makes it lose to a frequency prior.

**Architecture:** Offline, cluster the mined reaction rules into a canonical ~100–200 radius-0 **type** vocabulary + a `rule_id→type_id` map, and extract per-pair dense `(type, reacting-site)` labels. A `FactorizedGenerator` adds a graph-level **type head** and an atom-level **site head** on the shared `GraphEncoder`, trained by cross-entropy/BCE (not PU). At inference, select types by `P(type|s)`, apply each selected type's rules broadly via RDKit, and rank the product pool by `P(type)·P(site)·filter_score`, emitting a precision-calibrated variable-size set.

**Tech Stack:** Python, RDKit, torch, torch-geometric; reuses `scripts/mine_rules.py`, `grail_metabolism/model/som.py`, `grail_metabolism/model/_graph.py`, `grail_metabolism/workflows/*`.

## Global Constraints

- Existing data only (~4,787 train substrates; no new labels; no external rule bank import).
- Never clobber `grail_metabolism/resources/extended_smirks.txt` or deployed checkpoints — new artifacts use suffixed/new paths.
- Route runtime wiring through `grail_metabolism/workflows/ensemble.py:EnsembleWorkflow.run_bundle`; construct models via `grail_metabolism/workflows/factory.py`.
- Select on val, touch test once. Headline = tautomer-InChIKey macro recall@15 + the recall-at-precision PR frontier.
- `make test` (`pytest grail_metabolism/tests -q`) stays green; new behavior gets a guard test in `grail_metabolism/tests/`.
- Preserve interpretability: every prediction stays an RDKit-applied SMIRKS product with atom-level provenance.
- Python interpreter for running: `/Users/nikitapolomosnov/anaconda3/bin/python`.
- NO Claude/AI/Co-Authored-By attribution in any commit or doc byline.
- Decision gate (Task 6): the factorized pipeline must beat the **0.413** broad+filter baseline on the clean test with paired-bootstrap CI separation; else the contribution reframes to precision-only (abstention frontier) and Stage-4 fallback (coarse-type bank consolidation) ships instead.

---

### Task 1: Reaction-type vocabulary + `rule_id → type_id` map

**Files:**
- Create: `grail_metabolism/model/reaction_types.py`
- Create: `scripts/build_type_vocab.py`
- Test: `grail_metabolism/tests/test_reaction_types.py`
- Reads: `results/mined_rule_catalog_v2.json` (5,856 SMIRKS + `count` support), `grail_metabolism/resources/extended_smirks.txt`

**Interfaces:**
- Consumes: `scripts/redesign_gate_a.py:radius0_signature(smirks)` logic (bond-level element-typed broken/formed set).
- Produces:
  - `reaction_types.canonical_type(smirks: str) -> tuple | None` — a canonical, order-collapsed radius-0 signature (multiset of `(elements, before_order, after_order)` collapsed so ring/multi-bond over-splits merge).
  - `reaction_types.build_type_vocab(catalog: dict, min_pairs: int = 5) -> tuple[dict, dict]` — returns `(type_id_to_sig, rule_smirks_to_type_id)`, keeping types whose pooled train-pair support ≥ `min_pairs` and mapping rarer rules to a shared `type_id = -1` "other" bucket.
- Artifact: `resources/coarse_type_vocab.json` = `{"types": {type_id: {"signature": str, "n_rules": int, "n_pairs": int}}, "rule_to_type": {smirks: type_id}}`.

- [ ] **Step 1: Write the failing test**

```python
# grail_metabolism/tests/test_reaction_types.py
from grail_metabolism.model.reaction_types import canonical_type, build_type_vocab

def test_canonical_type_merges_periphery_variants():
    # two hydroxylations differing only in aromatic periphery share a radius-0 type
    a = "[cH:1][cH:2]>>[c:1][OH]"            # aromatic C-H -> C-OH (schematic)
    b = "[cH:1][c:2]([CH3])>>[c:1][OH]"
    ta, tb = canonical_type(a), canonical_type(b)
    assert ta is not None and ta == tb

def test_build_type_vocab_buckets_rare_rules():
    catalog = {
        "[CH2:1][OH:2]>>[CH:1]=[O:2]": {"count": 40},   # frequent -> its own type
        "[cH:1]>>[c:1][F]": {"count": 1},                # rare -> "other" (-1)
    }
    id2sig, rule2type = build_type_vocab(catalog, min_pairs=5)
    assert rule2type["[CH2:1][OH:2]>>[CH:1]=[O:2]"] >= 0
    assert rule2type["[cH:1]>>[c:1][F]"] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_reaction_types.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `reaction_types.py`**

Port `radius0_signature` from `scripts/redesign_gate_a.py` into `canonical_type` but collapse the changed-bond list into a **`collections.Counter` multiset** keyed by `(sorted_elements, before_order, after_order)` and return `tuple(sorted(counter.items()))` — so N-of-the-same changed bond (ring formation) becomes one `(descriptor, N)` entry rather than N positional entries. `build_type_vocab` groups catalog SMIRKS by `canonical_type`, sums `count` per signature, assigns dense `type_id` (0..K-1) to signatures with pooled support ≥ `min_pairs` (descending support), maps all others to `-1`, and returns the two dicts. Unparseable SMIRKS → `-1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_reaction_types.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Build + commit the vocab artifact**

`scripts/build_type_vocab.py` loads `results/mined_rule_catalog_v2.json`, calls `build_type_vocab`, writes `grail_metabolism/resources/coarse_type_vocab.json`, and prints `K types cover P% of train pairs`. Run it, confirm `K` is ~100–300 and coverage ≥ 80% of pairs (consistent with Gate A's 86.7%).

```bash
/Users/nikitapolomosnov/anaconda3/bin/python scripts/build_type_vocab.py
git add grail_metabolism/model/reaction_types.py scripts/build_type_vocab.py grail_metabolism/tests/test_reaction_types.py
git add -f grail_metabolism/resources/coarse_type_vocab.json
git commit -m "feat(redesign): radius-0 reaction-type vocabulary + rule->type map"
```

---

### Task 2: Dense `(type, site)` training-label dataset

**Files:**
- Create: `grail_metabolism/model/factorized_data.py`
- Test: `grail_metabolism/tests/test_factorized_data.py`

**Interfaces:**
- Consumes: `reaction_types` (Task 1); `grail_metabolism/model/som.py:derive_som_labels(sub, met)`, `_reacting_atoms`; `grail_metabolism/utils/transform.py:from_rdmol`; `grail_metabolism/utils/preparation.py:MolFrame` (has `.map: {sub: set[met]}`); the rule bank (`resolve_default_rule_bank`) + RDKit to find which rule/type produced each metabolite.
- Produces: `factorized_data.build_factorized_dataset(molframe, rule_to_type) -> list[Data]` where each `Data` (from `from_rdmol(sub)`) carries `.y_type: LongTensor[num_types]` multi-hot (types that yield a true metabolite for this substrate) and `.y_site: FloatTensor[num_atoms]` (per-atom 0/1 reacting-atom label from `derive_som_labels`).

- [ ] **Step 1: Write the failing test**

```python
# grail_metabolism/tests/test_factorized_data.py
import pandas as pd
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.factorized_data import build_factorized_dataset

def test_dense_type_and_site_labels():
    frame = MolFrame(pd.DataFrame([{"sub": "CCO", "prod": "CC=O", "real": 1}]))
    frame.full_setup(rules=["[CH2:1][OH:2]>>[CH:1]=[O:2]"], include_pair_graphs=False, include_morgan=False)
    rule_to_type = {"[CH2:1][OH:2]>>[CH:1]=[O:2]": 0}
    ds = build_factorized_dataset(frame, rule_to_type)
    assert len(ds) == 1
    d = ds[0]
    assert d.y_type.sum() >= 1          # at least one positive type
    assert d.y_site.sum() >= 1          # at least one reacting atom
    assert d.y_site.numel() == d.x.size(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_factorized_data.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `factorized_data.py`**

`build_factorized_dataset(molframe, rule_to_type, catalog=None)`. **Type labels come from the mining catalog's `source_pairs`, NOT a full-bank re-apply** (the catalog `results/mined_rule_catalog_v2.json` records, per SMIRKS, the exact train pairs it was mined from — reuse that instead of re-running the ~90-min ceiling pass). Precompute `sub_to_types: dict[str, set[int]]`: for each `smirks, entry` in the catalog, `t = rule_to_type.get(smirks, -1)`; if `t >= 0`, for each `(sub_smi, prod_smi)` in `entry["source_pairs"]` add `t` to `sub_to_types[sub_smi]` (canonicalize the source-pair substrate SMILES the same way `MolFrame` does so keys line up — reuse `grail_metabolism.utils.preparation._standardize_smiles_cached` if `MolFrame` keys are standardized; otherwise raw). Then iterate `molframe.map.items()`: build `data = from_rdmol(Chem.MolFromSmiles(sub))`, skip if None/empty; `num_types = 1 + max(rule_to_type.values())`; `y_type = zeros(num_types)`, set `y_type[t]=1` for `t in sub_to_types.get(sub, set())`; `y_site = zeros(num_atoms)`, set to 1 at the indices from `derive_som_labels(sub, mets)`. Attach both, return the list. When `catalog is None`, load it from `results/mined_rule_catalog_v2.json`. The test's tiny in-memory frame will have an empty `sub_to_types` for its substrate unless a catalog is passed, so the test passes a 1-rule catalog whose `source_pairs` includes `("CCO", "CC=O")` — assert `y_type.sum() >= 1` under that catalog. (Also keep a `derive_som_labels`-based `y_site` assertion, which needs no catalog.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_factorized_data.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/factorized_data.py grail_metabolism/tests/test_factorized_data.py
git commit -m "feat(redesign): dense (type, site) label dataset builder"
```

---

### Task 3: `FactorizedGenerator` model (type + site heads)

**Files:**
- Create: `grail_metabolism/model/factorized.py`
- Test: `grail_metabolism/tests/test_factorized_model.py`

**Interfaces:**
- Consumes: `grail_metabolism/model/_graph.py:GraphEncoder` (`.forward(data)->[B,out]` graph-level via `global_mean_pool`, `.forward_nodes(data)->[N,out]` per-atom); dims `SINGLE_NODE_DIM=16`, `EDGE_DIM=18`.
- Produces:
  - `FactorizedGenerator(num_types, in_channels=16, edge_dim=18, hidden_dims=(192,256), out_dim=128)`.
  - `.type_logits(data) -> Tensor[B, num_types]` (graph-level).
  - `.site_logits(data) -> Tensor[N]` (per-atom).
  - `.arch: dict` (for checkpoint round-trip, like `SoMPredictor.arch`).

- [ ] **Step 1: Write the failing test**

```python
# grail_metabolism/tests/test_factorized_model.py
import torch
from rdkit import Chem
from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.model.factorized import FactorizedGenerator

def test_head_shapes():
    data = from_rdmol(Chem.MolFromSmiles("CCO"))
    model = FactorizedGenerator(num_types=50)
    tl = model.type_logits(data)
    sl = model.site_logits(data)
    assert tl.shape[-1] == 50
    assert sl.numel() == data.x.size(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_factorized_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `factorized.py`**

`FactorizedGenerator(nn.Module)`: one shared `GraphEncoder(in_channels, edge_dim, list(hidden_dims), out_dim, conv_kind="gatv2")`; `type_head = nn.Linear(out_dim, num_types)`; `site_head = nn.Linear(out_dim, 1)`. `type_logits(data)` = `type_head(encoder.forward(data))`; `site_logits(data)` = `site_head(encoder.forward_nodes(data)).squeeze(-1)`. Store `self.arch = {...}` mirroring `SoMPredictor.arch` plus `num_types`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/nikitapolomosnov/anaconda3/bin/python -m pytest grail_metabolism/tests/test_factorized_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/factorized.py grail_metabolism/tests/test_factorized_model.py
git commit -m "feat(redesign): FactorizedGenerator type+site heads on shared GraphEncoder"
```

---

### Task 4: MLE training + val gate (type head beats the type-frequency prior)

**Files:**
- Create: `scripts/train_factorized.py`
- Modify: `grail_metabolism/model/factorized.py` (add `fit(dataset, epochs, lr)` + `save`/`load`)
- Test: `grail_metabolism/tests/test_factorized_model.py` (add a `fit`-reduces-loss test)

**Interfaces:**
- Consumes: Task 2 dataset, Task 3 model.
- Produces: `FactorizedGenerator.fit(dataset, epochs=15, lr=1e-4, batch_size=32) -> list[float]` (loss = `CE(type_logits, y_type as multi-label BCEWithLogits) + BCEWithLogits(site_logits, y_site)`); `save(path)`/`load(path)` with `arch`; a trained checkpoint `artifacts/factorized_v1/checkpoints/factorized.pt`; a val report `results/factorized_val.json` with `type_head_recall@N`, `type_frequency_prior_recall@N`, `site_hit@3`.

- [ ] **Step 1: Write the failing test** (loss decreases over a few steps on a 2-substrate toy set — mirror `test_puloss_trains_on_logits` structure). Assert `history[-1] < history[0]`.
- [ ] **Step 2: Run it, verify FAIL** (`fit` not defined).
- [ ] **Step 3: Implement `fit`/`save`/`load`** in `factorized.py`: standard Adam loop over `Batch.from_data_list`, the combined loss above; `save` stores `{"arch":..., "state_dict":...}`; `load` reconstructs.
- [ ] **Step 4: Run the test, verify PASS.** Then run `make test` — expected `... passed` (green).
- [ ] **Step 5: Commit** the model+test.
- [ ] **Step 6: Full training run + VAL GATE.** `scripts/train_factorized.py` loads the bundle via `workflows/data.load_dataset_bundle` (the deployed dataset config), builds the Task-2 dataset on TRAIN, trains, and on VAL computes: (a) `P(type|s)` top-N type recall vs the marginal type-frequency prior's top-N; (b) site top-3 vs Gate C's 0.807. Write `results/factorized_val.json`.

```bash
/Users/nikitapolomosnov/anaconda3/bin/python scripts/train_factorized.py --epochs 15 2>&1 | tail -20
```

**GATE:** proceed to Task 5/6 only if the learned type head **beats the type-frequency prior on val** (the direct Prop-2-escape signal). If it does not beat the prior, STOP and report — the PU degeneracy persists at the type level and Stage-4 fallback (coarse-type bank consolidation) is the outcome. Commit the checkpoint + val report (`git add -f`).

---

### Task 5: Precision-calibration target for the filter

**Files:**
- Modify: `grail_metabolism/model/filter.py:calibrate_threshold` (line ~360, `target` Literal)
- Test: `grail_metabolism/tests/test_audit_fixes.py` (add `test_filter_precision_calibration`)

**Interfaces:**
- Consumes: existing `calibrate_threshold(scores, labels, target)`.
- Produces: `target` accepts `"precision"` — pick the **lowest** threshold whose precision ≥ `min_precision` (new kwarg, default 0.5), maximizing recall subject to that precision; if none reaches it, pick the max-precision threshold. Existing `f1`/`mcc` behavior unchanged.

- [ ] **Step 1: Write the failing test** — synthetic scores/labels where a known threshold achieves precision ≥ 0.8; assert `calibrate_threshold(..., target="precision", min_precision=0.8)` sets `calibrated_threshold` to a value whose precision ≥ 0.8 on the given data.
- [ ] **Step 2: Run it, verify FAIL** (target not supported).
- [ ] **Step 3: Implement** the `"precision"` branch in the existing threshold sweep (it already computes `precision`/`recall` per threshold at lines 396–397) — track the recall-maximizing threshold whose precision clears `min_precision`.
- [ ] **Step 4: Run the test, verify PASS**; run `make test` green.
- [ ] **Step 5: Commit.**

---

### Task 6: Factorized inference + eval vs the 0.413 baseline (the decision gate)

**Files:**
- Create: `grail_metabolism/model/factorized_infer.py`
- Create: `scripts/eval_factorized.py`
- Test: `grail_metabolism/tests/test_factorized_model.py` (add an end-to-end `generate` shape test)

**Interfaces:**
- Consumes: trained `FactorizedGenerator` (Task 4), `rule_to_type` (Task 1), the frozen deployed `Filter` (`artifacts/full5000_single/checkpoints/filter.pt`), `metrics.aggregate_prediction_metrics`, `scripts/run_benchmark.load_test_map`.
- Produces: `factorized_infer.generate(model, filter_model, rule_by_type, sub, top_types=10, max_output=15) -> list[str]` — select top-`top_types` types by `P(type|s)`; apply each selected type's rules broadly (RDKit) → candidate pool; score each candidate by `P(type)·site_plausibility(site_logits at its reacting atom, via som.product_som_score-style)·filter_score`; dedup by tautomer-InChIKey; return top-`max_output`. `scripts/eval_factorized.py` runs it on the full clean test and reports `recall@15` (tautomer) + the recall-at-precision PR frontier + `mean_output`.

- [ ] **Step 1: Write the failing test** — `generate` returns a list of ≤ `max_output` SMILES for `"CCO"` with a toy 1-type model+map; assert length bound + all parse.
- [ ] **Step 2: Run it, verify FAIL.**
- [ ] **Step 3: Implement `factorized_infer.generate`** as described (reuse `safe_run_reactants`, `_tautomer_inchikey`, and the filter's `score_batch`).
- [ ] **Step 4: Run the test, verify PASS**; `make test` green. Commit code.
- [ ] **Step 5: DECISION-GATE eval.** Run `scripts/eval_factorized.py` on the full clean test.

```bash
/Users/nikitapolomosnov/anaconda3/bin/python scripts/eval_factorized.py 2>&1 | tail -15
```

Compare `recall@15` to the committed baselines: deployed **0.330**, broad+filter **0.413**, SyGMa **0.572**. **GO** (build the full Stage-3 wiring through `EnsembleWorkflow.run_bundle` behind a config flag + report the lift) **iff** it beats 0.413 with paired-bootstrap CI separation (reuse `grail_metabolism/stats.py:paired_diff_bootstrap_ci`). Otherwise reframe to the precision-only contribution (the PR frontier from Task 5) + Stage-4 fallback. Write `results/factorized_eval.json`; commit (`git add -f`).

---

## Self-Review

**Spec coverage:** Stage 1 → Tasks 1–2; Stage 2 → Tasks 3–4 (+ val gate = the type-head>prior test); Stage 3 → Tasks 5–6 (precision calibration + inference + PR frontier + `run_bundle` wiring gated on the decision); Stage 4 fallback → the no-go branch of Tasks 4/6. Decomposition target (dense type/site labels replacing the PU vector) → Tasks 1–3. Precision axis → Task 5 + Task 6 PR frontier. `op_head` intentionally dropped (YAGNI; the type's rules applied at the predicted site subsume it) — noted here so the spec's `op` mention has an explicit resolution.

**Placeholder scan:** no TBD/TODO; each code task carries real code or a precise reuse instruction; empirical tasks (4, 6) carry exact commands + numeric gates.

**Type consistency:** `rule_to_type` (dict smirks→int) produced by Task 1, consumed by Tasks 2 & 6; `y_type`/`y_site` produced by Task 2, consumed by Tasks 3–4; `type_logits`/`site_logits` produced by Task 3, consumed by Tasks 4 & 6; `num_types = 1 + max(rule_to_type.values())` used consistently.

**Note on realism:** Tasks 1–3, 5 are deterministic/TDD; Tasks 4 and 6 are empirical with explicit go/no-go numeric gates (type-head > type-prior; recall@15 > 0.413). A null result at either gate is a documented, shippable outcome (Stage-4 fallback + precision-only contribution), not a failure.
