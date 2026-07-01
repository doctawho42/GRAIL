# Set-GFlowNet (Stage 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-step Set-GFlowNet over the rule forest whose terminal object is a *set* of metabolites with a PU-set-coverage reward, using the Stage-2a reranker as the forward policy.

**Architecture:** Forest environment (root substrate → rule-children via `generate_scored_with_details`); forward policy `P_F` = softmax over reranker scores of the frontier's children plus a learned STOP; reward `R(S)=exp(β·(TP−λ|S|))` over the annotated set; analytic leaf-based backward `P_B=1/#leaves`; Trajectory-Balance loss with a learned scalar `logZ`. Gated by a cheap depth-2 census (M0) that decides whether the multi-step *recall* claim is pursued or we fall back to the diversity-only method claim.

**Tech Stack:** Python, PyTorch, torch-geometric (`Batch`), RDKit, existing GRAIL modules (`model/reranker.py`, `model/generator.py`, `model/multistep.py`, `utils/transform.py`, `metrics.py`).

## Global Constraints

- `numpy<2` (RDKit / torch-geometric stack) — copied verbatim from CLAUDE.md.
- Rule environment preserved: every emitted node is a valid RDKit rule product; never leave the rule env.
- PU data: unobserved-applicable products are positive-unlabeled — never scored as hard negatives. The only size control is the `λ|S|` term.
- Matching: tautomer-InChIKey (`metrics._tautomer_inchikey`) everywhere a "hit" is decided.
- Selection discipline: `β, λ, K, D` chosen on val; test touched once; mean±std over ≥2 seeds via `scripts/aggregate_seeds.py`.
- Child enumeration is via `generator.generate_scored_with_details(state, top_k=..., compute_sites=False)` (same path the Stage-2a pool uses) → `(smiles, gen_score, rule_id, firing_atoms)`. Do NOT re-enumerate through `MetabolicTree` (that is the beam baseline only).
- Reranker forward signature (do not change): `BiEncoderReranker.forward(sub_graph, prod_batch, rule_prior, gen_score) -> Tensor[N]`, where `sub_graph = from_rdmol(parent)`, `prod_batch = Batch` of `from_rdmol(child)` graphs, `rule_prior = generator.rule_prior_logits[rule_ids]`, `gen_score = tensor(gen_scores)`.
- No Co-Authored-By / self-attribution in commit messages (user requirement).
- `make test` stays green after every task.

---

## Phase 0 — M0 depth-2 census (go/no-go, local, cheap)

### Task 1: Depth-2 reachability census

**Files:**
- Create: `scripts/census_multistep.py`
- Test: `grail_metabolism/tests/test_census_multistep.py`

**Interfaces:**
- Consumes: `generator.generate_scored_with_details`, `metrics._tautomer_inchikey`, `MolFrame.map`.
- Produces: `census_depth2(sub, annotated_ik, generator, top_k, max_pool) -> {"n_annot": int, "depth1": int, "depth2_only": int, "unreach": int}` — per-substrate counts of annotated metabolites reachable at depth 1, reachable at depth 2 but NOT depth 1, and unreachable within depth 2.

- [ ] **Step 1: Write the failing test** (synthetic generator stub — no dataset needed)

```python
# grail_metabolism/tests/test_census_multistep.py
from scripts.census_multistep import census_depth2

class _StubGen:
    """generate_scored_with_details(sub) -> children of `sub` from a fixed graph."""
    def __init__(self, graph):
        self._g = graph  # dict: smiles -> list of (child_smiles, gen_score, rule_id)
    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        return list(self._g.get(sub, []))

def _ik(s):  # identity "InChIKey" for the test: the smiles itself
    return s

def test_depth2_only_counts_chain_not_reachable_in_one_step(monkeypatch):
    import scripts.census_multistep as m
    monkeypatch.setattr(m, "_tautomer_inchikey", _ik)
    # root R -> A (depth1). A -> B (so B is depth2-only). R does NOT reach B directly.
    gen = _StubGen({"R": [("A", 0.9, 1)], "A": [("B", 0.8, 2)]})
    annotated_ik = {"A", "B"}          # both A and B are annotated metabolites of R
    out = census_depth2("R", annotated_ik, gen, top_k=10, max_pool=10)
    assert out["n_annot"] == 2
    assert out["depth1"] == 1          # A
    assert out["depth2_only"] == 1     # B (reachable only via R->A->B)
    assert out["unreach"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_census_multistep.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'census_depth2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/census_multistep.py
"""M0 census: how many annotated metabolites are depth-2-reachable-in-the-rule-env
but NOT depth-1? Go/no-go for the Stage-2b multi-step RECALL claim (the diversity
method claim does not depend on this). Cheap, CPU, no training."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey


def _children_ik(sub, generator, top_k, max_pool):
    out = {}
    for smiles, gen_score, rule_id, *_ in generator.generate_scored_with_details(
        sub, top_k=top_k, max_pool=max_pool, compute_sites=False
    ):
        ik = _tautomer_inchikey(smiles)
        if ik is not None and ik not in out:
            out[ik] = smiles
    return out  # ik -> smiles


def census_depth2(sub, annotated_ik, generator, top_k=200, max_pool=150):
    d1 = _children_ik(sub, generator, top_k, max_pool)          # ik -> smiles at depth 1
    depth1_hits = set(d1) & annotated_ik
    d2 = set()
    for child_smiles in d1.values():                            # expand each depth-1 child once
        d2 |= set(_children_ik(child_smiles, generator, top_k, max_pool))
    depth2_only = (d2 & annotated_ik) - depth1_hits
    reached = depth1_hits | (d2 & annotated_ik)
    return {
        "n_annot": len(annotated_ik),
        "depth1": len(depth1_hits),
        "depth2_only": len(depth2_only),
        "unreach": len(annotated_ik - reached),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_census_multistep.py -v`
Expected: PASS.

- [ ] **Step 5: Add the dataset-driven `main()` (aggregates the census over a split)**

```python
# append to scripts/census_multistep.py
def main() -> None:
    from grail_metabolism.config import DatasetConfig, GeneratorConfig
    from grail_metabolism.model.grail import _read_checkpoint
    from grail_metabolism.workflows.data import load_dataset_bundle
    from grail_metabolism.workflows.factory import build_generator

    ap = argparse.ArgumentParser(description="M0 depth-2 reachability census")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--substrates", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-pool", type=int, default=150)
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/census_multistep.json"))
    args = ap.parse_args()

    state = _read_checkpoint(args.gen_ckpt)
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False); generator.eval()

    cfg = DatasetConfig(use_clean_splits=True, standardize=False, cache_preprocessed=False,
                        max_test_substrates=args.substrates + 60, sampling_seed=0)
    bundle = load_dataset_bundle(cfg)
    frame = getattr(bundle, args.split)

    agg = {"n_annot": 0, "depth1": 0, "depth2_only": 0, "unreach": 0, "n_subs": 0}
    for sub, prods in list(frame.map.items())[: args.substrates]:
        annotated_ik = {_tautomer_inchikey(p) for p in prods} - {None}
        if not annotated_ik:
            continue
        c = census_depth2(sub, annotated_ik, generator, args.top_k, args.max_pool)
        for k in ("n_annot", "depth1", "depth2_only", "unreach"):
            agg[k] += c[k]
        agg["n_subs"] += 1
    agg["depth2_only_frac"] = agg["depth2_only"] / max(agg["n_annot"], 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/census_multistep.py grail_metabolism/tests/test_census_multistep.py
git commit -m "feat(census): M0 depth-2 reachability census (go/no-go for multi-step recall)"
```

**GATE (M0 decision):** run `python scripts/census_multistep.py --split test` (clean test) and, on Colab, against the GLORYx set. If `depth2_only_frac` is ~0 on **both** → drop the recall claim, keep the diversity-only method claim (Phases 1–3 still build; only the applied-eval in Task 9 is skipped). If `>0` on the external set → pursue the full hybrid.

---

## Phase 1 — Set-GFlowNet core (local, unit-testable)

### Task 2: `ForestState` — the environment state

**Files:**
- Create: `grail_metabolism/model/set_gflownet.py`
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Produces: `ForestState` with `.add(parent_ik, child_ik) -> ForestState` (returns a new state), `.leaves() -> list[str]` (InChIKeys with no children), `.terminal_set() -> frozenset[str]` (all non-root IKs), `.frontier() -> list[str]` (nodes with depth < D and set-size < K), `.depth_of(ik) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# grail_metabolism/tests/test_set_gflownet.py
from grail_metabolism.model.set_gflownet import ForestState

def test_forest_leaves_and_terminal_set():
    s = ForestState(root="R", max_depth=3, max_size=10)
    s = s.add("R", "A").add("R", "B").add("A", "C")   # R->A,R->B,A->C
    assert s.terminal_set() == frozenset({"A", "B", "C"})
    assert set(s.leaves()) == {"B", "C"}              # A has child C, so A is not a leaf; R is root
    assert s.depth_of("C") == 2

def test_forest_flat_set_all_leaves():
    s = ForestState(root="R", max_depth=3, max_size=10).add("R", "A").add("R", "B")
    assert set(s.leaves()) == {"A", "B"}              # flat single-step set: every member is a leaf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_forest_leaves_and_terminal_set -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# grail_metabolism/model/set_gflownet.py  (part 1)
"""Multi-step Set-GFlowNet over the rule forest. Terminal = a set of metabolites;
reward = PU set-coverage; forward policy = the Stage-2a reranker; backward = analytic
1/#leaves. See docs/superpowers/specs/2026-07-01-set-gflownet-stage2b-design.md."""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class ForestState:
    root: str
    max_depth: int
    max_size: int
    parent: Dict[str, str] = field(default_factory=dict)   # child_ik -> parent_ik

    def add(self, parent_ik: str, child_ik: str) -> "ForestState":
        new_parent = dict(self.parent)
        new_parent[child_ik] = parent_ik
        return replace(self, parent=new_parent)

    def terminal_set(self) -> FrozenSet[str]:
        return frozenset(self.parent.keys())

    def depth_of(self, ik: str) -> int:
        d = 0
        while ik in self.parent:
            ik = self.parent[ik]; d += 1
        return d

    def leaves(self) -> List[str]:
        parents = set(self.parent.values())
        return [ik for ik in self.parent if ik not in parents]

    def frontier(self) -> List[str]:
        nodes = [self.root] + list(self.parent.keys())
        return [n for n in nodes
                if self.depth_of(n) < self.max_depth and len(self.parent) < self.max_size]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/tests/test_set_gflownet.py
git commit -m "feat(set_gflownet): ForestState env (leaves, terminal set, frontier)"
```

### Task 3: PU set-coverage reward

**Files:**
- Modify: `grail_metabolism/model/set_gflownet.py`
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Produces: `set_coverage_logreward(terminal_set, annotated_ik, beta, lam) -> float` returning `β·(TP − λ·|S|)` (log-reward; the trainer exponentiates only where needed). `TP = |terminal_set ∩ annotated_ik|`.

- [ ] **Step 1: Write the failing test**

```python
# append to grail_metabolism/tests/test_set_gflownet.py
import math
from grail_metabolism.model.set_gflownet import set_coverage_logreward

def test_set_coverage_logreward_pu_and_size_penalty():
    annotated = {"A", "B"}
    # set {A, X}: TP=1 (A hits; X is unlabeled, NOT penalized as false), |S|=2
    lr = set_coverage_logreward(frozenset({"A", "X"}), annotated, beta=2.0, lam=0.1)
    assert math.isclose(lr, 2.0 * (1 - 0.1 * 2))          # 2*(1-0.2)=1.6
    # empty set: TP=0, |S|=0 -> logreward 0
    assert set_coverage_logreward(frozenset(), annotated, beta=2.0, lam=0.1) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_set_coverage_logreward_pu_and_size_penalty -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# grail_metabolism/model/set_gflownet.py  (part 2)
def set_coverage_logreward(terminal_set, annotated_ik, beta: float, lam: float) -> float:
    """log R(S) = beta * (TP - lam*|S|). PU-aware: non-annotated members cost only lam
    (size), never a false-negative penalty."""
    tp = len(terminal_set & annotated_ik)
    return float(beta) * (tp - float(lam) * len(terminal_set))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/tests/test_set_gflownet.py
git commit -m "feat(set_gflownet): PU set-coverage log-reward beta*(TP-lam|S|)"
```

### Task 4: Analytic leaf-based backward log-`P_B`

**Files:**
- Modify: `grail_metabolism/model/set_gflownet.py`
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Produces: `log_pb_trajectory(states) -> float` = `Σ_t log(1 / #leaves(state_{t+1}))` over the post-ADD states of a trajectory (STOP contributes 0). For a flat single-step set of size m built one element at a time this equals `Σ_{i=1..m} log(1/i)` (`= -log(m!)`), the analytic set-DAG backward.

- [ ] **Step 1: Write the failing test**

```python
# append to grail_metabolism/tests/test_set_gflownet.py
from grail_metabolism.model.set_gflownet import ForestState, log_pb_trajectory

def test_log_pb_flat_set_is_minus_log_factorial():
    s0 = ForestState(root="R", max_depth=3, max_size=10)
    s1 = s0.add("R", "A")               # 1 leaf
    s2 = s1.add("R", "B")               # 2 leaves
    s3 = s2.add("R", "C")               # 3 leaves
    lp = log_pb_trajectory([s1, s2, s3])
    assert abs(lp - (math.log(1/1) + math.log(1/2) + math.log(1/3))) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_log_pb_flat_set_is_minus_log_factorial -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# grail_metabolism/model/set_gflownet.py  (part 3)
def log_pb_trajectory(post_add_states) -> float:
    """Sum of log(1/#leaves) over the states reached AFTER each ADD action. The last-added
    node of a forest must be a current leaf, so P_B(remove leaf)=1/#leaves is the exact
    analytic backward for forest construction."""
    total = 0.0
    for st in post_add_states:
        n_leaves = max(len(st.leaves()), 1)
        total += math.log(1.0 / n_leaves)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/tests/test_set_gflownet.py
git commit -m "feat(set_gflownet): analytic leaf-based backward log P_B"
```

### Task 5: `SetGFlowNetTrainer` — reranker `P_F`, forest rollout, TB loss

**Files:**
- Modify: `grail_metabolism/model/set_gflownet.py`
- Test: `grail_metabolism/tests/test_set_gflownet.py`

**Interfaces:**
- Consumes: `ForestState`, `set_coverage_logreward`, `log_pb_trajectory`, `generator.generate_scored_with_details`, `generator.rule_prior_logits`, `BiEncoderReranker.forward`, `utils.transform.from_rdmol`, `torch_geometric.data.Batch`.
- Produces:
  - `StopHead(nn.Module)` — pooled-frontier → scalar STOP logit.
  - `SetGFlowNetTrainer(generator, reranker, config, annotated_ik_fn)` with `.candidate_children(state_smiles) -> list[(child_smiles, gen_score, rule_id)]` (cached), `.policy_logits(state, cand) -> Tensor[N+1]` (N child logits from the reranker + 1 STOP logit), `.sample_forest(root) -> (ForestState, sum_log_pf, post_add_states)`, `.tb_loss(root) -> Tensor`, `.fit(substrates, epochs)`.
- `config` = the existing `GFlowNetConfig` extended with `lam: float`, `max_size: int`, `top_k: int` (add fields; defaults `lam=0.1`, `max_size=15`, `top_k=200`). `max_depth`, `beta`, `epsilon`, `lr`, `logz_lr`, `batch_substrates`, `epochs` already exist. (Adding these fields is part of this task's Step 3; update `config.py` and keep `make test` green.)

- [ ] **Step 1: Write the failing test** (stub generator + a tiny real reranker; assert loss is finite and gradients flow)

```python
# append to grail_metabolism/tests/test_set_gflownet.py
import torch
from grail_metabolism.model.reranker import BiEncoderReranker
from grail_metabolism.utils.transform import SINGLE_NODE_DIM
from grail_metabolism.config import GFlowNetConfig
from grail_metabolism.model.set_gflownet import SetGFlowNetTrainer

class _MiniGen:
    rule_prior_logits = torch.zeros(8)
    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        # R->A,B ; A->C ; others terminal. gen_scores arbitrary, rule_ids < 8.
        return {"CCO": [("CCO O", 0.9, 1), ("CC", 0.5, 2)], "CCO O": [("CCOO", 0.7, 3)]}.get(sub, [])

def test_tb_loss_is_finite_and_backprops():
    gen = _MiniGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1)
    cfg.lam = 0.1; cfg.max_size = 5
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())
    torch.manual_seed(0)
    loss = trainer.tb_loss("CCO")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in rr.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py::test_tb_loss_is_finite_and_backprops -v`
Expected: FAIL with `ImportError` (`SetGFlowNetTrainer`).

- [ ] **Step 3: Write minimal implementation** (real code; children enumerated via the generator, scored via the reranker)

```python
# grail_metabolism/model/set_gflownet.py  (part 4)
import torch
import torch.nn.functional as F
from torch import nn
from rdkit import Chem
from torch_geometric.data import Batch
from ..utils.transform import from_rdmol


class StopHead(nn.Module):
    """Scalar STOP logit from a pooled representation of the current frontier."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, frontier_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(frontier_embed).view(())


class SetGFlowNetTrainer:
    def __init__(self, generator, reranker, config, annotated_ik_fn, device=None):
        self.generator = generator
        self.reranker = reranker
        self.config = config
        self.annotated_ik_fn = annotated_ik_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_z = nn.Parameter(torch.zeros(1, device=self.device))
        self.stop_head = StopHead(reranker.embed_dim).to(self.device)
        self._child_cache = {}
        self.loss_history_ = []

    def candidate_children(self, state_smiles):
        if state_smiles not in self._child_cache:
            seen, out = set(), []
            for smiles, gscore, rid, *_ in self.generator.generate_scored_with_details(
                state_smiles, top_k=self.config.top_k, compute_sites=False
            ):
                if smiles in seen:
                    continue
                seen.add(smiles); out.append((smiles, float(gscore), int(rid)))
            self._child_cache[state_smiles] = out
        return self._child_cache[state_smiles]

    def _reranker_child_logits(self, parent_smiles, children):
        mol = Chem.MolFromSmiles(parent_smiles)
        sub_graph = from_rdmol(mol)
        prod_batch = Batch.from_data_list(
            [from_rdmol(Chem.MolFromSmiles(c)) for c, _, _ in children]
        ).to(self.device)
        rule_prior = self.generator.rule_prior_logits.to(self.device)[
            torch.tensor([rid for _, _, rid in children], device=self.device)
        ]
        gen_score = torch.tensor([g for _, g, _ in children], device=self.device)
        return self.reranker(sub_graph.to(self.device), prod_batch, rule_prior, gen_score)  # [N]

    def sample_forest(self, root):
        cfg = self.config
        state = ForestState(root=root, max_depth=cfg.max_depth, max_size=getattr(cfg, "max_size", 15))
        ik = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
        smiles_of = {root: root}
        sum_log_pf, post_add = [], []
        for _ in range(getattr(cfg, "max_size", 15)):
            # Gather candidate ADD actions across the frontier (parent, child, gscore, rid).
            actions = []
            for parent_ik in state.frontier():
                p_smiles = smiles_of.get(parent_ik, parent_ik)
                kids = self.candidate_children(p_smiles)
                for c_smiles, g, rid in kids:
                    c_ik = ik(c_smiles)
                    if c_ik in state.terminal_set() or c_ik == root:
                        continue
                    actions.append((parent_ik, p_smiles, c_smiles, c_ik, g, rid))
            # Build the logit vector: [reranker child logits...] + [stop logit].
            stop_logit = self.stop_head(self._frontier_embed(state, smiles_of)).view(1)
            if not actions:
                break
            by_parent = {}
            for a in actions:
                by_parent.setdefault(a[1], []).append(a)
            child_logits = []
            for p_smiles, group in by_parent.items():
                child_logits.append(self._reranker_child_logits(
                    p_smiles, [(c, g, rid) for _, _, c, _, g, rid in group]))
            child_logits = torch.cat(child_logits) if child_logits else torch.zeros(0, device=self.device)
            logits = torch.cat([child_logits, stop_logit])
            log_probs = F.log_softmax(logits, dim=0)
            idx = self._sample_index(log_probs.detach())
            sum_log_pf.append(log_probs[idx])
            if idx == len(actions):   # STOP
                break
            flat = [a for p in by_parent.values() for a in p]  # same order as child_logits
            parent_ik, _, c_smiles, c_ik, _, _ = flat[idx]
            state = state.add(parent_ik, c_ik); smiles_of[c_ik] = c_smiles
            post_add.append(state)
        total = torch.stack(sum_log_pf).sum() if sum_log_pf else torch.zeros((), device=self.device)
        return state, total, post_add

    def _frontier_embed(self, state, smiles_of):
        # Mean of the reranker's substrate encoding over frontier nodes (cheap pooled rep).
        embs = []
        for ik_ in state.frontier():
            mol = Chem.MolFromSmiles(smiles_of.get(ik_, ik_))
            if mol is not None:
                embs.append(self.reranker.encode_substrate(from_rdmol(mol).to(self.device)))
        return torch.stack(embs).mean(0) if embs else torch.zeros(self.reranker.embed_dim, device=self.device)

    def _sample_index(self, log_probs):
        n = int(log_probs.shape[0])
        if self.config.epsilon > 0.0 and float(torch.rand(())) < self.config.epsilon:
            return int(torch.randint(0, n, (1,)))
        return int(torch.multinomial(log_probs.exp(), 1))

    def tb_loss(self, root):
        state, sum_log_pf, post_add = self.sample_forest(root)
        log_r = set_coverage_logreward(
            state.terminal_set(), set(self.annotated_ik_fn(root)),
            self.config.beta, getattr(self.config, "lam", 0.1))
        log_pb = log_pb_trajectory(post_add)
        return (self.log_z.squeeze() + sum_log_pf - log_pb - log_r) ** 2

    def fit(self, substrates, epochs=None, verbose=False):
        cfg = self.config
        epochs = epochs if epochs is not None else cfg.epochs
        subs = [s for s in substrates if s]
        params = [p for p in self.reranker.parameters() if p.requires_grad] + list(self.stop_head.parameters())
        opt = torch.optim.Adam([{"params": params, "lr": cfg.lr},
                                {"params": [self.log_z], "lr": cfg.logz_lr}])
        self.loss_history_ = []
        for epoch in range(epochs):
            self.reranker.train()
            order = torch.randperm(len(subs)).tolist()
            ep, nb = 0.0, 0
            for start in range(0, len(subs), cfg.batch_substrates):
                batch = [subs[i] for i in order[start:start + cfg.batch_substrates]]
                losses = [self.tb_loss(s) for s in batch]
                loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep += float(loss.item()); nb += 1
            self.loss_history_.append(ep / max(nb, 1))
            if verbose:
                print(f"setgfn epoch={epoch+1} tb_loss={self.loss_history_[-1]:.4f} logZ={float(self.log_z):.3f}", flush=True)
        return self
```

- [ ] **Step 4: Add the two small reranker helpers the trainer needs**

In `grail_metabolism/model/reranker.py`, expose (a) `self.embed_dim` (the GraphEncoder output dim, already computed in `__init__` — assign it to `self.embed_dim`), and (b) `def encode_substrate(self, sub_graph) -> Tensor` returning the pooled substrate embedding (factor the existing substrate-encoding path in `forward` into this method and call it from `forward`). Do not change `forward`'s external signature.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_set_gflownet.py -v`
Expected: PASS (loss finite, gradients flow).

- [ ] **Step 6: Run the full suite**

Run: `make test`
Expected: all pass (existing 81 + new set_gflownet tests).

- [ ] **Step 7: Commit**

```bash
git add grail_metabolism/model/set_gflownet.py grail_metabolism/model/reranker.py grail_metabolism/tests/test_set_gflownet.py grail_metabolism/config.py
git commit -m "feat(set_gflownet): SetGFlowNetTrainer (reranker P_F, forest rollout, TB loss)"
```

---

## Phase 2 — bootstrap + evaluation

### Task 6: Intermediate-node bootstrap pairs

**Files:**
- Modify: `grail_metabolism/workflows/reranker.py`
- Test: `grail_metabolism/tests/test_reranker.py`

**Interfaces:**
- Produces: `build_intermediate_pairs(generator, molframe, n_substrates, top_k) -> list[_BiExample]` — for each substrate whose annotated set contains a metabolite `m2` that is depth-2-only (via the Task-1 `census_depth2` logic: reachable as `root→m1→m2`, not depth-1), emit a `_BiExample` rooted at `m1` with `m2` as the positive hit among `m1`'s rule-children. These are concatenated with the existing depth-1 examples for the reranker fine-tune.

- [ ] **Step 1: Write the failing test** (stub generator; assert the depth-2 pair is rooted at the intermediate)

```python
# append to grail_metabolism/tests/test_reranker.py
from grail_metabolism.workflows.reranker import build_intermediate_pairs

def test_intermediate_pairs_rooted_at_intermediate(tiny_generator, tiny_molframe_depth2):
    ex = build_intermediate_pairs(tiny_generator, tiny_molframe_depth2, n_substrates=5, top_k=10)
    assert any(e.sub == "m1_smiles" and e.hit_mask.any() for e in ex)
```

(Define `tiny_generator` / `tiny_molframe_depth2` fixtures in the test file mirroring the existing reranker-test fixtures, with one `root→m1→m2` chain where `m2` is annotated and depth-2-only.)

- [ ] **Step 2: Run test to verify it fails** — `ImportError: build_intermediate_pairs`.
- [ ] **Step 3: Implement** `build_intermediate_pairs` reusing `census_depth2` (import from `scripts.census_multistep`) to find `(m1, m2)` chains, then `_bi_example_from_pool(m1, pool_of_m1, {m2}, prior, num_rules)` (existing helper) to build each example.
- [ ] **Step 4: Run test — PASS.**
- [ ] **Step 5: Commit** `feat(reranker): intermediate depth-2 bootstrap pairs for the forest policy`.

### Task 7: Diversity / coverage / modes metrics

**Files:**
- Create: `grail_metabolism/eval/diversity.py`
- Test: `grail_metabolism/tests/test_diversity.py`

**Interfaces:**
- Produces: `modes_discovered(sampled_sets, annotated_ik) -> int` (distinct annotated IKs found across all sampled sets); `mean_pairwise_tanimoto(smiles_list) -> float`; `n_unique_scaffolds(smiles_list) -> int`; `set_size_calibration(sampled_sets, annotated_ik) -> float` (mean |S| − mean |annotated|).

- [ ] **Step 1: Write the failing test**

```python
# grail_metabolism/tests/test_diversity.py
from grail_metabolism.eval.diversity import modes_discovered, mean_pairwise_tanimoto

def test_modes_discovered_counts_distinct_hits_across_sets():
    sets = [frozenset({"A", "X"}), frozenset({"B", "Y"}), frozenset({"A"})]
    assert modes_discovered(sets, annotated_ik={"A", "B", "C"}) == 2   # A, B found; C never

def test_mean_pairwise_tanimoto_identical_is_one():
    assert abs(mean_pairwise_tanimoto(["CCO", "CCO"]) - 1.0) < 1e-6
```

- [ ] **Step 2: Run — FAIL (ImportError).**
- [ ] **Step 3: Implement** using RDKit Morgan fingerprints (`AllChem.GetMorganFingerprintAsBitVect`, `DataStructs.TanimotoSimilarity`) and `MurckoScaffold` for scaffolds.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(eval): diversity/coverage/modes metrics for set generation`.

---

## Phase 3 — orchestration and runs (Colab GPU for M1/M2)

### Task 8: `scripts/run_gflownet.py` — train + dual-eval + JSON

**Files:**
- Create: `scripts/run_gflownet.py`

**Interfaces:**
- Consumes: everything above. Mirrors `scripts/run_reranker_gate.py` structure (load generator + reranker ckpts, build/cache pools, construct `SetGFlowNetTrainer`, fit, eval, write JSON tagged by seed/split like `reranker_gate_bi*`).

- [ ] **Step 1: Write `main()`** with args `--train-substrates --test-substrates --eval-split {val,test} --seed --beta --lam --max-depth --max-size --top-k --epochs --workers --out`. Load the Stage-2a reranker checkpoint as the `P_F` init (optionally fine-tuned via Task 6 pairs first). Build `annotated_ik_fn` from the split's `MolFrame.map` via `_tautomer_inchikey`.
- [ ] **Step 2: Evaluation** — for each eval substrate: sample M forests (config M, default 32); report recall@K (single highest-log-reward set truncated to K), plus `modes_discovered`, `mean_pairwise_tanimoto`, `n_unique_scaffolds`, `set_size_calibration`; and the baselines at matched K (reranker top-K from Stage 2a, `MetabolicTree.beam_search`, temperature-sampling). Write results JSON with a `config`/`metrics` block matching `reranker_gate_bi*.json` so `aggregate_seeds.py` works unchanged.
- [ ] **Step 3: Smoke-run locally** on ~20 substrates, depth≤2, M=4 — assert JSON is written and recall/diversity fields are present.

Run: `python -u scripts/run_gflownet.py --train-substrates 20 --eval-split val --max-depth 2 --max-size 8 --epochs 2 --top-k 50`
Expected: JSON at `results/gflownet_*.json` with `reranker`/`beam`/`gflownet` recall + diversity fields.

- [ ] **Step 4: Commit** `feat(scripts): run_gflownet.py — train Set-GFlowNet + dual eval matrix`.

### Task 9: Colab notebook + M1 (small) then M2 (scale-up)

**Files:**
- Create: `docs/benchmark/colab/gflownet_buildout.ipynb` (mirror `reranker_buildout.ipynb`: Drive mount + data symlinks + `git reset --hard origin/metabench-reranker`).

- [ ] **M1 (small GPU):** `--train-substrates 300 --max-depth 2 --max-size 10 --epochs 10 --eval-split val`. GATE: TB loss decreases and converges; sampled-set mean log-reward > a random-policy control; `modes_discovered` > beam at matched K. If not, reduce `max_size`/apply SubTB (config flag) before scaling.
- [ ] **M2 (scale-up):** `--train-substrates 1200 --eval-split test --test-substrates 2000 --seed 0/1/2`, plus the external multi-gen eval (only if the M0 gate passed). Then `python scripts/aggregate_seeds.py --glob 'results/gflownet_*seed*.json'` for mean±std. Ingest the diversity/coverage table + (if applicable) the external recall lift into `docs/benchmark/stage2b_setgflownet.md` and memory.
- [ ] **Commit** the notebook: `feat(colab): gflownet_buildout notebook (M1 small -> M2 scale-up)`.

---

## Self-review notes (author)

- **Spec coverage:** §3.1 env → Task 2; §3.2 `P_F` → Task 5; §3.3 reward → Task 3; §3.4 `P_B`+TB → Tasks 4–5; §3.5 bootstrap → Task 6; §4 eval → Tasks 7–8; §5 milestones → Task 1 (M0), Task 9 (M1/M2); §6 components covered; §7 risks: SubTB flag (Task 9 M1 gate), leaf-`P_B` analytic (Task 4), PU reward (Task 3), M0 fallback (Task 1 gate).
- **Fallback wiring:** if the Task-1 M0 gate says "diversity-only", Task 9's external recall eval is skipped; everything else (the method claim) still ships.
- **Type consistency:** `ForestState` (root/parent/leaves/terminal_set/frontier), `set_coverage_logreward(terminal_set, annotated_ik, beta, lam)`, `log_pb_trajectory(post_add_states)`, `SetGFlowNetTrainer(generator, reranker, config, annotated_ik_fn)` used consistently across Tasks 2–5, 9.
