# way2drug deploy build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A file-to-file CLI (`grail-metabolites INPUT OUTPUT`) that predicts probable metabolites with scores using the deployed GRAIL model, its release ranking, and a bank with the BioTransformer templates excluded.

**Architecture:** Promote the release ranking into the library; build a released generator checkpoint by subsetting the deployed one to the released bank (no retraining); wrap both in a thin CLI that reads a SMILES file and writes a TSV of ranked metabolites.

**Tech Stack:** Python, PyTorch, torch-geometric, RDKit (pinned), numpy<2.

**Spec:** `docs/superpowers/specs/2026-09-12-way2drug-deploy-design.md`

## Global Constraints

- Uses the released bank `grail_metabolism/resources/extended_smirks_released.txt` (6970 templates); never ships or loads `extended_smirks.txt`.
- Excluding the 611 BioTransformer templates costs zero references (`results/licence_removal_cost__clean_test.json`); the released generator must reproduce the full generator's per-rule scores on kept rules exactly.
- No Claude/AI attribution in any commit or file. All comments/docstrings in English.
- The filter has no per-rule parameters and ships unchanged.
- Reproducible from tracked inputs: released bank + released checkpoints, no full bank or corpus needed at deploy.

## File Structure

- `grail_metabolism/model/ranking.py` (create) — the one release-ranking implementation (`competition_ranks`, `reciprocal_rank_fusion`, `release_order`).
- `scripts/typed_edit/_rrf.py` (modify) — re-export from the library so there is one implementation.
- `scripts/build_released_checkpoint.py` (create) — subset the deployed generator to the released bank; write `artifacts/full5000_released/checkpoints/{generator,filter}.pt`.
- `grail_metabolism/deploy/__init__.py`, `grail_metabolism/deploy/predict_cli.py` (create) — the CLI: a testable core plus an argparse entry point.
- `pyproject.toml` (modify) — add the `grail-metabolites` console script.
- `deploy/README.md`, `deploy/requirements-deploy.txt`, `deploy/environment.yml`, `deploy/NOTICE.md` (create) — packaging and docs.
- `grail_metabolism/tests/test_deploy.py` (create) — guards for ranking parity, checkpoint subset, and the CLI core.

---

### Task 1: Release ranking in the library

**Files:**
- Create: `grail_metabolism/model/ranking.py`
- Modify: `scripts/typed_edit/_rrf.py`
- Test: `grail_metabolism/tests/test_deploy.py`

**Interfaces:**
- Produces: `competition_ranks(items, score) -> list[int]`; `reciprocal_rank_fusion(cands, k=60, filter_key="filter", generator_key="generator") -> list[dict]` (returns the candidate dicts in fused order); `release_order(pool, self_key, cap=100, rrf_k=60) -> list[dict]` (ordered candidate dicts, parent dropped).

- [ ] **Step 1: Write the failing test** — parity with the existing analysis order.

```python
# grail_metabolism/tests/test_deploy.py
from grail_metabolism.model.ranking import release_order

def _pool():
    # filter, generator, key; two candidates share a key to exercise dedup
    return [
        {"smiles": "A", "filter": 0.9, "generator": 0.2, "key": "k1"},
        {"smiles": "B", "filter": 0.1, "generator": 0.9, "key": "k2"},
        {"smiles": "A2", "filter": 0.5, "generator": 0.5, "key": "k1"},
        {"smiles": "S", "filter": 0.8, "generator": 0.8, "key": "parent"},
    ]

def test_release_order_dedups_caps_fuses_and_drops_parent():
    ordered = release_order(_pool(), self_key="parent", cap=100, rrf_k=60)
    keys = [c["key"] for c in ordered]
    assert "parent" not in keys          # parent dropped
    assert keys.count("k1") == 1         # deduped by key
    assert set(keys) == {"k1", "k2"}     # only non-parent keys survive
    # k1 kept the higher product row (0.9*0.2=0.18 vs 0.5*0.5=0.25 -> keeps A2)
    kept_k1 = next(c for c in ordered if c["key"] == "k1")
    assert kept_k1["smiles"] == "A2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_release_order_dedups_caps_fuses_and_drops_parent -v`
Expected: FAIL with `ModuleNotFoundError: grail_metabolism.model.ranking`.

- [ ] **Step 3: Write minimal implementation** — transcribe the registered fusion and the deployed order of operations.

```python
# grail_metabolism/model/ranking.py
"""The one implementation of the release ranking (H7 fusion, H9 cap, P2 dedup order).

Order of operations, which is not interchangeable: dedup by tautomer key in descending order of the
product of the two component scores, then cap by generator score, then fuse by reciprocal rank,
then drop the parent. Promoted here from the analysis scripts so there is a single implementation
the deploy and the measurements share.
"""
from __future__ import annotations

from typing import Callable, Sequence

RRF_K = 60   # Cormack, Clarke and Buettcher 2009; not tuned
CAP = 100    # H9


def competition_ranks(items: Sequence, score: Callable) -> list:
    """1-based competition ranks, descending by `score`; tied items share the lower rank."""
    order = sorted(range(len(items)), key=lambda i: -score(items[i]))
    out, prev, cur = [0] * len(items), None, 1
    for pos, i in enumerate(order, 1):
        v = score(items[i])
        if v != prev:
            prev, cur = v, pos
        out[i] = cur
    return out


def reciprocal_rank_fusion(cands, k=RRF_K, filter_key="filter", generator_key="generator"):
    """Order candidate dicts by reciprocal rank fusion of their two component scores."""
    rf = competition_ranks(cands, lambda c: c[filter_key])
    rg = competition_ranks(cands, lambda c: c[generator_key])
    idx = sorted(range(len(cands)), key=lambda i: -(1.0 / (k + rf[i]) + 1.0 / (k + rg[i])))
    return [cands[i] for i in idx]


def release_order(pool, self_key, cap=CAP, rrf_k=RRF_K):
    """The deployed order, returning the ordered candidate dicts (parent dropped)."""
    cands = sorted(pool, key=lambda c: -(c["filter"] * c["generator"]))
    seen, dedup = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        dedup.append(c)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:cap]
    return [c for c in reciprocal_rank_fusion(keep, k=rrf_k) if c["key"] != self_key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_release_order_dedups_caps_fuses_and_drops_parent -v`
Expected: PASS.

- [ ] **Step 5: Make `_rrf.py` re-export, and add a parity guard against the old implementation**

Replace the bodies in `scripts/typed_edit/_rrf.py` with imports, keeping the module's public names so its callers are unchanged:

```python
# scripts/typed_edit/_rrf.py
"""Re-exports the one fusion implementation, now in the library. Import from here or from
grail_metabolism.model.ranking; they are the same functions."""
from grail_metabolism.model.ranking import RRF_K, competition_ranks, reciprocal_rank_fusion

def rrf_order(cands, k=RRF_K, filter_key="filter", generator_key="generator"):
    return reciprocal_rank_fusion(cands, k=k, filter_key=filter_key, generator_key=generator_key)
```

Add a parity test that the library `release_order` key-sequence equals the pre-existing
`dehydrogenation_diagnostic.deployed_order` on the same pool:

```python
def test_release_order_matches_the_analysis_deployed_order():
    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from dehydrogenation_diagnostic import deployed_order  # the analysis order
    from grail_metabolism.model.ranking import release_order
    pool = [
        {"smiles": "A", "filter": 0.9, "generator": 0.2, "key": "k1"},
        {"smiles": "B", "filter": 0.1, "generator": 0.9, "key": "k2"},
        {"smiles": "C", "filter": 0.4, "generator": 0.7, "key": "k3"},
    ]
    lib = [c["key"] for c in release_order(pool, self_key="parent")]
    assert lib == deployed_order(pool, "parent")
```

- [ ] **Step 6: Run both tests, then commit**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py -v`
Expected: both PASS.

```bash
git add grail_metabolism/model/ranking.py scripts/typed_edit/_rrf.py grail_metabolism/tests/test_deploy.py
git commit -m "feat: release ranking promoted into the library, one implementation"
```

---

### Task 2: Build the released generator checkpoint (subset)

**Files:**
- Create: `scripts/build_released_checkpoint.py`
- Test: `grail_metabolism/tests/test_deploy.py`

**Interfaces:**
- Consumes: `artifacts/full5000_implicit/checkpoints/{generator,filter}.pt`, `extended_smirks.txt`, `extended_smirks_released.txt`.
- Produces: `artifacts/full5000_released/checkpoints/{generator,filter}.pt`; a pure function `subset_rule_tensors(state_dict, kept_idx) -> dict` that selects rows of the per-rule tensors.

**Per-rule tensors** (rows indexed by bank order, from `grail_metabolism/model/generator.py`): top-level `bias`, `rule_prior_logits`, `pos_weight`, `propensity_weight`; and the RuleParse embedding whose state key ends with `id_embedding.weight` (rows = rules, shape `(num_rules, embed_dim)`). Every other tensor is rule-agnostic and copies unchanged. `rule_support`/`rule_meta` are non-persistent buffers and are absent from the state dict.

- [ ] **Step 1: Write the failing test** for the pure subset function.

```python
def test_subset_rule_tensors_selects_rows_for_per_rule_tensors_only():
    import torch
    from scripts.build_released_checkpoint import subset_rule_tensors
    sd = {
        "bias": torch.tensor([10.0, 11.0, 12.0]),
        "rule_prior_logits": torch.tensor([0.0, 1.0, 2.0]),
        "pos_weight": torch.ones(3),
        "propensity_weight": torch.ones(3),
        "parse.id_embedding.weight": torch.tensor([[1.0], [2.0], [3.0]]),
        "encoder.lin.weight": torch.eye(4),   # rule-agnostic, must be untouched
    }
    out = subset_rule_tensors(sd, kept_idx=[0, 2])
    assert out["bias"].tolist() == [10.0, 12.0]
    assert out["rule_prior_logits"].tolist() == [0.0, 2.0]
    assert out["parse.id_embedding.weight"].tolist() == [[1.0], [3.0]]
    assert torch.equal(out["encoder.lin.weight"], sd["encoder.lin.weight"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_subset_rule_tensors_selects_rows_for_per_rule_tensors_only -v`
Expected: FAIL with `ModuleNotFoundError` / no `subset_rule_tensors`.

- [ ] **Step 3: Write the subset function and the build script.**

```python
# scripts/build_released_checkpoint.py
"""Build a released generator checkpoint by subsetting the deployed one to the released bank.

The deployed generator was trained on the full 7581-template bank and its per-rule tensors are
indexed to that bank's order. The released bank is an exact, order-preserving subset (6970). This
selects the kept rows of every per-rule tensor and writes a checkpoint whose rule list is the
released bank, so the loader matches it against the released bank. The removed 611 templates cost
zero references, so the released generator reproduces the full generator's scores on kept rules;
that is asserted, not assumed.

    python scripts/build_released_checkpoint.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FULL_BANK = ROOT / "grail_metabolism/resources/extended_smirks.txt"
RELEASED_BANK = ROOT / "grail_metabolism/resources/extended_smirks_released.txt"
SRC = ROOT / "artifacts/full5000_implicit/checkpoints"
DST = ROOT / "artifacts/full5000_released/checkpoints"

# top-level per-rule tensors; the id embedding is matched by suffix because its module prefix
# depends on the parser attribute name
PER_RULE_EXACT = {"bias", "rule_prior_logits", "pos_weight", "propensity_weight"}
PER_RULE_SUFFIX = "id_embedding.weight"


def _is_per_rule(name: str) -> bool:
    return name in PER_RULE_EXACT or name.endswith(PER_RULE_SUFFIX)


def subset_rule_tensors(state_dict: dict, kept_idx) -> dict:
    idx = torch.as_tensor(list(kept_idx), dtype=torch.long)
    out = {}
    for name, tensor in state_dict.items():
        out[name] = tensor.index_select(0, idx) if _is_per_rule(name) else tensor
    return out


def _rules(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    full = _rules(FULL_BANK)
    released = _rules(RELEASED_BANK)
    pos = {r: i for i, r in enumerate(full)}
    missing = [r for r in released if r not in pos]
    if missing:
        print(f"REFUSING: {len(missing)} released rules are not in the full bank", file=sys.stderr)
        return 1
    kept_idx = [pos[r] for r in released]
    # order-preserving check: the kept rows, in full order, are exactly the released bank
    assert [full[i] for i in kept_idx] == released

    payload = torch.load(SRC / "generator.pt", map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    payload["state_dict"] = subset_rule_tensors(sd, kept_idx)
    payload["rules"] = released   # loader matches this against the released bank
    DST.mkdir(parents=True, exist_ok=True)
    torch.save(payload, DST / "generator.pt")
    shutil.copy2(SRC / "filter.pt", DST / "filter.pt")  # rule-agnostic, unchanged

    # verify: the subset tensors equal the source rows for kept rules
    for name, tensor in sd.items():
        if _is_per_rule(name):
            assert torch.equal(payload["state_dict"][name], tensor.index_select(0, torch.tensor(kept_idx)))
    print(f"Wrote {DST}/generator.pt ({len(released)} rules) and copied filter.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test, then run the build**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_subset_rule_tensors_selects_rows_for_per_rule_tensors_only -v`
Expected: PASS.
Run: `python scripts/build_released_checkpoint.py`
Expected: prints `Wrote .../generator.pt (6970 rules) and copied filter.pt`, exit 0.

- [ ] **Step 5: Add a forward-parity smoke** (kept-rule scores identical) and commit.

```python
def test_released_generator_scores_match_full_on_kept_rules():
    import torch, pathlib
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.workflows.factory import build_generator
    root = pathlib.Path(__file__).resolve().parents[2]
    def load(ckpt, bank):
        rules = [l.strip() for l in open(bank) if l.strip()]
        p = torch.load(ckpt, map_location="cpu", weights_only=False)
        g = build_generator(GeneratorConfig(**p["arch"]), rules); g.load_state_dict(p["state_dict"], strict=False); g.eval()
        return g, rules
    full_g, full_rules = load(root / "artifacts/full5000_implicit/checkpoints/generator.pt",
                              root / "grail_metabolism/resources/extended_smirks.txt")
    rel_g, rel_rules = load(root / "artifacts/full5000_released/checkpoints/generator.pt",
                            root / "grail_metabolism/resources/extended_smirks_released.txt")
    sub = "CCO"
    with torch.no_grad():
        fs, _ = full_g.score_rules(sub, return_mask=True)
        rs, _ = rel_g.score_rules(sub, return_mask=True)
    pos = {r: i for i, r in enumerate(full_rules)}
    kept = [pos[r] for r in rel_rules]
    assert torch.allclose(torch.as_tensor(rs), torch.as_tensor(fs)[kept], atol=1e-5)
```

Run: `python -m pytest grail_metabolism/tests/test_deploy.py -v`
Expected: PASS (skip this test with `pytest.mark.skipif` if the full checkpoint is absent in the checkout).

```bash
git add scripts/build_released_checkpoint.py grail_metabolism/tests/test_deploy.py
git add -f artifacts/full5000_released/checkpoints/generator.pt artifacts/full5000_released/checkpoints/filter.pt
python scripts/sync_tracked_artifacts.py
git add .gitignore
git commit -m "feat: released generator checkpoint subset to the released bank"
```

---

### Task 3: Deploy CLI core + argparse entry

**Files:**
- Create: `grail_metabolism/deploy/__init__.py`, `grail_metabolism/deploy/predict_cli.py`
- Modify: `pyproject.toml`
- Test: `grail_metabolism/tests/test_deploy.py`

**Interfaces:**
- Consumes: `release_order` (Task 1); the released pair (Task 2).
- Produces: `predict_rows(model, rows, top_k, timeout_seconds) -> list[dict]` where a model exposes `.rank(smiles, top_k) -> list[(metabolite_smiles, score)]`; `read_input(path) -> list[(id, smiles)]`; `write_tsv(path, out_rows)`; `main(argv)`.

The core takes a `model` with a `.rank` method so it is testable with a stub (no torch/RDKit in the unit test). The real model adapter (loads checkpoints, generates, filters, calls `release_order`) is exercised by the slower end-to-end smoke.

- [ ] **Step 1: Write the failing test** for the core with a stub model.

```python
def test_predict_rows_handles_ok_and_unparseable_and_limits_top_k():
    from grail_metabolism.deploy.predict_cli import predict_rows
    class Stub:
        def rank(self, smiles, top_k):
            if smiles == "BAD":
                raise ValueError("unparseable")
            return [("m1", 0.9), ("m2", 0.8), ("m3", 0.7)][:top_k]
    rows = predict_rows(Stub(), [("s1", "CCO"), ("s2", "BAD")], top_k=2, timeout_seconds=1)
    assert [(r["parent_id"], r["rank"], r["metabolite_smiles"], r["status"]) for r in rows] == [
        ("s1", 1, "m1", "ok"), ("s1", 2, "m2", "ok"),
        ("s2", 0, "", "no_parse"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_predict_rows_handles_ok_and_unparseable_and_limits_top_k -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the core.**

```python
# grail_metabolism/deploy/predict_cli.py
"""File-to-file metabolite prediction for way2drug: read a SMILES file, write a TSV of ranked
metabolites with scores, using the deployed model, the release ranking, and the released bank."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HEADER = ["parent_id", "rank", "metabolite_smiles", "score", "status"]


def read_input(path: str):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            ident, smiles = line.split("\t", 1)
        else:
            ident, smiles = str(n), line
        rows.append((ident.strip(), smiles.strip()))
    return rows


def predict_rows(model, rows, top_k, timeout_seconds):
    out = []
    for ident, smiles in rows:
        try:
            ranked = model.rank(smiles, top_k=top_k)
        except TimeoutError:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "timeout"})
            continue
        except Exception:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "no_parse"})
            continue
        if not ranked:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "no_metabolites"})
            continue
        for rank, (met, score) in enumerate(ranked, 1):
            out.append({"parent_id": ident, "rank": rank, "metabolite_smiles": met,
                        "score": round(float(score), 6), "status": "ok"})
    return out


def write_tsv(path, out_rows):
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=str(Path(path).parent), suffix=".tmp")
    try:
        tmp.write("\t".join(HEADER) + "\n")
        for r in out_rows:
            tmp.write("\t".join(str(r[c]) for c in HEADER) + "\n")
        tmp.close()
        os.replace(tmp.name, path)   # atomic
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="grail-metabolites",
                                 description="Predict metabolites for a file of SMILES.")
    ap.add_argument("input", help="input file: one SMILES per line, or id<TAB>SMILES")
    ap.add_argument("output", help="output TSV path")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--timeout-seconds", type=int, default=120)
    ap.add_argument("--format", choices=["tsv"], default="tsv")
    args = ap.parse_args(argv)

    from grail_metabolism.deploy.model_adapter import load_released_model  # heavy import, deferred
    model = load_released_model(timeout_seconds=args.timeout_seconds)
    rows = read_input(args.input)
    out = predict_rows(model, rows, top_k=args.top_k, timeout_seconds=args.timeout_seconds)
    write_tsv(args.output, out)
    n_ok = len({r["parent_id"] for r in out if r["status"] == "ok"})
    print(f"wrote {args.output}: {len(rows)} substrates, {n_ok} with metabolites", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py::test_predict_rows_handles_ok_and_unparseable_and_limits_top_k -v`
Expected: PASS.

- [ ] **Step 5: Add the console script to `pyproject.toml`** under `[project.scripts]`:

```toml
[project.scripts]
grail = "grail_metabolism.cli:main"
grail-metabolites = "grail_metabolism.deploy.predict_cli:main"
```

(If a `grail` entry already exists, add only the `grail-metabolites` line.)

- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/deploy/__init__.py grail_metabolism/deploy/predict_cli.py pyproject.toml grail_metabolism/tests/test_deploy.py
git commit -m "feat: deploy CLI core (file-to-file, testable rank adapter)"
```

---

### Task 4: Model adapter + end-to-end smoke

**Files:**
- Create: `grail_metabolism/deploy/model_adapter.py`
- Test: `grail_metabolism/tests/test_deploy.py`

**Interfaces:**
- Consumes: released pair (Task 2), `release_order` (Task 1), `generate_scored` / `score_batch` from the model.
- Produces: `load_released_model(timeout_seconds) -> ReleasedModel` with `.rank(smiles, top_k) -> list[(smiles, score)]`.

- [ ] **Step 1: Write the adapter.** It reproduces the pool-build + release order: generator scores candidates, filter scores each pair, `release_order` ranks, and `.rank` returns the top-k `(smiles, score)` where `score = filter * generator`.

```python
# grail_metabolism/deploy/model_adapter.py
"""Load the released (bank, generator, filter) pair and rank metabolites with the release order."""
from __future__ import annotations

from pathlib import Path

import torch

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.model.ranking import release_order
from grail_metabolism.workflows.factory import build_filter, build_generator

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "grail_metabolism/resources/extended_smirks_released.txt"
CKPT = ROOT / "artifacts/full5000_released/checkpoints"


def _load(path, build_fn):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = build_fn(p["arch"], p.get("rules"))
    m.load_state_dict(p["state_dict"], strict=False)
    m.calibrated_threshold = p.get("calibrated_threshold")
    m.eval()
    return m


class ReleasedModel:
    def __init__(self, generator, filter_model, timeout_seconds):
        self.generator = generator
        self.filter = filter_model
        self.timeout_seconds = timeout_seconds

    def rank(self, smiles, top_k):
        from grail_metabolism.metrics import _tautomer_inchikey
        # apply the WHOLE released bank (not the generator's default top_k): the release ranking is
        # defined on whole-bank pools, so top_k here is the rule count, and the CLI's --top-k caps
        # the RANKED metabolites afterwards, below.
        scored = self.generator.generate_scored(smiles, top_k=self.generator.num_rules)
        if not scored:
            return []
        products = [s for s, _ in scored]
        filter_scores = self.filter.score_batch(smiles, products)     # aligned to products
        cands, self_key = [], _tautomer_inchikey(smiles)
        for (prod, gen), filt in zip(scored, filter_scores):
            try:
                key = _tautomer_inchikey(prod)
            except Exception:
                continue
            cands.append({"smiles": prod, "generator": float(gen), "filter": float(filt), "key": key})
        ordered = release_order(cands, self_key=self_key)[:top_k]
        return [(c["smiles"], c["filter"] * c["generator"]) for c in ordered]


def load_released_model(timeout_seconds=120):
    rules = [l.strip() for l in open(BANK) if l.strip()]
    gen = _load(CKPT / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r or rules))
    filt = _load(CKPT / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))
    return ReleasedModel(gen, filt, timeout_seconds)
```

- [ ] **Step 2: Verify `score_batch`'s signature** before relying on it.

Run: `python -c "import inspect; from grail_metabolism.model.filter import Filter; print(inspect.signature(Filter.score_batch))"`
Expected: a `(self, substrate, products, ...)`-shaped signature returning one score per product. If it differs (e.g. returns a tensor, or takes a different argument order), adapt the `score_batch` call and the zip accordingly; do not proceed on an assumed signature.

- [ ] **Step 3: Write the end-to-end smoke** (skipped when the released pair is absent).

```python
import pathlib, pytest
_HAVE = (pathlib.Path(__file__).resolve().parents[2] / "artifacts/full5000_released/checkpoints/generator.pt").exists()

@pytest.mark.skipif(not _HAVE, reason="released checkpoint not built in this checkout")
def test_cli_end_to_end_small(tmp_path):
    from grail_metabolism.deploy.predict_cli import main
    inp = tmp_path / "in.smi"; inp.write_text("s1\tCCO\nBADSMILES\n")
    out = tmp_path / "out.tsv"
    assert main([str(inp), str(out), "--top-k", "5"]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == ["parent_id", "rank", "metabolite_smiles", "score", "status"]
    ids = {ln.split("\t")[0] for ln in lines[1:]}
    assert ids == {"s1", "2"}                      # line-2 id defaulted to its line number
    assert any(ln.split("\t")[4] == "no_parse" for ln in lines[1:])  # BADSMILES flagged
```

- [ ] **Step 4: Run the smoke**

Run: `python -m pytest grail_metabolism/tests/test_deploy.py -v`
Expected: PASS (the e2e test runs if the released pair exists; otherwise it is skipped and reported as such).

- [ ] **Step 5: Measure and record latency** on 3 substrates.

Run: `python -c "import time; from grail_metabolism.deploy.model_adapter import load_released_model as L; m=L(); [ (lambda t: (m.rank(s,15), print(s, round(time.time()-t,1),'s')))(time.time()) for s in ['CCO','c1ccccc1O','CC(=O)Nc1ccc(O)cc1'] ]"`
Note the per-substrate seconds; they go into the README in Task 5.

- [ ] **Step 6: Commit**

```bash
git add grail_metabolism/deploy/model_adapter.py grail_metabolism/tests/test_deploy.py
git commit -m "feat: released model adapter and end-to-end deploy smoke"
```

---

### Task 5: Packaging and docs

**Files:**
- Create: `deploy/README.md`, `deploy/requirements-deploy.txt`, `deploy/environment.yml`, `deploy/NOTICE.md`

- [ ] **Step 1: Write `deploy/requirements-deploy.txt`** with the pins the stack needs.

```text
# GRAIL deploy pins. RDKit is pinned because tautomer canonicalisation moves between releases and
# the matching key depends on it; a different RDKit's measured drift (scripts/kaggle_b1.py) is 3 of
# 3000 structures and 1 of 3000 keys, recorded rather than hidden.
numpy<2
rdkit==2022.09.5
torch
torch-geometric
```

- [ ] **Step 2: Write `deploy/NOTICE.md`** stating the bank exclusion and provenance.

```markdown
# Deploy notice

This build uses `extended_smirks_released.txt` (6970 templates), which excludes the 611 templates
that appear verbatim in BioTransformer's published set. Removing them costs zero references
(`results/licence_removal_cost__clean_test.json`). The generator checkpoint shipped here is the
deployed generator subset to this bank; the filter is unchanged. No corpus and no full bank are
shipped or required.
```

- [ ] **Step 3: Write `deploy/environment.yml`** (optional self-contained env).

```yaml
name: grail-deploy
channels: [conda-forge]
dependencies:
  - python=3.10
  - numpy<2
  - rdkit=2022.09.5
  - pip
  - pip: [torch, torch-geometric]
```

- [ ] **Step 4: Write `deploy/README.md`** with the exact contract, an example, and the measured latency from Task 4 Step 5.

```markdown
# GRAIL metabolite prediction — deploy CLI

Predict probable metabolites with scores for a file of substrate SMILES.

## Usage
    grail-metabolites INPUT OUTPUT [--top-k 15] [--timeout-seconds 120]

- INPUT: one substrate per line, `SMILES` or `id<TAB>SMILES`. Blank lines ignored; a line with no id
  is given its 1-based line number.
- OUTPUT: TSV with header `parent_id  rank  metabolite_smiles  score  status`. `score` is the deployed
  combined confidence (filter * generator). `status` is one of `ok`, `no_parse`, `timeout`,
  `no_metabolites`.

## Example
    printf 'p1\tCCO\np2\tc1ccccc1O\n' > in.smi
    grail-metabolites in.smi out.tsv --top-k 15

## Environment
Install `requirements-deploy.txt` into a Python 3.10 environment, or use `environment.yml`. The model
files (released bank + released checkpoints) ship with the package.

## Performance
Per-substrate latency (measured, top-k 15): <fill from Task 4 Step 5, e.g. "CCO ~2s; larger
substrates up to ~120s, bounded by --timeout-seconds">.

## Bank
See NOTICE.md: the released bank excludes the BioTransformer templates at zero coverage cost.
```

- [ ] **Step 5: Commit**

```bash
git add deploy/README.md deploy/requirements-deploy.txt deploy/environment.yml deploy/NOTICE.md
git commit -m "docs: deploy packaging, pinned requirements, and notice"
```

---

## Notes for the executor

- SDF output (`--format sdf`), listed as optional in the spec, is intentionally deferred: the
  requirement is metabolites with scores, which TSV satisfies, so `--format` ships with `choices=["tsv"]`
  and SDF is a later, separate follow-up. This is a deliberate scope trim, not an oversight.
- The heavy background job (match_scale sweep) may be running; these tasks are code + a one-time
  checkpoint build and do not need the GPU, but Task 2's build and Task 4's smoke load the model on
  CPU — run them when they will not contend badly with an in-flight generation job.
- Every artifact added under `artifacts/` or `results/` must be `git add -f`'d and registered via
  `python scripts/sync_tracked_artifacts.py` before commit (Task 2), then `--check` must pass.
