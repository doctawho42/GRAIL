"""The group scorer H8 registers: build features, train on train, select on validation.

H8 fixes what the model may consume and what it optimises, and this implements exactly that and
nothing more. For a substrate and one formula group of its pool the features are

  the substrate encoding the generator already produces,
  the group's size and the maximum and mean of its members' filter and generator scores,
  the elemental difference between the group's formula and the substrate's,

and the objective is softmax cross-entropy over the groups of one substrate against the
indicator of which groups hold an annotated reference, normalised to a distribution because more
than one group can. The scorer sets the order of the groups; inside a group the members keep the
order rank fusion gives them, which H8 leaves alone because reordering there is worth 0.011.

Nothing here is chosen on the population the result is reported on. Training reads the training
split, every hyperparameter is chosen by micro recall@15 on validation, and the 291 are touched
once, by h8_verdict.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

# the elements the bank can add or remove, fixed here so the vector's length never depends on
# the data a particular run happens to see
ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I")
GROUP_FEATURES = 5 + len(ELEMENTS)


def _counts(formula: str) -> dict:
    """Element counts from an RDKit molecular formula string such as C9H8O4 or C6H5ClN+."""
    import re
    out = defaultdict(int)
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym:
            out[sym] += int(num) if num else 1
    return out


class Featuriser:
    """Turns a pool into (group order, feature matrix, per-group member order)."""

    def __init__(self, generator):
        self.g = generator
        self._enc: dict = {}
        self._form: dict = {}

    def formula(self, smiles: str) -> str:
        if smiles not in self._form:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            m = Chem.MolFromSmiles(smiles)
            self._form[smiles] = rdMolDescriptors.CalcMolFormula(m) if m else smiles
        return self._form[smiles]

    @torch.no_grad()
    def encode(self, sub: str) -> np.ndarray:
        if sub not in self._enc:
            _, graph = self.g._graph_for_substrate(sub)
            if graph is None:
                self._enc[sub] = np.zeros(self.enc_dim, dtype=np.float32)
            else:
                from torch_geometric.data import Batch
                v = self.g.substrate_encoder(Batch.from_data_list([graph]))
                self._enc[sub] = v.squeeze(0).cpu().numpy().astype(np.float32)
        return self._enc[sub]

    @property
    def enc_dim(self) -> int:
        """Derived by encoding one trivial molecule, not read off an attribute name.

        The encoder's output width is a property of how it was built, and reading it from a
        layer attribute couples this to the encoder's internals. Encoding ethanol once and
        measuring costs nothing and cannot be wrong about the tensor that will actually arrive.
        """
        if not hasattr(self, "_dim"):
            _, graph = self.g._graph_for_substrate("CCO")
            from torch_geometric.data import Batch
            with torch.no_grad():
                self._dim = int(self.g.substrate_encoder(
                    Batch.from_data_list([graph])).shape[-1])
        return self._dim

    def groups(self, sub: str, pool: list):
        """Groups in fusion order of their best member, with each group's members in that order."""
        fused = rrf_order(pool)
        order = {id(c): i for i, c in enumerate(fused)}
        by_g = defaultdict(list)
        for c in fused:
            by_g[self.formula(c["smiles"])].append(c)
        names = sorted(by_g, key=lambda g: min(order[id(c)] for c in by_g[g]))
        return names, by_g

    def features(self, sub: str, pool: list):
        names, by_g = self.groups(sub, pool)
        enc = self.encode(sub)
        sub_counts = _counts(self.formula(sub))
        rows = []
        for g in names:
            members = by_g[g]
            f = [c["filter"] for c in members]
            q = [c["generator"] for c in members]
            gc = _counts(g)
            delta = [float(gc.get(e, 0) - sub_counts.get(e, 0)) for e in ELEMENTS]
            rows.append(np.concatenate([
                enc,
                np.array([float(len(members)), max(f), sum(f) / len(f),
                          max(q), sum(q) / len(q)], dtype=np.float32),
                np.array(delta, dtype=np.float32)]))
        return names, by_g, np.stack(rows).astype(np.float32)


class GroupScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_examples(pools, refs, feat, limit=None):
    """One example per substrate: features per group and the target distribution over groups."""
    out = []
    subs = sorted(s for s in pools if refs.get(s) and pools[s])
    if limit:
        subs = subs[:limit]
    for n, s in enumerate(subs, 1):
        if n % 25 == 0:
            print(f"  featurising {n}/{len(subs)}", file=sys.stderr, flush=True)
        real = set(refs[s])
        names, by_g, X = feat.features(s, pools[s])
        y = np.array([1.0 if any(c["key"] in real for c in by_g[g]) else 0.0 for g in names],
                     dtype=np.float32)
        if y.sum() == 0 or len(names) < 2:
            continue                       # nothing to rank, or no signal to rank it by
        out.append({"sub": s, "X": X, "y": y / y.sum(), "names": names,
                    "by_g": by_g, "real": real})
    return out


def reorder_recall(model, examples, k=15, device="cpu"):
    """Micro recall@k with groups ordered by the model and members left in fusion order."""
    hit = tot = 0
    model.eval()
    with torch.no_grad():
        for e in examples:
            s = model(torch.from_numpy(e["X"]).to(device)).cpu().numpy()
            order = np.argsort(-s)
            keys = [c["key"] for i in order for c in e["by_g"][e["names"][i]]]
            hit += len(set(keys[:k]) & e["real"]); tot += len(e["real"])
    return hit / tot if tot else 0.0


def fusion_recall(examples, k=15):
    hit = tot = 0
    for e in examples:
        keys = [c["key"] for g in e["names"] for c in e["by_g"][g]]
        hit += len(set(keys[:k]) & e["real"]); tot += len(e["real"])
    return hit / tot if tot else 0.0
