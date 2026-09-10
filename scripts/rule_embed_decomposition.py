#!/usr/bin/env python3
"""How the rule representation's variance splits across its three components.

The rule representation is LayerNorm(graph_encoded + gate * id_embedding + meta). This measures how
much of the pre-norm sum's variance each component carries, which is the mechanism check the
support-gate is registered against (docs/RULE_REPRESENTATION_PREREGISTRATION.md): at lambda = 0 the
id holds about 82% of it, and if the gate works that share must fall and the graph's must rise. A
recall gain with an unchanged split is not the chemistry learning to score rare rules.

The gate is applied as the model applies it, so the split is of what actually enters the sum: at
lambda > 0 the id component is gate * id, not the raw id.

    python scripts/rule_embed_decomposition.py --gen artifacts/full5000_implicit/checkpoints/generator.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B  # noqa: E402
from grail_metabolism.config import GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "rule_embed_decomposition.json"))
    args = ap.parse_args()

    st = torch.load(args.gen, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**st["arch"]), st.get("rules"))
    gen.load_state_dict(st["state_dict"], strict=False)
    gen.eval()
    parser = gen.parser
    lam = float(getattr(parser, "id_gate_lambda", 0.0))

    with torch.no_grad():
        batch = parser.rule_batch
        encoded = parser.encoder(batch)
        ids = parser.id_embedding.weight[: encoded.size(0)]
        meta = parser.meta_encoder(parser.rule_meta)
        if lam > 0.0:
            n = parser.rule_support[: encoded.size(0)]
            ids = (n / (n + lam)).unsqueeze(1) * ids
        comps = {"graph": encoded, "id": ids, "meta": meta}
        totvar = {k: float(v.var(dim=0, unbiased=False).sum()) for k, v in comps.items()}
        meannorm = {k: float(v.norm(dim=1).mean()) for k, v in comps.items()}

    s = sum(totvar.values())
    rep = {"config": {**B._code_version(), "generator": B._rel(args.gen), "id_gate_lambda": lam},
           "total_variance": {**totvar, "sum": s},
           "mean_component_norm": meannorm,
           "id_share_of_variance": round(totvar["id"] / s, 4) if s else None,
           "graph_share_of_variance": round(totvar["graph"] / s, 4) if s else None,
           "meta_share_of_variance": round(totvar["meta"] / s, 4) if s else None}
    print(f"  id_gate_lambda = {lam}")
    for k in ("graph", "id", "meta"):
        print(f"    {k:<6} variance {totvar[k]:>10.3f}  share {totvar[k]/s:.4f}")
    print(f"\n  id share {rep['id_share_of_variance']}, graph share {rep['graph_share_of_variance']}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
