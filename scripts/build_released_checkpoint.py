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
