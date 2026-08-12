"""A worked example of the contract, not registered: modules starting with _ are skipped.

Copy the shape. Every check must be one a perturbation of the manuscript would break, and the
figure must identify exactly one leaf of one artifact -- not "some leaf equals this number".
"""
from __future__ import annotations

import re


def register(ctx) -> None:
    art = ctx.art("robust_order.json")
    if art is None:
        return
    m = re.search(r"seven systems on their \$([\d,{}]+)\$ common reactions", ctx.flat)
    if not m:
        ctx.checks.append((False, "example, the sentence is present", "present", "missing",
                           "scripts/gates/_example.py"))
        return
    ctx.check("example, the reactions the seven-system group shares",
              m.group(1).replace("{,}", ""), art["leaderboards"]["cluster0"]["n_items"],
              "results/robust_order.json")
