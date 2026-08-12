"""Per-appendix checks, one module per appendix file.

``verify_paper_numbers.py`` grew to nine hundred checks written against the body and the passages
the body leans on, and it reads every numeral the body prints. The appendices are where the numbers
those passages summarise actually live, and six hundred and eighty-four of their numerals were read
by nothing. Splitting the appendix checks into one module per appendix file keeps them addable
without a merge conflict per author and makes the coverage question answerable file by file.

A module exposes ``register(ctx)`` and adds to ``ctx.checks``. ``ctx`` carries:

    flat      the whole manuscript, whitespace collapsed, which every pattern matches against
    root      the repository root, for loading artifacts
    check     check(name, printed, computed, note="") -- printed-precision comparison
    checks    the list every check appends (ok, name, printed, computed, note) to
    art       art("name.json") -> parsed artifact, cached, or None when it is absent
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Ctx:
    flat: str
    root: Path
    check: Callable
    checks: list
    art: Callable
    tex: dict = field(default_factory=dict)


def register_all(ctx: Ctx) -> list[str]:
    """Import every module beside this one and let it add its checks. Returns the names run."""
    ran = []
    for mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"{__name__}.{mod.name}")
        if hasattr(m, "register"):
            m.register(ctx)
            ran.append(mod.name)
    return sorted(ran)
