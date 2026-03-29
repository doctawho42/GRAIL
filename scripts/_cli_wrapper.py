from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cli(command: str, argv: Sequence[str] | None = None) -> int:
    from grail_metabolism.cli import main

    return main([command, *(list(argv) if argv is not None else sys.argv[1:])])
