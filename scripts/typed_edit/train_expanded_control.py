#!/usr/bin/env python3
"""The control arm for the label-convention retraining.

The retrained generator gains +0.18 of ensemble recall@15 over the deployed checkpoint, and
none of that is attributable yet. The deployed checkpoint is from 29 June; five commits have
touched the generator since, one of them propensity-scored positive weighting, which is a
separate intervention aimed at the same failure. `both arms move at once` is the objection commit
6f41168 raised against this very experiment, and it applies to the comparison as run.

So this trains TODAY's code on the OLD label convention. The difference between this and
`full5000_implicit` is the label matrix and nothing else; the difference between this and the
deployed checkpoint is the code and nothing else. Two subtractions, one attribution.

LABEL_PRESENTATION is a module global with no config field, deliberately: the constant stays
where it was so that a dozen artifacts do not change meaning silently. It is patched here, in
one place, and the run records which value it used.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grail_metabolism.utils import preparation  # noqa: E402

preparation.LABEL_PRESENTATION = "expanded"
print(f"LABEL_PRESENTATION patched to {preparation.LABEL_PRESENTATION!r} for this run",
      file=sys.stderr, flush=True)

from grail_metabolism.cli import main  # noqa: E402

sys.argv = ["grail", "run-config", str(ROOT / "configs" / "full5000_expanded_control.yaml")]
raise SystemExit(main())
