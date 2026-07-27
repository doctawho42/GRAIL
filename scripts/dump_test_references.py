#!/usr/bin/env python3
"""Dump substrate -> reference metabolites for the clean test split.

Loading the split pulls in torch. Keying is CPU-bound and wants a process pool, and a pool in a
torch-loaded process stalls on this platform, so the two are separated: this writes the references
once, and the keying step then runs without torch present.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.factorize_recall import build_dataset_config
from grail_metabolism.workflows.data import load_dataset_bundle

OUT = ROOT / "results" / "test_references.json"

bundle = load_dataset_bundle(build_dataset_config(100000))
refs = {s: list(p) for s, p in bundle.test.map.items() if p}
OUT.write_text(json.dumps(refs))
print(f"wrote {OUT}: {len(refs)} substrates", flush=True)
