#!/usr/bin/env python3
"""Fetch MOSES' published generated sets for the models it ships samples for.

MOSES reports Unique@k as the count of distinct canonical SMILES among a model's generations.
That is a matching decision wearing a different name, so the same re-scoring the paper applies to
metabolite predictions applies here: vary the criterion over frozen outputs and see whether the
ordering of published models moves.

Samples live in Git LFS; the media host serves the content rather than the pointer.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "moses_samples.json"
BASE = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/samples"
MODELS = ["aae", "char_rnn", "combinatorial", "hmm", "jtn", "latent_gan", "ngram", "vae"]
SEED = 1  # one seed per model keeps the comparison like-for-like


def fetch(model: str) -> list[str]:
    url = f"{BASE}/{model}/{model}_{SEED}.csv"
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].upper().startswith("SMILES"):
        lines = lines[1:]
    if len(lines) < 1000:
        raise SystemExit(f"ERROR: {model} returned {len(lines)} rows -- fetched the LFS pointer, "
                         "not the data.")
    return lines


def main() -> int:
    out = {}
    for m in MODELS:
        try:
            out[m] = fetch(m)
            print(f"  {m:16} {len(out[m]):,} molecules", flush=True)
        except Exception as e:
            print(f"  {m:16} FAILED: {e}", flush=True)
    if len(out) < 4:
        raise SystemExit("ERROR: fewer than four models fetched; a ranking claim needs more.")
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({sum(len(v) for v in out.values()):,} molecules, {len(out)} models)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
