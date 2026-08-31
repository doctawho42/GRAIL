#!/usr/bin/env python3
"""The comparator this work excluded, run.

BioTransformer was left out of the comparison on the ground that it does not supply ranked
per-substrate predictions, and its template reach was reported instead. That ground does not
distinguish it from MetaPredictor, which is in the comparison and whose list is shorter than the
budget on nearly every substrate, and it is the wrong ground: the tool runs. It failed here only
because its bundled native InChI library is cached for one processor architecture and this machine
is another, which a Java 8 build resolves. It is also the comparator a reader most wants, since
611 of its 668 templates are inside this bank, so its column measures how much of this work's
reach is BioTransformer's reach carrying a different name.

Two settings are run, not one. `superbio` is the sequence its own documentation recommends for
human metabolism; `allHuman` at one step is the setting that matches what every other arm here is
allowed, a single enzymatic application. Reporting both is the same discipline the SyGMa scenario
received: an emission knob swept in the direction that helps the comparator, so a lead cannot be a
property of the setting it was run at.

BioTransformer emits no score, so its file order is its ranking, which is the treatment
MetaPredictor's unranked list already gets. The predictions are written out beside the other
comparators so the comparison can be recomputed without a Java runtime.

    python scripts/typed_edit/biotransformer_arm.py --csv superbio=/path/bt_superbio.csv \
        --csv 'allHuman s1=/path/bt_allhuman_s1.csv' --index-map /path/index_map.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0


def parse_csv(path: Path, index_map: dict, substrates: set) -> dict:
    """substrate SMILES -> metabolite SMILES in the order BioTransformer wrote them."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        m = Chem.MolFromSmiles(s) if s else None
        return Chem.MolToSmiles(m) if m is not None else None

    # A precursor is named by the SDF title on the first step and by a metabolite id on later
    # ones; only first-step rows carry a title this map knows, and the rest are reached through
    # the precursor structure. Both routes are tried, in that order.
    by_structure = {}
    for name, smiles in index_map.items():
        c = canon(smiles)
        if c:
            by_structure[c] = smiles
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            parent = index_map.get((row.get("Precursor ID") or "").strip())
            if parent is None:
                parent = by_structure.get(canon((row.get("Precursor SMILES") or "").strip()))
            if parent is None or parent not in substrates:
                continue
            smiles = (row.get("SMILES") or "").strip()
            if smiles:
                out.setdefault(parent, []).append(smiles)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", default=[],
                    help="label=path, repeatable; the label names the setting in the report")
    ap.add_argument("--index-map", required=True,
                    help="JSON of SDF title -> substrate SMILES, as the input was written")
    ap.add_argument("--preds-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "biotransformer_arm.json"))
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey

    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"]); refs.update(blob["references"])
    subs = sorted(s for s in pools if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}
    index_map = json.loads(Path(args.index_map).read_text())

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    ours = {s: drop_parent([c["key"] for c in rrf_order(
        sorted(pools[s], key=lambda c: -c["generator"])[:CAP])], s) for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(order, k=None):
        return np.array([len(set(order[s] if k is None else order[s][:k]) & real[s])
                         for s in subs], dtype=float)

    settings, written = {}, {}
    for spec in args.csv:
        if "=" not in spec:
            print(f"--csv wants label=path, got {spec!r}", file=sys.stderr)
            return 2
        label, path = spec.split("=", 1)
        raw = parse_csv(Path(path), index_map, set(subs))
        lists = {s: drop_parent(_dedup(raw.get(s, []), CAP + 5), s) for s in subs}
        slug = label.replace(" ", "_").lower()
        target = Path(args.preds_dir) / f"biotransformer_{slug}_preds.json"
        target.write_text(json.dumps({s: raw.get(s, []) for s in subs}, indent=1))
        written[label] = str(target.relative_to(ROOT))

        row = {"source_csv": Path(path).name,
               "predictions": written[label],
               "mean_emitted": round(float(np.mean([len(lists[s]) for s in subs])), 1),
               "substrates_with_no_prediction": int(sum(1 for s in subs if not lists[s])),
               "recall": {str(k): round(float(hits(lists, k).sum() / U.sum()), 4) for k in KS}}
        for k in (15, 30, 50):
            d = hits(ours, k) - hits(lists, k)
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            row[f"exhaustive_minus_biotransformer_at_{k}"] = {
                "gap": round(float(d.sum() / U.sum()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
        # The length control this work applies to every other comparator: both arms cut to the
        # number of candidates BioTransformer returned on that substrate.
        matched = np.array([len(set(ours[s][:len(lists[s])]) & real[s]) for s in subs],
                           dtype=float)
        theirs = hits(lists)
        d = matched - theirs
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        row["matched_length"] = {
            "recall_ours": round(float(matched.sum() / U.sum()), 4),
            "recall_theirs": round(float(theirs.sum() / U.sum()), 4),
            "gap": round(float(d.sum() / U.sum()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "mean_slots": round(float(np.mean([len(lists[s]) for s in subs])), 2)}
        settings[label] = row

    if not settings:
        print("no --csv given", file=sys.stderr)
        return 2

    best = max(settings, key=lambda lab: settings[lab]["recall"]["30"])
    report = {
        "provenance": stamp(__file__),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum())},
        "version": "BioTransformer 3.0.0, the distribution whose templates this bank carries",
        "runtime_note": ("the bundled JNI InChI artefact is cached for MAC-X86_64 and this "
                         "machine is arm64; the run uses an x86_64 Java 8 runtime, which is what "
                         "the earlier smoke output on this machine was produced with"),
        "ranking": "BioTransformer emits no score; its file order is its ranking",
        "criterion": "tautomer-aware InChIKey, as everywhere else",
        "convention": "a prediction equal to the substrate is dropped, for both arms",
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "by_setting": settings,
        "setting_taken_forward": best,
        "predictions_written": written,
        "reading": (
            "This closes the exclusion. The comparator that shares 611 templates with this bank "
            "is measured on the same substrates under the same criterion, at both the setting its "
            "own documentation recommends and the one-step setting every other arm is held to, "
            "and with the same length control the other comparators receive."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'setting':16s} {'emitted':>8s} {'r@15':>7s} {'r@30':>7s} {'r@50':>7s}"
          f"   exhaustive minus BioTransformer at 30")
    for label, row in settings.items():
        c = row["exhaustive_minus_biotransformer_at_30"]
        print(f"{label:16s} {row['mean_emitted']:8.1f} {row['recall']['15']:7.4f} "
              f"{row['recall']['30']:7.4f} {row['recall']['50']:7.4f}"
              f"   {c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'  separates' if c['excludes_zero'] else ''}")
    print("\nmatched to BioTransformer's own list length:")
    for label, row in settings.items():
        m = row["matched_length"]
        print(f"  {label:16s} ours {m['recall_ours']:.4f}  theirs {m['recall_theirs']:.4f}  "
              f"{m['gap']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]"
              f"{'  separates' if m['excludes_zero'] else ''}  ({m['mean_slots']:.1f} slots)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
