#!/usr/bin/env python3
"""Build the MetaTox submission set: the 291 test substrates GRAIL has predictions for.

The grant-deciding comparison is GRAIL vs MetaTox (the way2drug incumbent). GRAIL already
holds ranked predictions for all 291 test substrates (artifacts/full5000_single). SyGMa raw
pools exist for a 150-substrate subset (all 150 are within the 291), so this one file also
supports the tighter 3-way (GRAIL/SyGMa/MetaTox) on the shared 150.

GRAIL's substrate SMILES come out of standardize_mol tautomer-canonicalized, which for ~27% of
this set stores an UNNATURAL iminol/imidic-acid tautomer (amide `NC(=O)` written as `N=C(O)`;
e.g. the benzamide of a taxane). Submitting those artifacts to an external incumbent tool would
make MetaTox reject or mis-handle them -- unfair to MetaTox. So the SUBMISSION SMILES is the
tautomer-canonicalized natural form (rdMolStandardize.TautomerEnumerator), while the JOIN KEY
stays the GRAIL string. We ASSERT every submission form shares the eval key's _tautomer_inchikey
(the exact function our scoring matches on) -- so re-tautomerizing changes only the drawn tautomer,
never the molecule, and the join back to ground truth stays valid.

Emits, under results/metatox_input/:
  - substrates.smi        `<submission_smiles>\t<id>` per line  -- SMILES upload
  - substrates.sdf        molecules (natural tautomer) with _Name = id  -- SDF upload
  - substrate_map.csv     id, substrate_smiles (EVAL KEY, verbatim), submission_smiles,
                          in_shared_150, retautomerized  -- the join key
  - README.md             what to run in MetaTox and the exact format to return

IDs are SUB0001.. assigned by sorted(eval-key) so the mapping is deterministic/reproducible.
The eval key is the substrate string AS IT APPEARS in the GRAIL prediction CSV -- that is what
run_match_sensitivity.py joins on, so MetaTox predictions must come back keyed by that id
(molecule name) or by that exact substrate string.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")

GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
SYGMA_RAW = ROOT / "results" / "match_sens_cache" / "sygma_preds_a7bf90dad9de0e5b.json"
OUTDIR = ROOT / "results" / "metatox_input"


def main() -> int:
    # One code path for every batch. A second builder would let the submission format drift between
    # batches, and the join back to ground truth is exactly what must not drift.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(GRAIL_CSV),
                    help="GRAIL prediction CSV whose substrate column supplies the eval keys")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--split", default="test", help="named in the README so a batch says what it is")
    ap.add_argument("--purpose", default="the GRAIL vs MetaTox comparison",
                    help="one line for the README, so a returned file can be traced to its request")
    ap.add_argument("--limit", type=int, default=0, help="0 sends every substrate available")
    ap.add_argument("--seed", type=int, default=0, help="only used when --limit subsamples")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # eval keys = substrate strings exactly as GRAIL's prediction CSV holds them
    with open(args.predictions) as fh:
        subs = sorted({row["substrate"] for row in csv.DictReader(fh) if row["substrate"]})
    if args.limit and args.limit < len(subs):
        # a subsample is unrecoverable without its cap and seed, and this repository has paid for
        # that once; both are written into the map and the README
        import random
        random.Random(args.seed).shuffle(subs)
        subs = sorted(subs[: args.limit])
    shared150 = set(json.loads(SYGMA_RAW.read_text())) if SYGMA_RAW.exists() else set()
    print(f"GRAIL {args.split} substrates: {len(subs)}  |  "
          f"shared-with-SyGMa: {len(shared150 & set(subs))}", flush=True)

    te = rdMolStandardize.TautomerEnumerator()
    rows = []
    unparseable = []
    identity_broken = []  # tautomerization changed the molecule (must never happen) -> excluded
    writer = Chem.SDWriter(str(outdir / "substrates.sdf"))
    for i, key in enumerate(subs, 1):
        sid = f"SUB{i:04d}"
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            unparseable.append(key)
            continue
        # natural-tautomer submission form (fixes GRAIL's iminol artifacts)
        sub_mol = te.Canonicalize(mol)
        submission = Chem.MolToSmiles(sub_mol)
        # HARD GUARANTEE: re-tautomerizing changes only the drawn tautomer, not the molecule,
        # under the SAME key our scoring joins/matches on. If it doesn't, drop it -- never
        # silently submit a different molecule than we score.
        try:
            same_mol = _tautomer_inchikey(key) == _tautomer_inchikey(submission)
        except Exception:
            same_mol = False
        if not same_mol:
            identity_broken.append((key, submission))
            continue
        sub_mol.SetProp("_Name", sid)
        writer.write(sub_mol)
        rows.append({
            "id": sid,
            "substrate_smiles": key,             # the EVAL KEY (verbatim) -- join on this
            "submission_smiles": submission,     # natural tautomer -- what MetaTox gets
            "in_shared_150": int(key in shared150),
            "retautomerized": int(submission != Chem.MolToSmiles(mol)),
        })
    writer.close()

    # .smi : submission SMILES <TAB> id
    with open(outdir / "substrates.smi", "w") as fh:
        for r in rows:
            fh.write(f"{r['submission_smiles']}\t{r['id']}\n")

    # join map
    with open(outdir / "substrate_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "substrate_smiles", "submission_smiles", "in_shared_150", "retautomerized"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n150 = sum(r["in_shared_150"] for r in rows)
    shared_note = (f"\n{n150} of them are also in the SyGMa-shared subset (`in_shared_150=1`), so "
                   f"this set supports the tighter 3-way (GRAIL / SyGMa / MetaTox) as well."
                   if n150 else "")
    fallback = (f"Prioritize the {n150} rows with `in_shared_150=1` -- that subset still answers "
                f"the grant question against **both** GRAIL and SyGMa."
                if n150 else
                "Any contiguous prefix of `substrate_map.csv` by `id` is a usable subset: the ids "
                "are assigned by sorted eval key, so a prefix is reproducible from this file alone "
                "and I can score whatever comes back. Tell me where you stopped.")
    n_retaut = sum(r["retautomerized"] for r in rows)
    readme = f"""# MetaTox submission set ({n} substrates, GRAIL {args.split} split)

Purpose: {args.purpose}.
GRAIL already has ranked predictions for all {n} of these substrates.{shared_note}

## Files
- `substrates.smi`  -- `<SMILES>\\t<id>`, one per line. Use for a SMILES/text upload.
- `substrates.sdf`  -- molecules with the id as the molecule name. Use for an SDF upload.
- `substrate_map.csv` -- the join key. `substrate_smiles` is the exact string our evaluation
  joins on; `submission_smiles` is what's in the .smi/.sdf; `id` (SUB0001..) is the stable handle.

SMILES here are the **natural tautomer** (e.g. amides drawn `NC(=O)`, not the iminol `N=C(O)`).
{n_retaut} of {n} were re-tautomerized from GRAIL's internal form for chemically-sane submission;
each is guaranteed the same molecule as its eval key under our tautomer-InChIKey matching.

## What to run in MetaTox
Predict metabolites (Phase I + Phase II as MetaTox offers) for each of the {n} substrates.
**Please run the SMIRKS-rule variant**, the same configuration as the previous batch: that is the
one whose predictions we scored, and mixing configurations between batches would make the two
populations incomparable in the one way that matters here.

## What to return to me (so I can score it apples-to-apples)
A JSON keyed by **id** (preferred) or by the exact **submission_smiles**, mapping each substrate
to its **ranked** list of predicted metabolite SMILES, best first:

```json
{{ "SUB0001": ["<metabolite_smiles_1>", "<metabolite_smiles_2>", ...],
   "SUB0002": [ ... ] }}
```

Keep MetaTox's own ranking order. If MetaTox returns a probability/score per metabolite,
include it as a parallel list or `[[smiles, score], ...]` and I'll rank by score. If it emits
an unranked set, that's fine too -- tell me, and I'll report it at matched output budget so an
unranked pool isn't flattered at large k.

## Fallback if the batch is too large
{fallback}

Notes: {n_retaut} of {n} re-tautomerized for submission (identity preserved under tautomer-InChIKey).
{len(unparseable)} substrates failed to parse (excluded); {len(identity_broken)} dropped because
tautomerization changed molecular identity (should be 0).
"""
    (outdir / "README.md").write_text(readme)

    print(f"wrote {n} rows -> {outdir}", flush=True)
    print(f"  substrates.smi / substrates.sdf / substrate_map.csv / README.md", flush=True)
    print(f"  shared-150 flagged: {n150}   re-tautomerized: {n_retaut}   "
          f"unparseable(excluded): {len(unparseable)}   identity-broken(excluded): {len(identity_broken)}", flush=True)
    for k in unparseable[:5]:
        print("  UNPARSEABLE:", k[:80], flush=True)
    for a, b in identity_broken[:5]:
        print("  IDENTITY-BROKEN:", a[:60], "->", b[:60], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
