# MetaTox submission set (994 substrates, GRAIL validation split)

Purpose: training a joint GRAIL x MetaTox ranker on a population disjoint from the 291 already scored.
GRAIL already has ranked predictions for all 994 of these substrates.

## Files
- `substrates.smi`  -- `<SMILES>\t<id>`, one per line. Use for a SMILES/text upload.
- `substrates.sdf`  -- molecules with the id as the molecule name. Use for an SDF upload.
- `substrate_map.csv` -- the join key. `substrate_smiles` is the exact string our evaluation
  joins on; `submission_smiles` is what's in the .smi/.sdf; `id` (SUB0001..) is the stable handle.

SMILES here are the **natural tautomer** (e.g. amides drawn `NC(=O)`, not the iminol `N=C(O)`).
4 of 994 were re-tautomerized from GRAIL's internal form for chemically-sane submission;
each is guaranteed the same molecule as its eval key under our tautomer-InChIKey matching.

## What to run in MetaTox
Predict metabolites (Phase I + Phase II as MetaTox offers) for each of the 994 substrates.
**Please run the SMIRKS-rule variant**, the same configuration as the previous batch: that is the
one whose predictions we scored, and mixing configurations between batches would make the two
populations incomparable in the one way that matters here.

## What to return to me (so I can score it apples-to-apples)
A JSON keyed by **id** (preferred) or by the exact **submission_smiles**, mapping each substrate
to its **ranked** list of predicted metabolite SMILES, best first:

```json
{ "SUB0001": ["<metabolite_smiles_1>", "<metabolite_smiles_2>", ...],
   "SUB0002": [ ... ] }
```

Keep MetaTox's own ranking order. If MetaTox returns a probability/score per metabolite,
include it as a parallel list or `[[smiles, score], ...]` and I'll rank by score. If it emits
an unranked set, that's fine too -- tell me, and I'll report it at matched output budget so an
unranked pool isn't flattered at large k.

## Fallback if the batch is too large
Any contiguous prefix of `substrate_map.csv` by `id` is a usable subset: the ids are assigned by sorted eval key, so a prefix is reproducible from this file alone and I can score whatever comes back. Tell me where you stopped.

Notes: 4 of 994 re-tautomerized for submission (identity preserved under tautomer-InChIKey).
0 substrates failed to parse (excluded); 0 dropped because
tautomerization changed molecular identity (should be 0).
