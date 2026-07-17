# MetaTox submission set (291 substrates)

The grant-deciding comparison: **GRAIL vs MetaTox** on the GRAIL test set.
GRAIL already has ranked predictions for all 291 of these substrates;
150 of them are also in the SyGMa-shared subset (`in_shared_150=1`), so this set
supports the tighter 3-way (GRAIL / SyGMa / MetaTox) as well.

## Files
- `substrates.smi`  -- `<SMILES>\t<id>`, one per line. Use for a SMILES/text upload.
- `substrates.sdf`  -- molecules with the id as the molecule name. Use for an SDF upload.
- `substrate_map.csv` -- the join key. `substrate_smiles` is the exact string our evaluation
  joins on; `submission_smiles` is what's in the .smi/.sdf; `id` (SUB0001..) is the stable handle.

SMILES here are the **natural tautomer** (e.g. amides drawn `NC(=O)`, not the iminol `N=C(O)`).
79 of 291 were re-tautomerized from GRAIL's internal form for chemically-sane submission;
each is guaranteed the same molecule as its eval key under our tautomer-InChIKey matching.

## What to run in MetaTox
Predict metabolites (Phase I + Phase II as MetaTox offers) for each of the 291 substrates.

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

## Fallback if batch submission is limited
Prioritize the 150 rows with `in_shared_150=1` -- that subset still answers the grant
question against **both** GRAIL and SyGMa.

Notes: 79 of 291 re-tautomerized for submission (identity preserved under tautomer-InChIKey).
0 substrates failed to parse (excluded); 0 dropped because
tautomerization changed molecular identity (should be 0).
