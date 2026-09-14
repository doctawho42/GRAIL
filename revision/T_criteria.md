# T_criteria: the implemented matching criteria and who uses them

One row per criterion implemented in this repository. The comparison each performs is read from
the code, not from the manuscript. The literature user is taken only where this repository itself
states it; no citation is invented, and every criterion for which no user could be found is marked
as such.

Dispatch: `grail_metabolism/metrics.py:_match_keys`. Citation keys are from `paper2/refs.bib`,
which both documents load (`\bibliography{refs}` in `paper2/grail_jcim.tex` and `paper2/si.tex`).

| criterion | exact comparison performed | implementation | user named by this repository | citable key | status of the citation |
|---|---|---|---|---|---|
| `exact` | equality of the raw prediction and reference strings, no normalisation | the fallback branch of `_match_keys` | none | none | **no user found.** Not a member of the declared protocol set in `scripts/run_match_sensitivity.py` |
| `canonical` | RDKit canonical SMILES with `isomericSmiles=False`, string equality | `_canonical_key` | LAGOM | `Larsson_2025` | cited by the manuscript already |
| `inchikey` | the full InChIKey, all blocks, including stereochemistry | `_inchikey`, falling back to the raw string when RDKit cannot parse | called "strict"; no tool named | none | **no user found** (see the search below) |
| `inchi_no_stereo` | the first block of the InChIKey only, so stereoisomers, protonation states and isotopologues collide | `_inchikey_skeleton` | GLORYx | `de_Bruyn_Kops_2020` | cited by the manuscript already |
| `tanimoto1` | a Morgan fingerprint used as an identity key, i.e. Tanimoto equal to one | `_morgan_key` | MetaTrans | none | **no citable entry.** MetaTrans is absent from all 14 `.bib` files in the repository, although `artifacts/tier2/metatrans_preds.json` holds its predictions |
| `inchikey_tautomer` | Cleanup, FragmentParent, uncharge, `TautomerEnumerator.Canonicalize` on both sides, then InChIKey; per-input fallback to the plain InChIKey | `_tautomer_inchikey` | this work ("ours") | `Dhaked_2020` as the basis, not as a user | Dhaked_2020 treats tautomerism in chemoinformatics and InChI; it motivates the criterion and does not use it for prediction matching |

## Where the attribution comes from

Two places in the repository state the protocol-to-tool mapping, and they agree:

- `scripts/run_match_sensitivity.py`, module docstring: *"GLORYx: InChI without stereo; MetaTrans:
  Tanimoto=1; LAGOM: canonical SMILES; strict: InChIKey; ours: tautomer-canonical InChIKey"*.
- the same file's `MODE_LABEL`: `canonical` → "canon-SMILES (LAGOM)", `inchikey` → "InChIKey
  (strict)", `inchi_no_stereo` → "no-stereo (GLORYx)", `tanimoto1` → "Tanimoto=1 (MetaTrans)",
  `inchikey_tautomer` → "tautomer-InChIKey (ours)".

A third statement is consistent with them: `grail_metabolism/tests/test_audit_fixes.py` asserts the
`inchi_no_stereo` behaviour under the comment "GLORYx stereo-blind", and
`grail_metabolism/metrics.py` line 135 calls the stereo-free canonical SMILES "the matching LAGOM
uses".

## What was searched before marking a criterion unattributed

| criterion | search performed | result |
|---|---|---|
| `tanimoto1` | all 14 `.bib` files in the repository, case-insensitive, for "metatrans" | no match |
| `inchikey` | the 40 titled entries of `paper2/refs.bib` for `inchi`, `identifier`, `standardi`, `fingerprint`, `tanimoto`, `canonical`, `tautomer` | 3 matches, none a user of a prediction-matching protocol: `Dhaked_2020` (tautomerism and InChI V2), `Hahnke_2018` (PubChem structure standardization), `Mansouri_2024` (QSAR-ready standardization workflow) |
| `exact` | the declared protocol set in `scripts/run_match_sensitivity.py` | not a member; no tool associated |

`Boyce_2022` compares the performance and coverage of in-silico liver-metabolism tools and was
inspected for this purpose; it is a comparison study and does not define a matching protocol.

## Two facts about the set of criteria

- **The declared set and the implemented set differ.** `grail_metabolism/config.py` declares
  `["exact", "inchi_no_stereo", "inchikey", "inchikey_tautomer", "tanimoto1"]`;
  `grail_metabolism/metrics.py` declares those and `canonical`. A configuration therefore cannot
  request `canonical` through the config type, while the metric implements it and the
  match-sensitivity sweep uses it.
- **Strictness, as the existing supplement asserts it** (`paper2/si_table_criteria.tex`):
  `canonical` strictest, then `inchikey`, then `inchi_no_stereo`, then `tanimoto1` loosest, with
  `inchikey_tautomer` as the default. That ordering is the supplement's claim and is not
  recomputed here.

## Precomputed key maps

`results/key_tables/` holds one SMILES-to-key map per criterion, consulted before any live
canonicalisation:

| map | entries |
|---|---|
| `canonical.json` | 108,967 |
| `inchikey.json` | 108,967 |
| `inchi_no_stereo.json` | 108,967 |
| `tanimoto1.json` | 108,967 |
| `inchikey_tautomer.json` | 116,882 |

One substrate of the 1,170, `O=[Se](O)O`, is absent from all five maps and is keyed live
(`MCAHWIHFGHIESP-UHFFFAOYSA-N` under the tautomer criterion). No reference SMILES is absent from
any map.
