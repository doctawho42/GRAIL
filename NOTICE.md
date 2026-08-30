# Third-party content in this repository, and what it constrains

This file records what this repository redistributes that it did not write, and the terms each
carries. It is not a licence. The repository's own licence is a decision the authors have not yet
recorded, and this file exists because that decision is constrained rather than free.

## The rule bank

`grail_metabolism/resources/extended_smirks.txt` holds 7,581 SMIRKS templates. Their provenance is
measured in `results/curated_provenance.json` and described in the Supporting Information:

| portion | count | origin |
|---|---:|---|
| curated, named | 1,233 | three collections that ship with this code |
| carried as curated, in fact extracted | 492 | an earlier machine extraction from the annotated pairs |
| mined | 5,856 | mined from the training split by `scripts/mine_rules.py` |

**763 of the 1,715 curated templates are somebody else's rules, verbatim** — 44.5% of the curated
half, measured by string equality against the deployed bank in `results/curated_third_party.json`:

| source | templates in this bank | of the source's own set | terms |
|---|---:|---:|---|
| BioTransformer | 611 | 611 of 668 | distributed with BioTransformer; redistribution requires permission |
| SyGMa 1.1.0 | 152 | 152 of 204 | GPL |

String equality cannot trace a rule that was rewritten, so both counts are lower bounds and the
remaining 952 curated templates are not thereby shown to be original.

This is the reason the repository carries no licence yet. Redistributing 152 GPL lines makes the
bank to that extent a derivative of a GPL work, and a permissive licence over the whole repository
would not be consistent with it; the BioTransformer templates carry a separate obligation that a
licence choice does not discharge, because permission is a thing to be obtained rather than
declared. It is a constraint to be resolved, not an oversight, and the resolution is the authors':
relicense compatibly and obtain BioTransformer's permission, remove the borrowed templates and
report what that costs the coverage ceiling, or obtain different grants. Removal is measurable in
the same instrument that found them.

## The corpus

The annotated substrate–metabolite corpus is assembled from ChEMBL (CC BY-SA 3.0), DrugBank
(CC BY-NC 4.0), MetXBioDB (distributed with BioTransformer, whose terms require permission for
redistribution) and the GLORYx reference set. ChEMBL's ShareAlike and DrugBank's NonCommercial
terms cannot both be satisfied by one derivative, which is why the Zenodo deposit carries no source
structure: every substrate and every annotated metabolite in it is replaced by a
tautomer-canonical InChIKey before the archive is built.

The corpus files themselves (`grail_metabolism/data/*.sdf`, `*_triples*.txt`) are **not** tracked
here; they are gitignored and are obtained from the sources under the reader's own licences.
`results/test_references.json` **is** tracked and holds the evaluated test substrates and their
annotated metabolites as SMILES, so that every recall figure in the paper can be recomputed.

## Comparators

Per-substrate predictions from SyGMa, MetaTox and MetaPredictor are tracked under `results/` so
the comparison can be recomputed without re-running any of them. Their digests and versions are
pinned in `paper2/split_manifest.json`. Each remains the property of its authors and is
redistributed here only as the frozen output of a run on our substrates.

## RDKit and the rest

RDKit is BSD-3-Clause. The remaining Python dependencies are listed in `requirements.txt` with
their own terms.
