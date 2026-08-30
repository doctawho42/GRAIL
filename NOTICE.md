# Third-party content in this repository, and what it constrains

This file records what this repository holds that it did not write, and the terms each carries. It
is not a licence. The repository's own licence is a decision the authors have not yet recorded, and
this file exists because that decision is constrained rather than free.

Two separate obligations run through what follows and they are easy to conflate. The **bank**
contains other people's templates, which puts the bank under their terms. The **repository** tracks
other people's files, which puts the release under the terms of everything inside them — including
the parts the bank never uses. Both are measured in `results/curated_third_party.json`.

## The rule bank

`grail_metabolism/resources/extended_smirks.txt` holds 7,581 SMIRKS templates: 5,866 mined from the
training split by `scripts/mine_rules.py` and 1,715 carried as curated. Of the curated portion,
1,233 come from three named collections and 492 from an earlier machine extraction that was carried
as curated and is not (`results/curated_provenance.json`).

**966 of the 1,715 curated templates — 56.3% — are somebody else's rules, verbatim.** Measured by
string equality against every published rule set held on disk:

| rightsholder | templates in the bank | terms, as its own distribution states them |
|---|---:|---|
| BioTransformer | 611 | LGPL (see below) |
| SyGMa 1.1.0 | 273 | "GPL", no version given |
| GLORYx | 82 | no licence text for it is held here |
| RetroSim | 0 | no licence text for it is held here |

The SyGMa count is 273 and not 152 because GLORYx's own rule file attributes 178 of its 260 rules to
SyGMa, in a notation the installed SyGMa package does not use. String equality cannot trace a
template that was rewritten, so every count above is a lower bound, and the 749 curated templates
that match nothing measured are not thereby shown to be original.

## The files this repository redistributes

`grail_metabolism/resources/external/` holds five files tracked by git. Three are byte-identical to
a file in the BioTransformer distribution. Redistributing a file carries the terms of everything in
it, whatever the bank uses:

| file | size | templates the bank uses | terms |
|---|---:|---:|---|
| `bt_database_metabolicReactions.json` | 779 KB | 611 | LGPL |
| `bt_database_ENVMICRO_metabolicReactions.json` | 172 KB | **2** | **CC BY-NC-SA 4.0** |
| `bt_database_standardizationReactions.json` | 11 KB | **0** | LGPL |
| `gloryx_reactionrules.csv` | 37 KB | 260 | not stated here |
| `retrosim_templates_general.json` | 120 KB | **0** | not stated here |

The ENVMICRO file is the sharp one. Its own header, and the `LICENSE` beside it upstream, put the
EAWAG data it holds under CC BY-NC-SA 4.0, licensed by EnviPath: NonCommercial and ShareAlike. Those
terms cannot be absorbed into a GPL release or a permissive one, and the file is redistributed here
verbatim. Its two templates that reach the bank are both also present in the LGPL core file, so the
bank does not depend on it.

Three of the five files contribute 2, 0 and 0 templates. `scripts/convention_census.py` reads all
five for a template-convention census reported in the companion manuscript, which is a use and not a
reason to redistribute: that census already reads SyGMa's rules from the installed package rather
than from a copy, and can read these the same way.

## BioTransformer's terms, exactly

`artifacts/tier2/biotransformer/LICENSE.md` and `README.md` grant the LGPL and say:

> Users are free to copy and redistribute the material in any medium or format. Moreover, they could
> modify, and build upon the material under the condition that they must give appropriate credit,
> provide links to the license, and indicate if changes were made. Furthermore, the above copyright
> notice and this permission notice must be included.

and:

> Use and re-distribution of the these resources, in whole or in part, for commercial purposes
> requires explicit permission of the authors.

So redistribution does **not** require permission; commercial use or redistribution does. Two things
about that are unsettled by the texts themselves. The prose names LGPL version 2.1 while the licence
appended below it and the README both name version 3, and the distribution does not reconcile them.
And "commercial purposes" is nowhere defined, which matters here because the released model is to be
deployed as a service on `way2drug.ru`: that is a use by the authors, not a downstream user's
question.

## SyGMa's terms

The installed distribution's metadata says `License: GPL` and nothing more; no licence text ships
with it and no version is named anywhere on disk. Which GPL version governs is therefore not settled
by anything in this checkout, and it decides whether a GPLv3 release is available. The upstream
repository is `https://github.com/ridderl/sygma`; checking whether it carries a `LICENSE` is the
cheapest way to close this.

## What this constrains

A permissive licence over the whole repository is not available while the bank contains GPL
templates. A GPL release is available only if SyGMa's version permits it, and it cannot include the
ENVMICRO file, whose NonCommercial and ShareAlike terms no GPL satisfies. In every case the LGPL
conditions have to be met in fact: credit, links to the licences, an indication that the templates
were extracted and merged, and the upstream notices retained. This repository currently holds a
licence text for only one of its four rightsholders, and no `LICENSE` file of its own.

None of this is an oversight to be corrected by a note. It is a set of decisions, and the ones that
cost nothing scientifically separate cleanly from the ones that cost something:

- the ENVMICRO file and the RetroSim file can be untracked at no cost to the bank;
- the standardisation file likewise;
- the templates themselves are load-bearing, and what removing them costs is measured in
  `results/licence_removal_cost.json`.

## The corpus

The annotated substrate–metabolite corpus is assembled from ChEMBL (CC BY-SA 3.0), DrugBank
(CC BY-NC 4.0), MetXBioDB (distributed with BioTransformer) and the GLORYx reference set. ChEMBL's
ShareAlike and DrugBank's NonCommercial terms cannot both be satisfied by one derivative, which is
why the Zenodo deposit carries no source structure: every substrate and every annotated metabolite
in it is replaced by a tautomer-canonical InChIKey before the archive is built.

The corpus files themselves (`grail_metabolism/data/*.sdf`, `*_triples*.txt`) are **not** tracked
here; they are gitignored and are obtained from the sources under the reader's own licences.
`results/test_references.json` **is** tracked and holds the evaluated test substrates and their
annotated metabolites as SMILES, so that every recall figure in the paper can be recomputed.

## Comparators

Per-substrate predictions from SyGMa, MetaTox and MetaPredictor are tracked under `results/` so the
comparison can be recomputed without re-running any of them. What can be pinned about each — a
version where one exists, the configuration, the frozen predictions with their digest, and the date
each entered this repository — is recorded in `results/comparator_provenance.json`; BioTransformer's
jar digest and SyGMa's version are additionally in `paper2/split_manifest.json`. Each remains the
property of its authors and is redistributed here only as the frozen output of a run on our
substrates.

## RDKit and the rest

RDKit is BSD-3-Clause. The remaining Python dependencies are listed in `requirements.txt` and are
not redistributed here; each carries its own terms.
