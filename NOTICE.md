# Third-party content in this repository, the terms it carries, and the licence that follows

This file records what this repository holds that it did not write, the terms each carries, and how
the repository's own licence follows from them. **The licence is GPLv3** (`LICENSE`), and it is not
a preference: the bank redistributes reaction templates published under the GPL and the LGPL, and
the GPL is the licence that satisfies both. A permissive release is not available while those
templates are in the bank, and what removing them would cost is measured below rather than guessed.

Two separate obligations run through what follows and they are easy to conflate. The **bank**
contains other people's templates, which puts the bank under their terms. The **repository** tracked
other people's files, which would put the release under the terms of everything inside them,
including the parts the bank never uses. Both are measured in `results/curated_third_party.json`.
The second obligation is now discharged: of the five third-party rule files, one remains tracked and
four do not.

Three decisions were taken and each is stated with what it cost:

| decision | why | measured cost |
|---|---|---|
| GPLv3 for the code and the bank | SyGMa's templates are GPL, BioTransformer's are LGPL, and GPLv3 satisfies both | none; a permissive licence was never available |
| stop tracking four of the five third-party rule files | one is CC BY-NC-SA and no GPL release can carry it; two carry no licence text held here; one is redistributable but contributes no template the bank uses | none: the bank is unchanged, and the files stay obtainable from their own projects |
| do not distribute the corpus | its four sources' terms do not combine and the assembly recorded no per-record provenance, so no subset can be shown free of either | stated below, and it is not zero |

## The rule bank

`grail_metabolism/resources/extended_smirks.txt` holds 7,581 SMIRKS templates: 5,856 mined from the
training split by `scripts/mine_rules.py` and 1,725 carried as curated. Of the curated portion,
1,233 come from three named collections and 492 from an earlier machine extraction that was carried
as curated and is not (`results/curated_provenance.json`).

**966 of the 1,725 curated templates are somebody else's rules, verbatim**, and every one of them
falls inside the named 1,233, none among the 492. The share to read is therefore **78.3% of the
collections described as curated chemistry**, not 56% of the half they sit in. Measured by string
equality against every published rule set held on disk:

| rightsholder | templates in the bank | terms, as its own distribution states them |
|---|---:|---|
| BioTransformer | 611 | LGPL (see below) |
| SyGMa 1.1.0 | 273 | "GPL", no version given |
| GLORYx | 82 | no licence text for it is held here |
| RetroSim | 0 | no licence text for it is held here |

The SyGMa count is 273 and not 152 because GLORYx's own rule file attributes 178 of its 260 rules to
SyGMa, in a notation the installed SyGMa package does not use. String equality cannot trace a
template that was rewritten, so every count above is a lower bound, and the 759 curated templates
that match nothing measured are not thereby shown to be original.

## The files, and which of them this repository still redistributes

`grail_metabolism/resources/external/` holds five third-party rule files on disk. Redistributing a
file carries the terms of everything in it, whatever the bank uses, so four of the five are no
longer tracked. They stay on disk for anyone who has them, `results/curated_third_party.json`
records which were found, and each is obtainable from its own project:

| file | size | templates the bank uses | terms | tracked |
|---|---:|---:|---|---|
| `bt_database_metabolicReactions.json` | 779 KB | 611 | LGPL, redistribution permitted | **yes** |
| `bt_database_ENVMICRO_metabolicReactions.json` | 172 KB | 2 | **CC BY-NC-SA 4.0** | no |
| `bt_database_standardizationReactions.json` | 11 KB | 0 | LGPL | no |
| `gloryx_reactionrules.csv` | 37 KB | 260 | none held here | no |
| `retrosim_templates_general.json` | 120 KB | 0 | none held here | no |

The ENVMICRO file was the sharp one and it is the reason the rule was drawn this way. Its own
header, and the `LICENSE` beside it upstream, put the EAWAG data it holds under CC BY-NC-SA 4.0,
licensed by EnviPath: NonCommercial and ShareAlike, which no GPL release can carry. Its two
templates that reach the bank are both also present in the LGPL core file, so nothing here depends
on it.

The one that remains is the LGPL core, and it stays for a reason beyond permission: it is the
evidence for the largest single attribution this bank owes, 611 templates, and a reader checking
that claim needs the file the claim is made against. The other four were an obligation with no such
return: two contribute no template at all, and GLORYx's own attribution column says 178 of its 260
rules are SyGMa's, which the installed SyGMa package supplies directly.

`scripts/convention_census.py` reads all five when they are present, which is a use and not a
reason to redistribute: it already reads SyGMa's rules from the installed package rather than from
a copy, and reads these the same way.

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
with it and no version is named anywhere on disk. A program offered under "the GPL" with no version
is conventionally read as offered under any version, at the recipient's choice, which is how it is
read here and why GPLv3 is available. That reading is a convention and not a statement by SyGMa's
authors: the upstream repository is `https://github.com/ridderl/sygma`, and if it carries a
`LICENSE` naming a version, that version governs and this file should be corrected to it. The
reading is recorded rather than assumed so that a correction has something to correct.

## What this constrained, and the licence that follows

A permissive licence over the whole repository was never available while the bank contains GPL
templates. A GPL release is available: SyGMa's distribution says "GPL" without naming a version,
which is read here the way the Free Software Foundation reads it, as any version at the recipient's
choice, and BioTransformer's LGPL is compatible in this direction. What a GPL release cannot carry
is the ENVMICRO file, whose NonCommercial and ShareAlike terms no GPL satisfies, and that file is
no longer tracked.

**The licence is therefore GPLv3**, in `LICENSE`, covering the code and the rule bank. The LGPL
conditions are met in fact and not only in principle: credit is given above and in `LICENSE`, the
licence text for BioTransformer is at `artifacts/tier2/biotransformer/LICENSE.md`, this file states
that templates were extracted from those distributions and merged into one bank, and the upstream
notices are retained. Two obligations are recorded and not discharged, because they are questions
for the rightsholders rather than for a file in this repository: BioTransformer's own README
requires explicit permission for *commercial* use or redistribution, and the released model is to
be deployed as a service, which the authors should settle with its authors directly; and GLORYx's
82 templates carry no licence text this repository holds, so the cost of removing them is measured
below and the decision is stated rather than assumed.

None of this is an oversight to be corrected by a note. It is a set of decisions, and the cost of
each is measured rather than guessed (`results/licence_removal_cost__clean_test.json`, one
uncapped pass of each bank variant over the 1,170 evaluated test substrates):

| option | templates dropped | references lost of 2,597 | change in reach |
|---|---:|---:|---|
| drop BioTransformer's | 611 | **0** | 0.0000 |
| drop SyGMa's | 152 | 14 | −0.0054 [−0.0086, −0.0026] |
| drop every borrowed template | 763 | 14 | −0.0054 [−0.0086, −0.0026] |

**The 611 BioTransformer templates cost nothing.** Every reference they reach is reached by
something else in the bank, so the largest borrowing here is the one that can be given up for
free. The whole obligation to BioTransformer, the templates and the tracked files together, can
therefore be discharged by removal at no measurable cost to the science.

SyGMa's 152 cost 14 references of 2,597, 0.54% of the reach. That is a real but small price, and
it is the only one of these decisions where anything is being traded.

The unused files cost nothing to untrack and are untracked. What is kept is the LGPL core, whose
611 templates the bank uses and whose redistribution the LGPL permits.

## The corpus

The annotated substrate–metabolite corpus is assembled from ChEMBL (CC BY-SA 3.0), DrugBank
(CC BY-NC 4.0), MetXBioDB (distributed with BioTransformer) and the GLORYx reference set. ChEMBL's
ShareAlike and DrugBank's NonCommercial terms cannot both be satisfied by one derivative, which is
why the Zenodo deposit carries no source structure: every substrate and every annotated metabolite
in it is replaced by a tautomer-canonical InChIKey before the archive is built.

The corpus files themselves (`grail_metabolism/data/*.sdf`, `*_triples*.txt`) are **not** tracked
here; they are gitignored and are obtained from the sources under the reader's own licences.

`results/test_references.json` held the evaluated test substrates and their annotated metabolites
as SMILES: 1,170 substrates and 2,597 metabolites of that same corpus. It is no longer tracked, and
what replaced it is `results/test_reference_descriptors.json`.

**Keying it with one hash was not available, and the reason is this work's own contribution.** The
candidate pools could be keyed because a recall figure is computed on keys. Fixing the reference to
a single key fixes the *matching criterion* with it, and this work sweeps five, so four of them
would become uncomputable and the demonstration the paper is largely about would go with them.

Every one of the five is nonetheless decided by something that is not a structure, which is what
the replacement carries per reference:

| criterion | decided by | in the file |
|---|---|---|
| canonical SMILES equality | equality of a string | its SHA-256 |
| full InChIKey | a hash | the InChIKey |
| stereochemistry-blind first block | a hash | the InChIKey's first 14 characters |
| tautomer-aware key (the default) | a hash | the key |
| Tanimoto = 1 on Morgan fingerprints | the fingerprint, a lossy irreversible descriptor | its on-bits |

`scripts/typed_edit/reference_descriptors.py --verify` checks the substitution rather than
asserting it: on every pair it compares, each criterion's verdict on the descriptors matches its
verdict on the structures, and it reports **0 disagreements**. The file reconstructs no metabolite:
a cryptographic hash and a folded 1,024-bit fingerprint are not the molecule.

The substrates stay as structures, for a reason that has nothing to do with licences: reproducing
the comparison means running a predictor on a substrate, and a hash cannot be run on. They are 1,170
drug and xenobiotic structures, each individually a published fact obtainable by name, and it is the
annotation rather than the compound list that is the corpora's contribution.

What this costs a reader who does not hold the source licences: they can recompute every recall
figure and every cell of the criterion sweep, and they cannot re-run the structure-level analyses
in the Supporting Information, the transformation-class split, the composite-step instruments and
the stereochemistry census, because those read the reference structure itself. Those need the corpus,
which such a reader already needed.

## Comparators

Per-substrate predictions from SyGMa, MetaTox, MetaPredictor and BioTransformer are tracked under
`results/` so the comparison can be recomputed without re-running any of them. BioTransformer's are
the output of its own jar on our substrates, at `allHuman` for one step and again on the natural
tautomer; the jar is LGPL and redistributing what it produced on our own inputs is not a
redistribution of the tool. What can be pinned about each, a
version where one exists, the configuration, the frozen predictions with their digest and the date
each entered this repository, is recorded in `results/comparator_provenance.json`; BioTransformer's
jar digest and SyGMa's version are additionally in `paper2/split_manifest.json`. Each remains the
property of its authors and is redistributed here only as the frozen output of a run on our
substrates.

## RDKit and the rest

RDKit is BSD-3-Clause. The remaining Python dependencies are listed in `requirements.txt` and are
not redistributed here; each carries its own terms.
