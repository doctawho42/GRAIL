# What changed since the first round

Every condition the editor set, and what was done. Numbers here are all generated from pinned
artifacts and appear in the documents through macros.

## Blocking

**1. CORPUS ASSEMBLY, written as a real subsection (SI S1).** The corpus files were written in
one run months before this repository's first commit; no script among its 839 commits reads any
source or writes the structure files, so the extraction cannot be read off code and is stated as
unrecoverable rather than omitted. What IS established, each measured: 464,966 records, and the
source-accession field is empty in every one, so per-source counts cannot exist; the label
convention checked against the triples rather than asserted, with the finding that the
positive-unlabelled negatives are stored in the corpus and not derived at load; 234 duplicated
substrate structures and 0 whose annotations disagree, so there was no conflict for a conflict
rule to arbitrate; splits substrate-disjoint and not molecule-disjoint, no split seed anywhere,
partition carried forward by digest. A verification tool (`scripts/verify_rebuilt_corpus.py`)
ships in place of the unshippable rebuild: it tells a reader holding their own licences exactly
how their corpus stands to ours, including whether their structures are drawn in the same
dialect, which a digest comparison cannot see.

**2. The worked example, repaired — and it did not survive.** The substrate SMILES is now stated
in the paper. On gemcitabine as a chemist draws it the interactive mode finds 0 of 4 rather than
1 of 4: the template it had been credited with, mined rule 4913, requires the exocyclic imine and
cannot fire on the correct structure. The exhaustive mode still finds 4 of 4, at ranks 6, 17, 24,
39 against 8, 13, 16, 21, and the deamination is found by curated template 251 at rank 39 instead
of a mined one at 13. Both drawings are reported.

**3. Substrate presentation, promoted to a declared axis and measured (SI S8).** The mechanism is
established: every corpus structure is a fixed point of the InChI round-trip (378,228 of 378,238;
41,591 of 41,591; 45,133 of 45,133, the ten exceptions being salts), and the round-trip places a
mobile hydrogen on oxygen. It is NOT the declared standardiser's output: run under ten RDKit
releases spanning four years, `standardize_mol` returns the amide in ten of ten, from both forms.
Census: 345 of 1,170 evaluated substrates are not fixed points of the standardiser, every move a
pure tautomer move; per split 2,892/9,194, 281/1,063, 350/1,200. Bank census: the mined half
requires an imidic reactant 684 times against 628 for a real amide, the curated half 62 against
170, and no SyGMa template requires it. The head-to-head sweep under both drawings is in SI Table S10 with the coverage ceiling on the
same population beside it.

The sweep itself, on all 291 substrates with both arms rebuilt: the largest effect is -0.0180 for
each arm, two of nine cells separate from zero for the exhaustive mode and one of nine for the
interactive, four of 54 arm-against-comparator verdicts move, and none of the four is a lead the
paper claims — the exhaustive mode's at k=30 and k=50 and SyGMa's at k<=5 all stand under both
drawings. The coverage ceiling on that population goes 0.8105 to 0.8030, a difference of -0.0075
with the interval covering zero. So the axis decides an individual prediction — it decides the
whole worked example — and barely moves the aggregate. That is reported as it came out.

Two gates had to pass before any of it was believed. The stored column reproduces the comparison
table of the main text at all 45 of its cells, which is the check that the sweep measures the same
quantity and not a near neighbour of it. Pointing both dialects at the same pools returns exactly
zero everywhere with every interval collapsed, which is the check that the pairing is right.

## The rest

**4. Composite share.** 8.9% is re-described as the disconnected-centre share and is a floor, not
a bound. A second instrument was registered as H16 in a commit of its own, with threshold E>=5
fixed from enumerated single-enzyme chemistry, BEFORE the instrument was written; the two commits
are in the released history in that order. It flags 903; the two agree on only 130; the union is
1,295 of 5,855, 22.1%, against a registered bar of 13.4%. Reconciled with the depth-two result
explicitly.

**4b. What the corpus's drawing cost the comparator.** SyGMa is the one comparator that can be
re-run. Handed the substrate as the declared standardiser draws it, its micro recall rises at
every budget above the first: +0.0135 at k=15, +0.0195 at k=30, +0.0211 at k=50, with four of the
nine intervals excluding zero — and they are the four widest budgets, which are the budgets at
which the paper claims a lead over SyGMa. It does not reverse the comparison; the exhaustive
mode's lead at k=30 is several times the correction. It does mean that lead is over a SyGMa
handed a drawing its rules were not written for, and the SI says a reader should subtract this
first. Found because condition 3 forced the question, and reported against our own interest.

**5. SyGMa containment, disclosed in both places** — beside Table 2 in the Results and where the
bank is described in SI S7. 152 of 175, 86.9% of SyGMa, all in the curated half and none in the
mined one, with what it means for reading Figure 1 stated rather than left to the reader.

**6. CURATED RULES, resolved.** The 492 are not curated. 477 of them are verbatim lines of a file
committed as extracted reaction rules; they are in that file's notation and no hand-written
collection's; and they carry the corpus's imidic drawing at 0.118 where the named expert
collections carry it at 0.003. The bank holds two machine extractions and one curated body of
1,233.

**7. Licence, decided.** The conflict does not resolve, so the deposit stops raising it: every
substrate and every annotated metabolite is replaced by its tautomer-canonical InChIKey before
the archive is built, so no source record is redistributed. 349 and 293 substrates keyed, 0
unkeyable, 0 collisions, counts in the manifest. What remains is our own output and carries CC BY
4.0. The tool refuses to build an archive from an unkeyed pool. The Zenodo DOI is the author's
action and its bracket stands.

**8. Front matter.** The Introduction's "artifacts sufficient to recompute any cell" is replaced
by the precise statement. The abstract's "rather than of our implementation" is gone; the bound's
convention dependence is measured at 9 references in 2,597 with an interval excluding zero, and
the abstract now claims only the true weaker thing. The hydrogen convention is named in the
Methods with the band 469-477 and the note that the public entry point defaults to the other
presentation and returns 688.

**9. Cross-references.** All nine were wrong. The manuscript now loads `xr` against the SI's own
aux and every pointer is a `\ref`, so a wrong one prints ?? and fails the build. The checker
refuses any hand-typed `Table~S<n>` in any source INCLUDING the table generators, where two
literals were still living and would have come back on the next regeneration.
Perturbation-tested in both directions.

**10. Provenance.** 59 pinned artifacts, all current. Table S8 now carries the paired interval on
the fusion-minus-product contrast for both populations, so the selection penalty reads as two
overlapping intervals rather than a difference of point estimates. No released artifact carries
an absolute filesystem path any more (47 were removed across 24 files, including both
verification manifests). Nothing disclosed was trimmed.

## Found while doing the above, and reported rather than filed

- **The arms did not all meet the same drawing.** The MetaTox submission re-tautomerises 79 of
  the 291 substrates to the natural form, on the stated and correct ground that sending an
  unnatural imidic acid to an external service would be unfair to it. The other four arms met the
  corpus string. So SyGMa, whose 175 templates are written entirely in amide notation, did not
  get the drawing its rules expect and the external service did.
- **The annotation was never described** in a paper entirely about measuring against it. SI now
  carries the census: 2,597 references over 1,170 substrates, 2,292 distinct; 790 smaller than
  their parent, 283 the same size, 1,524 larger; 8 equal to their own substrate.
- **The preregistration gate had never run on this paper.** It knew only H-identifiers and this
  paper numbers its predictions P1 upward, so it reported all sixteen absent. Fixed, and all
  sixteen are now accounted for.
- **The generated-macro claim was false of the SI**, where 37 measurements are typed by hand. The
  claim now states what is true and its own three figures are generated.
- The coverage census's invariance argument covers 337 of the 475, not all of them. Said so.
- "GRAIL ahead at k=30" was a maximum over our own two modes; the arm is now named everywhere,
  including that the interactive mode trails MetaTox at those budgets.


## Two further passes after the conditions were closed

**A full proofread of both documents** found 36 defects, most of them introduced by the night's
own edits, and all were fixed: three stale prediction counts, a bank still described as four
curated collections three sentences before the section disproving it, "three quarters of the bank
with no expert provenance" that is five sixths after the reattribution, a P4 grid that was 5x10 in
the table and 5x11 in the text, a worked-example caption carrying timings from two producers ago,
a subsection heading asserting what its own subsection denies, "the counts are reported for every
arm" followed by three of five, a load correction a reader could not derive from the medians
beside it, two populations under one word, a licence macro pasted into a slot whose wording it did
not fit and which compiled to nonsense, and a dozen repairs to spliced prose. Four labelled floats
were never cited; three now are.

**Cross-references now resolve in both directions.** The manuscript's pointers into the SI were
converted to \ref through xr; the five running the other way were still literals with a
disclaimer attached. A disclaimer is not a fix. The SI now loads xr against the manuscript, both
documents are built twice so each reads the other's numbering, and the gate refuses a hand-typed
float pointer in either document. Perturbation-tested in both directions.

## What remains open, and why

Two brackets. The Zenodo DOI is the author's action; the deposit is built, hash-keyed and
verified, and minting the record is a step this work will not take on the author's behalf. The
provenance of the three named curated collections is an external fact about published rule sets,
not a measurement, and is the one item the repository genuinely cannot supply.

37 hand-typed measurements remain in the Supporting Information, enumerated in the checker where
the next person can see them. That is stated in the paper rather than smoothed over, and it is the
exact statement of what is left to wire.
