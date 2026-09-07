# Cover letter

**To the Editors, *Journal of Chemical Information and Modeling***

**Manuscript:** Metabolite Predictors Have a Multiverse Problem

**Authors:** Nikita L. Polomoshnov (corresponding, nikitapol@fbb.msu.ru), Anastasia V. Rudik

**Type:** Article

---

Dear Editors,

We submit the manuscript above for consideration as an Article.

## What the paper is

It is an evaluation study, not a report of a new predictor that outperforms its predecessors. The
question it starts from is that recall figures for metabolite structure predictors are quoted
against one another and are not comparable, because recall depends on two decisions made outside
any model: what counts as a match between a predicted structure and a reference, and how many
candidates a system is allowed to emit. Papers in this area report neither consistently, and the
systems compared here differ by more than a factor of twenty in what they emit by default.

We promote both decisions to declared axes and sweep them, running five published predictors and
two configurations of a rule-grounded system of our own over the same 291 substrates carrying 665
annotated metabolites, and we print the whole grid rather than a cell. The comparison does not
resolve into a single ordering, and we say so: the verdict moves with the matching criterion at six
of nine budgets, and cutting every arm to the comparator's own list length retires three of the four
leads the sweep produces at wide budgets. We then bound what one application of one template bank
can reach, and decompose the shortfall into what the bank misses and what the corpus never
contained.

We are aware that the author guidelines ask a new modelling approach to be benchmarked across tasks
and to show superiority consistently. We would ask that the manuscript be read as what it is rather
than against that expectation, because it does not make that claim and its contribution does not
rest on one. The system in it is the instrument that makes the two axes measurable; the result is
the measurement. We think the paper is useful to this journal's readers precisely where it declines
to produce a winner, since the ordering it fails to produce is the ordering the literature currently
quotes.

## Why this journal

The tools compared here were published in this journal and its neighbours, and the protocol
question the paper addresses was raised in this journal a year ago by Ash and co-workers, whose
prescription for method comparison we cite and build on. An independent benchmark of this same tool
class appeared here in 2026; we use its published tables, without re-running anything of theirs, to
show that the confound we describe is present in it as well, and we report that check in the
Supporting Information.

## Declarations

**Concurrent submission, at the level of the shared repository.** A separate manuscript by the
corresponding author is under review elsewhere. It is a different paper on a different subject: a
re-scoring study over 24 published leaderboards in four domains, asking what part of a published
rank order survives when the evaluation choices behind it are declared and varied. It contains no
part of the present manuscript and reaches no conclusion about metabolite prediction as such.

What the two share is the repository. `github.com/doctawho42/GRAIL` hosts the code and the deposited
artifacts of both, because both were developed in it, and one of the leaderboards that other study
re-scores is the metabolite comparison this manuscript reports. In consequence the frozen
per-substrate predictions released with this submission are also read by that study, in the same
form and without modification. The 111 artifacts the present manuscript's figures are drawn from are
listed in its Data and Software Availability section, and each records the script and source digest
that produced it, so a reader can establish which numbers belong to which manuscript without relying
on our word for it. We raise this because the overlap is real at the level of files, and we would
rather state it than have it discovered.

**Competing interest.** A.V.R. is an author of MetaTox, one of the comparators, and is a co-author
here. This is stated in the manuscript. MetaTox is also the incumbent service the system reported
here is intended to succeed. The comparator arm was obtained from the public service and its
configuration and its limits are recorded in the Supporting Information, including what about the
service's settings cannot be recovered from the outside.

**Data and Software Availability.** The manuscript carries the required section. Source code, the
deployed checkpoints, the released rule bank, the split manifest, the frozen per-substrate
predictions of every comparator, the evaluation harness and the preregistration are in the
repository named there, and the candidate pools too large to commit are deposited separately. One
point deserves the editor's attention: the bank we release is not the bank we measure. Our figures
are measured on 7,581 templates, of which 611 are present verbatim in BioTransformer's published
reaction set, whose distribution requires explicit permission to redistribute that we have not
sought. Those are removed and 6,970 ship. We measured what the removal costs rather than assuming
it: zero references on the evaluated test set. The reason, the counts per rightsholder and the cost
of removing each are set out in the manuscript, in the Supporting Information and in the
repository's own notice file.

**Prior publication.** No part of this work has been published previously, and it is not under
consideration by another journal.

**Preregistration.** Sixteen predictions were registered with thresholds and falsification
conditions before they were checked. The register is released and the manuscript reports the ones
that failed alongside the ones that held.

## Suggested reviewers

We suggest referees with experience of rule-based metabolite prediction and of evaluation
methodology, and we ask that referees be free of a working relationship with the MetaTox group. We
have no objection to any particular referee.

Thank you for considering the manuscript.

Nikita L. Polomoshnov
Faculty of Bioengineering and Bioinformatics, Lomonosov Moscow State University
nikitapol@fbb.msu.ru
