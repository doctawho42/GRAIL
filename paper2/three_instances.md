# Five numbers of our own, one of which was never wrong

The claim this work is built on is that a published number often refers to a different object
than a reader takes it to refer to. The claim is easier to make about other people's papers than
to live under. While building the instruments for this one we produced five numbers of exactly
that kind. Each was caught, each is fixed, and they are set out here because they are the only
honest explanation of why the checks in this repository sit where they sit.

The first three share a pattern. A quantity was computed, committed, and read afterwards as
though it described the object its name suggested. It did not. Nothing in any of the three was a
mistake in reasoning; each was a mismatch between what a name meant to one piece of code and
what it meant to another.

The fourth is a different kind: a number computed against nothing at all, which came out close to
its theoretical maximum and carried an interval that excluded zero.

The fifth is a different kind again, and it is the one worth the reader's attention, because
**no number in it was wrong**. There was no error. Every figure was correct, every check passed,
and the artifact was silent about a distribution it had computed and discarded. It is included
because a document listing only mistakes teaches that authors make mistakes, which nobody
doubts, while this one names a failure mode that survives a clean audit.

A sixth section records a defect that did not happen, for the same reason.

---

## 1. A rule bank of 7,581 that was silently a rule bank of 844

**What it said.** A step-0 report on the bank: type counts, singleton shares, combinatorics.

**What it described.** The 844 rules of the bank that contain no `#` character.

**The mechanism.** The loader stripped comments with `line.split("#")[0]`. In SMARTS, `#` is the
atomic-number primitive: `[#6:1]` is a carbon. The bank is written almost entirely in that
notation and carries no comment lines at all, so the strip truncated 6,737 of 7,581 rules into
fragments that failed to parse and were dropped. The survivors were the hand-written fifth, in
the older `[C:1]` notation — that is, the strip did not sample the bank, it selected a
provenance.

**What caught it.** Not an invariant. The loader printed `rules parsed: 844` next to the file's
7,581 lines, and the two numbers were side by side. Six invariants in the self-test were green
throughout, and could not have been otherwise: the fixture was written by hand in `[C:1]`
notation, so no fixture rule contained a `#`.

**The fix.** A comment is now `#` at the start of a line or after whitespace, never inside a
token. The loader returns its accounting — lines seen, parsed, unparseable — and the report
carries it. The self-test writes a probe file containing a `[#6:1]` rule and a real trailing
comment, and fails by name if the loader mishandles either. Reverting the fix makes it fail.

**What it cost.** Nothing, because the count was read before the numbers were. Had it not been,
every number in the step-0 verdict would have described the curated fifth of a bank whose
central finding is that its two halves behave differently.

---

## 2. A ceiling of 0.7284 that had been 0.8007 for months

**What it said.** `ceiling_on_this_subset: 0.7284` in a committed artifact, beside a pool
coverage of 0.7602 on the same 245 substrates.

**What it described.** The quantity as an earlier version of the code computed it. A correction
had since taken the same quantity, on the same substrates, to 0.8007.

**The mechanism.** The value was written into the artifact as a literal at the moment it was
produced. The function that produces it was later corrected. The artifact was not regenerated,
and nothing that read it asked which version of the code had written it. Every consumer saw a
committed JSON with a plausible number in a field with the right name.

**What caught it.** The docstring of the corrected function, which says so in as many words --
and only because someone was reading the function for another reason. That is not a mechanism;
it is luck. The number had already been used, in this project, to argue that a 245-substrate
subsample was systematically harder than the split it was drawn from and that a multi-hour
re-measurement was therefore needed. It is not: the subsample is a genuine random draw
(`rng.choice`, `replace=False`, seed 44, indices spread across the whole file) and the real gap
is 0.8007 against 0.8171, ordinary sampling variation on n = 245.

**The fix.** Artifacts here were gated on write -- a harness refuses to emit until it reproduces
a committed figure -- and read blind. The other half now exists. Writers stamp an artifact with
the digest of their own source; readers verify that the producer has not moved since. For
artifacts written before stamping, the producer is recovered from the commit that added the
artifact and hashed, which is labelled as an inference and not as a recorded fact. Where the
producer has moved, the sweep prints the diff and, when the two parse trees are identical with
docstrings removed, proves the change could not have altered a number.

**What it cost.** One wrong argument, made in this project's own working notes, for a
measurement that was worth making for a different reason.

---

## 3. A stratum of 277 substrates that was a stratum of 118

**What it said.** The membership file for hypothesis H1: substrates whose reference
transformation needs a label that is sparse when rules are the labels and dense when types are.
277 of 1,170 substrates, 376 of 2,536 references.

**What it described.** That, plus every reference whose mined SMIRKS happened not to be written
the same way as the catalog entry for the same rule.

**The mechanism.** The rule-level lookup was an exact string match between the SMIRKS derived
from a test transformation and the SMIRKS in the mined catalog. Two templates for the same
transformation, written by the same miner at different times or with different explicit
hydrogens, are different strings. 972 of 2,536 references found no catalog entry and were
recorded at support zero, which is indistinguishable from a rule that exists and is rare. This
is the comparison the second paper of this series is about, applied to the stratum a hypothesis
was to be registered on.

**What caught it.** A reviewer asking why the join was by string, on objects the work itself
proves are written inconsistently. The re-join is the measurement: under a notation-blind key at
radius 2, 252 of the 972 do find a catalog rule, and the stratum falls to 149 pairs over 124
substrates; at radius 1, to 143 over 118. 227 of the original 376 pairs were in the stratum by
notation.

**The fix.** Neither end is the truth -- a string join understates support, and a radius-2
signature pools over 4,264 keys for 5,856 catalog rules and overstates it -- so the stratum is
not a number but three, and H1 is registered as an enrichment factor that must hold under every
one of them, with the intersection named in advance as the primary definition. The sensitivity
is a committed artifact rather than a footnote.

**What it cost.** Nothing yet, because it was found before the freeze. Had it not been, H1 would
have been registered on a denominator uncertain by a factor of 2.2, and its verdict would have
been a fact about a join.


---

## 4. A gap of +0.4901 against a comparator that had made no predictions

**What it said.** On the validation split the three-way fusion arm led MetaTox by $+0.4901$ of
micro recall@15, with a paired-bootstrap interval of $[+0.4204, +0.5620]$ excluding zero.

**What it described.** That arm's own recall.

**The mechanism.** MetaTox was run on the 291-substrate comparison set and nowhere else. On
validation every one of its prediction lists is empty, and an empty list scores zero hits exactly
as a very bad method does. Subtraction cannot tell the two apart, and a bootstrap over the
difference puts a tight interval around the confusion. Nothing in the arithmetic was wrong.

**What caught it.** Reading the output. The figure sat in a table beside the arm's own recall of
$0.4901$, and the two being identical is not a coincidence any comparison can produce honestly.

**The fix.** A contrast now requires the number of items its comparator covers as a mandatory
argument and refuses to return a number when that is zero: an absent comparator is not a beaten
comparator. It also flags any gap that consumes almost all of an arm's own score, because the
other side contributing nothing is usually a measurement error rather than a triumph.

---

## 5. A distribution that was computed twelve times and never written down

**What it said.** The coverage decomposition: of $475$ uncovered references, $337$ need a reaction
type the bank does not contain, $98$ need one it does, $40$ admit no type.

**What was wrong with it.** Nothing. Every one of those numbers is correct, was correct on the
first run, and reproduces exactly.

**What it did not say.** How many *distinct* types those $337$ misses are. That question decides
the project's direction: $337$ misses over forty types with twenty of them carrying half the mass
is a fortnight of hand-written rules and a closed problem; $337$ misses over three hundred types
seen once each is a boundary to be declared. The two futures are not close and the artifact
distinguished neither.

**The mechanism.** The classifier types every uncovered pair and compares that type against the
bank. It has always known which type each miss carries, because that is how it decides the miss is
novel. It kept the count and discarded the label, twelve runs in a row, because the question its
author asked was "what fraction of the gap is new chemistry" and not "what is the new chemistry
made of". An artifact answers the question put to its author and is silent about the rest, and
that silence is indistinguishable from the data never having existed.

**What caught it.** Nothing in the repository. A reader asked the question, and the answer was
three lines of change and a re-run: the labels were still derivable because the inputs were
frozen. Had the inputs not been frozen, the twelve runs would have been unrecoverable.

**The answer, once recorded.** The $337$ misses are $312$ distinct types; $294$ are seen once and
carry $87.2\%$ of the mass. It is the second future.

**The fix, and it is mechanical.** If a classifier assigns a label in order to produce a count,
the label is what gets written, not the count. A distribution that has already been computed is
recorded whether or not the current question needs it. This costs one line and it is the only one
of the five defects that a numbers-versus-artifact audit cannot catch by construction, because
both sides of every statement were correct.

---

## 6. One that did not happen, and the check that stopped it

A document that lists only the defects which fired teaches that its authors make mistakes. This
one is here because it teaches which check works.

Bounding the tautomer enumerator was going to move the median substrate from fifty seconds to
five. The enumerator is a module-level singleton in the preprocessing library, and the function
that standardises a candidate is memoised on its SMILES alone. The matching key calls that same
memoised function. Lowering the budget for the generator would therefore have lowered it for the
key as well, and quietly redefined what counts as a hit, in a project whose first paper is about
preprocessing conventions moving published orderings.

It was caught by asking, before the run, which other callers share the object being changed. The
standardisation of survivors now uses a private enumerator with a private cache; the global pair
is untouched; and the check that the matching keys are unchanged was run on ninety-six molecules
before the sweep started rather than inferred afterwards.

The general form: **before changing a parameter, enumerate its readers.** A shared mutable default
is the mechanism by which a change to one stage silently redefines the measurement of another.

---

## What they have in common, and what follows

In each case the defect was invisible to the checks that existed, and for the same reason: the
checks tested whether the code did what it said, and the defect was that the object had changed
identity. A green invariant suite is evidence about logic. It is not evidence about population,
about code version, or about identity of key.

Five kinds of check answer five questions, and this repository runs all five.

| question | instrument | the instance it would have caught |
|---|---|---|
| is the population the one I meant? | a count printed at every load boundary | the 844 rules |
| is this number from the code I am reading? | a provenance stamp verified on read | the 0.7284 ceiling |
| is the key the same object on both sides? | the same join under a notation-blind key | the 277-substrate stratum |
| does the thing I am comparing against exist here? | coverage as a mandatory argument to a contrast | the +0.4901 gap |
| did I keep the distribution I already computed? | write the label, not the count | the 312 types |

The last row is the one that does not fit the others. The first four catch a number that is wrong.
The fifth catches a number that is right and alone, and no audit comparing numbers against
artifacts can reach it, because both sides agree.

A fourth thing follows about fixtures. Two of the three were invisible to a hand-written fixture
because the fixture was written in the author's own dialect: `[C:1]` where the bank uses
`[#6:1]`. A fixture that shares an author with the code shares the author's assumptions. Where a
check is cheap enough to run over the real artifact, it is run over the real artifact: the
invariants on the reactant-query rewriter run over all 7,580 bank templates, and that is how a
tightening bug in the admissibility gate was found that no hand-written case had exposed.

None of this makes the work more reliable than a reader is entitled to assume. It makes the
basis for that assumption checkable, which is the only claim any of these instruments supports.
