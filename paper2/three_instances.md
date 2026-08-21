# Three numbers of our own that did not describe what they appeared to describe

The claim this work is built on is that a published number often refers to a different object
than a reader takes it to refer to. The claim is easier to make about other people's papers than
to live under. While building the instruments for this one we produced three numbers of exactly
that kind. Each was caught, each is fixed, and the three are set out here because they are the
only honest explanation of why the checks in this repository sit where they sit.

The pattern is the same in all three. A quantity was computed, committed, and read afterwards as
though it described the object its name suggested. It did not. Nothing in any of the three was a
mistake in reasoning; each was a mismatch between what a name meant to one piece of code and
what it meant to another.

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

## What the three have in common, and what follows

In each case the defect was invisible to the checks that existed, and for the same reason: the
checks tested whether the code did what it said, and the defect was that the object had changed
identity. A green invariant suite is evidence about logic. It is not evidence about population,
about code version, or about identity of key.

Three kinds of check answer those three questions, and this repository now runs all three.

| question | instrument | the instance it would have caught |
|---|---|---|
| is the population the one I meant? | a count printed at every load boundary | the 844 rules |
| is this number from the code I am reading? | a provenance stamp verified on read | the 0.7284 ceiling |
| is the key the same object on both sides? | the same join under a notation-blind key | the 277-substrate stratum |

A fourth thing follows about fixtures. Two of the three were invisible to a hand-written fixture
because the fixture was written in the author's own dialect: `[C:1]` where the bank uses
`[#6:1]`. A fixture that shares an author with the code shares the author's assumptions. Where a
check is cheap enough to run over the real artifact, it is run over the real artifact: the
invariants on the reactant-query rewriter run over all 7,580 bank templates, and that is how a
tightening bug in the admissibility gate was found that no hand-written case had exposed.

None of this makes the work more reliable than a reader is entitled to assume. It makes the
basis for that assumption checkable, which is the only claim any of these instruments supports.
