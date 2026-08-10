# The engine axis is an incomplete loop, not a convention

Verified 2026-08-10 against RDKit 2022.09.5, the version the paper declares.

## What was checked

The paper's application loop expands the substrate with `AddHs`, fires the template, and sanitises
the product. It reports that 63% of the resulting fragments are structures RDKit will not read back,
against 0.4% without the expansion, and concludes: "The template is not wrong and the engine is not
wrong. They disagree about how a hydrogen is spelled."

Inserting one call before sanitisation — `RemoveHs(product, sanitize=False)` — changes that.

## On the paper's own worked example

Benzoic acid under aromatic hydroxylation, `[cH1:1]>>[c:1]O`:

| loop | products that sanitise |
|---|---|
| `AddHs → RunReactants → sanitize` (the paper's) | 0 of 5 |
| `AddHs → RunReactants → RemoveHs → sanitize` | 5 of 5 |

The three distinct products of the repaired expanded arm are identical to the three the implicit arm
produces: `O=C(O)c1ccc(O)cc1`, `O=C(O)c1cccc(O)c1`, `O=C(O)c1ccccc1O`.

## At scale

300 rules drawn from the bank at seed 0, against five substrates, products built from the expanded
substrate:

| | share sanitising |
|---|---|
| as the paper's loop does it | 40 of 557 = 7.2% |
| with one `RemoveHs` first | 376 of 557 = 67.5% |

Of the firings that produce anything under either arm, 58 of 66 (88%) give the repaired expanded arm
exactly the implicit arm's product set.

## Measured on the full split

`results/completed_loop_reach__clean_test.json`, the 152 shared rules over all 1,170 clean test
substrates, three arms of the same loop:

| arm | reach |
|---|---|
| expanded, loop as it stood | 0.1998 [0.182, 0.219] |
| expanded, loop completed | 0.3423 [0.320, 0.365] |
| implicit | 0.3897 [0.367, 0.413] |

The published engine term of +0.1898 [0.173, 0.208] splits into **+0.1425 [0.125, 0.160] from the
missing call** and **+0.0474 [0.038, 0.057] that survives it**. Both intervals exclude zero. Three
quarters of what section 4 called a convention was an unfinished loop.

`results/explicit_h_mechanism__clean_test.json` adds a third arm to the mechanism and inverts its
story:

| arm | fragments | unreadable |
|---|---|---|
| loop as it stood | 21,905 | 13,804 (63.0%) |
| loop completed | 20,601 | **0 (0.0%)** |
| implicit | 21,106 | 87 (0.4%) |

Completing the loop removes every unreadable fragment, cleaner than the unexpanded arm. So the
product-breakage mechanism is not a convention effect at all. What survives is the opposite kind of
failure: the completed loop fires the same templates and yields 20,601 fragments against 21,106, so
a substrate carrying its hydrogens matches slightly less. The section's sentence "what the switch
costs is not matching" was right about the defect and backwards about the convention.

## What this costs the paper

The engine axis is one of the three the title counts and the one the abstract leads with as a
genuine reversal. If most of the +0.188 engine term is the missing call rather than the convention,
then:

- the mechanism sentence is false as written: the engine is missing a step, and saying neither side
  is wrong is the one reading the measurement does not support;
- the +0.188 term, the 63%-vs-0.4% gap and the SyGMa/BioTransformer exchange all compare a complete
  loop against an incomplete one and have to be re-measured with the loop completed;
- if the term collapses, the engine axis leaves the title, the abstract and the count of three, and
  returns as what it still is: a coverage figure is not comparable across papers that leave the
  application procedure unstated, which the 51,368-template survey and the transcription-convention
  finding establish without needing a reversal.

## What is not affected

The census across six libraries, the three-construct taxonomy corrected to six, the transcription
control, the criterion axis, the budget axis, the population null, the decomposition and the
coverage ceiling are all measured elsewhere and do not pass through this loop.

## What has to happen next

Re-run every arm of section 4 and of the engine-reach appendix with `AddHs → RunReactants →
RemoveHs(sanitize=False) → sanitize`, and report what survives, whichever way it comes out.
