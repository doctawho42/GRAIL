# Per-template hydrogen dispatch: the rule, the predictions, and what would kill it

Written and committed **before** the measurement runs. A dispatch policy that is chosen after seeing
which setting wins is an oracle, and an oracle beating the better of two global settings is
arithmetic rather than a result. What follows is fixed in advance and can lose.

## The claim under test

A rule bank is a mixed-convention object: some templates spell out the hydrogen they consume and
some leave it implicit. Every engine in this literature applies one convention to the whole bank, so
a published coverage figure is the value of a (bank, convention) pair rather than of the bank.
Dispatching the convention per template should make the figure a property of the bank alone.

## The rule, fixed

For each template, expand the substrate with explicit hydrogens **iff the reactant side of that
template contains a bracket-hydrogen atom token** — `[H]` or `[#1]`, optionally atom-mapped or
charged. This is `needs_explicit_hydrogen` in `scripts/explicit_h_mechanism.py`, unchanged, and it
is a syntactic test on the rule file that touches no molecule and no outcome.

Three things it explicitly does not do, each of which would make it an oracle:

- it does not read the product side;
- it does not use the hydrogen-**count** primitive inside `[CH3]` or `[C;H2]`, which is unaffected
  by expansion;
- it does not treat a negation such as `[!#1]` as a hydrogen.

**Unclassifiable templates.** The token test cannot see inside recursive SMARTS, `$(...)`. There are
238 such templates: 126 ours, 76 SyGMa's, 36 BioTransformer's (`results/explicit_h_mechanism.json`).
Their policy is frozen here as **the majority convention of the bank they belong to** — implicit for
ours and SyGMa's, explicit for BioTransformer's — and never as whichever setting turns out to win.

## Predictions, in advance

| bank | prediction | why |
| --- | --- | --- |
| SyGMa, 175 | **exactly 0.5273** | none of its templates carries a hydrogen atom, so dispatch must reduce to the identity map and reproduce the implicit-hydrogen arm to four decimals |
| ours, 7581 | strictly above **0.7989** | 675 templates want the expansion that the implicit arm denies them |
| BioTransformer, 994 | strictly above **0.4727** | 304 templates want the implicit form that the deployed arm denies them |

The SyGMa row is the null and it is the one that matters. If dispatch returns anything other than
0.5273 there, the classifier is wrong and every other number in the run is uninterpretable.

## The deliverable is not the win

Beating the better global setting is a precondition, not the finding. The quantity that carries the
claim is the **residual convention-dependence** per bank:

    residual = reach(dispatch) - max(reach(all explicit), reach(all implicit))

A residual at zero for a bank says that bank is single-convention and picking the right global
setting is all there is; a residual above zero says the bank is genuinely mixed and no global
setting can express it. The claim "reach is a property of the bank" is worth making only for banks
whose residual is positive, and its size is the measure of how much a global convention costs.

## What would kill it

- SyGMa deviates from 0.5273 → the classifier is buggy; stop and fix before reading anything else.
- BioTransformer fails to clear 0.4727 → the syntactic rule does not capture what a template needs,
  dispatch would require per-template empirical calibration, and calibration on the outcome is the
  oracle this pre-registration exists to forbid. In that case reach stays a pair, the manuscript
  keeps its present framing, and this direction is reported as a failed repair.
- All three residuals at zero → banks are single-convention after all, the mixed-convention premise
  is wrong, and the correct statement is the weaker one the paper already makes.

## What this does not change

The deployed configuration stays as reported. GRAIL applies one convention globally, that is what
its published numbers measure, and dispatch is a proposed repair with a measured benefit rather than
a correction to them. No headline figure is restated on its account.
