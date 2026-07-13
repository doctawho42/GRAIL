# GRAIL: rule-based metabolite-structure prediction, a coverage×selection×ranking diagnosis, and the TAME evaluation protocol

> **Draft status (2026-07-13):** first assembled full draft. Numbers sourced from `docs/GRAIL_FRAMING.md` / `results/*.json`. Compute-gated values are marked `[PENDING: ...]`; unverified citations are marked `[cite: ...]`. Venue target: JCIM / J. Cheminformatics.

## Abstract
> _[STUB — Task 11]_

## 1. Introduction
> _[STUB — Task 11]_

## 2. Related Work
> _[STUB — Task 10]_

## 3. Methods — GRAIL architecture

GRAIL predicts xenobiotic metabolite structures with a three-stage, rule-based-plus-learned
pipeline. The first stage, the **generator**, is a learned multi-label rule selector that
approximates `P(r|s)` — the probability that rule `r` applies to substrate `s` — over a curated
bank of **7,581 SMIRKS** rules; the default scorer is retrieval-based, combining cross-attention
between substrate and rule graph embeddings, an embedding-similarity term, and an MLP head. The
second stage, **RDKit rule application**, mechanically applies every rule the generator selects
and enumerates the resulting candidate products, retaining provenance of which rule produced which
candidate. The third stage is a **PU-trained, MCS-aware pair filter**: a binary classifier scoring
each (substrate, product) pair. Its training data are positive-unlabeled — annotated true
metabolites are the only positives, while rule-applicable products lacking a positive annotation
are treated as *unlabeled*, not as confirmed negatives, since absence of an annotation does not
certify a transformation does not occur. The filter is accordingly trained in the logit domain
(`return_logits=True`) so a PULoss/nnPU surrogate operates on raw classifier outputs rather than
post-sigmoid probabilities, and the generator itself down-weights unobserved-but-applicable rules
rather than penalizing them as hard negatives. Featurization uses fixed-width graphs: single-molecule
graphs feed the generator encoder with 16-dim node features, while the pair filter operates on
merged substrate–product graphs with 18-dim nodes and 18-dim edges plus a 1024-dim Morgan
fingerprint branch; cross-edges linking substrate and product atoms come from an element-aware
maximum common substructure (MCS) atom correspondence rather than sorted or arbitrary indices,
preserving chemically meaningful atom mappings across the reaction. At deployment,
`ModelWrapper.generate` runs all three stages and ranks the candidate set by
`filter_score × generator_score`, combining the filter's pair-plausibility judgment with the
generator's rule-selection confidence into a single ranking signal. Throughout this paper, unless
stated otherwise, structure matching between predicted and reference metabolites uses
`inchikey_tautomer` as the default match mode. The three stages together form an interpretable
instrument — the selected rule, the enumerated product, and the filter's pair judgment are each
inspectable — and the contribution we claim is interpretable learned rule selection paired with a
PU-aware pair filter, not recall supremacy over other metabolite predictors.

## 4. Methods — Formal framework
> _[STUB — Task 3]_

## 5. Methods — TAME evaluation protocol
> _[STUB — Task 4]_

## 6. Results — Rule-bank coverage ceiling
> _[STUB — Task 5]_

## 7. Results — External validity of the ceiling
> _[STUB — Task 5]_

## 8. Results — Recall decomposition
> _[STUB — Task 6]_

## 9. Results — Honest-anchor certification
> _[STUB — Task 7]_

## 10. Results — Diagnosis: levers and three propositions
> _[STUB — Task 8]_

## 11. Results — Match-sensitivity and cross-method comparison
> _[STUB — Task 9]_

## 12. Limitations
> _[STUB — Task 12]_

## 13. Data & Code Availability
> _[STUB — Task 12]_

## 14. Conclusion
> _[STUB — Task 12]_

## Figure 1 — pipeline schematic
> _[FIGURE 1: GRAIL 3-stage pipeline schematic — TO BUILD]_ A left-to-right schematic: (i) substrate + 7,581-rule SMIRKS bank → learned retrieval-scored **generator** selecting rules; (ii) **RDKit rule application** enumerating candidate products; (iii) **PU-trained MCS-aware pair filter** scoring (substrate, product) pairs; deployment ranks by `filter_score × generator_score`. Real schematic is a post-draft task.

## Draft TODO / open items
> _[STUB — Task 12 seeds this; final content is the out-of-scope track-list]_
