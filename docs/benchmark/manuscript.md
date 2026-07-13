# GRAIL: rule-based metabolite-structure prediction, a coverage×selection×ranking diagnosis, and the TAME evaluation protocol

> **Draft status (2026-07-13):** first assembled full draft. Numbers sourced from `docs/GRAIL_FRAMING.md` / `results/*.json`. Compute-gated values are marked `[PENDING: ...]`; unverified citations are marked `[cite: ...]`. Venue target: JCIM / J. Cheminformatics.

## Abstract
> _[STUB — Task 11]_

## 1. Introduction
> _[STUB — Task 11]_

## 2. Related Work
> _[STUB — Task 10]_

## 3. Methods — GRAIL architecture
> _[STUB — Task 2]_

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
