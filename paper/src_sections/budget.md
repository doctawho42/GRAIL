**Budget-matched view.** recall@15 is not a budget-fair comparison: SyGMa reaches 0.554 by emitting
~74 predictions per substrate, whereas GRAIL reaches 0.365 at ~8.7. Normalising by output —
precision, recall per prediction — reorders the board: SyGMa's precision is **0.029**, roughly 4×
below the low-output methods (GRAIL **0.109**, BioTransformer 0.127, MetaTrans 0.122, MetaPredictor
0.116). Precision is a pessimistic lower bound under incomplete annotation (§Methods), but the ~7×
output-volume gap it reflects is real. At a matched small budget (recall@5) the recall spread also
compresses (SyGMa 0.465 ≈ MetaPredictor 0.470; GRAIL 0.321). This corroborates the independent
SyGMa-overproduction finding of Boyce et al. 2022 (§2): SyGMa's coverage is bought with a large
prediction volume, which the budget-matched view controls for.

**Table 4.** Budget-matched leaderboard, tautomer-InChIKey (`results/budget_matched_leaderboard.json`):
recall at matched cut-offs, precision (recall per prediction), and mean output size. Same populations
as Table 3 — GRAIL and SyGMa on n≈1170, the tier-2 methods on the n=150 shared subset.

| method | recall@5 | recall@10 | recall@15 | precision | mean_output |
|---|---|---|---|---|---|
| **GRAIL** | 0.321 | 0.356 | 0.365 | 0.109 | 8.7 |
| SyGMa | 0.465 | 0.533 | 0.554 | 0.029 | 74.2 |
| BioTransformer | 0.315 | 0.396 | 0.444 | 0.127 | 10.8 |
| MetaPredictor | 0.470 | 0.575 | 0.585 | 0.116 | 11.2 |
| MetaTrans | 0.424 | 0.533 | 0.561 | 0.122 | 12.7 |

**Budget-matched frontier and cross-method decomposition (we measure our own instrument first).**
Table 4's cut-offs already place GRAIL below SyGMa at every k; a fine-grained frontier confirms this
is not an artifact of the k=15 operating point. Ranking each method's own candidate pool and reading
recall@k for k = 1…64 on the shared substrates (`results/budget_matched_frontier.json`, tautomer),
**SyGMa dominates GRAIL at every budget with no crossover** — from k=1 (**0.233 vs 0.127**) through
k=64 (**0.520 vs 0.410**). GRAIL does *not* win at a matched output budget; its learned
selector/ranker is the binding weakness (Proposition 1), which the budget-matched view isolates from
the output-volume confound. The mechanism is visible when recall is decomposed **per method**, not
only for GRAIL (`results/cross_method_decomposition.json`, shared n=150): SyGMa applies its whole
applicable rule set and emits it, so its **selection retention is 1.0 by construction** — it has no
selection stage (precision 0.076 at ~64 outputs) and its recall is simply its bank's realised
coverage, **0.520**. GRAIL's rule bank in fact covers **more** of these transformations — bank
ceiling **0.701** on this subset (cf. the 0.735 full-test ceiling, §6) > SyGMa's realised 0.520 —
yet GRAIL realises only recall@15 0.318 here because its selection + ranking retains just **0.454**
of that reachable coverage. The gap to SyGMa is therefore not a worse rule bank; it is a selection
tax that recall charges only to a method that *has* a selection stage — a tax GRAIL pays and loses.
We report this because a benchmark whose authors run their own pipeline through their own instrument,
and find it loses at every budget, is more trustworthy than one whose method wins.

