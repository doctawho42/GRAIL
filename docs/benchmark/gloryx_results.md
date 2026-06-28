# GLORYx-37 results — match-sensitivity on the literature shared set

The GLORYx external set (37 drugs; reference metabolites as a per-generation tree, flattened
to all generations — 205 entries, ~136 unique by InChIKey) is the field's de-facto shared
hold-out. We score every method's *fixed* predictions under all five matching protocols
(`scripts/eval_on_gloryx.py`; data `docs/benchmark/data/gloryx_test.json`, escapes fixed on
load). GRAIL = full5000_priors checkpoint at val-selected `prior_strength=8`, top-15.

## Recall@15 by matching protocol (3 methods run under one protocol)

| method | canonical (LAGOM) | InChIKey (strict) | no-stereo (GLORYx) | Tanimoto=1 (MetaTrans) | tautomer-InChIKey (ours) |
|---|---|---|---|---|---|
| SyGMa | 0.498 | 0.492 | 0.498 | 0.500 | 0.498 |
| BioTransformer | 0.373 | 0.346 | 0.373 | 0.373 | 0.373 |
| GRAIL | 0.237 | **0.116** | 0.243 | 0.237 | 0.243 |

recall@k (tautomer): SyGMa 0.347/0.461/0.483/0.498 · BioTransformer 0.175/0.297/0.336/0.373
· GRAIL 0.182/0.219/0.228/0.243 (@5/10/12/15).

**Match-sensitivity is driven by stereochemistry handling.** GRAIL strips stereo during
generation (`standardize_mol`, `isomericSmiles=False`), so under the only *stereo-aware*
protocol (full InChIKey) it drops to 0.116, but under every *stereo-blind* protocol
(canonical isomeric-free, no-stereo InChIKey, Tanimoto=1, tautomer) it scores ~0.24 — **a
2.1× swing from a single protocol choice.** BioTransformer also dips under strict InChIKey
(0.346 vs 0.373). SyGMa preserves stereo and is protocol-robust (~0.49–0.50).

**Two reorderings the protocol/metric reveal.** (i) Among these three (well-separated)
methods the *order* is stable across match protocols, but the *magnitudes* swing 1.5–3×, so
for closer methods the protocol would flip the leaderboard (the choice of stereo-awareness
alone moves GRAIL 2.1×). (ii) **k-sensitivity:** at recall@5 GRAIL ≈ BioTransformer (0.182 vs
0.175) but by @15 BioTransformer pulls ahead (0.373 vs 0.243) — it trades top-rank precision
for more candidates at higher k. *How you match and at which k both change who wins.*

## Cross-distribution and protocol-vs-published caveats (for the paper)

1. **GRAIL generalizes worse to GLORYx than to its own clean test** (0.24 vs ~0.38 at
   `prior_strength=8`): GLORYx is 37 out-of-distribution drugs and its references include many
   multi-generation / multi-step metabolites a single-step generator cannot reach.
2. **Standardized re-evaluation is stricter than published numbers.** Under our protocol
   SyGMa is 0.498@15; its published recall is 0.68 (uncapped, ~22 predictions/drug). The
   recurring published leaderboard (GLORYx 0.77, SyGMa 0.68, MetaPredictor 0.47, LAGOM 0.43,
   MetaTrans 0.35) is **not** one measurement — it mixes three incomparable axes and two of
   its numbers are mis-attributed. See the provenance table below.

## The published "leaderboard" is not one measurement (provenance)

The recurring leaderboard cited for this task — GLORYx 0.77, SyGMa 0.68, MetaPredictor 0.47,
LAGOM 0.43, MetaTrans 0.35 — comes from **LAGOM (Larsson et al. 2025) Table 2**. Tracing each
number to its source (`data/published_provenance.json`; agent-extracted, cross-checks
internally consistent, e.g. GLORYx 105/136 = 0.77 recall and 105/1724 = 0.061 precision) shows
it is **not a single measurement**: it conflates three incomparable axes — *matching
protocol*, *k / prediction budget*, and *test set* — and **two of its five numbers are
mis-attributed**.

| quoted | what the number actually is | k / budget | match | test set | source |
|---|---|---|---|---|---|
| **GLORYx 0.77** | uncapped recall, 105/136 TP from **1724** predictions (precision 0.061) | uncapped (~47/drug) | InChI-no-stereo | GLORYx-37 (**= our set**) | de Bruyn Kops 2021, Table 5 |
| **SyGMa 0.68** | uncapped recall, ~800 predictions (precision 0.12); GLORYx authors' re-eval | uncapped (~22/drug) | InChI-no-stereo | GLORYx-37 (**= our set**) | de Bruyn Kops 2021, Table 5 |
| **MetaPredictor 0.47** | ⚠ **not its number** — ≈ SyGMa's top-5 (47.4%) in MetaPredictor's own table / LAGOM re-run. Its *own* recall is **0.544@5 → 0.739@15** | top-5…15 | Tanimoto=1 | own 135-drug/283-met | Zhu 2024, BiB Table 1 |
| **LAGOM 0.43** | top-10 recall, ~328 predictions | top-10 | canonical SMILES | GLORYx-136 pairs | Larsson 2025, Table 2 |
| **MetaTrans 0.35** | ⚠ **not its number** — LAGOM's canonical-SMILES re-run. Its *own* recall is **0.576@10** | top-10 (re-run) | canonical SMILES | LAGOM's GLORYx-136 | Larsson 2025, Table 2 |

**LAGOM Table 2 is itself a mix:** GLORYx (0.77) and SyGMa (0.68) carry footnote *"a = values
obtained from de Bruyn Kops et al."* (quoted; uncapped recall over 1724 / 800 predictions),
while LAGOM/MetaTrans/MetaPredictor/Chemformer were re-run at top-10 over ~328 predictions
under canonical SMILES. **The 0.77-vs-0.43 spread is dominated by the uncapped-vs-top-10
prediction budget, not method quality** — comparing them as a leaderboard is a category error.

**Concrete demonstration (same method, same set, same matching — only the budget changes).**
On GLORYx-37, SyGMa's published 0.68 is *uncapped* (~22 predictions/drug). Under our protocol
*capped at top-15* the same tool scores **recall@15 = 0.498**, and SyGMa is protocol-robust
across all our match modes (~0.49–0.50, table above). So almost the entire **0.68 → 0.50** gap
is the prediction budget — not the matching, not the model. This is exactly the confound a
standardized protocol (fixed k, fixed match, fixed set) removes.

## Pending (raw predictions → full rank-flip table)

The full cross-method rank-flip needs each tool's raw predictions on this set. Next:
BioTransformer 3.0 (Java JAR, weights bundled) and MetaPredictor (weights in-repo) — both run
on CPU and drop into `run_match_sensitivity.py` as prediction files. GLORYx-run (needs FAME3
weights), MetaTrans (dependency rot), LAGOM (no checkpoint) are harder; cite their published
numbers as context where running is infeasible.
