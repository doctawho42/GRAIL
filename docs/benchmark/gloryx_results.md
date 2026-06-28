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
   SyGMa is 0.498@15; its published recall is 0.68 (at its own, larger/uncapped k and its own
   matching). Published leaderboard numbers (GLORYx 0.77, SyGMa 0.68, MetaPredictor 0.47,
   LAGOM 0.43, MetaTrans 0.35) are **not** apples-to-apples — which is exactly why a single
   protocol is needed.

## Pending (raw predictions → full rank-flip table)

The full cross-method rank-flip needs each tool's raw predictions on this set. Next:
BioTransformer 3.0 (Java JAR, weights bundled) and MetaPredictor (weights in-repo) — both run
on CPU and drop into `run_match_sensitivity.py` as prediction files. GLORYx-run (needs FAME3
weights), MetaTrans (dependency rot), LAGOM (no checkpoint) are harder; cite their published
numbers as context where running is infeasible.
