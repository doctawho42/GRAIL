## 3. Methods — GRAIL architecture

GRAIL predicts xenobiotic metabolite structures with a three-stage, rule-based-plus-learned
pipeline. The first stage, the **generator**, is a learned multi-label rule selector that
approximates `P(r|s)` — the probability that rule `r` applies to substrate `s` — over a curated
bank of **7,581 SMIRKS** rules; the default scorer is retrieval-based, combining cross-attention
between substrate and rule graph embeddings, an embedding-similarity term, and an MLP head. The
second stage, **RDKit rule application**, mechanically applies every rule the generator selects
and enumerates the resulting candidate products, retaining provenance of which rule produced which
candidate. The third stage is a **PU-trained structural filter**: a binary classifier scoring
each (substrate, product) pair (neural architecture detailed below). Its training data are positive-unlabeled — annotated true
metabolites are the only positives, while rule-applicable products lacking a positive annotation
are treated as *unlabeled*, not as confirmed negatives, since absence of an annotation does not
certify a transformation does not occur. The filter is accordingly trained in the logit domain
(`return_logits=True`) so a PULoss/nnPU surrogate operates on raw classifier outputs rather than
post-sigmoid probabilities, and the generator itself down-weights unobserved-but-applicable rules
rather than penalizing them as hard negatives. Featurization uses fixed-width graphs — 16-dim single-molecule nodes, 18-dim edges — and a
1024-bit Morgan fingerprint per molecule; the two learned stages, whose architectures are detailed
in the two subsections below, build on this shared featurization. At deployment,
`ModelWrapper.generate` runs all three stages and ranks the candidate set by
`filter_score × generator_score`, combining the filter's pair-plausibility judgment with the
generator's rule-selection confidence into a single ranking signal. Throughout this paper, unless
stated otherwise, structure matching between predicted and reference metabolites uses
`inchikey_tautomer` as the default match mode. The three stages together form an interpretable
instrument — the selected rule, the enumerated product, and the filter's pair judgment are each
inspectable — and the contribution we claim is interpretable learned rule selection paired with a
PU-aware structural filter, not recall supremacy over other metabolite predictors.

**Generator architecture (learned rule selector).** The generator scores the entire rule bank
against a substrate in one forward pass. A shared GATv2 graph encoder — three message-passing
layers with node dimensions 16→192→256→128, each layer BatchNorm→ReLU→dropout(0.1) and a
mean-pooled readout — embeds the substrate's single-molecule graph into a 128-dim vector; a
parallel MLP embeds its 1024-bit Morgan fingerprint and a gated fusion (GELU MLPs, LayerNorm)
combines the two into the substrate representation. Each SMIRKS rule is embedded independently by a
rule-graph encoder of the same GATv2 family (hidden dimensions 128→192→128) together with a
six-dimensional global rule-feature branch, so the full 7,581-rule bank is re-encoded on every
forward pass — the dominant compute cost, and the reason inference scales with bank size. The
per-rule logit is a sum of four learned terms: (i) a *cross-attention interaction* in which the rule
embedding queries the substrate's atom states (single-head attention over node keys/values) to form
a rule-specific local context, which is concatenated with the pooled embeddings and passed through a
three-layer MLP; (ii) a *global* cosine similarity between the substrate and rule embeddings; (iii) a
*local* cosine similarity between the attention context and the rule; and (iv) an SMARTS-*match*
term — each scaled by a learned scalar. To this logit the model adds a per-rule bias and a
frequency-prior term (weight `prior_strength = 0.4`) — the log-odds that a rule yields a true
metabolite, estimated from its TRAIN statistics — then
subtracts a large penalty (7.5) for rules whose SMARTS does not match the substrate — an
applicability mask that zeroes out inapplicable rules — before a sigmoid yields `P(r|s)`. When a rule
fires at several sites, its per-firing probabilities are combined by noisy-OR. Training is
multi-label rule classification with a margin-ranking auxiliary (rank weight 0.25, margin 0.45);
critically, rules that are SMARTS-applicable but not annotated as productive are *down-weighted*
(weight 0.5), not treated as hard negatives, reflecting the positive-unlabeled nature of the labels.
Optimization is Adam (lr 1e-4, ≤20 epochs, early stopping).

**Filter architecture (structural plausibility).** The deployed filter — the checkpoint behind every
GRAIL number in this paper — is the *single-encoder* variant: it embeds the substrate and the
candidate product **independently**, each with its own three-layer GATv2 encoder over the molecule's
16-dim single-graph (16→192→256→128, mean-pooled), then concatenates the two 128-dim graph
embeddings with both molecules' 1024-bit Morgan fingerprints and maps the resulting 2304-dim vector
through a three-layer MLP (→128→64→1, ReLU, dropout 0.1) to a single logit. The codebase also
implements a *pair* variant — the configuration default — that instead encodes one **merged**
substrate–product graph whose 18-dim nodes are joined by cross-edges placed from an element-aware
maximum-common-substructure (MCS) atom correspondence, rather than sorted or arbitrary indices,
preserving chemically meaningful atom mappings across the reaction; we deploy the single-encoder
filter because — as we show below — it is both more accurate and far cheaper than the MCS pair
variant, which is implemented but dominated. Both variants train on **positive-unlabeled** data — annotated
metabolites are the only positives, while rule-applicable-but-unannotated products are treated as
*unlabeled*, not confirmed negatives, since a missing annotation does not certify a transformation
does not occur — using a non-negative PU (nnPU) risk estimator with class-prior 0.75, evaluated in
the **logit domain** (`return_logits=True`) so the surrogate operates on raw classifier outputs
rather than a doubly-applied sigmoid. Optimization is Adam (lr 1e-4, ≤20 epochs, early stopping,
patience 7).

**Single vs. pair filter — the deployment choice, measured.** Rather than assume the single-encoder
filter is the right default, we trained both variants on an identical 800-substrate subset (same
negatives, seed, and optimizer; only the filter `mode` differs) and evaluated them on the full clean
test split with the same generator and candidate pool. The single-encoder filter is the better
model: it beats the MCS pair filter on paired recall@15 (single 0.366 vs pair 0.357; pair−single
**−0.0090, 95% CI [−0.017, −0.001]**, paired bootstrap, n=1170) and by a wider margin at recall@5
(0.282 vs 0.259). It is also roughly 20× cheaper: the pair filter's per-pair MCS featurization makes
full-data training ≈14 h (against ~40 min for single) and imposes an MCS at every inference
candidate. Both accuracy and cost therefore favour independent encoders — the MCS pair filter is a
*dominated*, not merely unused, alternative (`results/filter_compare_matched_sub800.json`; this is a
matched-subset comparison, so both absolute recalls sit below the full-data deployed filter and only
the paired delta is the claim).

**Rule-bank construction.** Unlike the hand-curated expert systems GRAIL is benchmarked against
(SyGMa, Meteor), its SMIRKS rule bank is built by an automated, self-validated mining procedure
over annotated substrate→metabolite pairs rather than by manual reaction-rule authoring. The
bank-construction pipeline (`scripts/mine_rules.py`, `process_pair`) reads only the clean,
molecule-disjoint **TRAIN** split — never validation or test — and, per annotated pair: (i) locates
a ring-aware maximum-common-substructure anchor (RDKit FMCS with `ringMatchesRingOnly`,
`completeRingsOnly`, `bondCompare=CompareAny`, `atomCompare=CompareElements`), requiring the MCS to
cover at least 40% of the smaller molecule's atoms; (ii) derives a reaction center from that anchor
(`find_reaction_center`) by flagging matched atoms whose element, formal charge, H-count,
aromaticity, or degree changed, atom pairs whose connecting bond order changed, and any
leaving/entering atoms together with their matched neighbours — trying multiple substrate/product
match combinations and keeping the one that minimises the resulting center size; (iii) expands the
center by a 1-bond radius (`expand_center`) to capture the immediate reactive environment; (iv)
atom-maps the expanded substrate/product fragments by their MCS positional correspondence into a
candidate SMIRKS template (`build_smirks`); (v) **self-tests** the template — a rule survives only
if applying it to its own source substrate regenerates, after canonicalization, the annotated
source product (`self_test`), otherwise it is discarded; and (vi) passes the surviving, deduplicated
templates through a **selectivity filter** (`filter_rule_candidates`) that rejects templates that
are unselective (mean more than 200 distinct products across a 50-substrate sample) or
simultaneously too general and too prolific (applicable to more than 90% of sampled substrates
while still producing more than 50 products on average). The codebase implements a second
atom-mapping backend alongside the MCS-positional correspondence above: `combine_reaction` in
`grail_metabolism/utils/reaction_mapper.py` detects a reaction center via RASCAL MCES
(`rdRascalMCES.FindMCES`, falling back to RDKit FMCS) plus a 1-bond environment, then emits a mapped
SMIRKS using **RXNMapper**, a neural attention-based atom-mapper
(`get_attention_guided_atom_maps`) — so both a neural and a rule-based atom-mapping route to SMIRKS
construction are available in the codebase. Because mining reads exclusively from the clean TRAIN
split, no test-split metabolite can leak into a mined rule, so the rule-bank coverage ceiling
reported as a held-out quantity (§6, 0.735) stays honest. The deployed bank,
`grail_metabolism/resources/extended_smirks.txt` (**7,581** SMIRKS), is the deduplicated union of
four prior curated banks — `smirks.txt` (473), `merged_smirks.txt` (656), `compressed_rules.smarts`
(500), and `notebooks_rules.txt` (1,051) — with the newly mined, self-tested, selectivity-filtered
templates. Mining scale and support are heavy-tailed: the MCS-positional miner alone extracts
**5,856** unique, self-tested SMIRKS templates from the clean TRAIN positives
(`results/mined_rule_catalog_v2.json`), of which **4,274 of 5,856 (73%)** are supported by a single
training pair and only **457** by five or more (the single most-supported template covers **1,531**
pairs) — deduplicated together with the four prior curated banks above, this reaches the deployed
**7,581**-rule bank, so the bank is dominated by highly specific, near-single-example templates
rather than broad general reactions. Re-running the miner independently over the same clean TRAIN
split re-derives templates that are already all present in the deployed bank (**5,856/5,856; 0
new**; `grail_metabolism/resources/mined_only_v2.txt` against `extended_smirks.txt`) — the mining
procedure is deterministic and, on this TRAIN corpus, source-saturated: it surfaces no additional
coverage beyond what the bank already contains, so the rule-bank coverage ceiling (**0.735**, §6)
is bounded by the diversity of the training-reaction *source*, not by mining incompleteness.
Expanding coverage would therefore require new reaction corpora or more general templates — which
the selectivity filter above deliberately resists — rather than further re-mining of the same data,
consistent with §10's finding that GRAIL's residual gap to SyGMa is a compound of over-aggressive
rule-selection breadth and a residual rule-set coverage limit — not a ranking deficiency. The selectivity filter is a design-time counterpart to the
selection↔precision trade-off
quantified empirically in §10's selection-breadth ablation: rejecting over-general templates at
construction time is the same lever as narrowing `top_k` at inference time, applied before
deployment rather than at the deployed operating point.

