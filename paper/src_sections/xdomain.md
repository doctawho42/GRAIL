## 12. Limitations

We state these plainly rather than let them surface in review. First, **no recall win**: GRAIL
does not win on recall anywhere in this paper. It loses to SyGMa by a certified, significant
margin (§9: paired Δ = **−0.242**, 95% CI [−0.271, −0.212]; McNemar p ≈ 1.7×10⁻⁴⁴) and to the
learned transformer baselines on the shared n=150 subset (MetaPredictor 0.585, MetaTrans 0.561,
both above GRAIL's 0.365 under the same tautomer protocol, §11). Nothing in this paper should be
read as a state-of-the-art claim; the contribution is the ceiling, the decomposition, and the
protocol, with GRAIL as one honestly diagnosed row. Second, **precision is a pessimistic lower
bound** throughout: because negatives are rule-applicable, non-annotated products rather than
confirmed non-metabolites (§3), an unannotated true metabolite is scored as a false positive, so
every precision number in this paper understates the pipeline's true precision by an unknown,
non-annotation-corrected amount — we lead with recall and `mean_output_size` for this reason and
do not report precision as a headline metric.

Third, the deployed headline (recall@15 = 0.261
micro / 0.330 macro, §8–§9) is a **single checkpoint**, not a seed-averaged estimate; the
honest-anchor certification (§9)
bounds *evaluation* variance — resampling and discordance over a fixed model — not *training*
variance across independently seeded runs, and a three-seed retraining confirms the headline is
seed-stable (macro 0.344 ± 0.010, micro 0.269 ± 0.006; §9). Fourth, the five-method,
five-protocol match-sensitivity comparison (§11, Table 3) mixes sample sizes: the three tier-2
comparators (BioTransformer, MetaPredictor, MetaTrans) are scored only on the **n=150** shared
subset for which frozen third-party predictions exist, while GRAIL and SyGMa's headline numbers
elsewhere in the paper are reported on the full **n≈1170** clean test split; within Table 3 itself
GRAIL and SyGMa are re-scored on the matching n=150 subset so the five columns are paired on
identical substrates. MetaPredictor has since been re-scored on the full test (recall@15 0.568,
≈ SyGMa, `results/metapredictor_1170.json`), confirming its n=150 value is representative and
leaving only BioTransformer and MetaTrans on n=150; a full single-n, five-method rerun remains
future work (BioTransformer's exact n=150 configuration is unscripted and MetaTrans's inference
pipeline is no longer reproducible in-tree).

Fifth, the external-validity ceiling (§7) is measured on only **37** GLORYx parent substrates, so
its 95% CI is wide by design ([0.531, 0.733]) and the composition-effect regression that partially
explains the internal–external gap is fit on the same small external population — we frame it as
suggestive, not as a transferable law. Sixth, Proposition 2's explanatory model rests on an
**unmeasured assumption**: the propensity model `e(r) ∝ π(r)` (labeling probability proportional
to the marginal rule-firing rate) is asserted, not estimated from data, so the identifiability
argument for why the learned selector underperforms a frequency prior is consistent with, but not
proof of, the stated mechanism (§10). Finally, Proposition 3's paradigm bound is **single-step
conditional**: depth-2 rule application recovers only +0.012 ceiling at 8.5× candidate cost (§10),
so multi-step and out-of-bank chemistry remain an open, unresolved coverage lever rather than a
closed question — the bound constrains the single-step paradigm, not the metabolite-prediction
problem in general.

We also built and evaluated a factorized-generator redesign of Stage-1 as a direct intervention
(§10); it does **not** beat the deployed pipeline as a generator (coverage-bound at recall@15
0.256 < deployed 0.365) — its only deployable value is a modest, paired-significant re-ranking
gain (+0.0165, 95% CI [0.006, 0.027], full clean test, n=1170) — and closing the larger gap to SyGMa
still requires a broader rule bank, out of scope here, consistent with the coverage-binding
finding.

Finally, a **scope-and-generality** limitation that doubles as this paper's clearest future
direction: our rank-flip result (§11) is established **within metabolite prediction only**. Whether
match-protocol choice reorders leaderboards *across* molecular-ML tasks — retrosynthesis, molecular
generation — is an open, testable hypothesis, and we name its falsifying run explicitly: re-score
≥2 comparable published models' *frozen* predictions in another domain under the same five match
conventions; if the method ordering does not move (interaction CI covering zero), the generality
claim fails. A clean instance requires ≥2 models of **comparable strength but differing
match-sensitivity** whose raw ranked predictions are public; we found they are not (only same-family
or strength-mismatched checkpoints are readily runnable), so the multi-model cross-domain rank-flip
is deferred rather than forced. A first single-model probe (ReactionT5v2 on the USPTO-50k test,
n=200; `results/xdomain_retro_protocol.json`) in fact suggests the effect is **domain-dependent, not
universal**: the *same* predictions move only ~**0.02** in top-1 accuracy across {canonical,
stereo-stripped, InChIKey, tautomer} matching (0.695–0.715) — roughly an order of magnitude below the
metabolism interaction (§11) — plausibly because retrosynthesis references are canonical-clean small
molecules, whereas the metabolism rule engine emits tautomer/stereo variants of its references. So
the honest cross-domain reading is **not** that protocol choice universally reorders leaderboards,
but that its magnitude is a function of how much a generator's outputs diverge tautomerically/
stereochemically from the references. Establishing whether protocol choice *reorders* (not merely
shifts) leaderboards in a domain with comparable tautomer ambiguity remains the named open test, and
needs only public multi-model predictions in a re-scorable form — which do not yet exist.

