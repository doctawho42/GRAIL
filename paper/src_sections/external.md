## 7. Results — External validity of the ceiling

The internal ceiling `coverage_bank = 0.735` (the **micro**, pooled ratio-of-sums estimate `Σhit /
Σtrue` — not a per-substrate mean; within-substrate pairs are dependent so a mean-of-ratios would
misrepresent the uncertainty) carries a cluster-bootstrap 95% CI of **[0.709, 0.762]** (resampling
substrates, 10,000 resamples). Recomputed apples-to-apples on an **external** set — the 37 GLORYx
parent substrates, one uncapped full-bank depth-1 `apply_rules` pass, no pool cap, the same
tautomer match — the external ceiling is **0.633** (95% CI **[0.531, 0.733]**, n=37, wide by
design at this sample size). This is far above the previously committed figure of **0.3715**,
which is **pool-capped** — a generator-budget artifact inherited from `gloryx_oracle.json` — and
must **never** be read as "the external ceiling."

Internal (0.735) and external (0.633) both sit in the **micro** frame. A separate question is
whether the internal-external gap is compositional: GLORYx parents tend to be larger,
more-conjugated drugs than the internal test set, and the ΔMW long tail noted in §4/§10 suggests
coverage should degrade with molecular complexity. One OLS regression of **per-substrate**
coverage on molecular descriptors (molecular weight, ring count, aromatic-atom count, heteroatom
count, a conjugation-site count, and `n_true`) fit across both populations predicts **macro**
(per-substrate-mean) coverages of **~0.79** internal and **~0.74** external, in-sample — the macro
frame, distinct from the micro ceilings quoted above. Note honestly that this ~0.74 in-sample
external prediction sits *above* the measured external macro coverage of 0.697, so the in-sample
fit alone overstates transferability. Holding the external points fully out of the fit
(fit-internal-only → predict-external) instead lands at **0.738 out-of-sample**, recovering
**~56%** of the internal→external macro-coverage gap
(`regression.predicted_external_mean_oos = 0.738`, `regression.gap_recovery_frac = 0.565`). This
out-of-sample number, not the in-sample one, is the transferable claim.

We frame this as a **suggestive partial composition effect at n=37** — consistent with, but not
proof of, molecular composition driving part of the internal-external coverage gap — and
explicitly *not* a fitted or transferable law, and *not* a defect in the rule bank or the matching
protocol.

> _[FIGURE: internal-vs-external coverage scatter — optional, post-draft]_

*Source: `results/ceiling_external_validity.json`.*

