# Related work & novelty positioning (TAME / GRAIL-primary paper)

> Literature scan 2026-07-13 (Consensus + PubMed) to (a) verify the novelty claims, (b) fix the
> comparator citations, (c) pre-empt the referee's "this has been done" attacks. Sources cited with
> DOIs; the prior multi-tool comparisons and the tautomer/leakage prior art are corroborating, not
> threatening — but the novelty must be **scoped precisely**.

## 1. The comparator methods (verified citations)

| Method | Type | Citation | Key numbers to cite |
|---|---|---|---|
| **SyGMa** | rule-based (phase-1+2, empirical prob. score) | Ridder & Wagener 2008, *ChemMedChem* 3(5):821–32, doi:10.1002/cmdc.200700312 | covers ~70% of human biotransformations; **68% of test metabolites reproduced, 30% in top-3** (the source of our "SyGMa 0.68 on its own set") |
| **GLORYx** | SoM (FAME-3 ML) + reaction rules | de Bruyn Kops et al. 2020, *Chem Res Toxicol* 34(2):286–299, doi:10.1021/acs.chemrestox.0c00224 | **recall 77%, AUC 0.79**; explicitly finds **phase-2 ranking harder than phase-1** |
| **BioTransformer (3.0)** | knowledge + ML | Djoumbou-Feunang et al. 2019, *J Cheminform* (≈417 cites) | broad-scope in-silico metabolism + identification |
| **MetaTrans** | end-to-end NMT (SMILES→SMILES) | Litsa, Das, Kavraki 2020, *Chem Sci* (KavrakiLab) — **verify exact vol/DOI** (not PubMed-indexed) | 6-model transformer ensemble; emits an unranked set |
| **MetaPredictor** | transformer ensemble | **exact citation UNVERIFIED — confirm before submission** | recall@15 ≈ 0.585 (our re-run) |

*(Note: PubMed PMID 32082146 "Deep Learning Based Drug Metabolites Prediction" (Wang et al. 2020,
Front Pharmacol, doi:10.3389/fphar.2019.01586) is a DIFFERENT SMARTS+DL method, NOT MetaTrans — do
not conflate.)* According to PubMed.

## 2. Prior multi-method comparisons EXIST — so we are NOT "first comparison"

Two prior studies already benchmarked overlapping tool sets. We must cite them and **differentiate on
the protocol, not the comparison**:

- **Scholz et al. 2023**, *Sci Total Environ* — benchmarked **SyGMa, GLORY, GLORYx, BioTransformer 3.0,
  MetaTrans** on 85 agrochemical parents (rat metabolism). Findings that **corroborate ours**: recovered
  ~⅓–⅔ of metabolites; **precision low (~18% first-gen, ~2% over three generations)**; **predictions
  differ strongly between rule-based vs ML tools → ensembles promising**; the bottleneck is data
  quantity/quality. (Consensus [1] below.)
- **Boyce et al. 2022**, *Comput Toxicol* 21:1–15, doi:10.1016/j.comtox.2021.100208 — compared **SyGMa,
  Meteor Nexus, BioTransformer, TIMES, OECD Toolbox, CTS** on 37 chemicals. **SyGMa had the highest
  coverage but was "prone to significant overprediction" (5,125 metabolites = 54.7% of all
  predictions); precision 1.1–29%, sensitivity 14.7–28.3%.** According to PubMed. — This is **direct
  external support for our output-budget confound**: SyGMa's high recall is bought with a huge
  prediction volume, exactly what TAME's budget-matched view controls.

**Takeaway:** prior comparisons each used their **own ad-hoc matching + own test set** and reported
raw recall/precision. None (i) standardized the structure-match definition, (ii) audited leakage, or
(iii) asked whether the ranking is *stable* under the match choice. That triad is our contribution.

## 3. Tautomer-aware matching — prior art JUSTIFIES our protocol (cite, don't fear)

- **Dhaked et al. 2019**, *JCIM*, doi (verify) — 86 tautomer transforms; **standard InChI normalizes
  only a SUBSET of tautomers**; full handling would ~triple affected compounds. This is **the citation**
  that motivates tautomer-aware matching: plain InChIKey provably misses keto/enol + carbon-shift
  tautomers (our empirical 8/8 merge check + the 0.718→0.735 ceiling gap). (Consensus [1], §"tautomer".)
- **Hähnke et al. 2018** (*J Cheminform*, PubChem standardization): **60% of PubChem structures differ
  from the InChI form, primarily due to a different tautomer** — structure identity is materially
  tautomer-choice-dependent.
- **Mansouri et al. 2024** (*J Cheminform*, QSAR-ready workflow): standardization pipeline (desalt, strip
  stereo, standardize tautomers/nitro) — precedent for our `standardize_mol` path.

**Positioning:** tautomer standardization is a known *preprocessing* step; **no prior metabolite-
prediction evaluation adopts it as the MATCHING protocol and quantifies how much the leaderboard moves
because of it.** That quantification (differential match-sensitivity, interaction CI [+0.073,+0.171])
is new.

## 4. Leakage-audited splits — general tools exist; ours is task-specific

- **DataSAIL 2023** (Joeres et al., bioRxiv) — splits datasets to minimize cross-split similarity (BLP);
  benchmarks vs DeepChem/LoHi/GraphPart on MoleculeNet. General leakage-aware splitting.
- **Hermann et al. 2024** (bioRxiv) — pretraining-aware vs naive splits distort protein thermostability.
- **Strobel et al. 2025** (*BMC Bioinformatics*) — MS/MS-similarity eval methodology with
  similarity-graded train/test splits + released pipelines (analogous *shape* to ours, different task).

**Positioning:** we do **not** claim to invent leakage-aware splitting; we contribute the
**metabolite-prediction-specific molecule-disjoint audit** (substrate↔metabolite overlap is the leak
here) with a machine-checkable `leakage_fix_report.json`, and we show val≈test (0.327 vs 0.330) as
corroboration.

## 5. "Rankings flip with the evaluation choice" — precedent in ML/NLP, new in this domain

- **Mishra et al. 2021** ("How Robust are Model Rankings") — difficulty-weighting reorders leaderboards;
  "top models may not be best." (Consensus [1], §"leaderboard".)
- **Rodriguez et al. 2021** ("Evaluation Examples are not Equally Informative") — item-response leaderboard;
  ranking reliability.
- **Sheikhi et al. 2026** (*IEEE Access*) — survey: data contamination, "leaderboard obsession", benchmark
  lifecycle.

**Positioning:** the *general* insight "evaluation choices change rankings" is established for ML/NLP
leaderboards. Our contribution is the **domain-specific axis**: in metabolite **structure** prediction
the load-bearing, previously-unexamined choice is **how a predicted structure is matched to the
reference** (canonical / InChIKey / no-stereo / Tanimoto=1 / tautomer), and we show it reorders the
leaderboard in two independent method pairs with a non-monotone (MetaTrans) response.

## 6. Scoped novelty statement (defensible)

> We are **not** the first to compare metabolite-structure predictors (Scholz 2023; Boyce 2022) nor to
> standardize chemical structures (PubChem; QSAR-ready) nor to build leakage-aware splits (DataSAIL).
> TAME's contribution is their **first joint instantiation for metabolite structure prediction**: a
> **standardized, tautomer-aware structure-matching protocol**, a **leakage-audited molecule-disjoint
> split**, a **match-sensitivity ("rank-flip") analysis** showing the leaderboard is not match-invariant,
> and — via GRAIL as one honest, interpretable row — a **coverage × selection × ranking decomposition**
> of *where* the rule-based paradigm's headroom is lost. GRAIL is a diagnosed instrument, never a SOTA
> claim.

## 7. Referee-risk register (novelty attacks → rebuttals)

| Attack | Rebuttal |
|---|---|
| "Scholz 2023 / Boyce 2022 already benchmarked these tools." | Correct — cited. They used ad-hoc per-tool matching on their own sets and reported raw recall/precision. We standardize the *match protocol*, audit *leakage*, and quantify *match-sensitivity* — and our findings (low precision, SyGMa overproduction, rule-vs-ML divergence) **replicate theirs**, strengthening both. |
| "Eval-choice-changes-ranking is known (Mishra/Rodriguez)." | Yes, for ML/NLP leaderboards. We give the concrete **structure-match** axis in a chemistry task, with CIs, not a generic claim. |
| "Tautomer standardization is standard preprocessing." | As preprocessing, yes (PubChem, QSAR-ready). No metabolite-prediction eval adopts it as the *matching* protocol or measures the ranking impact (interaction CI [+0.073,+0.171]). |
| "Leakage-aware splitting exists (DataSAIL)." | General tools, yes. We contribute the metabolite-specific molecule-disjoint audit + a machine-checkable report; not a new splitting algorithm. |
| "GRAIL loses on recall — why include it?" | Not a SOTA claim. GRAIL is the interpretable instrument that makes the decomposition possible (rule-grounded, exposes coverage vs selection vs ranking). One honest row. |

## 8. Corroboration we should actively cite (turns 'prior work' into support)
- Scholz 2023 low precision + rule-vs-ML divergence → our precision-lower-bound framing + rank-flip.
- Boyce 2022 SyGMa overproduction (54.7% of all predictions) → our output-budget confound + budget-matched view.
- GLORYx 2020 phase-2 harder than phase-1 → our ΔMW conjugate long-tail (glucuronide/sulfate/glutathione).
- SyGMa 2008 "70% of biotransformations covered" → contextualizes our 0.735 rule-bank ceiling.

## Open items before submission
- Verify exact **MetaTrans** (Litsa 2020, Chem Sci) and **MetaPredictor** citations + DOIs.
- Pull the **Gao 2026** metabolite benchmark the earlier plan referenced (not surfaced this scan — the
  surfaced 2026 item, Giné et al. JASMS, is MS/MS *annotation*, a different task; confirm which "Gao 2026" was meant).
- Confirm Dhaked 2019 + DataSAIL DOIs for the reference list.
