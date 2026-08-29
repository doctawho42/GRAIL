

====================================================================================================
AGENT 1
====================================================================================================

All facts verified against source, `si.aux`/`grail_jcim.aux`, and both compiled PDFs (`pdftotext`). Paths below are relative to `/Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746/`.

# 1. True SI numbering

`paper2/si.tex` contains **no `\setcounter`, no `\begin{table}`/`\begin{figure}` of its own, and no nested table environments**. All nine tables arrive via `\input` of standalone files, each holding exactly one `table` environment, and they number in `\input` order. `si.tex:14-15` renumber the counters as `S\arabic{table}` / `S\arabic{figure}`.

| True № | `\input` at | Table env | `\label` | Caption (opening) |
|---|---|---|---|---|
| **S1** | `paper2/si.tex:29` | `paper2/si_table_splits.tex:1` | `tab:si-splits` | "The three substrate-disjoint splits. Triples count every (substrate, product, label) row…" |
| **S2** | `paper2/si.tex:90` | `paper2/si_table_criteria.tex:1` | `tab:si-criteria` | "The five declared matching criteria. The references carry no stereochemistry…" |
| **S3** | `paper2/si.tex:98` | `paper2/si_table_parentdrop.tex:1` | `tab:si-parentdrop` | "What the parent-drop convention gives each arm… No cell separates from zero." |
| **S4** | `paper2/si.tex:115` | `paper2/si_table_precision.tex:1` | `tab:si-precision` | "Micro precision at each budget on the comparison set, with the parent-drop convention of Table 2…" |
| **S5** | `paper2/si.tex:119` | `paper2/si_table_intervals.tex:1` | `tab:si-intervals` | "Every difference between a GRAIL arm and a comparator on the comparison set, in micro recall, with its paired bootstrap 95% interval…" |
| **S6** | `paper2/si.tex:130` | `paper2/si_table_criterion.tex:1` | `tab:si-criterion` | "The verdict of the comparison under each declared matching criterion…" |
| **S7** | `paper2/si.tex:284` | `paper2/si_table_oracle.tex:1` | `tab:si-oracle` | "An oracle that orders candidate groups by whether they contain a reference, under four partitions…" |
| **S8** | `paper2/si.tex:300` | `paper2/si_table_ranking.tex:1` | `tab:si-ranking` | "Micro recall when the same candidate pool is ordered five ways…" |
| **S9** | `paper2/si.tex:360` | `paper2/si_table_case.tex:1` | `tab:si-case` | "The first twenty of the 100 candidates the exhaustive mode returns for the worked example…" |

**SI figures: there are none.** Zero `\begin{figure}` and zero `\includegraphics` in `si.tex` or any `si_table_*.tex`; `pdftotext si.pdf` contains the word "Figure" exactly once, and that is a pointer to the *main text's* Figure 1 inside the S5 caption. So any `Figure S<n>` pointer would be dangling — none exists.

Cross-checks, both independent of my reading of the source:
- `paper2/si.aux:18,21,24,27,30,34,46,49,55` — `\newlabel{tab:si-splits}{{S1}…table.1}`, `{tab:si-criteria}{{S2}…table.2}`, `{tab:si-parentdrop}{{S3}…table.3}`, `{tab:si-precision}{{S4}…table.4}`, `{tab:si-intervals}{{S5}…table.5}`, `{tab:si-criterion}{{S6}…table.6}`, `{tab:si-oracle}{{S7}…table.7}`, `{tab:si-ranking}{{S8}…table.8}`, `{tab:si-case}{{S9}…table.9}`.
- `pdftotext -layout si.pdf` prints the captions as `Table S1:` … `Table S9:` in exactly this order. `si.pdf` (21:45) is newer than `si.tex` (21:43), so it is current.

SI **sections** (no pointer currently targets one by number, but for completeness, from `si.aux:15-66`): S1 corpus/splits, S2 matching criteria, S3 parent-drop, S4 Precision, S5 differences and intervals, S6 sensitivity to criterion, S7 rule bank, S8 architectures (S8.1 Generator, S8.2 Filter, S8.3 PU objective), S9 what a single step is, S10 registered predictions, S11 group ranking and oracle, S12 stage ablations (S12.1 ties), S13 when a ranking comparison can measure anything, S14 choice of evaluation population, S15 worked example, S16 runtime, S17 comparators, S18 provenance.

# 2. Every textual pointer, and what actually carries the content

Exhaustive sweep of `\bS[0-9]+` over every `.tex` compiled into either document (`grail_jcim.tex`, `body.tex`, `si.tex`, `numbers.tex`, `markers.tex`, `table_{modes,sweep,case,hypotheses,grain}.tex`, all nine `si_table_*.tex`) yields **exactly 9 hits**, all of the form `Table~S<n>`, all hardcoded. No `Figure S<n>`, no `Section S<n>` anywhere. `pdftotext grail_jcim.pdf` confirms 8 rendered `Table S…` strings and zero `Figure S`/`Section S`. The editor's count is right, and the two in "captions of Tables 2 and 4" live in the included files `table_sweep.tex` (= main Table 2) and `table_hypotheses.tex` (= main Table 4), confirmed by `grail_jcim.aux:73` `\newlabel{tab:sweep}{{2}…}` and `grail_jcim.aux:86` `\newlabel{tab:hyp}{{4}…}`.

**All nine are wrong.** One by one:

**(1) `paper2/body.tex:142`** — "On the pool the deployed system actually ranks the same contrast, same population and same budget, is $+\numHSevenDeployed$, 95\% CI $[+\numHSevenDeployedLo, +\numHSevenDeployedHi]$ **(Table~S6)**."
Wants: the P1 contrast (RRF fusion vs. their product) recomputed on the deployed, capped pool. Carried by **S8** (`si_table_ranking`). `numbers.tex:378` `\numHSevenDeployed = 0.0504`; `si_table_ranking.tex:17,20` give validation draw *n*=293, *k*=15: fusion 0.4748 − their product 0.4244 = **0.0504** exactly. S6 is the criterion-sensitivity grid, which contains no such contrast.

**(2) `paper2/body.tex:221`** — "Precision is reported in **Table~S2** but is not used to order systems…"
Wants: the precision table. That is **S4** (`si_table_precision`, caption "Micro precision at each budget on the comparison set"). S2 is the list of five matching criteria.

**(3) `paper2/body.tex:230`** — "the largest effect on recall anywhere in the grid is `\numPdropMaxEffect{}` and no cell separates from zero **(Table~S4)**."
Wants: the parent-drop sweep. That is **S3** (`si_table_parentdrop`), whose caption ends "No cell separates from zero" and whose largest cell is `+.0030` = `numbers.tex:428` `\numPdropMaxEffect = 0.003`. S4 is precision.

**(4) `paper2/body.tex:233`** — "The sweep is `\numCritBudgets{}` budgets wide and **Table~S4** is `\numCritCriteria{}` criteria deep…"
Wants: the criterion grid. That is **S6** (`si_table_criterion`): 5 criterion rows × 9 budget columns, matching `\numCritCriteria = 5` (`numbers.tex:40`) and `\numCritBudgets = 9` (`numbers.tex:39`). S4 (precision) is 9 budgets × 5 *methods*, not criteria.

**(5) `paper2/body.tex:262`** — "Every difference behind that division, with its paired interval, is **Table~S1**."
Wants: the per-budget differences with paired intervals. That is **S5** (`si_table_intervals`, caption "Every difference between a GRAIL arm and a comparator … with its paired bootstrap 95% interval"). S1 is the split-size table.

**(6) `paper2/body.tex:366`** — "…moves the verdict at up to `\numCritMovedMax{}` of `\numCritBudgets{}` budgets **(Table~S3)**."
Wants: the criterion grid. **S6**. Its caption states "the verdict moves at 6 of 9 budgets under `canonical`", and `numbers.tex:44` `\numCritMovedMax = 6`. S3 is parent-drop.

**(7) `paper2/table_sweep.tex:21`** (caption of **main Table 2**) — "…the verdicts are read from the paired intervals in **Table~S3** rather than from these levels."
Wants: the intervals table. **S5** (`si_table_intervals`: "$^{*}$ marks an interval excluding zero, which is the condition under which the paper claims a lead or a trail"). S3 is parent-drop.

**(8) `paper2/table_hypotheses.tex:18`** (caption of **main Table 4**) — "…the same contrast for P1 on the deployed pool is in **Table~S6**."
Same target as (1): **S8** (`si_table_ranking`).

**(9) `paper2/si.tex:57-58`** (internal) — "…so a third party can reproduce the tautomer column of **Table~S4** and not the rest of it."
Wants: the per-criterion results grid, of which only the tautomer criterion is reproducible from released InChIKeys. That is **S6** (`si_table_criterion`), the only SI object holding results under all five criteria. S4 is precision (single criterion, no criterion axis).

# 3. Correction map

| # | File | Line | Current | Correct | Evidence for the correct target |
|---|---|---|---|---|---|
| 1 | `paper2/body.tex` | 142 | `Table~S6` | `Table~S8` | `si_table_ranking.tex:17,20` validation draw *k*=15: 0.4748 − 0.4244 = 0.0504 = `\numHSevenDeployed` (`numbers.tex:378`) |
| 2 | `paper2/body.tex` | 221 | `Table~S2` | `Table~S4` | `si_table_precision.tex:18` is the only precision table; `si.aux:27` → S4 |
| 3 | `paper2/body.tex` | 230 | `Table~S4` | `Table~S3` | `si_table_parentdrop.tex:16` "No cell separates from zero"; max cell `+.0030` = `\numPdropMaxEffect` (`numbers.tex:428`); `si.aux:24` → S3 |
| 4 | `paper2/body.tex` | 233 | `Table~S4` | `Table~S6` | `si_table_criterion.tex:9-13` = 5 criterion rows; `\numCritCriteria=5`, `\numCritBudgets=9`; `si.aux:34` → S6 |
| 5 | `paper2/body.tex` | 262 | `Table~S1` | `Table~S5` | `si_table_intervals.tex:30` caption is verbatim "Every difference between a GRAIL arm and a comparator…"; `si.aux:30` → S5 |
| 6 | `paper2/body.tex` | 366 | `Table~S3` | `Table~S6` | `si_table_criterion.tex:16` "the verdict moves at 6 of 9 budgets"; `\numCritMovedMax=6` |
| 7 | `paper2/table_sweep.tex` | 21 (caption, main Table 2) | `Table~S3` | `Table~S5` | `si_table_intervals.tex:30` "$^{*}$ marks an interval excluding zero, which is the condition under which the paper claims a lead or a trail" |
| 8 | `paper2/table_hypotheses.tex` | 18 (caption, main Table 4) | `Table~S6` | `Table~S8` | same as row 1 |
| 9 | `paper2/si.tex` | 58 | `Table~S4` | `Table~S6` (→ `Table~\ref{tab:si-criterion}`) | only SI table with a criterion axis; `si.tex:56-57` "That key supports the default matching criterion and not the other four" |

**Why the drift happened** (reconstructed from `git log -- paper2/si.tex`, and it corroborates every target above). The SI table order changed five times and the pointers were bumped by hand, always one insertion behind:

- `ea240c6`: S1 splits, S2 criteria, S3 criterion, S4 oracle, **S5 ranking**, S6 case
- `3e9861a`: intervals inserted at 3 → S1 splits, S2 criteria, S3 intervals, S4 criterion, S5 oracle, **S6 ranking**, S7 case
- `9d5202f`: precision inserted at 3 → …, S3 precision, S4 intervals, S5 criterion, S6 oracle, **S7 ranking**, S8 case
- `6dd64aa` (HEAD): parentdrop inserted at 3 → the current S1–S9 above, **S8 ranking**

Pointer (1) was written at `3e9861a` as `Table~S5`, which was `ranking`'s number *one commit earlier*; it was bumped once to `S6` at `9d5202f` and never again — hence off by two now. Pointer (6) was written at `f339217` as `Table~S3` when `criterion` genuinely was S3 (`git show f339217:paper2/si.tex`), and was never bumped through three insertions — hence off by three. Pointers (2), (4), (5), (7) were each written using the pre-insertion numbering of the very commit that added a table. Pointer (3) is the newest (`6dd64aa`, the commit that added the parent-drop table itself) and points at the table it displaced. This is a mechanical-renumbering defect throughout, not a content mix-up — which is why every intended target is recoverable and unambiguous.

# 4. Labels exist — convert all nine to `\ref`

**Yes.** Every SI table carries a `\label`, all inside the `table` environment immediately after `\caption`, so `\ref` resolves to the table number in all nine cases:

```
tab:si-splits      si_table_splits.tex:13       → S1
tab:si-criteria    si_table_criteria.tex:15     → S2
tab:si-parentdrop  si_table_parentdrop.tex:17   → S3
tab:si-precision   si_table_precision.tex:19    → S4
tab:si-intervals   si_table_intervals.tex:31    → S5
tab:si-criterion   si_table_criterion.tex:17    → S6
tab:si-oracle      si_table_oracle.tex:15       → S7
tab:si-ranking     si_table_ranking.tex:25      → S8
tab:si-case        si_table_case.tex:30         → S9
```

Plus one section label: `sec:si-sensitivity` at `si.tex:128` → S6 (`si.aux:32`). It is already used correctly at `si.tex:94` via `Section~\ref{sec:si-sensitivity}`, and `si.tex:92`, `102`, `121`, `133`, `150`, `287`, `295`, `302` already use `\ref` for SI tables. **Pointer (9) is the only hardcoded `S<n>` left inside `si.tex`** and can be fixed today with `Table~\ref{tab:si-criterion}`.

For the eight in the manuscript, a caveat I can state as fact rather than guess: `body.tex` and `si.tex` compile as **two independent documents** (`grail_jcim.tex:50` `\input{body}`; `si.tex` is its own `\documentclass`), and neither loads `xr`/`xr-hyper` (verified: no `\usepackage{xr}` in either preamble). So `\ref{tab:si-ranking}` in `body.tex` will render `??` as things stand. Converting the manuscript's eight requires adding to `grail_jcim.tex`'s preamble

```latex
\usepackage{xr}
\externaldocument[SI-]{si}
```

and then writing `Table~\ref{SI-tab:si-ranking}`, with `si.aux` present at the time `grail_jcim.tex` is compiled. Absent that, the eight must stay hardcoded and be corrected to the literals in the map above.

# 5. Adjacent findings (not numbering; report, don't silently fix)

- **`si.tex:57` says "column" but the criteria are rows.** In `si_table_criterion.tex` the five criteria are the *rows* (lines 9-13) and the budgets are the columns. So even after renumbering to S6, "the tautomer column" should read "the tautomer row". The same slip appears at **`si.tex:150`**, "The reference column of Table~\ref{tab:si-criterion}" — that `\ref` resolves correctly to S6, but S6 has no column named "reference"; the intended object is the `tautomer (default)` row (line 13), the reference criterion.
- **The four pointers running the other way — SI into the main text — are all correct.** I checked them because they are hardcoded too and share the defect class: `si.tex:100` "Table~2 declares that a prediction equal to the substrate is dropped" ✓ (`table_sweep.tex:21` caption ends with exactly that sentence; `grail_jcim.aux:73` → Table 2); `si_table_precision.tex:18` "the parent-drop convention of Table~2" ✓ (same); `si_table_intervals.tex:30` "in Figure~1" ✓ (`grail_jcim.aux:71` `fig:sweep` → Figure 1, caption "Shading is read from the paired intervals"); `si_table_intervals.tex:30` "The estimator is Equation~6" ✓ (the paired-bootstrap $\delta^{(b)}$ at `body.tex:214-217` renders as `(6)` in `grail_jcim.pdf`).
- **Both PDFs are current** relative to the sources (`si.tex` 21:43 / `si.pdf` 21:45; `body.tex` 21:43 / `grail_jcim.pdf` 21:45), and `body.tex:230` — added in the HEAD commit — appears in the compiled main PDF, so the eight rendered `Table S…` strings I matched are the shipping state, not a stale build.


====================================================================================================
AGENT 2
====================================================================================================

## 1. Where the curated/named/unnamed accounting is computed

`/Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746/scripts/paper2_numbers.py:260-282` is the only code that produces these numbers. It is not written to a `results/` artifact — it writes straight into `paper2/numbers.tex`.

```python
# paper2_numbers.py:262-264 (comment)
# superseded earlier cut of 5,866. Three curated collections ship with the code; the rest of
# the curated half comes from a fourth whose file is not in the repository.
    curated_files = {                                    # :269-273
        "hydroxylation": "grail_metabolism/data/smirks.txt",
        "merged":        "grail_metabolism/data/merged_smirks.txt",
        "notebooks":     "grail_metabolism/resources/notebooks_rules.txt",
    }
    n["curated.total"]   = len(bankset - minedset)       # :279
    n["curated.named"]   = len(named)                    # :280
    n["curated.unnamed"] = len(bankset - minedset - named)  # :281
```

I re-ran this logic against the files. It reproduces exactly: `curated.total` 1725, `curated.named` 1233, `curated.unnamed` 492, with per-file contributions 474 / 656 / 1051 — matching `paper2/numbers.tex:48-53`. The consuming prose is `paper2/si.tex:175-177` ("The remaining 492 come from a fourth collection whose file is not in the repository … treat the curated half as unattributed"), with the open marker at `paper2/markers.tex:10`.

**The comment at `paper2_numbers.py:264` and the SI sentence are both false.** Three other places in the same repository name the fourth collection and its file:

- `docs/benchmark/manuscript.md:332` — "four prior curated banks — `smirks.txt` (473), `merged_smirks.txt` (656), `compressed_rules.smarts` (500), and `notebooks_rules.txt` (1,051)"
- `scripts/mine_rules.py:45` — `"compressed_rules.smarts": ROOT / "grail_metabolism" / "compressed_rules.smarts"`
- `scripts/measure_coverage.py:35` — same path in `_ALL_RULE_BANK_SPECS`

(`scripts/convention_census.py:140-142` also drops it, calling only the other three "the hand-written collections the curated half was assembled from".)

## 2. The four collections and where their files live

All four are present on disk **and tracked in git** (`git ls-files` matches each; both the bank and `compressed_rules.smarts` are clean against HEAD):

| collection | absolute path | rules | added |
|---|---|---|---|
| hydroxylation | `…/grail_metabolism/data/smirks.txt` | 474 | `c2eb482` 2025-05-13 |
| merged | `…/grail_metabolism/data/merged_smirks.txt` | 656 | `2764220` 2025-06-18 |
| notebooks | `…/grail_metabolism/resources/notebooks_rules.txt` | 1051 | `7b03e52` 2026-03-29 |
| **compressed** | `…/grail_metabolism/compressed_rules.smarts` | **500** | `7b03e52` 2026-03-29 |

`smirks.txt` has no trailing newline, so `wc -l` reports 473 — hence the 473/474 disagreement between `manuscript.md:332` and `numbers.tex:48`. 474 non-blank lines is correct.

The four are nested, which is why 474+656+1051+500 = 2681 deduplicates to 1715:
- `smirks.txt` **==** `merged_smirks.txt[:474]`, order-exact, and `smirks.txt ⊂ notebooks_rules.txt`
- `notebooks_rules.txt ∩ merged_smirks.txt` **==** `smirks.txt` exactly
- `compressed_rules.smarts ∩ merged_smirks.txt` = 18; `∩ smirks.txt` = 0; `∩ notebooks_rules.txt` = 0
- so the bank's curated half decomposes as 474 (shared) + 182 (merged-only) + 577 (notebooks-only) + 482 (compressed-only) = 1715

The bank file is block-structured and confirms this: `extended_smirks.txt` lines 1–656 are `merged_smirks.txt` **verbatim and in order**; 657–1138 are the compressed collection; 1139–1715 the notebooks tail; 1716–7581 mined.

No deleted provenance file exists. `git log --all --diff-filter=D` over all commits shows the only rule-like file ever deleted is `grail_metabolism/data/reactions.txt`, deleted in a revert and restored. Nothing was removed to make the fourth collection disappear; it was simply omitted from the dict at `paper2_numbers.py:269`.

## 3. The 492 unnamed rules — traced

Concretely (dumped with line numbers to `/private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-GRAIL--claude-worktrees-hungry-pasteur-25d746/507ecf1d-5e06-4076-9717-52ed7f41592a/scratchpad/unnamed_492.txt`), the 492 are **two disjoint groups**:

**(a) 482 rules, `extended_smirks.txt` lines 657–1138, contiguous.** These are exactly `compressed_rules.smarts` minus its 18 rules that also sit in `merged_smirks.txt`. Membership test over every file in the repository (exact string match on extracted templates):

```
u492  blk482  file
 492     482  grail_metabolism/resources/extended_smirks.txt
 482     482  grail_metabolism/data/reactions.txt
 482     482  grail_metabolism/compressed_rules.smarts
 477     477  grail_metabolism/data/xtracted.txt
  10       0  grail_metabolism/resources/mined_only.txt
```

All **500** of `compressed_rules.smarts` are in `grail_metabolism/data/reactions.txt` (4,533 templates, added `f03bc21` 2025-06-16); 495 are also in `grail_metabolism/data/xtracted.txt` (4,534, added `f588cff` 2025-05-20 "added extracted reaction rules"). A third file of the same lineage, `extracted_grail.txt` (3,343 templates, ⊂ `reactions.txt`), is untracked in the main checkout at `/Users/nikitapolomosnov/PycharmProjects/GRAIL/grail_metabolism/data/extracted_grail.txt`; 368 of the 500 are in it.

**(b) 10 rules, scattered singletons** at bank lines 2015, 2019, 2020, 3409, 3653, 4109, 4744, 5255, 6284, 6720. All ten are in `resources/mined_only.txt` and none in `mined_only_v2.txt` — they are exactly `set(mined_only) − set(mined_only_v2)` (5866 − 5856 = 10). These are **mined templates miscounted as curated** by the v2-based partition at `paper2_numbers.py:268`. Under the v1 partition the counts are curated 1715 / unnamed 482. `results/bank_overlap_sygma.json` and `results/convention_census.json` use the v1 partition (curated 1715); `numbers.tex` uses v2 (1725). Both are in the paper's ecosystem.

So `U(492) = compressed_rules.smarts_only(482) ⊎ (mined_v1 − mined_v2)(10)`, with zero residue — verified.

**None of the 482 come from any published rule set available here.** Exact-membership tests: SyGMa 0, GLORYx 0, BioTransformer 0, RetroSim (`resources/external/retrosim_templates_general.json`, 1,351 keys) 0, USPTO retro-template library (`grail_metabolism/uspto_templates.csv.gz`, 42,554) 0. (A naive substring test reports one USPTO hit; exact matching gives zero.)

**How `compressed_rules.smarts` was made — partially established, not fully.** The deleted notebook `notebooks/reduce_reactions.ipynb` (recoverable at `7b03e5234cc95ef18f8e5252579d18ec1bfe5d65^`) reads `data/reactions.txt` (cell 0, splitting on `"', '"` — a Python list repr, which is exactly the format of that file), unions it with `merged_smirks.txt`, and runs `cluster_smarts_rules(rules_all, target_clusters=500, tolerance=50, fp_size=16000, radius=4)` → `final_rules` (cell 20). That matches the file's size (500), its source set (all 500 ⊆ reactions.txt) and its 18-rule overlap with `merged_smirks.txt`. **But no cell in any recovered notebook writes `compressed_rules.smarts`, and `git log -S compressed_rules` finds no producing code anywhere in history.** The clustering is a documented-looking procedure I found in a notebook; I cannot state it as *the* procedure that wrote the shipped file.

**The upstream corpus of `reactions.txt`/`xtracted.txt` is NOT established.** What I can say: the dialect (`(frag.frag)>>(frag)`, atom maps on every atom) matches the output of `combine_reaction` in `grail_metabolism/utils/reaction_mapper.py:117-129`, which parenthesises multi-fragment sides exactly that way and maps with RXNMapper; and the main checkout holds `mapping_extr.csv` (4,533 data rows × 7,388 substrate columns) and `mapping_filtered.csv` (3,343 rows), whose row counts equal `reactions.txt` and `extracted_grail.txt` respectively — but their row keys are integer indices, so they establish alignment, not origin. It is **not** the USPTO retro-template library (0/42,554 exact overlap). What would settle it: the script or notebook that wrote `xtracted.txt` on 2025-05-20 (commit `f588cff` added only that file plus `notebooks/generator_tuning.ipynb`, which does not produce it), or the code that built `mapping_extr.csv`.

## 4. Reproducing 152/175 independently

`scripts/bank_overlap_sygma.py:43` resolves SyGMa's rules as `Path(sygma.__file__).parent / "rules"`, reading `phase1.txt` and `phase2.txt`, taking `line.split("\t")[0]`, skipping `#` and blanks (`:48-56`). **SyGMa is not installed in `test_grail`** — that script cannot run in the env you gave me. It is installed in the base env: `/Users/nikitapolomosnov/anaconda3/lib/python3.10/site-packages/sygma/rules/{phase1.txt,phase2.txt}`, **SyGMa 1.1.0, `License: GPL`, author Lars Ridder** (dist-info METADATA).

Reading those two files with the script's own parser and testing verbatim membership myself:

```
sygma raw lines 175, unique 175
in bank                        152   share 0.8686
in curated (mined_only v1)     152    in mined half   0
in curated (mined_only v2)     152    in mined_v2     0
share_of_grail_curated_that_is_sygma (152/1715) 0.0886
```

Every field of `results/bank_overlap_sygma.json`'s `containment` block reproduces exactly (175, 152, 152, 0, 0.8686, 0.0886). The 23 misses are real chemistry SyGMa has and the bank does not (e.g. `[#6:1][SH1:2]>>[#6:1][S:2]C`).

## 5. Are the 152 in the named 1,233 or the unnamed 492?

**All 152 are inside the named 1,233. Zero are in the unnamed 492.** Per file:

- `resources/notebooks_rules.txt`: **152 / 152**
- `data/smirks.txt`: 75 · `data/merged_smirks.txt`: 75 (the same 75, since `smirks ⊂ merged`)
- `compressed_rules.smarts`: **0** · mined half: **0**

Bank line positions of the 152: 6 … 1712, all inside the curated block. This refutes the reviewer inference recorded in `.review/round2/result.json` ("the balance lies in merged_smirks.txt, notebooks_rules.txt, or the 492-rule collection with no file") — the balance lies wholly in `notebooks_rules.txt`, and the 492 carry none of it.

## 6. What can be attributed, and what cannot

Verbatim exact-match of the whole bank against every published library shipped in or installed alongside the repo:

| library | source read | its rules | verbatim in bank | all inside |
|---|---|---|---|---|
| SyGMa 1.1.0 (GPL) | installed pkg `sygma/rules/*.txt` | 175 | **152** | `notebooks_rules.txt` (75 also in `smirks.txt`) |
| GLORYx | `…/resources/external/gloryx_reactionrules.csv` | 260 | **260 (all)** | `notebooks_rules.txt` (196 also in `smirks.txt`) |
| BioTransformer | `…/resources/external/bt_database_*.json` | 976 | **611** | `notebooks_rules.txt` (211 also in `smirks.txt`) |
| RetroSim | `…/resources/external/retrosim_templates_general.json` | 1351 | 0 | — |
| USPTO retro templates | `…/grail_metabolism/uspto_templates.csv.gz` | 42554 | 0 | — |

The GLORYx CSV carries its own `Rule source` column, so its 260 split further: **178 attributed to SyGMa** (all 178 in bank), **73 to GLORY** (all 73), **9 to GLORYx itself** (all 9). The BioTransformer JSONs carry a citation header naming Djoumbou Feunang et al., J. Cheminform. 2019, 11:2.

Union of the three libraries verbatim in the bank: **966 templates**, all 966 inside `notebooks_rules.txt`, 434 of them also in `smirks.txt`/`merged_smirks.txt`, **0 in `compressed_rules.smarts`, 0 in the mined half.**

So, of the 1,725 curated templates:

- **966 (56%) are attributable today** to SyGMa / GLORYx / GLORY / BioTransformer by verbatim identity against files already in the repository or an installed package. `notebooks_rules.txt` is 966/1051 accounted for; `smirks.txt` 434/474; `merged_smirks.txt` 434/656.
- **482 have a file and a lineage but no external attribution**: `compressed_rules.smarts` → all 500 ⊆ `data/reactions.txt`, 495 ⊆ `data/xtracted.txt`. They are machine-extracted templates from an in-house corpus, not a published library. Their *corpus* is unestablished.
- **180 more are also machine-extracted, and are currently mislabelled as hand-written**: `merged_smirks.txt` minus `smirks.txt` is 182 rules, of which **180 are in `reactions.txt ∪ xtracted.txt`**. `convention_census.py:140-142` calls `merged_smirks.txt` a "hand-written collection"; 180/656 of it is not. `smirks.txt` and `notebooks_rules.txt` have zero overlap with the extracted pool and are hand-written/collected as described.
- **10 are mined templates counted as curated**, a `mined_only.txt` vs `mined_only_v2.txt` bookkeeping seam.
- **267 of the 1,233 "named" match none of the five external libraries** and remain unattributed on their own.

What I could not establish, and what would establish it: (i) the corpus `data/xtracted.txt` / `data/reactions.txt` were extracted from — needs the extraction script from around 2025-05-20 or the builder of `mapping_extr.csv`; (ii) the exact command that wrote `compressed_rules.smarts` — `git log -S` finds no producer, only the `reduce_reactions.ipynb` cell whose parameters match; (iii) where `notebooks_rules.txt` lived before `7b03e52` — `mine_rules.py:46-48` falls back to `notebooks/rules.txt`, a path that never existed in git and does not exist in the main checkout; (iv) licences for the GLORYx CSV and the BioTransformer JSONs beyond the citation header in the latter (SyGMa's GPL is recorded in its dist-info).


====================================================================================================
AGENT 3
====================================================================================================

# Recompute path and cost: substrate tautomer presentation as a declared axis

Everything below is read from the worktree at `/Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746`. Timings marked **measured** were run here with `/Users/nikitapolomosnov/anaconda3/envs/test_grail/bin/python`; timings marked **claimed** are docstring text; timings marked **inferred** come from file mtimes and are labelled as such.

---

## 0. Two blocking facts before any command below runs

**(i) `sygma` is not installed and the ceiling script cannot import.** `scripts/engine_knobs.py:52` is a bare module-level `import sygma`, and `scripts/coverage_gap_types.py:40` does `from engine_knobs import apply_with`. Verified:

```
$ .../test_grail/bin/python scripts/coverage_gap_types.py --help
ModuleNotFoundError: No module named 'sygma'   (engine_knobs.py line 52)
```

`sygma` is not importable from any of the 20 `~/anaconda3/envs/*`, from `/opt/homebrew/Caskroom/miniforge/base`, from the `metapredictor` env, or from system `python3`. It is declared as an optional extra: `pyproject.toml:65` `sygma = { version = "1.1.0", optional = true }`, `pyproject.toml:71` `baselines = ["sygma"]`. So `pip install -e .[baselines]` (or `pip install sygma==1.1.0`) is a precondition for the ceiling arm — even though the ceiling arm never calls SyGMa. Whether that install succeeds needs network and I did not attempt it.

**(ii) `scripts/typed_edit/case_study.py` is modified in the working tree, uncommitted, and already carries this exact axis.** `git status` shows ` M scripts/typed_edit/case_study.py`. The diff adds `--present {stored,standardised}` (line 54), standardises the substrate at line 78-81 while looking references up under the unchanged corpus key (line 96), and records `presentation` / `substrate_presentation` in the artifact (lines 146, 150). A review workflow is also live in this tree (`.review/README.md` modified, `.review/round2/`, untracked `scripts/typed_edit/dialect_census.py`). The file changed between two reads inside this session, so treat my line numbers for that one file as a snapshot.

---

## 1. The ceiling (0.8171, 475 uncovered)

### 1a. Which script actually produces the number

`\numCeilingCoverage` is generated, not typed: `scripts/paper2_numbers.py:174` `n["ceiling.coverage"] = cov["coverage"]`, where `cov = art("coverage_gap_types.json")` (`paper2_numbers.py:36`). So the number the paper prints comes from **`results/coverage_gap_types.json`**, written by **`scripts/coverage_gap_types.py`** (`--out` default at line 121). Contents: `coverage 0.8171`, `covered_pairs 2122`, `uncovered_pairs 475`, `gap {novel_type 337, known_type 98, untypeable 40}`, `n_substrates 1170`, `n_rules 7581`, `n_bank_types 4417`.

The artifact you named, **`results/typed_edit_known_pairs.json`**, is a **second, independent recomputation of the same quantity** by `scripts/typed_edit/known_type_recovery.py` (mapped in `scripts/audit_artifact_provenance.py:49`). Its merge step is gated against the committed decomposition (`known_type_recovery.py:236-248`) and the artifact records `reproduces_committed_decomposition: true`. I summed its shards `results/gaptypes/a*.json` myself: 8 shards tiling `[0,147] … [1026,1170]`, totalling 1170 substrates, `covered 2122 / uncovered 475`, `novel 337 / known 98 / untypeable 40`, **coverage 0.8171**. So both scripts are on the same path and either can carry the sweep; `known_type_recovery.py --phase a` is the better vehicle because it also dumps the 98 known-type and 337 novel-type pairs.

### 1b. Where the substrate enters

Three lines, and only three:

| file:line | code |
|---|---|
| `scripts/coverage_gap_types.py:161` | `sub_mol = Chem.MolFromSmiles(sub)` |
| `scripts/coverage_gap_types.py:166` | `products = {k: {0} for k in apply_with(sub_mol, rules, False, "canonical", False)}` |
| `scripts/engine_knobs.py:119` | `substrate = Chem.AddHs(Chem.Mol(mol)) if add_hs else Chem.Mol(mol)` |

and the mirror pair in the second producer: `known_type_recovery.py:86` (`MolFromSmiles`) and `:89` (`apply_with(sub_mol, rules, False, "canonical", False)`).

The substrate string itself comes from `scripts/run_benchmark.py:78 load_test_map` → `test_triples_clean.txt` + `test.sdf`, keyed by the corpus SMILES (`run_benchmark.py:86-90`). The matcher side is untouched by this axis: references are keyed with `_tautomer_inchikey(met)` (`coverage_gap_types.py:174`), which already standardises both sides.

### 1c. What would have to change

`engine_knobs.apply_with` (`scripts/engine_knobs.py:109-139`) already exposes four discretionary switches (`add_hs`, `norm`, `drop_invalid`, `remove_hs`). Adding a fifth in the same style is faithful to its design and is **one line of body**:

```python
def apply_with(mol, rules, add_hs, norm, drop_invalid, remove_hs=False,
               standardise_substrate: bool = False):          # new keyword
    base = standardize_mol(Chem.Mol(mol)) if standardise_substrate else Chem.Mol(mol)
    substrate = Chem.AddHs(base) if add_hs else base          # replaces line 119
```

`standardize_mol` (`grail_metabolism/utils/preparation.py:127`) accepts and returns a `Chem.Mol`, so no SMILES round-trip is needed. **14 scripts route through `apply_with`** — `coverage_gap_types.py`, `known_type_recovery.py`, `ceiling_by_provenance.py`, `ceiling_norm_check.py`, `ceiling_external_validity.py`, `ceiling_convention_matched.py`, `completed_loop_reach.py`, `explicit_h_mechanism.py`, `dispatch_paired_ci.py`, `bank_engine_replication.py`, `provenance_knob_attribution.py`, `factorize_recall.py`, plus `engine_knobs.py` itself — so a single default flip (or a module constant read from an env var) switches the whole ceiling family at once, exactly as `DEFAULT_APPLICATION_PRESENTATION` (`preparation.py:369`) already does for hydrogens.

### 1d. Cost — measured

Timed here on 6 clean-test substrates sampled every 195th (one uncontended process, whole 7,581-rule bank, `add_hs=False, norm="canonical"`, exactly `apply_with`'s body):

```
heavy  n_products  t_apply  t_key   t_standardize_substrate
 28       493       3.00     57.5      0.006
 20      1322       3.74      8.8      0.002
 38       630       2.91    112.4      0.010
 22       428       2.09     25.6      0.002
 31       351       2.06     63.3      0.007
 18       254       1.04     24.4      0.002
median            2.50     41.6      0.004
```

- **Rule application is not the cost.** Median 2.50 s/substrate.
- **Tautomer-InChIKey-ing every product is the cost.** Median 41.6 s/substrate, range 8.8–112.4 s. This matches the note at `scripts/bank_without_selection.py:120`: "roughly seven structures a second."
- **Standardising the substrate is free.** Median 0.004 s. Over the whole split: **43.9 s for all 1,170 substrates** (measured separately).

So per-substrate ≈ **44 s median** → **~14 h single-threaded for 1,170 substrates**. The committed run was sharded: `results/gaptypes/a*.json` are 8 shards of 147/144, mtimes Aug 28 05:51 → 09:21, i.e. a **~3.5 h spread** (inferred, not recorded — no timing field exists in either artifact). The earlier 6-shard run (`results/gapshards/shard*.json`, 195 each) spans Aug 8 01:47 → 07:09.

**The added cost of the sweep is a whole second run, not the standardisation.** 29.5% of substrates change presentation (§4), so ~70% produce byte-identical product sets; if both arms run inside one shard process the `_tautomer_inchikey` `lru_cache(131072)` (`grail_metabolism/metrics.py:77`) would make the unchanged 70% nearly free on the second arm — a shard of 147 substrates × ~580 products ≈ 85k entries fits under the cache ceiling, a single-process 1,170-substrate run (~680k) does not. That saving is a design choice available to you, not something the current code does.

---

## 2. The head-to-head at k=30 and k=50

### 2a. Which artifact holds the cells the paper prints

`paper2/table_sweep.tex` rows `30` and `50` (all five columns) are generated by `scripts/paper2_tables.py:18` from **`results/deployment_table.json`**, written by `scripts/typed_edit/deployment_table.py`. The SI paired intervals (`paper2/si_table_intervals.tex`) come from the same artifact (`scripts/paper2_si_tables.py:159`).

Committed cells:

| k | GRAIL exh. | GRAIL int. | MetaTox | SyGMa | MetaPred. |
|---|---|---|---|---|---|
| 30 | 0.6391 | 0.4902 | 0.6015 | 0.5068 | 0.4797 |
| 50 | 0.7038 | 0.4902 | 0.6271 | 0.5113 | 0.4797 |

`results/vs_metatox.json` (from `scripts/typed_edit/vs_metatox.py`) is a **narrower** two-arm artifact (bank vs MetaTox only) on the same 291. It is pinned (`audit_artifact_provenance.py:59-60`) but it is not what feeds Table 3.

### 2b. Recomputed vs frozen, arm by arm

**Recomputed from GRAIL checkpoints (2 arms).** In `vs_metatox.py:103-111`: `generator.generate_scored_with_details(s, top_k=7581, …)` then `filt.score_batch(s, cands)`, ranked by `filter × generator`. Defaults `--gen-ckpt artifacts/full5000_priors/checkpoints/generator.pt`, `--filter-ckpt artifacts/full5000_single/checkpoints/filter.pt` (lines 78-79). For `deployment_table.py` the two GRAIL arms are **not recomputed at all** — they are read from pre-built pool files (`deployment_table.py:58-59`):
- `results/widepools_implicit/w*.json` (6 shards, 291 substrates, "whole bank")
- `results/widepools_k30/all.json` (`top_k: 30`, "trained budget")

built by `scripts/typed_edit/build_wide_pools.py`.

> **Provenance gap, stated plainly.** The `widepools_implicit` shards carry only `slice`, `pools`, `references` — **no provenance stamp, no `top_k`, no checkpoint record**. `deployment_table.json` faithfully records the consequence: `configuration.top_k["whole bank"] = null`. Which checkpoint built them cannot be established from the artifacts; only the directory name says "implicit". I could not find any doc, shell script, or JSON in the repo recording the command that produced them.

**Frozen files (3 arms).**

| arm | file | written by | re-runnable here? |
|---|---|---|---|
| SyGMa | `results/sygma_fulltest_predictions.json` (7.9 MB, 1170 keys) | `scripts/sygma_fulltest_predictions.py` | **only after installing `sygma`** — not present in any env |
| MetaTox | `results/metatox_smirks_preds.json` (1.9 MB) | `scripts/metatox_smirks_ingest.py`, which *ingests a supplier delivery* and positionally joins it to `results/metatox_input/substrate_map.csv` (docstring lines 3-16) | **No.** There is no MetaTox engine in this repo. It is a third-party submission of 291 substrates; a re-presented substrate set would have to be re-submitted. |
| MetaPredictor | `artifacts/tier2_1170/metapredictor_preds.json` (849 KB) | `scripts/run_metapredictor_1170.sh` → `scripts/tier2_metapredictor_to_json.py` | **Yes.** Env `metapredictor` exists at `/opt/homebrew/Caskroom/miniforge/base/envs/metapredictor` with `onmt 2.3.0`, `torch 1.13.0`, `rdkit 2022.09.5`; repo at `artifacts/tier2/metapredictor_src` with **872 MB of model weights present**. The script itself warns "9-model CPU ensemble -> expect many hours." |

**Population join.** `vs_metatox.population()` (`vs_metatox.py:65-73`) intersects the *substrate SMILES keys* of `results/test_references.json`, `metatox_smirks_preds.json`, `scored_predictions.json`, `metapredictor_preds.json`, `sygma_fulltest_predictions.json`. **If the substrate string is standardised before it is used as a key, this intersection collapses to zero.** Any re-presentation must keep the corpus SMILES as the join key and change only what is handed to the engine — which is precisely the pattern the in-flight `case_study.py` patch adopts (`corpus_key` vs `s`, lines 78-96). I verified all 291 are inside the 1,170-substrate clean-test map.

---

## 3. Exact command lines

Run everything from the repo root with `PY=/Users/nikitapolomosnov/anaconda3/envs/test_grail/bin/python`. Where the committed run was sharded I give the shard geometry actually recorded in the artifacts.

### (a) The ceiling

Prerequisite (blocking, see §0i): `$PY -m pip install -e '.[baselines]'`

**a1 — single process, the artifact the paper reads:**
```bash
$PY scripts/coverage_gap_types.py --sample 0 --seed 42 \
    --out results/coverage_gap_types.json
```

**a2 — sharded exactly as committed (6 × 195):**
```bash
mkdir -p results/gapshards
for i in 0 1 2 3 4 5; do
  s=$((i*195)); e=$((s+195))
  $PY scripts/coverage_gap_types.py --start $s --end $e \
      --counts-out results/gapshards/shard$i.json &
done; wait
$PY scripts/coverage_gap_types.py --merge 'results/gapshards/shard*.json' \
    --out results/coverage_gap_types.json
```
(the merge glob must be quoted — `coverage_gap_types.py:125` globs it itself; `--counts-out`'s directory must exist, line 195 is a bare `write_text`.)

**a3 — the pairs-carrying twin (`results/typed_edit_known_pairs.json`), sharded 8 × 147/144 as committed, with its gate against a1:**
```bash
mkdir -p results/gaptypes
for i in 0 1 2 3 4 5 6 7; do
  s=$((i*147)); e=$((s+147)); [ $i -ge 6 ] && { s=$((882+(i-6)*144)); e=$((s+144)); }
  $PY scripts/typed_edit/known_type_recovery.py --phase a --start $s --end $e \
      --out results/gaptypes/a$i.json &
done; wait
$PY scripts/typed_edit/known_type_recovery.py --merge 'results/gaptypes/a*.json' \
    --out results/typed_edit_known_pairs.json
# phase B (relaxation ladder), reads the merged phase-A dump:
$PY scripts/typed_edit/known_type_recovery.py --phase b \
    --known-in results/typed_edit_known_pairs.json \
    --out results/typed_edit_known_type_recovery.json
```
Note: the merge gate at `known_type_recovery.py:236-248` compares against `results/coverage_gap_types.json`. **Under a swept presentation that gate will fail by construction** — it must be re-pointed at the same-arm a1 output, or the whole thing runs under a per-arm `--out` pair, or you get a red gate that means nothing.

**Then regenerate downstream:**
```bash
$PY scripts/typed_edit/novel_type_census.py --shards 'results/gaptypes/a*.json'
$PY scripts/typed_edit/uspto_type_overlap.py --shards 'results/gaptypes/a*.json'
```

### (b) The vs_metatox pools and table

**b1 — the two-arm artifact (bank vs MetaTox), all flags explicit:**
```bash
$PY scripts/typed_edit/vs_metatox.py \
    --gen-ckpt    artifacts/full5000_priors/checkpoints/generator.pt \
    --filter-ckpt artifacts/full5000_single/checkpoints/filter.pt \
    --threads 4 \
    --out results/vs_metatox.json
# writes results/vs_metatox_pools.json alongside (vs_metatox.py:152)
```
`--limit N` truncates for a smoke test **and disarms the MetaTox reproduction gate** (`vs_metatox.py:142-145`) — never quote a `--limit` run.

**b2 — the pools the five-column table actually reads (this is the eight hours):**
```bash
mkdir -p results/widepools_implicit results/widepools_k30
for i in 0 1 2 3 4 5; do
  s=$((i*49)); e=$((s+49)); [ $i = 5 ] && e=291
  $PY scripts/typed_edit/build_wide_pools.py --start $s --end $e --top-k 7581 \
      --gen-ckpt artifacts/full5000_priors/checkpoints/generator.pt \
      --filter-ckpt artifacts/full5000_single/checkpoints/filter.pt \
      --out results/widepools_implicit/w$i.json &
done; wait
$PY scripts/typed_edit/build_wide_pools.py --start 0 --end 291 --top-k 30 \
    --gen-ckpt artifacts/full5000_priors/checkpoints/generator.pt \
    --filter-ckpt artifacts/full5000_single/checkpoints/filter.pt \
    --out results/widepools_k30/all.json
```
Caveat: the checkpoint pair above is `build_wide_pools.py`'s own default (lines 87-88); the committed `widepools_implicit` shards do not record which pair was used, so this is a reconstruction, not a reproduction (§2b).

**b3 — the table and its intervals:**
```bash
$PY scripts/typed_edit/deployment_table.py \
    --whole-bank 'results/widepools_implicit/w*.json' \
    --trained    results/widepools_k30/all.json \
    --out        results/deployment_table.json
$PY scripts/paper2_tables.py        # writes paper2/table_sweep.tex
$PY scripts/paper2_si_tables.py     # writes paper2/si_table_intervals.tex etc.
$PY scripts/paper2_numbers.py       # writes results/paper2_numbers.json
$PY scripts/paper2_macros.py        # writes paper2/numbers.tex
$PY scripts/check_paper2_numbers.py
$PY scripts/verify_paper_numbers.py --tol 5e-4
$PY scripts/audit_artifact_provenance.py --all
```
`deployment_table.py` gates itself against `results/four_method_291.json` for all three comparators — same caveat as a3: under a swept presentation that gate is only meaningful for the frozen arms.

**b4 — the frozen arms, if you want them re-presented too:**
```bash
# SyGMa — needs the install from §0i; no CLI, output path hardcoded (line 27)
$PY scripts/sygma_fulltest_predictions.py
# MetaPredictor — separate env, many hours
bash scripts/run_metapredictor_1170.sh
# MetaTox — NOT re-runnable locally; the file is a supplier delivery ingested by:
$PY scripts/metatox_smirks_ingest.py   # ingest only, does not predict
# then rebuild the population-defining artifact:
$PY scripts/dump_scored_predictions.py --out results/scored_predictions.json
$PY scripts/four_method_291.py --out results/four_method_291.json
```

### (c) The case study

```bash
$PY scripts/typed_edit/case_study.py \
    --substrate "N=c1ccn(C2OC(CO)C(O)C2(F)F)c(O)n1" \
    --present stored \
    --gen-ckpt    artifacts/full5000_implicit/checkpoints/generator.pt \
    --filter-ckpt artifacts/full5000_priors/checkpoints/filter.pt \
    --top-k 30 --cap 100 --show 15 \
    --out results/case_study.json
# the swept arm — same command, --present standardised, different --out
$PY scripts/typed_edit/case_study.py ... --present standardised \
    --out results/case_study_standardised.json
```
`--present` exists **only in the uncommitted working tree**; on `HEAD` the flag does not exist. It requires `results/widepools_implicit/w*.json` for references (line 94) and now hard-fails if the corpus key carries none (line 97-98).

---

## 4. Cost, honestly

**The eight hours is 291 substrates, single-threaded.** Two independent docstrings: `scripts/typed_edit/vs_metatox.py:25` ("Building them costs eight hours") and `scripts/typed_edit/build_wide_pools.py:4` ("about eight hours single-threaded"). Both scripts build over `vs_metatox.population()` = **291 substrates** (`results/vs_metatox.json` records `population.n = 291`). That is **≈ 99 s per substrate**.

**Independent instrument agreeing with it.** `scripts/typed_edit/cost_envelope.py` times exactly the vs_metatox inner loop (`generate_scored_with_details(top_k=7581)` + `score_batch`, lines 46-52) on 106 validation substrates sampled every 3rd by size, deadline 600 s:

- 95 finished: median **36.0 s**, mean **56.3 s**, p90 **103.4 s**, max **552.3 s**, sum 5,346 s
- **11 of 106 (10.4%) did not finish inside 600 s**

Effective per-substrate ≥ (5,346 + 11×600)/106 = **112.7 s** — a lower bound, since the 11 exceeded the cap. 291 × 112.7 s = **9.1 h**, consistent with the docstrings' eight.

**Per-substrate re-run estimates for a 291-substrate arm:**

| basis | s/substrate | 291 substrates, serial | 6 shards |
|---|---|---|---|
| docstring | 99 | 8.0 h | ~1.4 h |
| cost_envelope, finished-only mean | 56.3 | 4.6 h | ~0.8 h |
| cost_envelope, deadline-inclusive lower bound | ≥112.7 | ≥9.1 h | ≥1.5 h |
| `widepools_implicit` mtimes (inferred: Aug 24 16:07→20:03, 6 shards × ~49) | — | — | ~4 h wall |

The mtime-inferred 4 h wall for 6 parallel shards implies ~24 CPU-hours, i.e. **3× the docstring**. I cannot reconcile those from the artifacts — neither `vs_metatox.json` nor the widepool shards carry a timing field, and the mtime reading assumes all six started together. Budget the pool build at **8–24 CPU-hours per arm**, not eight flat.

**The ceiling arm, measured above:** ~44 s/substrate median → **~14 h serial for 1,170**, or ~3.5 h wall on 8 shards (matching the `gaptypes` mtime spread).

**Total for a two-arm sweep, both cells:**

| piece | per arm | ×2 arms |
|---|---|---|
| ceiling, 1170 subs, 8 shards | ~3.5 h wall / ~14 CPU-h | ~7 h wall / ~28 CPU-h |
| whole-bank pools, 291 subs, 6 shards | ~1.5–4 h wall / 8–24 CPU-h | ~3–8 h wall / 16–48 CPU-h |
| k=30 pools, 291 subs | ~0.2 h (top_k 30, median 0.39 s/sub per `mode_timings.json`) | ~0.4 h |
| substrate standardisation itself | 44 s for 1170 (**measured**) | negligible |
| MetaPredictor re-run, if re-presented | "many hours" (claimed, `run_metapredictor_1170.sh:4`) | — |
| SyGMa re-run, if re-presented | not measurable — package absent | — |
| MetaTox re-run | **impossible locally** | — |

**One caution on the second arm being cheap.** 10.4% of substrates blew a 600 s deadline in `cost_envelope` and `mode_timings.json` records a 291-heavy-atom validation peptide the exhaustive arm "has never finished at any budget" (max 231.01 s in interactive mode). A changed substrate presentation changes which molecules are pathological; there is no basis in these artifacts for assuming the swept arm has the same tail.

---

## 5. The smallest faithful change

**There is no single function today.** The substrate is parsed at four independent sites, and the ceiling family and the model family do not share one:

| family | parse site | consumer |
|---|---|---|
| ceiling (14 scripts) | `scripts/engine_knobs.py:119` | rule firing |
| generator | `grail_metabolism/model/generator.py:998` (`_graph_for_substrate`) | feeds *both* the scored graph and the `mol` that `safe_run_reactants` fires on (`generator.py:1324` → `1338`) |
| filter | `grail_metabolism/model/filter.py:479` (`score_batch`) | pair/single graphs |
| deployed wrapper | `grail_metabolism/utils/preparation.py:384` | `apply_rules_to_molecule` — *not* on the `generate_scored_with_details` path |

**Recommendation — two edits, both one line of body, and neither is a fork:**

1. **Ceiling family: `scripts/engine_knobs.py:119`, function `apply_with` (defined line 109).** Add `standardise_substrate: bool = False` and apply `standardize_mol` before the `AddHs` branch (code in §1c). One function, 14 callers, zero call-site churn. This is exactly the shape the function already has for its other four knobs, and the file's own docstring (lines 5-17) is a manifesto for measuring precisely this kind of discretionary choice.

2. **Model family: the call site, not the library.** In `vs_metatox.py`, `build_wide_pools.py`, and `build_val_pools.py`, the same string `s` is handed to `generate_scored_with_details` *and* `score_batch`, so standardising it **once, before the loop**, switches the whole run — the generator's `_applicability_cache` is keyed on the SMILES string (`generator.py:562`), so a different string is a different cache entry and nothing is corrupted. The pattern is already written in the working tree at `case_study.py:78-81`:

   ```python
   corpus_key = s
   if args.present == "standardised":
       s = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(corpus_key)))
   ```
   with references and the population join kept on `corpus_key` (`case_study.py:96`). Three scripts × ~5 lines, plus a `--present` flag and a `presentation` field in the artifact.

If you want strictly one function for both families, the place is **`grail_metabolism/utils/preparation.py` beside line 369**, where `DEFAULT_APPLICATION_PRESENTATION` already establishes the precedent — including the comment explaining why such a constant must *not* be flipped globally, because "every artifact in `results/` was produced through the historical default." Add `SUBSTRATE_PRESENTATION` and a `present_substrate(smiles) -> str` that returns `_standardize_smiles_cached(smiles)` (line 162, already `lru_cache`d) or the input unchanged, then call it at `engine_knobs.py:119`, `generator.py:998`, and `filter.py:479`. That is one function, three call sites, one env-var flip per run — at the cost of touching the library rather than only the scripts.

---

## 6. How big is the axis (measured, so the sweep can be scoped)

| population | n | move under `standardize_mol` | fraction |
|---|---|---|---|
| clean test split (the ceiling) | 1,170 | **345** | **0.2949** |
| the four-method comparison set | 291 | **81** | **0.2784** |

All 291 are inside the 1,170. Zero parse or standardisation failures. Whole-split standardisation cost 43.9 s (median 0.003 s, max 2.43 s per substrate). So roughly **three substrates in ten are actually presented differently**; the other seven in ten will return byte-identical product sets and are, in principle, cache-free on the second arm within a shard (§1d).

---

## What I could not establish

- **The wall-clock of the committed ceiling run.** Neither `coverage_gap_types.json` nor `typed_edit_known_pairs.json` carries a timing field; the shard mtimes are the only evidence and they do not record start times. My 44 s/substrate is a fresh measurement on 6 substrates in this environment, not a recovery of what the run cost.
- **Which checkpoints built `results/widepools_implicit/w*.json`.** The shards carry no provenance stamp, no `top_k`, no checkpoint record; `deployment_table.json` propagates the hole as `top_k: null`. No script, shell file, or doc in the repo records the command.
- **Whether `sygma` can be installed.** It is absent from every interpreter I checked and declared only as an optional extra. Installing needs network; I did not try.
- **Whether the eight-hour docstring or the ~24 CPU-hour mtime reading is right.** Both are in the record and they differ by 3×.
- **Whether MetaTox could be re-run under a swept presentation.** There is no MetaTox engine in this repo — only an input builder (`make_metatox_input.py`) and an ingest (`metatox_smirks_ingest.py`). It is a supplier submission, so a re-presented arm is an external dependency, not a compute cost.
