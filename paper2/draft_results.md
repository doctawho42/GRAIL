On the 291 shared test substrates carrying 665 annotated metabolites, micro recall by budget. A prediction equal to the substrate is dropped before the budget for every method alike, the convention `results/four_method_291.json` uses.

| $k$ | GRAIL exhaustive | GRAIL interactive | MetaTox | SyGMa | MetaPredictor |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0902 | 0.1053 | 0.0406 | **0.1534** | 0.1398 |
| 3 | 0.2271 | 0.2406 | 0.1323 | **0.2887** | 0.2767 |
| 5 | 0.3098 | 0.3203 | 0.2180 | **0.3744** | 0.3549 |
| 8 | 0.4150 | 0.4015 | 0.3113 | 0.4271 | **0.4346** |
| 10 | 0.4526 | 0.4331 | 0.3880 | 0.4556 | **0.4707** |
| 15 | **0.5353** | 0.4797 | 0.5143 | 0.4842 | 0.4797 |
| 20 | **0.5789** | 0.4872 | 0.5519 | 0.4992 | 0.4797 |
| 30 | **0.6391** | 0.4902 | 0.6015 | 0.5068 | 0.4797 |
| 50 | **0.7038** | 0.4902 | 0.6271 | 0.5113 | 0.4797 |

Mean list length: GRAIL exhaustive 98.1, GRAIL interactive 15.6, MetaTox 30.9, SyGMa 40.5, MetaPredictor 10.7.

Read against the strongest comparator at each budget, the picture divides in three.

**GRAIL trails, interval excluding zero:** $k=1$ (SyGMa 0.1534 against GRAIL interactive 0.1053), $k=3$ (SyGMa 0.2887 against GRAIL interactive 0.2406), $k=5$ (SyGMa 0.3744 against GRAIL interactive 0.3203).

**Neither separates:** $k=8$ (MetaPredictor 0.4346 against GRAIL exhaustive 0.4150), $k=10$ (MetaPredictor 0.4707 against GRAIL exhaustive 0.4526), $k=15$ (MetaTox 0.5143 against GRAIL exhaustive 0.5353), $k=20$ (MetaTox 0.5519 against GRAIL exhaustive 0.5789).

**GRAIL leads, interval excluding zero:** $k=30$ (MetaTox 0.6015 against GRAIL exhaustive 0.6391), $k=50$ (MetaTox 0.6271 against GRAIL exhaustive 0.7038).

The advantage is at depth and not at the head of the list. SyGMa leads at the tightest budgets and MetaPredictor in the middle; both saturate, MetaPredictor at 0.4797 from $k=15$ on a mean list of 10.7, SyGMa at 0.5113 on 40.5. GRAIL's exhaustive mode keeps rising because it keeps having candidates.

Against MetaTox alone, the incumbent web service and the system this work set out to replace, the exhaustive mode leads at every budget: $k=1$ +0.0496*, $k=3$ +0.0947*, $k=5$ +0.0917*, $k=8$ +0.1038*, $k=10$ +0.0647*, $k=15$ +0.0211, $k=20$ +0.0271, $k=30$ +0.0376*, $k=50$ +0.0767*, where an asterisk marks an interval excluding zero.

Where a method runs out of candidates the budget stops measuring ranking, and the counts are reported for every arm: at $k=15$, 181 of 291 interactive lists are shorter than the budget, 273 of MetaPredictor's and 34 of MetaTox's; at $k=50$ the counts are 282, 291 and 249.
