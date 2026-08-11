# Pre-submission audit: what the paper claims about itself

Three defects this manuscript shipped for weeks were of one kind, and none was a wrong number:

- an *external* set that was 65% inside the training split,
- *regenerable from committed artifacts* where the central artifact was gitignored,
- a *budget-matched* comparison that was not budget-matched.

A numbers-versus-artifact audit cannot catch this class by construction — both sides of each
statement were correct. What was never checked is the manuscript's claims about **itself**. Each was
found only because something unrelated forced it.

Run every row before submitting. A row is not passed by remembering it was true once.

---

## 1. Every set called external, third-party or held-out is external

**Check:** tautomer-key match of each such set's substrates against every split GRAIL trained on.

```bash
python scripts/external_overlap_audit.py
```

**Status: PASS, with the overlap disclosed.** GLORYx overlaps on 24 of 37 drugs (64.9%); GRAIL's
figures on it are reported on the 13 unseen. Shared subset 0/150. The audit runs in a minute and is
implied by none of the other checks here.

## 2. All numbers are regenerable from committed artifacts

**Check:** diff every `results/` path written by `scripts/` against `git ls-files`. `results/` is
gitignored and the tracked files are there only by historical `git add -f`, so **new artifacts are
untracked silently** — nothing fails, nothing warns.

Scan every path a script reads or writes, not just `results/`. The first version of this command
looked only at `results/` and therefore missed `artifacts/tier2_1170/metapredictor_preds.json` — the
frozen full-split MetaPredictor predictions behind `set_metrics_by_criterion.py` (the paper's two
certified reversals), both cardinality tables, the budget curves and the propensity bounds. **A
check whose scope is narrower than the claim's scope reports a pass it has not earned.**

```bash
python - <<'PY'
import re, pathlib, subprocess
tracked = set(subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split())
refs = set()
for p in pathlib.Path("scripts").glob("*.py"):
    t = p.read_text()
    refs |= set(re.findall(r'["\']((?:results|artifacts|configs)/[A-Za-z0-9_./-]+\.(?:json|csv|txt|pt|sdf))["\']', t))
    for m in re.finditer(r'ROOT\s*/\s*"(artifacts|results)"\s*((?:/\s*"[^"]+"\s*)+)', t):
        refs.add("/".join([m.group(1)] + re.findall(r'"([^"]+)"', m.group(2))))
for r in sorted(refs):
    if r not in tracked and pathlib.Path(r).is_file():
        print(f"UNTRACKED {pathlib.Path(r).stat().st_size/1e6:8.2f} MB  {r}")
PY
```

**Status: PASS as of 2026-08-01, after fixing 18 violations in the first round and 2 in the
latest** — including `set_metrics_by_criterion.json`, the source of the paper's two certified
reversals, and, this round, `gloryx_criterion_grid.json` and `ceiling_gap_by_similarity.json`.

Three classes stay untracked, and each is untracked for a different reason, so the command's raw
output is not the verdict:

- **Regeneration caches**, which hold no numbers: `key_tables` 272M, `moses_keys` 431M,
  `rule_collapse_cache` 11M, `match_sens_cache`, `metatox_input`, `artifacts/preprocessed`.
- **Trained checkpoints** (`artifacts/*/checkpoints/*.pt`). The reproducibility statement says the
  *anonymised archive* holds these, not the git tree, and names which analyses need them — the ones
  that re-rank or re-select. Every split-level number regenerates without them.
- **`artifacts/tier2/biotransformer/database/*.json`**, a third-party file we deliberately do not
  redistribute. The reproducibility statement discloses this by name and says a reader points the
  script at an installed copy.

**Committing the file is necessary and not sufficient — the artifact must record the configuration
that produced it.** This bites only for results measured on a *subsample*: a full-split artifact is
recoverable because the substrate set is the whole split, while a subsampled one is recoverable only
from the cap and seed, and `_sample_triples` draws with `rng.choice(replace=False)`, so caps are not
nested and a wrong cap silently yields a different set.

```bash
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("results").glob("*.json")):
    try: d = json.loads(p.read_text())
    except Exception: continue
    if not isinstance(d, dict): continue
    n = d.get("n") or d.get("n_substrates")
    if isinstance(n, int) and n < 1000 and not any(
            k in d for k in ("config", "max_substrates", "sampling_seed")):
        print("NO CONFIG", p.name, "n =", n)
PY
```

**Status: PASS as of 2026-07-31; zero artifacts now carry a size with no provenance.** The flag was
too blunt to begin with. A subsampled artifact is unrecoverable only if *nothing else* pins its
substrate set, and for most of them something does:

- **Pinned by data or by a committed file.** The `n=37` GLORYx ladder is the whole external set; the
  `n=994` artifacts are the whole clean val split; the whole `n=150` family is defined by
  `artifacts/tier2/substrates.json`, a tracked list of exactly those 150 whose keyset the three
  tier2 prediction files match. Thirteen artifacts, now carrying a `population` field.
- **A different domain entirely.** `retro_transfer` and `xdomain_retro_protocol` are USPTO-50k
  retrosynthesis and have no metabolism split behind them.
- **Genuine seeded draws.** Four. Their cap is recoverable by search, because the draw is
  deterministic in (cap, seed) over a fixed pool: replicate `_sample_triples` and the map
  construction, validate the replica against caps whose yield was measured through the real loader,
  then search. Two resolve uniquely and carry `config_reconstructed`; two remain ambiguous over two
  to four caps and carry `config_candidates`.

```bash
python scripts/recover_subsample_config.py            # report
python scripts/recover_subsample_config.py --apply    # write the three fields
```

**The field names differ on purpose.** `config` is what a run recorded about itself,
`config_reconstructed` is an inference from a size, `config_candidates` is an inference that did not
resolve, and `population` is a fact about where a set came from. Collapsing them would be the
defect.

**The search manufactures provenance if it is not guarded, and it nearly did.** Run without a
declared population it proposed a metabolism val-split cap for the two retrosynthesis artifacts,
whose sizes are reachable by coincidence. A size that a cap can produce is not evidence that a cap
produced it. The population table in the script is the guard, and it is declared from evidence
rather than inferred.

## 3. Every comparison is matched on population, criterion and budget

**Check:** for each comparative claim, state the three settings both sides were computed under. Not
"is the number right" — both numbers are right — but "under which setting of the parameters this
paper itself declares free".

**This is the trigger the other rows do not cover.** Once the manuscript names a parameter free,
every unmatched comparison in it is a self-contradiction, and the reader finds it before the author
does. Known instances: precision@15 quoted beside untruncated output size 81; a full-split reranker
figure labelled n=245; a curation emitting 6.0 compared against a model emitting 10.6.

```bash
python scripts/population_matching_scan.py
```

The scan does not replace the reading, it bounds it. The prose pass reports every paragraph that
compares and names more than one population, which is where an unmatched comparison can hide; three
paragraphs qualify and all three declare their move. The artifact pass is a hard gate: a file that
declares a population must not source a number from one that declares another, and it resolves the
question by reading the referenced file rather than its name, so a file holding several populations
contradicts nobody and a stratified file is not flagged for having strata.

**Status: PASS, after the read-through found one violation and one latent one.** Re-running the row
against the manuscript as it now stands, the provenance appendix declared itself measured *on the
same substrates as the ceiling itself* while reporting the 245-substrate subsample, and the paper's
ceiling is on the 1,170. Every comparison inside that appendix was matched --- all of them on the
subsample --- so the row's own wording passed while the section named the wrong population to the
reader. Underneath it, `provenance_knob_attribution.py` measured its cells on whichever population
it was given and read its endpoints from a file named for a different one, writing two populations
into one artifact; two further endpoints were frozen literals with no population at all. The script
now measures every endpoint on the population it is asked for, and the appendix reports the split.

**The count-based version of this check would not have caught it.** The copied endpoints were bare
floats carrying no sample size, so a scan comparing the `n` values inside a file finds nothing here
and flags every stratified artifact instead --- eighteen of them, all correct by construction. What
made the defect visible was the provenance string in the config naming its source file. A check
that fires on the wrong things and stays silent on the right one is worse than no check, because it
is the one that gets ignored.

## 3a. A reordering is a description; only an interval makes it a claim

**Check:** wherever the paper counts orderings that change, ask what estimand the count belongs to
and whether that estimand was tested. A count of reorderings is a summary of the data, not evidence
that anything reorders.

**This row exists because the population axis failed it.** The paper reported that changing only
which substrates are in reorders 10 of 60 comparisons on our split, and offered that as one of four
headline choices. It carried no interval, and the reason it carried none is structural: the smaller
population is nested inside the larger, so the two gaps are dependent and their difference has no
honest interval. The fix is the complement --- what the larger population has and the subset does
not --- which is disjoint from the subset and admits one.

Tested that way, in both instances, the axis produces nothing. On our split none of the 60
interactions has an interval excluding zero. On eleven released retrosynthesis files, where two
clusters share 490 reactions nested inside each cluster's five thousand, 44 of 380 orderings change
and 18 have intervals excluding zero --- about what 380 tests give by chance --- and none survives
Holm. The paper now reports the population as undeclared and ambiguous, and as not shown to move a
ranking, which is a different and smaller claim than the one it made before.

**Status: PASS after the correction.** The criterion axis passes the same test, which is what makes
the contrast worth stating: 109 of its 448 interactions survive the same correction under the same
estimator.

```bash
python -c "
import json
for f in ('population_axis','retro_population_axis'):
    d = json.load(open(f'results/{f}.json'))
    print(f, 'reordered', d['reordered'], 'of', d['comparisons'],
          '| Holm survivors', d['holm_survivors'])"
```

## 3b. The manuscript reads as a paper, not as a record of how it was written

**Check:** the prose carries no trace of its own production. No file names or code paths, no address
to a reader, no deferral to later work, no sentence that recounts what an earlier version said, and
no construction that marks machine drafting. The abstract states qualitative conclusions and carries
no figures; the introduction enters the field before it enters the method and ends with a roadmap;
related work describes prior work and names the present study only in its final paragraph; captions
describe what is shown.

```bash
python - <<'EOF'
import pathlib, re
PAT = {
 "code path": r"\\texttt\{[^}]*\.(?:py|json|txt|md|sh|csv)\}|scripts/|docs/|results/",
 "reader address": r"\b(?:a reader|the reader|referee|reviewer)\b",
 "deferral": r"\b(?:future work|we leave|will be addressed)\b",
 "hedge": r"\b(?:we do not claim|the honest|caveat travels|cannot sign|worth stating)\b",
 "em dash": r"---",
}
for f in [pathlib.Path("paper/grail_iclr.tex")] + sorted(pathlib.Path("paper/app").glob("*.tex")):
    t = f.read_text()
    hits = {k: len(re.findall(v, t, re.I)) for k, v in PAT.items()}
    if any(hits.values()):
        print(f.name, {k: n for k, n in hits.items() if n})
EOF
```

**Status: PASS.** The document holds no match on any of the five patterns. The abstract and the
introduction contain no numeric expression at all.

**Two failures are recorded because the checks that caught them were added afterwards.** A pass that
varied one repeated construction computed replacement offsets against a string it was mutating and
corrupted sixty sites, each losing the last letter of the preceding word; LaTeX compiled and all
number checks passed throughout, because the damage fell entirely inside prose, and only a targeted
scan found it. Separately, the fourth of the four choices the paper names was asserted in the
abstract, the introduction, the contributions and the limitations while its measurement lived only
in an appendix, which reviewers are under no obligation to read. Neither failure is visible to a
check that reads numbers.

## 4. Every comparative claim carries an interval on the *difference*

**Check:** marginal intervals overlapping says nothing about the paired difference in either
direction — a certified paired gap can hide behind overlapping marginals (it did: +0.123
[+0.014,+0.245]). Bootstrap the difference, paired on the substrate.

```bash
python - <<'PY'
import re, pathlib
txt = re.sub(r"(?m)^\s*%.*$", "", pathlib.Path("paper/grail_iclr.tex").read_text())
body = txt[txt.find(r"\begin{abstract}"):txt.find(r"\subsubsection*{Reproducibility")]
COMP = re.compile(r"\b(above|below|leads?|behind|ahead|erases?|drops?|higher|lower|beats?|exceeds?)\b", re.I)
for s in re.split(r"(?<=[.;])\s+", body):
    s = " ".join(s.split())
    if COMP.search(s) and re.search(r"\$[-+]?[01]?\.\d+\$", s) and not re.search(r"\[[^\]]*,[^\]]*\]", s):
        print("-", s[:180])
PY
```

**Status: PASS as of 2026-07-29 — both open items computed and both survive.** Seven sentences
flag; five are fine (they state `n.s.` or `certified`, or are table fragments split by the scanner).
The two genuine ones carried no interval anywhere in the manuscript and now do:

- coverage ceiling **0.542 against 0.735** → paired **−0.193 [−0.213, −0.175]**
  (`scripts/ceiling_gap_ci.py`). Read off artifacts that already existed; nothing was re-derived.
  The paired interval is 0.038 wide against marginals of 0.053 and 0.051 — the substrate difficulty
  both banks face is shared, and pairing removes it.
- learned filter **0.413 against the prior's 0.374** → paired **+0.039 [+0.007, +0.072]**
  (`scripts/filter_vs_prior_ci.py`). Clears zero, and not by much. The two arms order one identical
  pool, so they can differ only where the pool exceeds the fifteen-candidate budget: the filter is
  ahead on 22 substrates, behind on 10, and tied on 213. Both facts belong next to the claim.

Each script gates before it reports: every arm must reproduce its published marginal, and the
per-substrate vector being bootstrapped must average to what the shipped aggregator returns for the
same rows — otherwise the interval describes a lookalike. The second script's breadth gate
(mean pool must reproduce 107.6) is what caught the wrong substrate set that sent row 2 above.

## 5. Splits are leakage-free and selection never touched test

**Check:** `*_triples_clean.txt` in use (`DatasetConfig.use_clean_splits`); every preset, threshold
and hyperparameter chosen on `evaluate_ensemble_val` / `ensemble_val.f1`.

**Status: PASS.** The one place k was cross-fitted inside test was removed rather than reworded.

## 6. Released artifacts reproduce the deployed model

**Check:** score dump vs deployed ranking, via `factorize_recall.build_deployed_model`.

**Status: PASS, 1170/1170.** This check caught three silent divergences (generator normalization,
generator threshold, calibrated thresholds from the payload) that no other check would have.

## 7. Anonymised for double-blind

Scan **everything tracked**, not the manuscript. Anonymity leaks through paths, not prose: the
`config` blocks row 2 asks for record checkpoint locations, and an absolute one names the author's
home directory in a committed artifact.

```bash
git ls-files -z | xargs -0 grep -lIE "$(printf '%s|%s|%s|%s' '<author-surname>' '@gm''ail' '/Us''ers/' '/ho''me/[a-z]')" 2>/dev/null
```

**Status: PASS as of 2026-07-31, after fixing 14 files across two rounds.** The manuscript was
always clean. The first round found ten tracked files naming the author's home directory: five
result artifacts recording absolute checkpoint paths — including `factorized_eval.json`, cited in
row 2 as the good example of config recording, and `filter_vs_prior_ci.json`, written the same day
*for* this audit — three scripts with a hard-coded path, and two planning documents.

**The second round found the one that mattered, and only because the pattern widened.** The first
scan grepped the author's surname and home directory. Widening it to any absolute home path and any
mail address turned up `pyproject.toml` carrying `authors = ["<name> <address>"]` — the author's
full name and email, in plain text, in the repository that is the anonymised archive. No amount of
path-stripping would have found it, because it is not a path. Also `results/grail_vs_metatox.json`,
holding a scratch directory whose name embeds the username.

**The check matched itself twice**, once through the pattern in its own command and once through the
prose describing what the pattern found. Both are now written so it does not, which is worth doing
rather than ignoring: a check with a known false positive gets skimmed, and a real hit gets skimmed
with it. That is not hypothetical here --- the third round found
`results/factorized_eval_subset250.json` carrying absolute paths, tracked by a bulk `git add -f`
after the previous round had passed. **Run this row after adding files, not before.**

**The fourth round caught two more, and only because the row was re-run after an add.**
`bank_overlap_sygma.py` and `reach_engine_vs_bank.py` --- the two scripts behind the system-reach
result in \S4, the largest finding in the paper --- each hard-coded
`/<home>/<user>/anaconda3/.../sygma/rules`. Both now resolve the path from the installed package
(`Path(sygma.__file__).parent / "rules"`, overridable by `$SYGMA_RULES`), which fixes two defects at
once: the scripts ran on one machine only, and they named the author. Verified equivalent rather
than assumed: the resolved directory is the same one, yielding the same $148+27=175$ rule lines the
paper cites.

**The rows interact, and this pair inverts.** Row 2 asks artifacts to record where their inputs
live; done naively that is exactly what breaks this row. Neither check catches it alone — row 2 sees
a config block and passes, this row saw only `paper/*.tex` and passed. A check narrower than its
claim, again.

## 8. Main body within the page limit

**Check:** everything through the Conclusion must fit in 9 pages. The ICLR 2026 author guide sets
the submission limit at 9 pages of main text, raised to 10 at camera-ready, and excludes references
and an optional reproducibility or ethics statement from the count. The appendix is unlimited, and
the same guide states that reviewers are not required to read it, which is the reason every
load-bearing claim is stated in the nine pages rather than pointed at.

```bash
pdftotext -layout paper/grail_iclr.pdf - | awk 'BEGIN{RS="\f"} NR==9' | tail -3
```

**Status: PASS.** Conclusion completes on page 9; page 10 opens the reproducibility statement.

## 9. Every citation resolves and supports what is attributed to it

**Status: PASS**, `results/citations_verified.json`. Re-run after any citation is added.

## 10. A re-run reproduces the artifact, not just the point estimate

**Check:** run a script twice and diff. Row 2 asks whether the artifact is *committed*; this asks
whether the committed file is what the code *produces*, which is the paper's actual claim.

```bash
python scripts/bank_overlap_sygma.py && cp results/bank_overlap_sygma.json /tmp/r1.json \
  && python scripts/bank_overlap_sygma.py && diff /tmp/r1.json results/bank_overlap_sygma.json
```

**Status: PASS as of 2026-08-01, after fixing four scripts.** Found by accident, which is the point:
a stray invocation rewrote `bank_overlap_sygma.json` and git showed the interval moving from
`[0.1512,0.2302]` to `[0.1510,0.2312]` while the point estimate reproduced to four decimals.

The cause is not randomness — the seed is fixed. `imap_unordered` returns rows in completion order,
so each run bootstrapped a differently ordered array. A sum over rows is order-invariant, which is
why every point estimate was exact; resampling row *indices* is not. `bank_overlap_sygma.py`,
`ceiling_by_provenance.py`, `decompose_biotransformer.py` and `reach_engine_vs_bank.py` now sort
before resampling. `sygma_depth_matched_reach.py` looked identical and is not affected: it keys into
a dict and iterates a fixed list.

Two published intervals moved and were corrected to the deterministic values: the engine effect
$+0.196$ went from $[+0.156,+0.239]$ to $[+0.155,+0.240]$, and the provenance table's three.

**No point estimate in the paper changed**, which is exactly why nothing else caught this. Rows 2
and 4 both pass on an artifact whose interval is not reproducible: one asks whether the file is
committed, the other whether the difference carries an interval at all. A check narrower than its
claim, for the fourth time in this document.

## 10a. Every artifact identifies code that still exists

**Check:** each artifact records the commit it was produced at. Ask git whether that commit is in
this history, and whether the tree was clean when it ran.

```bash
python - <<'EOF'
import json, pathlib, subprocess
for p in sorted(pathlib.Path("results").glob("*.json")):
    try: c = json.loads(p.read_text()).get("config", {})
    except Exception: continue
    if not isinstance(c, dict) or "git_commit" not in c: continue
    sha = c["git_commit"]
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                         capture_output=True).returncode == 0
    if not anc or c.get("git_dirty"):
        print(f"{p.name:52} {sha[:8]} ancestor={anc} dirty={c.get('git_dirty')}")
EOF
```

**Status: PARTIAL, and the partial part is stated rather than repaired.** Of 51 artifacts recording
a commit, one named a commit reachable from no branch: `decompose_biotransformer.json`, produced at
`022781d`, which a later amend removed from the history. The code that made an appendix table was
therefore not in the release. Re-run from the current tree it reproduces every value outside the
config block exactly, so nothing was wrong with the number; the artifact now records a commit that
exists.

Forty-four of the fifty-one record `git_dirty: true`. That flag is set by any uncommitted change in
the tree, including to an unrelated artifact being written by another run, so it is a weak signal and
most of these are benign. It is not nothing: a dirty flag means the recorded commit is a lower bound
on the code that ran, not an identification of it, and this document should not claim more than that.
What holds the numbers is not the commit field but Row 10, which asks whether a re-run reproduces the
file, and the widening to the full split re-ran a third of these under that question.

## 10b. A measured difference is between two procedures, and both have to be complete

**Check:** when a result is a difference between two ways of doing something, ask whether the losing
arm is a defensible alternative or an unfinished version of the winning one. A comparison against an
incomplete procedure measures the incompleteness.

```bash
python - <<'EOF'
from rdkit import Chem
from rdkit.Chem import rdChemReactions
sub = Chem.AddHs(Chem.MolFromSmiles("O=C(O)c1ccccc1"))
rxn = rdChemReactions.ReactionFromSmarts("[cH1:1]>>[c:1]O")
for label, repair in (("as the loop stands", False), ("with RemoveHs first", True)):
    ok = 0
    for tup in rxn.RunReactants((sub,)):
        for p in tup:
            try:
                q = Chem.RemoveHs(Chem.Mol(p), sanitize=False) if repair else Chem.Mol(p)
                Chem.SanitizeMol(q); ok += 1
            except Exception:
                pass
    print(f"{label}: {ok} of 5 products sanitise")
EOF
```

**Status: FAIL, found by an area chair and confirmed here.** Section 4 reports that the same rules
through two engines differ by $+0.188$ of reach and calls the difference a hydrogen convention,
concluding that "the template is not wrong and the engine is not wrong". The expanding arm never
contracts the product. One call, `RemoveHs(sanitize=False)` before sanitisation, takes the worked
example from 0 of 5 products sanitising to 5 of 5, and the three it then yields are byte-for-byte
the three the unexpanded arm yields. At 300 rules against five substrates the share that sanitises
goes from 7.2% to 67.5%, and 88% of firings give exactly the unexpanded arm's products.

The engine axis is therefore a comparison between a complete loop and an incomplete one until
measured otherwise. The re-measurement is running on the full split; whichever way it lands, the
mechanism sentence is wrong as written, because the engine is missing a step.

**What this does not touch:** the census over six libraries, the construct taxonomy, the
transcription control, the criterion axis, the budget axis, the population null, the decomposition
and the coverage ceiling are measured elsewhere and do not pass through this loop.

**The general form.** This document already records gates that certify against a frozen literal, a
value from the wrong population, and a re-derivation of the manuscript's own arithmetic. This is the
fourth: a measurement whose control arm is not the alternative it is named after. None of the four is
caught by asking whether a number matches its artifact, which is what every check here does.

## 11. Every numeric passage traces back to the artifact that produced it

**Check:** walk each passage of the manuscript that prints four or more numerals back to the
artifact behind it, by opening candidate files rather than inferring from names. Rows 2 and 10 ask
about the *script's* outputs; this asks about the *paper's* numbers, and the two are not the same
question.

**This cannot be automated, and the failed attempts are worth recording so nobody rebuilds them.**
The obvious check — collect the manuscript's numerals, look each one up across all artifacts —
is vacuous. There are about 288,000 distinct values across 229 committed artifacts, so essentially
every three-decimal number in [0,1] matches something. Run against the manuscript as it stood
before the budget family was computed, it reported **zero** unmatched: all fourteen genuinely
missing figures matched some unrelated artifact by coincidence. The second attempt asked whether
one artifact holds a passage's numbers *together*, on the theory that coincidences do not
co-occur. Also vacuous: a single 2,717-value artifact covers all thirteen numerals of the budget
passage both before and after the family is stripped. Both versions were deleted rather than
shipped green.

What works is a passage-by-passage walk with the artifact opened and confirmed, run as a fan-out
over the 34 qualifying passages with an adversarial second pass over everything reported missing.

**Status: 27 gaps reported, 11 closed, and it corrected three figures.** Two of the 27 are not
gaps on inspection — the tautomer step of $0.038$ and the $0.002$ reversal margin are subtractions
of two levels the artifacts do hold, checkable from the record though not from the page. The
corrections are the argument for the row:

- **Main text**, the pipeline converts $35.4\%$ of its ceiling, not $35.5\%$: the published figure
  was `0.261/0.735` off the *rounded* values, where the artifact fields give `0.354450`.
- **Main text**, a median of five rules per substrate carries a positive label and not "one to
  three", and $0.07\%$ of the label space and not $0.03\%$. Measured on the label cache the
  generator trains against (`scripts/label_density.py`, gated on reproducing the four already
  published per-rule counts): mean $11.3$, median $5$, only $27.7\%$ of substrates in the
  one-to-three band. The error ran in the direction that flattered the paper's own framing.
- **Appendix**, a reranking arm rises from $0.388$ to $0.404$, not from $0.413$ to $0.430$: the
  start point was `baseline_broad_filter`, a free-standing field equal to the $n{=}245$ figure,
  while the $+0.0165$ delta it was added to runs between two $n{=}1170$ arms. A number spliced
  across two populations, and the same $0.413$-versus-$0.388$ confusion that had already been
  fixed once in the opposite direction.

Three more passages claimed more than they measured: a factor of $4.6$ described as best-versus-
worst budget when it is best-budget-versus-full-output, a "range" of eight to twelve that spliced
a mean emitted count with a mean pool size, and a trend in reference-set size that is a contrast
between endpoints of a non-monotone series.

**Eleven gaps closed by making the producer record what the paper quotes** — `budget_curves.py` and
`ceiling_by_provenance.py` (row 2's entry), `transfer_confound.py`, which printed its table to
stdout and wrote nothing, and three `artifacts/*/reports/metrics.json` that were simply untracked.

**Two of the four open items closed on 2026-08-05, and neither needed the lost run.** The
cardinality-versus-ranking split is recomputed from `cardinality_crossfit.json`, which holds the
out-of-fold version and says something sharper than the unsourced figures did: predicting the
per-substrate count recovers $-10\%$ of the oracle headroom for GRAIL and $+5\%$ for MetaPredictor,
so for two of three methods it is worse than the best constant. The propensity-weight distribution
is a closed-form function of the per-rule positive counts, the training-set size and two config
constants, so `scripts/propensity_weights.py` recomputes it exactly — range $0.196$ to $1.669$, mean
$1.377$, $81.5\%$ up-weighted, every figure matching what was published — behind a gate that the
per-rule counts reproduce `rule_train_positives.json`. It also shows what the passage had not said:
the up-weighting lands mostly on the 4,271 rules that never carry a positive, which take the
maximum weight.

**The other two are cut rather than carried.** A listwise reranker's held-out scores and oracle
ceiling, and a coarse-vocabulary generator variant with its reachability cap, both needed a training
run whose output was not kept. Both were labelled supporting evidence by the passages that used
them — the reranker paragraph said outright that "the proposition rests on the $n=1{,}170$ intervals
rather than on it" — so the paper loses two robustness asides and gains an unconditional
reproducibility claim. For a paper whose subject is evaluation discipline that is the better trade,
and it is the one a reviewer can check in a minute.

**Row 11 is therefore closed.** Every number in the manuscript regenerates from the committed
record. The right way to keep it closed is to run the passage walk again before submission, since
nothing enforces it: a number added tomorrow will not fail any check in this document.

## 11a. No number in the main text is an orphan

**Check:** the named checks verify that a number matches the artifact it cites. They cannot see a
number that cites nothing. This gate is the complement: every decimal printed in the nine pages must
round to some value recorded in some committed artifact.

```bash
python - <<'EOF'
import pathlib, re, json
s = pathlib.Path("paper/grail_iclr.tex").read_text()
body = re.sub(r"(?m)^\s*%.*$", "", s[s.index("\\section{Introduction}"):s.index("\\appendix")])
nums = sorted({m.group(1) for m in re.finditer(r"\$[+-]?(\d*\.\d+)\$", body)})
vals = set()
def collect(o):
    if isinstance(o, float): vals.add(round(o, 6))
    elif isinstance(o, dict):
        for k, v in o.items():
            if k != "per_substrate": collect(v)
    elif isinstance(o, list):
        for v in o[:4000]: collect(v)
for p in pathlib.Path("results").glob("*.json"):
    try: collect(json.loads(p.read_text()))
    except Exception: pass
bad = [n for n in nums if not any(abs(v - float(n)) <= 0.5 * 10 ** -len(n.split(".")[1]) + 1e-12
                                 for v in vals)]
print(f"{len(bad)} of {len(nums)} orphaned:", bad)
EOF
```

**What it establishes and what it does not.** It is necessary, not sufficient. With tens of thousands
of floats in the artifacts a coincidental match is likely, so a number passing this gate is not
thereby verified against the right source; that is what the named checks do. What it catches is the
case they structurally cannot: a figure that no artifact produced.

**Status: PASS, after it found one.** The engine result was stated against a bank-to-bank gap that
appeared in no artifact, in no appendix and in no check, inside one of the paper's headline
sentences. The sentence now reports the decomposition that was computed, every term with an
interval, and five checks hold it.

## 12. Every entry point that generates a comparator's predictions applies the same rules

**Check:** for each external tool, list every script that invokes it and diff what each does to the
raw output before scoring. Not "does the tool run" — both paths ran fine for months — but "do the
paths agree on what counts as a prediction".

```bash
grep -ln "to_smiles()" scripts/*.py | while read f; do
  printf "%-44s " "$f"
  grep -q "parent\|_tautomer_inchikey(sub\|!= pk" "$f" && echo "drops the parent" || echo "DOES NOT"
done
```

**Status: PASS as of 2026-08-05, after fixing three of five call sites.** SyGMa's tree is rooted at
the substrate and `to_smiles()` returns it first, scored 1.0. `run_benchmark.sygma_topk` and
`eval_on_gloryx` dropped it; `sygma_fulltest_predictions`, `sygma_depth_matched_reach` and
`reach_engine_vs_bank` did not. So every full-split analysis gave the comparator a guaranteed miss
in its first slot — on 399 of 400 substrates checked — while the subset and external analyses did
not. Two entry points to one tool disagreed for months and nothing failed, because each was
internally consistent.

**What it cost, and what it bought.** One published finding was entirely an artifact of it: the
appendix reported that GRAIL led at $k{=}1$ by $0.191$ and was the better choice when one or two
predictions were wanted. Corrected, the comparator leads there by $0.038$. A second result moved
from certified to not: the recall lead behind the aggregate reversal was $0.026$ $[0.003,0.049]$
and is now $0.023$ with a bound resting on zero, because the parent occupied a slot that a real
candidate now fills. Against that, the budget sweep gained a third distinct ordering, and the fix
made the two paths agree.

**The cost is measured rather than assumed:** exactly 8 of 2,597 references share a tautomer key
with their own substrate and are discarded with it. That is a property of the tautomer-aware
criterion worth knowing on its own — eight annotated metabolites our default criterion cannot
distinguish from no reaction at all.

**Blast radius, checked rather than hoped:** the 150-substrate cache and the external path were
already parent-dropped, so the five-method table, the rank-flip result and every external figure
are untouched. Only full-split analyses moved.

## 13. No check certifies its subject against a value frozen in the checker

**Check:** for every gate that compares a measurement against a reference number, ask where the
reference comes from. A literal in the source is not a reference; it is a snapshot of the answer at
the moment the gate was written, and it keeps passing after its subject has moved.

```bash
grep -rn "^[A-Z_]*\(CEILING\|COMMITTED\|EXPECTED\|TARGET\|BASELINE\)[A-Z_]* *=" scripts/*.py
# and the half the line above misses: a gate whose reference is READ from a neighbouring artifact
grep -ln "raise SystemExit" scripts/*.py | while read f; do
  r=$(grep -o 'results/[a-z_]*\.json' "$f" | sort -u | grep -v recall_factorization | tr '\n' ' ')
  [ -n "$r" ] && printf "%-32s reference may come from: %s\n" "$(basename "$f")" "$r"
done
```

**Status: PASS as of 2026-08-10, after unfreezing four literals of twelve and scoping four gates.** The paragraph above
this one first said "one gate, and the other two confirmed", written before the command below was
run. The command found twelve, and the count was not the only thing wrong with the sentence.

`ceiling_by_provenance.py` gated its rule-subset pass against `CEILING_SUBSET = 0.7284`, and the
gate held while the ceiling it names was corrected to $0.8171$ — passing self-consistently on a
superseded number, which is worse than failing, because a failure is visible and this was a green
tick. It now reads the target out of `results/recall_factorization.json`, restricted to the same
substrates, and reproduces it to four decimals on 245 of 245. Three more held the same superseded
value or an ancestor of it: `ceiling_norm_check.py` gated on `0.7284` and in the wrong convention,
`bank_without_selection.py` and `selection_ablation.py` reported a ceiling beside live numbers with
nothing to catch them, and `make_diagnostic_figures.py` drew one as a line on a figure. All four now
read the artifact.

**What the frozen gate was hiding is the largest single correction in this document.** Re-run in the
convention the deployed generator fires rules in, the appendix's provenance table does not shift, it
inverts: curated $0.660$ / mined $0.328$ becomes curated $0.471$ / mined $0.785$. The published
reading — that the hand-curated fifth earns the ceiling and the mined tail re-derives it — was an
artifact of measuring the rule bank through the data-preparation helper while the system it
describes uses the inference path. Both are correct arithmetic. They differ in a convention neither
subset is asked about.

**The second command exists because the first one's scope was narrower than the row's claim, for the
fifth time in this document.** It greps for literals, and the worst instance was not a literal.
`bank_engine_replication.py` read its ceiling out of `results/reach_engine_vs_bank.json`, which
looks like the responsible thing to do and is not: the value there was $0.7284$, superseded by the
convention correction *and* measured on the other population, and the gate kept passing. In the same
sweep `hydrogen_dispatch.py` was found computing its dispatch arm on whichever population it was
given while reading the two global arms it must beat out of subsample literals — so run on the full
split it reported a residual that was a difference between two populations. Both now measure or read
their reference for the population they are running on. The distinction that matters is not literal
versus lookup; it is whether the reference tracks the thing it certifies.

**Widening the measurements to the full split found two more, and both fired rather than being
spotted.** `engine_knobs.py` compared a full-split run against arms measured on the subsample, and
then required it to reproduce an engine term measured there; each now reads the peer artifact for
its own population and refuses to run when that artifact does not exist, rather than silently
falling back on the one that does. `dispatch_paired_ci.py` asserted a whole-bank pair against a
subset arm. Counting the two above, four gates in this family had been written to compare across
populations — which is the comparison the paper exists to warn about, made by its own checks.

The pattern across all four is worth stating once. A gate is safe when its reference is a *fixed
property* (a rule count, a corpus overlap) or a *published claim*; it is unsafe when the reference is
a measurement that can move, and every unsafe one here moved in the same way — the subject was
re-measured on a new population or convention and the reference was not.

The surviving literals are the legitimate use — a gate asserting a *fixed* property or a *published
claim*, which fires when the measurement moves instead of hiding it. `bank_engine_replication.py`
checks each bank parses to the rule count the paper reports; `external_ceiling_split.py` holds
`COMMITTED_SPLIT = (24, 13)` to check it reconstructed the same parents the overlap audit counted, a
property of two fixed corpora; `ordering_stability.py` holds the pairs whose reversal the paper
certifies; `hydrogen_dispatch.py` holds SyGMa's two arms so the identity map has something to fail
against. The distinction is not literal versus lookup. It is whether the literal is the claim being
tested or a stale copy of the answer.

**Rows 10, 11, 12 and 13 were each added after the defect they name had already shipped, which is
the pattern worth reading: every row here exists because something got through.** Every row passes, and every one of them passes only
because it was run: the previous round closed the subsample-provenance item and found an author name
and email in `pyproject.toml`, and this one --- run because new results were added, not because
anything looked wrong --- found two tracked scripts hard-coding the author's home directory, behind
the paper's largest finding.

The document is subject to its own rule. The paragraph above the row-2 command once read "every
`n=245` artifact fails this" while the command printed 21 files, and the closing line once read "no
open items remain" while row 2 was open. Both were caught by running the check rather than by
rereading the prose, which is the entire argument for keeping the commands in here next to the
claims they support.

**A third variant, found while promoting the reversal into the main text.** The four
recovered-reference counts in that sentence are checked, and the check computes them as the reach
recorded in the artifact, rounded to four decimals, times the reference total: which is exactly how
the sentence was written. Nothing was frozen and nothing was read from the wrong population, yet the
gate tests arithmetic rather than provenance, and it would go on passing if the underlying pass were
re-run and the counts changed, because both sides move together. At this reference total the rounding
is worth a tenth of a reference, so no number was wrong. The measurement now records the integers,
and the check compares the manuscript against a recorded count. The rule the three variants share:
a reference must be produced by the measurement, not reconstructed from the manuscript's own
arithmetic.

---

**A fourth variant, and it is the one the paper's own thesis predicts.** `guaranteed reach` was
defined as the least a bank recovers over the conventions it might be run under, and computed as a
minimum over the arms recorded in the artifact. One of those arms expands the substrate and never
contracts the product. Section 4 calls that a defect in our own loop, in the same section, three
sentences earlier. So the minimum ranged over two conventions and one bug, and reported the bug: for
SyGMa it printed $0.223$ where the conventions alone give $0.485$. Three reviewers found it
independently, by the arithmetic not matching the counts printed beside it.

```bash
python -c "
import json; b=json.load(open('results/hydrogen_dispatch__clean_test.json'))['banks']
for k,v in b.items():
    a=v['global_arms']; print(k, 'arms', a, 'legitimate min',
        min(x for n,x in a.items() if n!='all_explicit' and x is not None))"
```

The gate now refuses the arm by name and demands a completed loop, failing loudly on a bank that
lacks one rather than substituting. The residual that dispatch is measured by had the same defect and
has the same repair. **The rule this adds: a control arm must be a configuration someone might
choose. An unfinished version of the treatment is not a control, and a worst case taken over one
measures the implementation rather than the object.** This is the fourth kind of bad reference the
document records, after the frozen literal, the value from the wrong population, and the
re-derivation of the manuscript's own arithmetic.

**One word, checked mechanically.** The manuscript reserves *certified* for three confirmatory
comparisons and lets *separated* stand for an interval alone. That distinction is worth nothing
unless the text obeys it, so obedience is a gate:

```bash
python scripts/audit_claim_words.py
```

It enumerates every sentence carrying the stronger word, matches each against a declared family or
against a correction the sentence names itself, and exits non-zero otherwise. On the draft that went
to the eighth panel it found twenty-four sentences claiming the word without one, in a paper whose
thesis is that undeclared conventions corrupt comparisons.

**A gate must distinguish a claim that moved from a claim that was cut.** Cutting the body to nine
pages moved several claims into appendices, and five checks failed because they anchored on a
section rather than on a sentence. The repair is not to delete them: each now searches the whole
manuscript and passes silently when the claim is absent, so a moved claim is still checked where it
is made and a dropped one is not a false alarm. A gate that cannot tell the two apart is a gate
someone eventually switches off.

---

**A fifth variant, and it is the mirror of the fourth.** The dispatch instrument compares one arm
against the best of two global settings. The completed loop was added to the global settings and
not to the dispatch arm, which kept sending its templates to the expanded substrate and never
contracting what came back. So a repaired control was being compared against an unrepaired
treatment, and the residual reported that asymmetry rather than the policy. It was visible only
because the contraction itself was corrected: the sign of the residual moved, which is what a
comparison does when only one of its two sides has been fixed.

```bash
grep -n '"dispatch":' scripts/hydrogen_dispatch.py   # must build from the contracted half
```

**The rule: a repair applied to a control must be applied to the treatment in the same commit.**
Fixing one side of a comparison and not the other produces a number that looks like a measurement of
the thing being compared and is a measurement of the repair. The fourth variant was a control that
was an unfinished version of the treatment; this is a treatment left as an unfinished version of
itself. Both are found the same way, by asking of every arm in a comparison what would have to be
true for it to be a configuration someone would choose.

**And a note on how it was found.** Four reviewers reported that Section 4 and its appendix
disagreed about one number. The disagreement was real, and repairing it exposed two further defects
underneath: the contraction that both sides used, and the asymmetry above. A contradiction between
two statements of the same quantity is worth chasing past the point where the two agree, because
what made them disagree is rarely the last thing wrong.
