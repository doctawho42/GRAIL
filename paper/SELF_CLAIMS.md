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

**Status: PASS as of 2026-07-29, after fixing 18 violations** — including
`set_metrics_by_criterion.json`, the source of the paper's two certified reversals. Five
regeneration caches stay untracked by intent (`key_tables` 272M, `moses_keys` 431M,
`rule_collapse_cache` 11M, `match_sens_cache`, `metatox_input`): they hold no numbers.

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

**Status: 21 artifacts flagged; most are fine and the flag was too blunt.** A subsampled artifact is
unrecoverable only if *nothing else* pins its substrate set. Three cases, and only the first is a
real defect:

- **Pinned by data.** The `n=37` GLORYx ladder is the whole external set; the `n=994` val artifacts
  are the whole clean val split. Nothing to record.
- **Pinned by a committed file.** The whole `n=150` family — match-sensitivity ×4, rank-flip ×2,
  budget-matched frontier — is defined by `artifacts/tier2/substrates.json`, a tracked list of
  exactly those 150, and all three tier2 prediction files share that keyset exactly. `rank_flip_ci.py`
  even carries a guard for this, added after joining a mismatched cache silently moved SyGMa from
  0.514 to 0.286. Recoverable; say so rather than re-run.
- **Pinned by nothing — the real defect.** The `n=245` family: `prior_vs_learned.json` (the
  certified −0.144 the main text cites), `prior_vs_learned_propensity.json`,
  `selection_ablation.json`, `selection_ablation_prior300.json`,
  `selection_ablation_ranksignal.json`.

Recovering the substrate set for row 4 took a search over caps and then inference from a *sibling*
script's default (`prior_vs_learned.py` defaults to `--max-substrates 250`, which yields 245;
`selection_ablation.py` defaults to 200 and produced 245, so it was overridden on a command line
recorded nowhere). The reconstruction was confirmed only because the recomputed marginals landed on
the published values to three decimals, and the pool-breadth gate rejected the first attempt. That
is gates plus luck, not provenance.

**This one cannot be closed retroactively** — the invocations were not recorded, so for most of
these the cap is not recoverable at all, and back-filling a guessed `config` would be worse than the
gap. Two honest options before submitting: re-run the load-bearing subsampled artifacts through
scripts that write a `config` block, or state in the reproducibility statement that subsampled
results are regenerable only via the scripts' defaults. Do not leave it implied. New analysis
scripts write the block; `factorized_eval.json` shows the practice already existed here.

## 3. Every comparison is matched on population, criterion and budget

**Check:** for each comparative claim, state the three settings both sides were computed under. Not
"is the number right" — both numbers are right — but "under which setting of the parameters this
paper itself declares free".

**This is the trigger the other rows do not cover.** Once the manuscript names a parameter free,
every unmatched comparison in it is a self-contradiction, and the reader finds it before the author
does. Known instances: precision@15 quoted beside untruncated output size 81; a full-split reranker
figure labelled n=245; a curation emitting 6.0 compared against a model emitting 10.6.

**Status: PASS for the audited claims.** No automated check exists; this is a read-through.

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

```bash
grep -rniE "<author-surname>|<handle>|@gmail|github\.com/" paper/*.tex paper/app/*.tex
```

**Status: PASS.** Only hits are in the unused ICLR template file `iclr2026_conference.tex`.

## 8. Main body within the page limit

**Check:** the reproducibility statement does **not** count (ICLR author guide, max 1 page);
everything through the Conclusion must fit in 9.

```bash
pdftotext -layout paper/grail_iclr.pdf - | awk 'BEGIN{RS="\f"} NR==9' | tail -3
```

**Status: PASS.** Conclusion completes on page 9; page 10 opens the reproducibility statement.

## 9. Every citation resolves and supports what is attributed to it

**Status: PASS**, `results/citations_verified.json`. Re-run after any citation is added.

---

**One open item remains: the configuration half of row 2**, and it is a disclosure decision rather
than a computation. Every other row is passing as of 2026-07-29, and passing only because it was
checked — three of them were failing silently while the manuscript asserted them.

The document is subject to its own rule. The paragraph above the row-2 command once read "every
`n=245` artifact fails this" while the command printed 21 files, and the closing line once read "no
open items remain" while row 2 was open. Both were caught by running the check rather than by
rereading the prose, which is the entire argument for keeping the commands in here next to the
claims they support.
