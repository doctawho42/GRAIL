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

**Check:** the reproducibility statement does **not** count (ICLR author guide, max 1 page);
everything through the Conclusion must fit in 9.

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

---

**No open items remain as of 2026-08-01.** Every row passes, and every one of them passes only
because it was run: the previous round closed the subsample-provenance item and found an author name
and email in `pyproject.toml`, and this one --- run because new results were added, not because
anything looked wrong --- found two tracked scripts hard-coding the author's home directory, behind
the paper's largest finding.

The document is subject to its own rule. The paragraph above the row-2 command once read "every
`n=245` artifact fails this" while the command printed 21 files, and the closing line once read "no
open items remain" while row 2 was open. Both were caught by running the check rather than by
rereading the prose, which is the entire argument for keeping the commands in here next to the
claims they support.
