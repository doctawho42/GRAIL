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

```bash
python - <<'PY'
import re, pathlib, subprocess
tracked = set(subprocess.run(["git","ls-files","results/"],capture_output=True,text=True).stdout.split())
outs = {"results/" + (m.group(1) or m.group(2))
        for p in pathlib.Path("scripts").glob("*.py")
        for m in re.finditer(r'"results"\s*/\s*"([^"]+)"|results/([A-Za-z0-9_]+\.json)', p.read_text())}
for o in sorted(outs):
    if o not in tracked and pathlib.Path(o).exists():
        print("UNTRACKED", o)
PY
```

**Status: PASS as of 2026-07-29, after fixing 18 violations** — including
`set_metrics_by_criterion.json`, the source of the paper's two certified reversals. Five
regeneration caches stay untracked by intent (`key_tables` 272M, `moses_keys` 431M,
`rule_collapse_cache` 11M, `match_sens_cache`, `metatox_input`): they hold no numbers.

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

**Status: TWO OPEN ITEMS.** Seven sentences flag; five are fine (they state `n.s.` or `certified`,
or are table fragments split by the scanner). Two are genuine and no interval exists anywhere in the
manuscript for either:

- coverage ceiling **0.542 against 0.735** (`grail_iclr.tex:336`, `:529`) — stated as a bare pair;
- learned filter **0.413 against the prior's 0.374** (`grail_iclr.tex:577`, `app/props.tex:179`).

Both are computable from existing artifacts and neither is computed. Decide before submitting:
report the paired interval, or downgrade the wording to a description.

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

**The two open items in row 4 are the only unmet claims.** Everything else on this list is passing
as of 2026-07-29 and was passing only after being checked — three of these rows were failing
silently while the manuscript asserted them.
