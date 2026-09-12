# GRAIL deploy build for way2drug — design

**Goal:** A file-to-file command-line executable way2drug can wrap: it reads a file of substrate
SMILES and writes a file of probable metabolites with scores, using the deployed GRAIL model and its
release ranking, and using a rule bank that excludes the BioTransformer-derived templates.

**Architecture:** A thin deploy CLI over the existing model. It loads a matched (bank, checkpoint)
pair, generates candidates for each substrate, ranks them with the paper's release ranking (RRF over
the two component scores, on a pool deduplicated by tautomer key and capped by generator score), and
writes a table. The BioTransformer templates are excluded by shipping the released bank and a
generator checkpoint subset to it — no retraining, and the exclusion is measured to cost zero
references.

## Global constraints

- The deploy uses the **released bank** `grail_metabolism/resources/extended_smirks_released.txt`
  (6970 templates = the full 7581 minus 611 BioTransformer-verbatim templates). It never ships or
  loads `extended_smirks.txt`. Removing the 611 costs **zero references** at the depth-1 ceiling
  (`results/licence_removal_cost__clean_test.json`: whole bank 0.8856, without BioTransformer
  0.8856), because those templates are redundant with the rest of the bank.
- No Claude/AI attribution in any commit or file.
- All code comments and docstrings in English.
- Selection/measurement stays on validation; the comparison/test split is not touched to tune this.
- The deploy must be reproducible from tracked inputs: a clone with the released bank and the released
  checkpoints can build and run it with no access to the full bank or the corpus.

## The checkpoint problem, and its resolution

The deployed generator (`artifacts/full5000_implicit/checkpoints/generator.pt`) was trained on the
full 7581-template bank and uses a **per-template id embedding heavily** (id carries 0.82 of the rule
representation's variance at the deployed id_gate_lambda=0, `results/rule_embed_decomposition.json`).
Its id-embedding table, rule-prior and per-rule bias are therefore indexed to the full bank's rule
order. Building a generator on the 6970-template released bank and loading this state_dict would
mismatch those tensors (7581 rows vs 6970); `strict=False` would silently skip them and leave the id
table untrained, which breaks the model.

The filter (`.../filter.pt`) scores a (substrate, product) graph pair and has **no per-rule
parameters**, so it ships unchanged.

**Resolution — subset, do not retrain.** The released bank is an exact subset of the full bank
(`released ⊂ full`, verified). A one-time build step selects, for each released template, its row in
the full bank's per-rule tensors (id embedding, rule-prior logits, per-rule bias) and writes a
released generator checkpoint whose rule order matches the released bank. The shared, rule-agnostic
parameters (encoder, attention, rule MLP, scales) copy across unchanged. Because the removed
templates are the 611 that cost zero references, the subset generator reproduces the full generator's
recall on the kept chemistry; this is a verification gate, not an assumption.

## Components

### 1. `scripts/build_released_checkpoint.py` (one-time, dev-side)

- Input: the deployed full checkpoint + `extended_smirks.txt` (full) + `extended_smirks_released.txt`
  (released). Build in a checkout that has the full bank.
- Load the full generator (matched to the full bank via its saved `arch`, the `bank_without_selection._load` pattern: `build_generator(GeneratorConfig(**arch), full_rules)`).
- Compute the kept-index map: for each released template, its index in the full bank (order-preserving).
- Subset every per-rule tensor in the generator's state to the kept indices; leave rule-agnostic
  tensors untouched. Persist a released checkpoint carrying `arch` + the **released** rule list, so the
  loader's rule-match check passes against the released bank.
- Verify: load the released checkpoint on the released bank, run the deployed ranking over the
  validation pools, and assert recall@k matches the full checkpoint within a small tolerance (the
  removed templates cost zero references, so a real gap means the subset is misaligned). Fail loudly
  otherwise.
- Output: `artifacts/full5000_released/checkpoints/generator.pt` (+ a copy of the unchanged
  `filter.pt`), tracked and deposited via `sync_tracked_artifacts.py`.

### 2. Release ranking, promoted into the library

The release ranking currently lives only in analysis scripts (`scripts/typed_edit/_rrf.py` and the
`deployed_order` functions). Promote the single fusion implementation and the order of operations
into the shipped package, e.g. `grail_metabolism/model/ranking.py`:

- `reciprocal_rank_fusion(cands, k=60, ...)` — the one implementation of the H7 fusion (1-based
  competition ranks, tied scores share the lower rank), imported rather than re-written.
- `release_order(pool, self_key, cap=100, rrf_k=60)` — dedup by tautomer key in descending
  product(filter*generator) order, cap the survivors at 100 by generator score, fuse by RRF, drop the
  parent key. This is the exact deployed order (H7/H9/P2), and the value the paper reports (0.5353 at
  k=15 on the comparison set) is its output on the full bank; the deploy runs the same order on the
  released bank.
- The analysis scripts are refactored to import from here, so there is one implementation.

### 3. The deploy CLI — `grail_metabolism/deploy/predict_cli.py`, console script `grail-metabolites`

Usage: `grail-metabolites INPUT_SMILES_FILE OUTPUT_TSV [--top-k 15] [--format tsv|sdf] [--timeout-seconds N]`

- **Load** the matched released pair (released bank + released generator + filter) once, via saved
  `arch`. `eval()` mode.
- **Read** the input file: one substrate per line, either `SMILES` or `id<TAB>SMILES`; when no id is
  given, the id is the 1-based line number. Blank lines skipped.
- **Per substrate**: parse; if unparseable, emit a flagged row and continue. Generate candidates with
  the generator (whole released bank applied), score with the filter, rank with `release_order`, take
  the top `--top-k`. Apply a per-substrate wall-clock timeout (`--timeout-seconds`, default 120); on
  timeout emit a flagged row and continue so one large substrate cannot fail the batch.
  `slow_substrates.txt` names the known-slow ones.
- **Write** the output. Default TSV with header `parent_id\trank\tmetabolite_smiles\tscore\tstatus`,
  where `score` is the deployed combined confidence (filter * generator) for that candidate and
  `status` is `ok` / `no_parse` / `timeout` / `no_metabolites`. `--format sdf` writes an SD file with
  the same fields as SD properties.
- Normalisation matches the deployed model (`gen_normalization` the generator was trained under), so
  candidate keys and scores are in the model's distribution.

### 4. Deploy package layout — `deploy/`

- `deploy/README.md` — what the tool is, the exact CLI contract, the input/output formats, an example.
- `deploy/requirements-deploy.txt` — pinned: `numpy<2`, `torch`, and the RDKit pin with the measured
  drift note (`scripts/kaggle_b1.py` records: over 3000 substrates the standardised structure differs
  on 3 and the matching key on 1 under a different RDKit; recorded, not hidden).
- `deploy/environment.yml` — optional conda spec for a self-contained env.
- `deploy/NOTICE.md` — states the released bank excludes the BioTransformer templates, and the
  provenance of the checkpoints and bank.
- The tool references the released bank + released checkpoints from the package; it does not duplicate
  them.

## Data flow

`input.smi` → read (id, SMILES) rows → for each: generator (released bank) → candidates with
generator scores → filter → (filter, generator) per candidate → `release_order` (dedup/cap/RRF) →
top-k → rows (id, rank, smiles, score, status) → `output.tsv`.

## Error handling

- Unparseable SMILES: one row, `status=no_parse`, no metabolites; batch continues.
- Per-substrate timeout: `status=timeout`; batch continues. Known-slow substrates
  (`slow_substrates.txt`) are still attempted but bounded by the timeout.
- Substrate with no applicable rules / no candidates: `status=no_metabolites`.
- Missing bank or checkpoint at startup: fail fast with a clear message naming the missing file.
- The output file is written atomically (temp file then rename) so a partial run does not leave a
  half-written file way2drug might read.

## Testing

- `build_released_checkpoint.py`: a test that the kept-index map is order-preserving and that a
  round-trip (subset then load on the released bank) yields a generator whose per-rule tensors equal
  the full generator's rows for the kept rules. The recall-parity gate is part of the build itself.
- `ranking.py`: a guard test that `release_order` reproduces the current `deployed_order` output on a
  small fixed pool (byte-identical), so promoting it into the library changes nothing.
- CLI: a smoke test on a tiny SMILES file (2-3 substrates incl. one unparseable) asserting the output
  has the expected header, one block per input, correct `status` values, and ranks contiguous from 1;
  runs without the corpus, on the released pair.
- `make test` stays green; new guards live beside the existing ones.

## Performance / operating point

The release ranking uses tautomer-canonical normalisation, which is the dominant inference cost
(measured at 94-99% of generation time on the full pipeline). Latency per substrate will be measured
during implementation on a handful of substrates and reported in the README; `--top-k` bounds the
filter's work. Default `--top-k 15` matches the release budget. If latency is prohibitive for
way2drug's interactive use, a lower default or a faster normalisation path is a follow-up, recorded
rather than silently chosen.

## Decisions resolved

- Ranking: the paper's release RRF+cap+dedup, promoted into the library (one implementation).
- Output: TSV `parent_id, rank, metabolite_smiles, score, status` by default; SDF via `--format`.
- Budget: default `--top-k 15`, exposed as a flag.
- BioTransformer: excluded by shipping the released bank + a subset generator checkpoint; zero
  measured coverage cost; nothing BioTransformer-derived is shipped.
- Env: pinned `requirements-deploy.txt` + README; optional conda env; way2drug runs it in its own env.

## Out of scope

- Multi-step (depth>1) metabolism: the deploy is single-step, matching the release.
- Retraining a released-bank model: unnecessary given the subset reproduces recall at zero cost.
- A persistent service/API: the chosen form is a file-to-file CLI; a service is a later, separate
  request.
