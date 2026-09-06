# Retraining on the full split, on Kaggle

The released checkpoints saw 5,000 of 9,011 training substrates for eight epochs and their early
stopping never engaged. The manuscript reports that as an underfitting signature rather than a
converged run, and it is the one open question that touches the coverage bound: if the learned
stages are trained to convergence, does the ceiling move? Five seeds at the reported configuration
also give the retraining spread the register's effects should be read against.

This is what a run needs, and what it costs measured rather than guessed.

## Before anything: one decision that is not technical

The corpus is assembled from ChEMBL under a ShareAlike licence and DrugBank under a NonCommercial
one, and the manuscript's availability section explains why a derivative drawing on both cannot
satisfy either. This project does not redistribute it.

Uploading it to Kaggle as a private dataset is not publication, but it is a transfer to a third
party under that party's hosting terms. That is a decision for the authors, not for a script, and
`scripts/pack_for_kaggle.py` writes the archive with the note attached rather than making it.

The same decision was made and executed for Modal in July, where the corpus already sits in the
`grail-data` volume.

## What it costs

The arithmetic below comes from `artifacts/expandedlabels_multiseed_full5000_seed0/reports/runtime.json`,
which is the released configuration on this machine's CPU at 5,000 substrates and eight epochs:

| stage | seconds | per epoch |
|---|---|---|
| generator | 9,291 | 1,161 |
| filter | 3,114 | 389 |
| evaluation | 879 | |
| data preparation | 49 | a cache HIT, not a cache build |

Two things scale it up. The full split is 9,011 substrates, a factor of 1.80. And
`configs/paper_full_converged.yaml` gives each stage a budget of **40** epochs with a patience of
six, in place of the eight the released run had, because the point of the exercise is to let early
stopping engage. So the epoch count is not a constant between the two runs, it is the variable
under test, and a per-seed cost cannot be quoted as a single number:

| what the GPU buys per epoch | generator, 40 epochs | filter, 40 epochs | one seed |
|---|---|---|---|
| 3x over this CPU | 7.7 h | 2.6 h | over the 12 h session cap |
| 6x | 3.9 h | 1.3 h | 5 to 6 h |
| early stopping at ~20 epochs, 6x | 1.9 h | 0.7 h | 3 h |

The `data_prepare_seconds: 49` in that table is the trap: it is the time to READ a cache that
already existed at `artifacts/preprocessed`, not the time to build one. A first Kaggle session pays
the build, and nothing in this repository has ever measured it.

Two constraints follow, and they bind:

- **The session cap is 12 hours.** A batch run that reaches it is killed and its outputs are not
  saved, so a seed that does not finish leaves nothing behind, including the graph cache.
- **The quota is 30 GPU hours a week.** At 6 to 12 hours a seed, five seeds do not fit. Either the
  campaign is three seeds, or the stages are split across sessions, or the epoch budget comes down.
  Decide that against the first seed's measured cost, not against this table.

## Packaging

```bash
python scripts/pack_for_kaggle.py --out ~/kaggle_upload
```

Two archives and a manifest with their digests, so what arrives can be checked against what left:

- `grail-code.tar.gz`, about 1.4 MB. The package, the converged config, the runner and the measured
  rule bank, with the corpus filtered out of `grail_metabolism/data` and only the featurisation
  files kept.
- `grail-corpus.tar.gz`, about 78 MB. The three splits and their clean triples. This is the archive
  the decision above is about.

Upload each as a private Kaggle dataset. Both are already uploaded:

| dataset | holds |
|---|---|
| `polomoshnov/grail-corpus` | the six corpus files, already extracted by Kaggle |
| `polomoshnov/grail-pkg-v2` | the package, extracted, 112 files |

Three traps are worth knowing before making another archive.

Kaggle extracts what it is given and refuses a collision: an archive holding two members at one path
fails to create, and the API reports the upload as successful and then never produces the dataset.
The packer used to add the rule bank twice, once by walking the package it lives in and once by
name, and three uploads failed silently before the web interface printed the reason. The packer now
refuses to write an archive with a duplicate member.

A pickle in the archive fails the same way, silently. The four PCA featurisation files are excluded
for that reason; the converged config runs with `pca: false`, so they are not needed.

The corpus files in a working checkout are symlinks into the main checkout, so the corpus archive is
written with `dereference=True`. Without it the archive arrives as six broken names.

## The notebook

One cell, and the parts that are load-bearing are load-bearing for reasons worth stating.

**The dataset mount path is discovered, not assumed.** Kaggle mounts an attached dataset under
`/kaggle/input/datasets/<owner>/<slug>`, not the `/kaggle/input/<slug>` its documentation suggests.
A hardcoded path cost one session, so the script finds each dataset by a member it is known to hold:

```python
def find(marker):
    for p in pathlib.Path("/kaggle/input").rglob(marker):
        return p.parent
    raise SystemExit(f"REFUSING: no attached dataset holds {marker}")

corpus = find("train_triples_clean.txt")
pkg = find("pyproject.toml")
```

**The dependencies install one at a time, and the package installs without its own list.**
`requirements.txt` pins `rdkit==2022.9.5`, which does not build for Python 3.12; resolving the list
at all therefore fails and takes every other dependency down with it.

```python
for spec in ["numpy<2", "torch-geometric", "rdkit", "pandas", "scikit-learn", "scipy",
             "matplotlib", "tqdm", "pyvis", "Pillow", "PyYAML"]:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", spec])
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"])
```

**The RDKit pin is recorded, not enforced.** It exists because tautomer canonicalisation moves
between releases and the matching key every recall figure is scored under is a tautomer-canonical
InChIKey. It is also unsatisfiable on this platform, so refusing on it would mean never running
this. What the difference costs is measured instead, in `results/rdkit_version_drift.json`: over
3,000 training substrates the standardised structure differs on 3 and the matching key on 1.
`scripts/kaggle_b1.py` writes the version that actually built the graphs into the session report, so
what a checkpoint was trained under is a fact about the artifact. NumPy stays a refusal: the stack
does not run under 2.x at all.

**The package is copied into `/kaggle/working` before anything runs.** The training writes its graph
cache to `artifacts/preprocessed` relative to the working directory, and only `/kaggle/working`
survives as kernel output. Running from the read-only mount would discard the expensive half.

The training itself is one line, and `--dry-run` checks the environment without spending a session:

```python
!python scripts/kaggle_b1.py --seed 0
```

Repeat with `--seed 1` through `--seed 4` in later sessions, quota permitting.

## Bringing the result home

Each session writes `artifacts/paper_full_converged_seed<N>/` with the checkpoints and
`reports/{metrics,generator_training,filter_training}.json`, plus `artifacts/seed<N>_session.json`
recording the exit code, the wall time and the environment. The kernel collects every `.json`,
`.yaml`, `.pt` and `.csv` under `artifacts/` into `/kaggle/working/out`.

```bash
kaggle kernels output polomoshnov/grail-b1-seed0 -p /tmp/seed0
# then place artifacts/paper_full_converged_seed0/ and artifacts/seed0_session.json under artifacts/
python scripts/full_split_retraining.py --seeds 0 --allow-single
```

`scripts/full_split_retraining.py` is the reader. Neither `run_multiseed.py` nor
`multiseed_headline.py` can do this: both TRAIN a spread rather than aggregate finished runs, and a
previous version of this document claimed otherwise. The reader puts the released three-seed spread
against the converged one, reports the difference against both intervals, and answers the question
the manuscript leaves open by reading `early_stopped_epoch` and `epochs_trained` out of the training
reports. It refuses a spread that is not one: seeds under different RDKit versions, seeds whose
configs differ in anything but the seed, a non-zero exit, or a single run quoted with an interval.
`--self-check` exercises all of that on runs synthesised from a real one and needs no GPU data.

What the paper needs from this is two numbers and one check: the spread across seeds at the reported
configuration, and whether the coverage ceiling moves when the stages are trained to convergence. If
it does not move, that is the result and it is worth printing; the limitation section currently says
the question is open, and it should say what was measured instead.
