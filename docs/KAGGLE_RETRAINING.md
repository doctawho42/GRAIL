# Retraining on the full split, on Kaggle

The released checkpoints saw 5,000 of 9,011 training substrates for eight epochs and their early
stopping never engaged. The manuscript reports that as an underfitting signature rather than a
converged run, and it is the one open question that touches the coverage bound: if the learned
stages are trained to convergence, does the ceiling move? Five seeds at the reported configuration
also give the retraining spread the register's effects should be read against.

This is what a run needs. It is written down rather than done here because two of the steps are the
author's to take.

## Before anything: one decision that is not technical

The corpus is assembled from ChEMBL under a ShareAlike licence and DrugBank under a NonCommercial
one, and the manuscript's availability section explains why a derivative drawing on both cannot
satisfy either. This project does not redistribute it.

Uploading it to Kaggle as a private dataset is not publication, but it is a transfer to a third
party under that party's hosting terms. That is a decision for the authors, not for a script, and
`scripts/pack_for_kaggle.py` writes the archive with the note attached rather than making it.

The same decision was made and executed for Modal in July, where the corpus already sits in the
`grail-data` volume. If Modal becomes available again it needs no new decision.

## What it costs

Measured, not estimated: the released run took 7,021 s for the generator and 2,228 s for the filter
on this machine's CPU, at 5,000 substrates and eight epochs. The full split is 1.80 times the
substrates, and convergence with a patience of six should take fifteen to twenty-five epochs rather
than eight.

| | one seed | five seeds |
|---|---|---|
| a laptop CPU | 9 to 14 h | 43 to 72 h |
| one Kaggle GPU session | 1 to 3 h | 5 to 15 h |

Kaggle gives 30 GPU hours a week and stops a session at twelve, so a seed fits inside a session with
room and the campaign fits inside a week's quota. The generator and the filter both select
`cuda if available`, so nothing in the code needs changing.

One caveat on the first session: the graph cache is built from scratch and that is the slow part.
Write it to the working directory and attach it to later sessions as a dataset rather than rebuilding
it five times.

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
| `polomoshnov/grail-pkg-v2` | the package, extracted, 88 modules |

One thing about the archive is worth knowing before making another. Kaggle extracts what it is
given and refuses a collision: an archive holding two members at one path fails to create, and the
API reports the upload as successful and then never produces the dataset. The packer used to add the
rule bank twice, once by walking the package it lives in and once by name, and three uploads failed
silently before the web interface finally printed the reason. The packer now refuses to write an
archive with a duplicate member rather than leaving that to be discovered.

## The notebook

Three cells. The first installs the pins, and they are not optional:

```python
!pip install -q "numpy<2" "rdkit==2022.9.5" torch-geometric
```

`rdkit==2022.9.5` is exact because tautomer canonicalisation is not stable across releases and the
matching key every recall figure is scored under is a tautomer-canonical InChIKey. A model trained
on graphs a different RDKit built is not the model this paper reports. `numpy<2` is what the stack
pins.

The second puts the package where Python can import it and the corpus where the loader looks. Kaggle
has already extracted both datasets, so nothing is unpacked here: the corpus files are linked rather
than copied, because they are 1.4 GB and the working directory is not.

```python
import os, shutil, pathlib
shutil.copytree("/kaggle/input/grail-pkg-v2", "/kaggle/working/grail", dirs_exist_ok=True)
os.chdir("/kaggle/working/grail")
data = pathlib.Path("grail_metabolism/data")
for f in ["train.sdf", "val.sdf", "test.sdf", "train_triples_clean.txt",
          "val_triples_clean.txt", "test_triples_clean.txt"]:
    dst = data / f
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(f"/kaggle/input/grail-corpus/{f}")
!pip install -q -e .
```

The third runs one seed:

```python
!python scripts/kaggle_b1.py --seed 0
```

`kaggle_b1.py` checks the environment before it trains and refuses rather than producing a
checkpoint nobody can compare: wrong RDKit, wrong NumPy major, and it says plainly when no GPU is
visible. Add `--dry-run` to check the environment without spending the session.

Repeat with `--seed 1` through `--seed 4` in later sessions.

## Bringing the result home

Each session writes `artifacts/paper_full_converged_seed<N>/` with the checkpoints and
`reports/metrics.json`, plus `seed<N>_session.json` recording the exit code and the wall time.
Download those directories, put them under `artifacts/`, and the multiseed producer reads them the
way it reads the existing spread study.

What the paper needs from this is two numbers and one check: the spread across seeds at the reported
configuration, and whether the coverage ceiling moves when the stages are trained to convergence. If
it does not move, that is the result and it is worth printing; the limitation section currently says
the question is open, and it should say what was measured instead.
