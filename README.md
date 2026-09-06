# GRAIL

GRAIL (Graph-scored Rule Application with Inspectable Localisation) predicts the **structures** of
the metabolites a xenobiotic will produce, and says which reaction template produced each one and
which atoms it acted on. It runs in three stages:

1. a multi-label generator scores biotransformation SMIRKS templates against the substrate graph
   and selects which to apply;
2. RDKit applies the selected templates and enumerates candidate products;
3. a binary filter scores each (substrate, product) pair, and the two scores are combined by
   reciprocal rank fusion into the ranking a caller receives.

Every prediction therefore arrives with its provenance: not a score alone but the template and the
site, so a chemist can reject a candidate on mechanistic grounds.

This is research code backing two manuscripts, and it is organised for that: the numbers in those
papers are generated from artifacts, the artifacts record the script and source digest that wrote
them, and a suite of checks refuses a manuscript whose figures have drifted from the code. See
[`scripts/README.md`](scripts/README.md) for how that chain is arranged and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for how to re-run it.

## What is included

- `grail_metabolism.utils.preparation.MolFrame` for dataset assembly from DataFrames, mappings or SDF + triples.
- `grail_metabolism.utils.transform` for molecular graph, pair graph and SMARTS rule graph featurization.
- `grail_metabolism.model.Generator` and `grail_metabolism.model.Filter` for generator and filter stages.
- `grail_metabolism.workflows` for pretrain / train / eval / infer orchestration.
- `grail_metabolism.experiments` for experiment presets and ablations.
- `grail` CLI for quick prediction, preset export and experiment execution.
- Packaged rules/resources in `grail_metabolism/resources/`.
- YAML experiment configs in `configs/`.
- Project structure notes in [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md).
- Curated lightweight notebooks in `examples/notebooks/`.

Large training data is not packaged: the corpus under `grail_metabolism/data/` is external
research data, and it is not redistributed here for the licence reason [`NOTICE.md`](NOTICE.md)
sets out.

## Installation

### Poetry

```bash
poetry install
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[tuning]
```

Important:

- Use `numpy<2` with the RDKit / PyG stack used here.
- If you need Optuna support, install the tuning extra: `pip install .[tuning]`.
- If you need the Streamlit demo, install the app extra: `pip install .[app]`.

## Quick start

```python
import pandas as pd

from grail_metabolism import MolFrame, summon_the_grail

rules = ["[CH2:1][OH:2]>>[CH:1]=[O:2]"]
frame = MolFrame(
    pd.DataFrame(
        [
            {"sub": "CCO", "prod": "CC=O", "real": 1},
            {"sub": "CCO", "prod": "CCO", "real": 0},
        ]
    )
)
frame.full_setup(rules=rules)

model = summon_the_grail(rules)
model.filter.fit(frame, eps=10, nnPU=False, verbose=False)
print(model.generate("CCO", top_k=1))
```

## CLI

Print the bundled rules:

```bash
grail rules
```

Apply a rules file directly:

```bash
grail predict "CCO" --rules my_rules.txt
```

Run a preset experiment:

```bash
grail run-preset paper_minimal_baseline
```

Run a YAML config:

```bash
grail run-config configs/paper_full_ensemble.yaml
```

Run multiple ablations:

```bash
grail ablate paper_no_pretrain paper_filter_graph_only paper_generator_dot
```

Export preset configs:

```bash
grail presets --export-dir configs/generated
```

The `predict` subcommand uses the simple rule engine by default. The experiment-oriented commands use the full workflow shell.

## Experiment presets

Main shipped presets:

- `paper_full_ensemble`
- `paper_no_pretrain`
- `paper_filter_graph_only`
- `paper_filter_morgan_only`
- `paper_filter_single`
- `paper_generator_dot`
- `paper_generator_mlp`
- `paper_filter_gcn`
- `paper_filter_gin`
- `paper_minimal_baseline`

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the intended ablation matrix.

## Data format

`MolFrame` supports:

- `pandas.DataFrame` with columns `sub`, `prod`, `real`
- `dict[str, set[str]]` for positive substrate-product pairs
- `SDF + triples.txt` through `MolFrame.from_file(...)`

Triples format:

```text
id_substrate id_metabolite is_real
```

## Testing

```bash
python -m pytest grail_metabolism/tests -q
```

or, in a clean environment:

```bash
pytest -q
```

Research-shell smoke tests cover:

- public API
- featurization
- CLI basics
- config serialization
- preset export

There is also a compact developer entrypoint:

```bash
make test
make smoke
```

## Packaging

Build source and wheel distributions:

```bash
python -m build
```

Build them from a clean staging directory and validate metadata:

```bash
bash scripts/build_release.sh
```

or:

```bash
make release
```

Useful references:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DATASETS.md](docs/DATASETS.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md)
- [docs/PUBLICATION_GUIDE.md](docs/PUBLICATION_GUIDE.md)
- [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [docs/WORKFLOWS.md](docs/WORKFLOWS.md)

## Where things are

| path | what is in it |
|---|---|
| `grail_metabolism/` | the package: models, featurisation, workflows, config tree, CLI, tests |
| `scripts/` | gates, artifact producers and research probes, mapped in [`scripts/README.md`](scripts/README.md) |
| `results/`, `artifacts/` | the deposited artifacts every published number is drawn from |
| `paper2/`, `paper/` | the two manuscripts, their tables and figures generated from those artifacts |
| `configs/` | YAML experiment configurations |
| `docs/` | architecture, reproduction, and the design notes for the newer model components |

`results/` and `artifacts/` are ignored wholesale and the deposited files are re-included one by
one, so `git status` shows a new artifact that ought to be deposited rather than hiding it;
`scripts/sync_tracked_artifacts.py --check` keeps that list honest.

## Notes on pretrained assets

`PretrainedGrail` will try to load local weights if they exist in the repository, but the package does not ship giant checkpoints by default. For publication, weights should be versioned separately, for example via Zenodo or a model registry.

## Licence

GRAIL is released under the GNU General Public License, version 3 or later. The full text, and the
reason this licence and not a permissive one, are in [`LICENSE`](LICENSE).

The short reason is that the rule bank redistributes other people's templates and their terms
decide ours: 273 of SyGMa's templates are in the bank that ships and SyGMa's distribution states
"GPL".

**The bank that ships is not the bank the paper measures.** The measured bank holds 7,581
templates; 611 of them are present verbatim in BioTransformer's published reaction set. That
distribution is LGPL and grants redistribution with credit, a link to the licence, an indication of
changes and the notices retained, all of which [`NOTICE.md`](NOTICE.md) provides; what it reserves
is *commercial* use or redistribution. Those 611 are dropped from the shipped bank as a courtesy
rather than an obligation, to narrow what a commercial user would have to establish, so
`grail_metabolism/resources/extended_smirks_released.txt` ships with 6,970 and the file
`extended_smirks.txt` is not in the package. The removal costs nothing measurable: every reference
those templates reach on the evaluated test set is reached by another template in the bank.

Dropping them from the bank is not the same as not carrying them, and the repository says which it
means. Four tracked files hold at least one of the 611 -- a trained checkpoint, which persists the
bank it was built against so a loader can refuse a mismatched pair, and three curated collections
the bank was mined from. `scripts/check_no_withheld_templates.py` counts them and refuses if any
document claims otherwise.

[`NOTICE.md`](NOTICE.md) carries the per-rightsholder counts for both banks, the attribution each
rightsholder requires, and what removing each one's templates would cost. `scripts/build_released_bank.py`
builds the released bank and records both digests; `scripts/check_licence_files.py` holds these
files and the package configuration to what the release actually contains.

The annotated corpus is **not** covered by this licence and is not distributed: its sources' terms
do not combine, and the assembly recorded no per-record provenance, so no subset of it can be shown
free of either. `NOTICE.md` states what is released instead.
