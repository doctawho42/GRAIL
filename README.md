# GRAIL

GRAIL (Graph-scored Rule Application with Inspectable Localisation) is a research-oriented package for xenobiotic metabolism prediction. It combines:

1. A multi-label generator that scores biotransformation SMARTS rules for a substrate graph.
2. Rule application with RDKit to enumerate candidate metabolites.
3. A binary filter that ranks substrate-metabolite pairs.

The repository originally mixed models, datasets, notebooks and local virtual environments in one tree. The refactor now turns it into:

- a usable Python package
- a reproducible experiment shell
- a preset-based ablation framework
- a CLI for train / eval / infer
- lightweight notebooks for reviewers and exploratory work

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

Large training artefacts are intentionally not packaged. The 36 GB local dataset in `grail_metabolism/data` is treated as external research data.

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

## Repository hygiene

Tracked source lives in:

- `grail_metabolism/`
- `scripts/`
- `configs/`
- `docs/`

Generated or local-only content is ignored:

- `artifacts/`
- `results/`
- `notebooks/`
- heavyweight files under `grail_metabolism/data/`

Legacy one-off research helpers live under `scripts/legacy/` and are not part of the tested package surface.

## Notes on pretrained assets

`PretrainedGrail` will try to load local weights if they exist in the repository, but the package does not ship giant checkpoints by default. For publication, weights should be versioned separately, for example via Zenodo or a model registry.

## Licence

GRAIL is released under the GNU General Public License, version 3 or later. The full text, and the
reason this licence and not a permissive one, are in [`LICENSE`](LICENSE).

The short reason is that the rule bank redistributes other people's templates and their terms
decide ours: 273 of SyGMa's templates are in the bank that ships and SyGMa's distribution states
"GPL".

**The bank that ships is not the bank the paper measures.** The measured bank holds 7,581
templates; 611 of them are present verbatim in BioTransformer's published reaction set, whose
README requires explicit permission to redistribute, and that permission was not sought. Those are
removed, so `grail_metabolism/resources/extended_smirks_released.txt` ships with 6,970 and
`extended_smirks.txt` is not distributed. The removal costs nothing measurable: every reference
those templates reach on the evaluated test set is reached by another template in the bank.

[`NOTICE.md`](NOTICE.md) carries the per-rightsholder counts for both banks, the attribution each
rightsholder requires, and what removing each one's templates would cost. `scripts/build_released_bank.py`
builds the released bank and records both digests; `scripts/check_licence_files.py` holds these
files and the package configuration to what the release actually contains.

The annotated corpus is **not** covered by this licence and is not distributed: its sources' terms
do not combine, and the assembly recorded no per-record provenance, so no subset of it can be shown
free of either. `NOTICE.md` states what is released instead.
