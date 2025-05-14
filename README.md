# GRAIL: Graph Neural Networks and Rule-based Approach in Drug Metabolism Prediction
[![PyPI Version][pypi-image]][pypi-url]

**GRAIL** is an open-source tool for drug metabolism prediction, combining **SMARTS reaction rules** with **Graph Neural Networks** (GNNs). It is designed for researchers and developers working in cheminformatics, drug discovery, and computational biology.

---

## Key Features
- **Rule-based Predictions**: Leverages SMARTS reaction rules for accurate metabolic predictions.
- **Graph Neural Networks**: Utilizes cutting-edge GNN architectures for enhanced learning from molecular graphs.
- **Flexible Data Handling**: Supports data input from multiple formats, including pandas DataFrames, dictionaries, and SDF files.
- **Customizable Models**: Includes modular components (`Filter`, `Generator`, and others) for flexible model creation and training.
- **Hyperparameter Optimization**: Built-in support for Optuna for efficient hyperparameter tuning.

---

## Table of Contents
1. [Installation](#1-installation)
   - [From Source](#11-from-source-with-poetry)
   - [From PyPI](#12-from-pypi)
2. [Data Availability](#2-data-availability)
3. [Quick Start](#3-quick-start)
4. [Modules Overview](#4-modules-overview)
5. [Usage Examples](#5-usage-examples)

---

## 1. Installation

### 1.1 From Source with **Poetry**
Run the following command in the directory containing the `pyproject.toml` file:
```bash
poetry install
```

### 1.2 From **PyPI**
Install the library directly from PyPI:
```bash
pip install grail_metabolism
```

**IMPORTANT:** If you plan to run **GRAIL** with **CUDA**, execute the `install.py` script post-installation to set up the appropriate `torch-geometric`, `torch-scatter`, and `torch-sparse` versions:
```bash
python install.py
```

---

## 2. Data Availability
The dataset can be downloaded from [Zenodo](https://zenodo.org/records/15392504?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVmNWEwN2QyLWVlZTMtNDk2Ny1hYjg3LWExNDcwMDA5NTEyNSIsImRhdGEiOnt9LCJyYW5kb20iOi...).

**Note:** This dataset is still in draft form and is subject to updates.

---

## 3. Quick Start

**IMPORTANT:** Due to **RXNMapper** incompatibility with newer Python versions, use **Python 3.9 or lower** when creating new transformation rules.

For a quick demonstration of the library's capabilities, refer to the `notebooks/Unit_Tests.ipynb` file.

---

## 4. Modules Overview

### 4.1 MolFrame
The `MolFrame` class handles data preparation and is essential for working with metabolic maps and molecular data. It supports:
- **Initialization**:
  - From `pandas.DataFrame`
  - From dictionaries with metabolic maps
  - From SDF files
- **File Loading**:
  - Use the `MolFrame.from_file` method to load data.
  - Pre-process triples (substrate, metabolite, real_or_not) using `MolFrame.read_triples`.

### 4.2 Models
The `model` module contains key components:
- **Filter**: Implements GNN-based filters for molecular graphs.
- **Generator**: Handles the generation of reaction rules and transformations.

### 4.3 Utilities
- **Preparation**: Prepares and standardizes molecular data for training and evaluation.
- **OptunaWrapper**: Facilitates hyperparameter optimization using the Optuna library.
- **Transform**: Here you can transform molecular data into graphs
- **ReactionMapper**: Make your own set of reaction rules via RXNMapper

#### Graphs' base characteristics:
There are two types of molecular graphs in this library: singlegraphs
(just common molecular graphs) and pairgraphs, in which two molecular graphs (substrate and product)
are merged by those maximum common substructure. 

Node-edge dimensions:
- 16 and 18 in singlegraphs (`from_rdmol`)
- 10 and 6 in PCAed singlegraphs
- 17 and 18 in pairgraphs (`from_pair`)
- 12 and 6 in PCAed pairgraphs

There are also SMARTS reaction graphs (merged by atom mapping) `from_rule`

#### Reaction mapper
- `combine_reaction` to process reaction into SMARTS pattern
- `process` to process the MolFrame by `combine_reaction`
---

## 5. Usage Examples

### Example 1: Loading Data with `MolFrame`
```python
from grail_metabolism.utils.preparation import MolFrame
# Process triples 
# (substrate_index, metabolite_index, real_or_not)
triples = MolFrame.read_triples('triples.txt')
# Initialize from file
mol_frame = MolFrame.from_file('data.sdf', triples)
```

### Example 2: Training filter part
```python
from grail_metabolism.model.filter import Filter
from grail_metabolism.utils.preparation import MolFrame

# Setting hyperparameters
in_channels = ...
edge_dim = ...
arg_vec = [...] # len = 6

# Set up the data
train = MolFrame(...)
test = MolFrame(...)

model = Filter(in_channels, edge_dim, arg_vec, mode=...) # mode: pair or single
model.fit(train)
# Or
train.train_pairs(model, # or train_singles
                  test,
                  lr=...,
                  eps=...,
                  decay=...,
                  verbose=...)
sub = ...
met = ...
model.predict(sub, met)
```

### Example 3: Training the Generator
```python
from grail_metabolism.model.generator import Generator
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.utils.transform import from_rule
from rdkit import Chem

# Setting rule dict
rules = [...]
rule_dict = {rule: from_rule(rule) for rule in rules}
in_channels = ...
edge_dim = ...

# Set up the data
train = MolFrame(...)

model = Generator(rule_dict, in_channels, edge_dim)
model.fit(train)
model.generate(...)
```

### Example 4: Training a Complete Model
```python
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.grail import summon_the_grail

# Initialize model and datasets
rules = ...
model = summon_the_grail(rules, (..., ...), (..., ...))
train_set = MolFrame(...)
test_set = MolFrame(...)

# Train the model
trained_model = model.fit(train_set)

model.generate(...)
```

### Example 4: Hyperparameter Optimization
```python
from grail_metabolism.utils.optuna import OptunaWrapper

# Initialize OptunaWrapper
wrapper = OptunaWrapper(None, mode='pair')
# Or use OptunaWrapper.from_pickle for the ready Optuna study

# Run optimization
wrapper.make_study(train_set, test_set)

# Train optimal model
wrapper.train_on(train_set, test_set)
```

---

## 6. Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description.

[pypi-image]: https://badge.fury.io/py/grail_metabolism.svg
[pypi-url]: https://pypi.python.org/pypi/grail_metabolism
