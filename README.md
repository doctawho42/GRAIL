# GRAIL: Graph neural networks and Rule-based Approach In drug metaboLism prediction
[![PyPI Version][pypi-image]][pypi-url]

**GRAIL** is an open-source tool for drug metabolism 
prediction, based on SMARTS reaction rules and graph neural 
networks. 

## 1. Installation
### 1.1 From source with **Poetry**
Run `poetry install` from the directory with `pyproject.toml` file
### 1.2 From **PyPi**
`pip install grail_metabolism`

**IMPORTANT:** If you are going to run **GRAIL** with **CUDA**,
then after installation run `install.py` script to add 
proper versions of `torch-geometric`, `torch-scatter`
and `torch-sparse` to your environment.

## 2. Data availability
Data can be downloaded from [Zenodo](https://zenodo.org/records/15392504?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVmNWEwN2QyLWVlZTMtNDk2Ny1hYjg3LWExNDcwMDA5NTEyNSIsImRhdGEiOnt9LCJyYW5kb20iOiI1NDc1NDVhMmY5NTc3MzNhNWRiMmI3MjU4NjdiN2ZhZiJ9.ZS9JZ207ZRQ5b1zzvtLAxD71hOmaKIuLMCrhW5gDia1-MGrJJ287LCrVf1yyLQKm0Cr8Ls-L8OQ5HMdHbl_mOA)
draft. ATTENTION: This is not the final version of the dataset.

## 3. Quick start

**IMPORTANT:** Due to **RXNMapper** incompatibility with newer
versions of **Python**, use only **Python 3.9 or lower** if you want
to create your own set of transformation rules. All necessary
tools are in `grail.utils.reaction_mapper`

For a quick start you may look into the `notebooks/Unit_Tests.ipynb`.

### MolFrame
For the data uploading and further usage you should import `grail_metabolism.utils.preparation.MolFrame`.
It has three different variants of initialization: from pandas.DataFrame, from dictionaries with metabolic maps, and from SDF file.
For loading data from file use the `MolFrame.from_file` function, having previously read (substrate, metabolite, real_or_not) triples via `MolFrame.read_triples`.

### Models
In the `model` module you can find all necessary model classes, especially Filter and Generator.

[pypi-image]: https://badge.fury.io/py/grail_metabolism.svg
[pypi-url]: https://pypi.python.org/pypi/grail_metabolism
