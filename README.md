# GRAIL: Graph neural networks and Rule-based Approach In drug metaboLism prediction
[![PyPI Version][pypi-image]][pypi-url]

**GRAIL** is an open-source tool for drug metabolism 
prediction, based on SMARTS reaction rules and graph neural 
networks. 

## 1. Installation
### 1.1 From source with **Poetry**
Run `poetry install` from the directory with `pyproject.toml` file
### 1.2 From **Docker** image
Now is not available
### 1.3 From **PyPi**
`pip install grail_metabolism`

**IMPORTANT:** If you are going to run **GRAIL** with **CUDA**,
then after installation run `install.py` script to add 
proper versions of `torch-geometric`, `torch-scatter`
and `torch-sparse` to your environment.

## 2. Quick start

**IMPORTANT:** Due to **RXNMapper** incompatibility with newer
versions of **Python**, use only **Python 3.9 or lower** if you want
to create your own set of transformation rules. All necessary
tools are in `grail.utils.reaction_mapper`

[pypi-image]: https://badge.fury.io/py/grail_metabolism.svg
[pypi-url]: https://pypi.python.org/pypi/grail_metabolism
