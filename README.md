# GRAIL: Graph-based drug metabolism Reaction prediction via Attentive Interaction Layers
[![PyPI Version][pypi-image]][pypi-url]

**GRAIL** is an open-source tool for drug metabolism 
prediction, based on transformers and graph neural 
networks. 

## 1. Installation
### 1.1 From source with **Poetry**
### 1.2 From **Docker** image
### 1.3 From **PyPi**
`pip install grail`

**IMPORTANT:** If you are going to run **GRAIL** with **CUDA**,
then after installation run `install.py` script to add 
proper versions of `torch-geometric`, `torch-scatter`
and `torch-sparse` to your environment.

## 2. Quick start

**IMPORTANT:** Due to **RXNMapper** incompatibility with newer
versions of **Python**, use only **Python 3.9 or lower** if you want
to create your own set of transformation rules. All necessary
tools are in `grail.utils.reaction_mapper`

[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric

