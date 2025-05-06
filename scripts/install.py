#!/usr/bin/env python
import torch
import os

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

os.system(r'pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
os.system(r'pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
os.system(r'pip install -q git+https://github.com/pyg-team/pytorch_geometric.git')
