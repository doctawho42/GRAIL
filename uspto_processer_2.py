#!/usr/bin/env python
import pandas as pd
from rdkit import Chem

from grail.utils.preparation import extract, standardize_mol

from tqdm.auto import tqdm
import warnings
from typing import Union
import pickle as pkl
warnings.filterwarnings('ignore')
tqdm.pandas()

uspto = pd.read_csv('grail/data/USPTO_FULL.csv')
reactions = uspto['reactions'].str.split('>').apply(lambda x: (x[0], x[2]))
uspto = pd.DataFrame(reactions.to_list(), columns=['sub', 'prod'])

print('USPTO loaded')

print('\nMol objects generation\n')

uspto['sub_mol'] = uspto['sub'].progress_apply(Chem.MolFromSmiles)
uspto['prod_mol'] = uspto['prod'].progress_apply(Chem.MolFromSmiles)
uspto.dropna(subset=['sub_mol', 'prod_mol'], inplace=True)

print('\nStandardize molecules\n')

def standardize(x: Chem.Mol) -> Union[None, str]:
    try:
        return standardize_mol(x)
    except RuntimeError:
        return None
    except Chem.rdchem.KekulizeException:
        return None


import gc
gc.collect()

uspto['prod'] = uspto['prod_mol'].progress_apply(standardize)
gc.collect()

pkl.dump(uspto['prod'], open('prod.pkl', 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
print('Done')
