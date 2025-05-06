#!/usr/bin/env python
import pandas as pd
from rdkit import Chem

from grail_metabolism.utils.preparation import extract, standardize_mol

from tqdm.auto import tqdm
import warnings
from typing import Union
import pickle as pkl
warnings.filterwarnings('ignore')
tqdm.pandas()

uspto = pd.read_csv('grail_metabolism/data/USPTO_FULL.csv')
reactions = uspto['reactions'].str.split('>').apply(lambda x: (x[0], x[2]))
uspto = pd.DataFrame(reactions.to_list(), columns=['sub', 'prod'])

uspto['sub'] = uspto['sub'].progress_apply(extract)
uspto['prod'] = uspto['prod'].progress_apply(extract)
uspto.dropna(subset=['sub', 'prod'], inplace=True)

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

uspto['sub'] = uspto['sub_mol'].progress_apply(standardize)
gc.collect()

pkl.dump(uspto['sub'], open('sub.pkl', 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
print('Done')
