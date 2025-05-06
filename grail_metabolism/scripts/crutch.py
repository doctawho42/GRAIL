from tqdm.notebook import tqdm
from rdkit import Chem
from sys import stdout
from rdkit import RDLogger
RDLogger.DisableLog('rdApp*')

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

from grail_metabolism.utils.reaction_mapper import combine_reaction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('rxn', type=str, help='reaction')

args = parser.parse_args()

try:
    sub, met = tuple(map(Chem.MolFromSmiles, args.rxn.split('>>')))
    print(combine_reaction(sub, met), file=stdout)
except ValueError:
    print('', file=stdout)
