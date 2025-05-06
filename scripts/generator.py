#!/usr/bin/env python3

import argparse

import rdkit
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from tqdm import tqdm

from functools import wraps
import errno
import os
from threading import Timer
from itertools import chain
from typing import Union, List
import re

RDLogger.DisableLog('rdApp.*')

epilog = r'''Task file format:

substrates=/path/to/file.smi\n
output=/path/to/file.smi\n
rules=/path/to/file.txt'''

parser = argparse.ArgumentParser(
    prog='generate.py',
    description='Generates metabolites from substrates given in a SMILES file using SMIRKS-rules passed as a list in a text file. Takes task file as input.',
    epilog=epilog)

parser.add_argument('task_file', type=str, help='Path to task file')
args = parser.parse_args()

with open(args.task_file) as task:
    filepaths = [x.split('=')[1].strip() for x in task.readlines()]
    substrates = filepaths[0]
    output = filepaths[1]
    rules = filepaths[2]

with open(rules) as smirks_file:
    smirks = [x.strip() for x in smirks_file.readlines()]

with open(substrates) as subfile:
    subs = [x.rstrip() for x in subfile.readlines()]

uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
te = rdMolStandardize.TautomerEnumerator()  # canonicalize tautomer form

carbon_counter = lambda x: x.count('C') + x.count('c') - x.count('Cl') - x.count(
    'Ca') - x.count('Co') - x.count('Sc') - x.count('Cr') - x.count(
        'Cd') - x.count('Cs')
iscorrect = lambda x: carbon_counter(x) >= 3
atom_counter = lambda x: len(re.findall('((?<=\[)[A-Z][a-z])|((?!\[)[A-GI-Za-z])', x))


def extract(smiles: str) -> Union[str, None]:
    """Extract the biggest correct molecule from a given SMILES string"""
    xtr = max(smiles.split('.'), key=atom_counter)
    if iscorrect(xtr):
        return xtr
    else:
        return None


def timeout(seconds: float = 30, error_message: str = os.strerror(errno.ETIME)):

    def decorator(func):

        @wraps(func)
        def _handle_timeout(*args, **kwargs):

            def _raise_timeout():
                raise TimeoutError

            timer = Timer(seconds, _raise_timeout)
            timer.start()
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                with open('timeout.log', 'a') as logfile:
                    print(*args, sep='\t', file=logfile)
                return None
            finally:
                timer.cancel()
            return result

        return _handle_timeout

    return decorator


@timeout(seconds=30)
def standardize_mol(mol: Union[rdkit.Chem.rdchem.Mol, str], ph: float = None) -> str:
    """Standardize the RDKit molecule, select its parent molecule, uncharge it,
       then enumerate all tautomers."""

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)
    # if many fragments, get the "parent" (the actual mol we are interested in)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    # try to neutralize molecule and make one resonance structure

    # try to uncharge molecule
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    taut_uncharged_parent_clean_mol = te.Canonicalize(
        uncharged_parent_clean_mol)
    assert taut_uncharged_parent_clean_mol is not None

    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol, isomericSmiles=False)


def metaboliser(mol: rdkit.Chem.rdchem.Mol, rules: List[str] = smirks):
    """Applies all known biotransformation rules to the given substrate and returns
    products with corresponding SMIRKS"""

    mets = set()
    if mol is not None:
        mol = Chem.AddHs(mol)
    else:
        return None

    for i, rule in enumerate(rules):
        rxn = AllChem.ReactionFromSmarts(rule)
        try:
            mols_prebuild = list(chain.from_iterable(rxn.RunReactants((mol,))))
        except ValueError:
            continue
        if not mols_prebuild:
            continue
        else:
            mols_splitted = []
            for preb in mols_prebuild:
                mols_splitted += Chem.MolToSmiles(preb).split('.')
            mols_splitted = [x for x in mols_splitted if iscorrect(x)]
            mols_splitted = list(map(Chem.MolFromSmiles, mols_splitted))
            mols_splitted = [x for x in mols_splitted if x is not None]
            if not mols_splitted:
                continue
            try:
                mols_standart = list(map(standardize_mol, mols_splitted))
            except Chem.KekulizeException:
                continue
            except RuntimeError:
                continue
            except Chem.AtomValenceException:
                continue
            for stand in mols_standart:
                if stand != '':
                    mets.add(stand)

    return mets


def process(subs: List[str]) -> None:
    with open(output, 'w') as outfile:
        for sub in tqdm(subs):
            pred_mols = [x for x in metaboliser(Chem.MolFromSmiles(sub)) if x is not None]
            for gen in pred_mols:
                print(sub, gen, sep='\t', file=outfile)


process(subs)
