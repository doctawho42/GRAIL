from tqdm.auto import tqdm
import pandas as pd

try:
    from rxnmapper import RXNMapper
except ImportError:
    print('ATTENTION: RXNMapper is not installed. Please install it first if you want to create your own set of rules.')

from rdkit import Chem

import warnings
from typing import Iterable, Set, Union

from rdkit.Chem import rdFMCS, rdRascalMCES
warnings.filterwarnings('ignore')
tqdm.pandas()

rxn_mapper = RXNMapper()


def _mask_gen(mol: Chem.Mol, mol_map: Iterable[int], secondary = True):
    mask = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a not in mol_map or b not in mol_map:
            mask.add(a)
            mask.add(b)
    if not secondary:
        return mask
    temp_mask = list(mask)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in temp_mask or b in temp_mask:
            mask.add(a)
            mask.add(b)
    return mask


def _reactor(sub: Chem.Mol, prod: Chem.Mol, valences = True, secondary = True, test_mode = False) -> str:
    common = rdFMCS.FindMCS([sub, prod], matchValences=valences, timeout=10)
    common_struct = Chem.MolFromSmarts(common.smartsString)
    mol1_map = sub.GetSubstructMatch(common_struct)
    mol2_map = prod.GetSubstructMatch(common_struct)
    mask_1, mask_2 = _mask_gen(sub, mol1_map, secondary=secondary), _mask_gen(
        prod, mol2_map, secondary=secondary)
    sub = Chem.MolFragmentToSmarts(sub, atomsToUse=list(mask_1))
    prod = Chem.MolFragmentToSmarts(prod, atomsToUse=list(mask_2))
    unmapped_reaction = f'{sub}>>{prod}'
    if test_mode:
        return unmapped_reaction
    res = rxn_mapper.get_attention_guided_atom_maps([unmapped_reaction])[0]
    return res['mapped_rxn']


def _generate_rxn(mol1: Chem.Mol, mol2: Chem.Mol, secondary = True, test_mode = False) -> str:
    opts = rdRascalMCES.RascalOptions()
    opts.timeout = 10
    res = rdRascalMCES.FindMCES(mol1, mol2, opts)
    mat_1, mat_2 = [x[0] for x in res[0].bondMatches()], [x[1] for x in res[0].bondMatches()]
    at_1, at_2 = [x[0] for x in res[0].atomMatches()], [x[1] for x in res[0].atomMatches()]
    mol1_bonds, mol1_ats, mol2_bonds, mol2_ats = set(), set(), set(), set()
    for bond in mol1.GetBonds():
        if bond.GetIdx() not in mat_1:
            mol1_bonds.add(bond.GetIdx())
            mol1_ats.add(bond.GetBeginAtomIdx())
            mol1_ats.add(bond.GetEndAtomIdx())
    if secondary:
        temp_mask = list(mol1_ats)
        for bond in mol1.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in temp_mask or b in temp_mask:
                mol1_bonds.add(bond.GetIdx())
                mol1_ats.add(a)
                mol1_ats.add(b)
    for bond in mol2.GetBonds():
        if bond.GetIdx() not in mat_2:
            mol2_bonds.add(bond.GetIdx())
            mol2_ats.add(bond.GetBeginAtomIdx())
            mol2_ats.add(bond.GetEndAtomIdx())
    for bond in list(mol1_bonds):
        if bond in mat_1:
            also_reacted = mat_2[mat_1.index(bond)]
            mol2_bonds.add(also_reacted)
            reacted_bond = mol2.GetBondWithIdx(also_reacted)
            a = reacted_bond.GetBeginAtomIdx()
            b = reacted_bond.GetEndAtomIdx()
            mol2_ats.add(a)
            mol2_ats.add(b)
    if len(mol1_ats) != 0:
        for atom in mol1_ats:
            if atom in at_1:
                also_reacted = at_2[at_1.index(atom)]
                mol2_ats.add(also_reacted)
    if len(mol2_ats) != 0:
        for atom in mol2_ats:
            if atom in at_2:
                also_reacted = at_1[at_2.index(atom)]
                mol1_ats.add(also_reacted)
    sub = Chem.MolFragmentToSmarts(mol1,
                                   atomsToUse=list(mol1_ats),
                                   bondsToUse=list(mol1_bonds))
    prod = Chem.MolFragmentToSmarts(mol2,
                                    atomsToUse=list(mol2_ats),
                                    bondsToUse=list(mol2_bonds))
    unmapped_reaction = f'{sub}>>{prod}'
    if test_mode:
        return unmapped_reaction
    res = rxn_mapper.get_attention_guided_atom_maps([unmapped_reaction])[0]
    return res['mapped_rxn']


def combine_reaction(mol1: Chem.Mol, mol2: Chem.Mol, secondary = True, test_mode = False) -> Union[str, None]:
    try:
        rxn = _generate_rxn(mol1, mol2, secondary=secondary, test_mode=test_mode)
    except IndexError:
        rxn = _reactor(mol1, mol2, secondary=secondary, test_mode=test_mode)
    if '.' in rxn:
        sub, prod = rxn.split('>>')
        if '.' in sub:
            sub = f'({sub})'
            prod = f'({prod})'
            return f'{sub}>>{prod}'
    return rxn


def process(data: pd.DataFrame, secondary = True, test_mode = False) -> Set[str]:
    reactions = set()
    flag = False
    if 'sub_mol' in data.columns:
        flag = True
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if flag:
            sub = row['sub_mol']
            prod = row['prod_mol']
        else:
            sub = Chem.MolFromSmiles(row['sub'])
            prod = Chem.MolFromSmiles(row['prod'])
        try:
            reactions.add(combine_reaction(sub, prod, secondary=secondary, test_mode=test_mode))
        except ValueError:
            continue
        except TimeoutError:
            continue
        finally:
            continue
    return reactions

