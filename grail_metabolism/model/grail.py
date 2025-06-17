from grail_metabolism.utils.transform import from_rule
import typing as tp
from .generator import Generator
from .filter import Filter
from .wrapper import ModelWrapper, SimpleGenerator
from pathlib import Path
import base64
from PIL import Image
from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch


def summon_the_grail(rules: tp.List[str], node_dim: tp.Tuple[int, int], edge_dim: tp.Tuple[int, int]) -> ModelWrapper:
    r"""
    Create the ready-to-use model with generator and filter parts
    :param rules: list of SMARTS rules to use
    :param node_dim: a tuple with two integers: (node_dim_generator, node_dim_filter)
    :param edge_dim: a tuple with two integers: (edge_dim_generator, edge_dim_filter)
    :return: ready-to-use model
    """
    rule_dict = {}
    for rule in rules:
        rule_dict[rule] = from_rule(rule)
    generator = Generator(rule_dict, node_dim[0], edge_dim[0])
    arg_vec = [400] * 6
    filter = Filter(node_dim[1], edge_dim[1], arg_vec, mode='pair')
    return ModelWrapper(filter, generator)

class PretrainedGrail(ModelWrapper):
    r"""
    Get the full pretrained GRAIL model
    Data paths are grail_metabolism/data/*.pth
    """
    def __init__(self):
        gen = torch.load(Path(__file__).parent / '..' / 'data' / 'best_generator.pth')
        filt = torch.load(Path(__file__).parent / '..' / 'data' / 'best_filter_pair.pth')
        with open(Path(__file__).parent / '..' / 'data' / 'merged_smirks.txt') as file:
            rules = file.read().splitlines()
        rule_dict = {rule: from_rule(rule) for rule in rules}
        generator = Generator(rule_dict, 10, 6, arg_vec=[785, 519])
        generator.load_state_dict(gen)
        filter = Filter(12, 6, [500]*6, 'pair')
        filter.load_state_dict(filt)
        super().__init__(filter, generator)

    def draw(self, substrate_smiles: str) -> None:
        def generate_molecule_image(mol):
            drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            image_data = drawer.GetDrawingText()
            return base64.b64encode(image_data).decode("utf-8")

        prods = set(self.generate(substrate_smiles))
        net = Network(height="800px", width="1200px", notebook=True)
        # Load RDKit molecules
        sub = Chem.MolFromSmiles(substrate_smiles)
        mol1_image_base64 = generate_molecule_image(sub)
        net.add_node(1,
                     shape="circularImage",
                     image=f"data:image/png;base64,{mol1_image_base64}",
                     color="red",
                     borderWidth=3)
        for i, prod in enumerate(prods):
            mol2 = Chem.MolFromSmiles(prod)
            # Generate images of molecules
            mol2_image_base64 = generate_molecule_image(mol2)
            # Add molecules to network
            # - In this network molecules are indexed by indices 1 and 2. Index can also be a string.
            net.add_node(i,
                         shape="circularImage",
                         image=f"data:image/png;base64,{mol2_image_base64}",
                         color="blue",
                         borderWidth=3)
            # Make edge between molecules
            # - Edge is formed between nodes of index 1 and 2.
            net.add_edge(1, i+1, color="purple")
        # Visualize the network
        net.show("network.html")

