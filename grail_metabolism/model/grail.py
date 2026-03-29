from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

from .filter import Filter
from .generator import Generator
from .wrapper import ModelWrapper
from ..utils.preparation import load_default_rules
from ..utils.transform import from_rule

RDLogger.DisableLog("rdApp.*")


def _load_checkpoint_payload(module, path: Path) -> bool:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(state, dict):
        return False
    if "state_dict" in state:
        module.load_state_dict(state["state_dict"], strict=False)
        if "calibrated_threshold" in state:
            module.calibrated_threshold = state["calibrated_threshold"]
        return True
    module.load_state_dict(state, strict=False)
    return True


def summon_the_grail(
    rules: Sequence[str],
    node_dim: Tuple[int, int] = (16, 18),
    edge_dim: Tuple[int, int] = (18, 18),
) -> ModelWrapper:
    rule_dict = {rule: from_rule(rule) for rule in rules}
    generator = Generator(rule_dict, node_dim[0], edge_dim[0])
    filter_model = Filter(node_dim[1], edge_dim[1], [128, 256, 128, 128, 64, 32], mode="pair")
    return ModelWrapper(filter_model, generator)


class PretrainedGrail(ModelWrapper):
    def __init__(
        self,
        generator_weights: Optional[str] = None,
        filter_weights: Optional[str] = None,
        rules_path: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        package_root = Path(__file__).resolve().parent.parent
        candidate_rules = [Path(rules_path)] if rules_path else []
        candidate_rules.extend(
            [
                package_root / "data" / "merged_smirks.txt",
                package_root / "data" / "smirks.txt",
            ]
        )

        selected_rules_path = next((path for path in candidate_rules if path.exists()), None)
        if selected_rules_path is not None:
            with open(selected_rules_path) as handle:
                rules = [line.strip() for line in handle if line.strip()]
        else:
            rules = load_default_rules()

        if not rules:
            raise FileNotFoundError("No rules were found for PretrainedGrail")

        rule_dict = {rule: from_rule(rule) for rule in rules}
        generator = Generator(rule_dict, 16, 18)
        filter_model = Filter(18, 18, [128, 256, 128, 128, 64, 32], mode="pair")
        super().__init__(filter_model, generator)

        gen_candidates = [Path(generator_weights)] if generator_weights else []
        filt_candidates = [Path(filter_weights)] if filter_weights else []
        gen_candidates.extend(
            [
                package_root / "best_generator.pth",
                package_root / "data" / "best_generator.pth",
            ]
        )
        filt_candidates.extend(
            [
                package_root / "best_filter_single.pth",
                package_root / "filter.pth",
                package_root / "data" / "best_filter_pair.pth",
            ]
        )

        loaded_generator = False
        loaded_filter = False
        for path in gen_candidates:
            if path.exists():
                if _load_checkpoint_payload(generator, path):
                    loaded_generator = True
                    break
        for path in filt_candidates:
            if path.exists():
                if _load_checkpoint_payload(filter_model, path):
                    loaded_filter = True
                    break

        if strict and not (loaded_generator and loaded_filter):
            raise FileNotFoundError("Failed to load pretrained weights")

    def draw(self, substrate_smiles: str, output_html: str = "network.html") -> Path:
        from pyvis.network import Network

        def generate_molecule_image(mol: Chem.Mol) -> str:
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")

        products = self.generate(substrate_smiles)
        network = Network(height="800px", width="1200px", notebook=False, directed=False)

        substrate = Chem.MolFromSmiles(substrate_smiles)
        if substrate is None:
            raise ValueError(f"Invalid substrate SMILES: {substrate_smiles}")

        network.add_node(
            0,
            shape="circularImage",
            image=f"data:image/png;base64,{generate_molecule_image(substrate)}",
            label=substrate_smiles,
            color="#b22222",
        )
        for index, product in enumerate(products, start=1):
            mol = Chem.MolFromSmiles(product)
            if mol is None:
                continue
            network.add_node(
                index,
                shape="circularImage",
                image=f"data:image/png;base64,{generate_molecule_image(mol)}",
                label=product,
                color="#1f77b4",
            )
            network.add_edge(0, index)

        output_path = Path(output_html)
        network.show(str(output_path))
        return output_path
