from __future__ import annotations

import base64
import warnings
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


def _read_checkpoint(path: Path) -> Optional[dict]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        warnings.warn(f"Failed to read checkpoint {path}: {exc}")
        return None
    return state if isinstance(state, dict) else None


def _load_checkpoint_payload(module, path: Path, strict: bool = False) -> bool:
    """Load weights into ``module`` and VERIFY the match.

    Previously this used load_state_dict(strict=False) and swallowed every error, so
    a wrong/partial checkpoint loaded silently leaving modules at random init. Now we
    inspect missing/unexpected keys and warn (or, with strict=True, fail) so a bad
    checkpoint is never mistaken for a successful load.
    """
    state = _read_checkpoint(path)
    if state is None:
        return False
    payload = state.get("state_dict", state)
    if not isinstance(payload, dict):
        warnings.warn(f"Checkpoint {path} has no usable state_dict")
        return False
    try:
        result = module.load_state_dict(payload, strict=strict)
    except Exception as exc:
        warnings.warn(f"Checkpoint {path} is incompatible with the model: {exc}")
        return False
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    expected = len(module.state_dict())
    matched = expected - len(missing)
    if matched <= 0:
        warnings.warn(f"Checkpoint {path} matched 0/{expected} parameters; treating as not loaded")
        return False
    if missing or unexpected:
        warnings.warn(
            f"Checkpoint {path}: loaded {matched}/{expected} parameters "
            f"({len(missing)} missing, {len(unexpected)} unexpected). "
            "Missing parameters keep their freshly-initialized values."
        )
    if "calibrated_threshold" in state:
        module.calibrated_threshold = state["calibrated_threshold"]
    return True


def _build_filter_from_arch(arch: dict):
    from ..config import FilterConfig
    from ..workflows.factory import build_filter

    return build_filter(FilterConfig(**arch))


def _build_generator_from_arch(arch: dict, rules: Sequence[str]):
    from ..config import GeneratorConfig
    from ..workflows.factory import build_generator

    return build_generator(GeneratorConfig(**arch), list(rules))


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
        # Prefer an explicit rules file; otherwise fall through to the SAME shared
        # resolver the training presets use, so packaged inference and training agree
        # on the rule bank / label space.
        explicit_rules_path = Path(rules_path) if rules_path else None
        if explicit_rules_path is not None and explicit_rules_path.exists():
            with open(explicit_rules_path) as handle:
                rules = [line.strip() for line in handle if line.strip()]
        else:
            rules = load_default_rules()

        if not rules:
            raise FileNotFoundError("No rules were found for PretrainedGrail")

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

        gen_path = next((path for path in gen_candidates if path.exists()), None)
        filt_path = next((path for path in filt_candidates if path.exists()), None)

        rule_dict = {rule: from_rule(rule) for rule in rules}

        # Reconstruct each component from the architecture saved alongside the weights
        # when present, so a checkpoint trained with non-default dims (e.g. the preset's
        # 192-wide hidden) loads instead of crashing against the hardcoded defaults.
        gen_state = _read_checkpoint(gen_path) if gen_path else None
        gen_arch = gen_state.get("arch") if isinstance(gen_state, dict) else None
        if gen_arch:
            try:
                generator = _build_generator_from_arch(gen_arch, rules)
            except Exception as exc:
                warnings.warn(f"Could not rebuild generator from saved arch ({exc}); using defaults")
                generator = Generator(rule_dict, 16, 18)
        else:
            generator = Generator(rule_dict, 16, 18)

        filt_state = _read_checkpoint(filt_path) if filt_path else None
        filt_arch = filt_state.get("arch") if isinstance(filt_state, dict) else None
        if filt_arch:
            try:
                filter_model = _build_filter_from_arch(filt_arch)
            except Exception as exc:
                warnings.warn(f"Could not rebuild filter from saved arch ({exc}); using defaults")
                filter_model = Filter(18, 18, [128, 256, 128, 128, 64, 32], mode="pair")
        else:
            filter_model = Filter(18, 18, [128, 256, 128, 128, 64, 32], mode="pair")

        super().__init__(filter_model, generator)

        loaded_generator = _load_checkpoint_payload(generator, gen_path, strict=strict) if gen_path else False
        loaded_filter = _load_checkpoint_payload(filter_model, filt_path, strict=strict) if filt_path else False

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
