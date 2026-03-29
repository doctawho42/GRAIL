from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from torch import nn

if TYPE_CHECKING:
    from grail_metabolism.utils.preparation import MolFrame


class GFilter(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.mode: Optional[Literal["single", "pair"]] = None
        self.calibrated_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True, **kwargs) -> "GFilter":
        raise NotImplementedError

    @abstractmethod
    def predict(self, sub: str, prod: str, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def score(self, sub: str, prod: str, **kwargs) -> float:
        raise NotImplementedError


class GGenerator(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.calibrated_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True, **kwargs) -> "GGenerator":
        raise NotImplementedError

    @abstractmethod
    def generate(self, sub: str, **kwargs) -> List[str]:
        raise NotImplementedError


class ModelWrapper:
    def __init__(
        self,
        filter: GFilter,
        generator: Union[GGenerator, Literal["simple"]],
        rules: Optional[Sequence[str]] = None,
    ) -> None:
        self.filter = filter
        if generator == "simple":
            if not rules:
                raise ValueError("rules are required for the simple generator")
            self.generator = SimpleGenerator(list(rules))
            self.rules = list(rules)
        else:
            self.generator = generator
            self.rules = list(getattr(generator, "rules", {}).keys()) if hasattr(generator, "rules") else list(rules or [])

    def fit(
        self,
        data: MolFrame,
        generator_lr: float = 1e-4,
        filter_lr: float = 1e-4,
        generator_epochs: int = 10,
        filter_epochs: int = 10,
        verbose: bool = True,
    ) -> "ModelWrapper":
        if not data.single or not data.reaction_labels:
            data.full_setup(
                rules=self.rules,
                include_pair_graphs=False,
                include_morgan=False,
                single_smiles=data.map.keys() if self.filter.mode == "pair" else None,
            )
        self.generator.fit(data, lr=generator_lr, eps=generator_epochs, verbose=verbose)
        self.filter.fit(data, lr=filter_lr, eps=filter_epochs, verbose=verbose)
        return self

    def generate(
        self,
        sub: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_threshold: Optional[float] = None,
    ) -> List[str]:
        from grail_metabolism.utils.preparation import standardize_mol
        rule_threshold = threshold if threshold is not None else getattr(self.generator, "calibrated_threshold", None)
        effective_filter_threshold = (
            float(filter_threshold)
            if filter_threshold is not None
            else float(getattr(self.filter, "calibrated_threshold", 0.5) or 0.5)
        )

        if hasattr(self.generator, "generate_scored"):
            scored_candidates = self.generator.generate_scored(sub, top_k=top_k, threshold=rule_threshold)
        else:
            scored_candidates = [(candidate, 1.0) for candidate in self.generator.generate(sub, top_k=top_k, threshold=rule_threshold)]
        accepted = []
        evaluated = []
        for candidate, generator_score in scored_candidates:
            try:
                normalized = str(standardize_mol(candidate))
            except Exception:
                normalized = candidate
            filter_score = float(self.filter.score(sub, normalized))
            evaluated.append((normalized, filter_score * float(generator_score), filter_score, float(generator_score)))
            if filter_score < effective_filter_threshold:
                continue
            accepted.append((normalized, filter_score * float(generator_score), filter_score, float(generator_score)))
        ranked_candidates = accepted
        if not ranked_candidates and evaluated:
            fallback_limit = max(1, min(top_k or 3, 3, len(evaluated)))
            ranked_candidates = sorted(evaluated, key=lambda item: (-item[1], -item[2], -item[3], item[0]))[:fallback_limit]
        else:
            ranked_candidates = sorted(ranked_candidates, key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        seen = set()
        ranked = []
        for candidate, _, _, _ in ranked_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ranked.append(candidate)
        return ranked

    def f1_score(self, sub: str, prods: Iterable[str]) -> float:
        from grail_metabolism.utils.preparation import standardize_mol

        real = {str(standardize_mol(product)) for product in prods}
        pred = set(self.generate(sub))
        true_positive = len(real & pred)
        false_positive = len(pred - real)
        false_negative = len(real - pred)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class SimpleGenerator(GGenerator):
    def __init__(self, rules: Sequence[str]) -> None:
        super().__init__()
        self.rules = list(rules)

    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True, **kwargs) -> "SimpleGenerator":
        del data, lr, verbose, kwargs
        return self

    def generate(
        self,
        sub: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        from grail_metabolism.utils.preparation import safe_run_reactants, standardize_mol

        del top_k, threshold
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            return []

        out: List[str] = []
        seen = set()
        for rule in self.rules:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(rule)
            except Exception:
                continue
            outcomes = safe_run_reactants(rxn, mol)
            for product_tuple in outcomes:
                for product in product_tuple:
                    try:
                        smiles = Chem.MolToSmiles(product)
                    except Exception:
                        continue
                    for fragment in smiles.split("."):
                        candidate = fragment.strip()
                        if not candidate:
                            continue
                        try:
                            candidate = str(standardize_mol(candidate))
                        except Exception:
                            continue
                        if candidate not in seen:
                            seen.add(candidate)
                            out.append(candidate)
        return out

    def generate_scored(
        self,
        sub: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[tuple[str, float]]:
        del top_k, threshold
        return [(candidate, 1.0) for candidate in self.generate(sub)]
