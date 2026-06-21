from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from torch import nn

if TYPE_CHECKING:
    from grail_metabolism.utils.preparation import MolFrame
    from grail_metabolism.config import MultiStepConfig


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
        som: Optional[nn.Module] = None,
    ) -> None:
        self.filter = filter
        # Optional site-of-metabolism prior (model.som.SoMPredictor). When set and used
        # with som_beta>0, generate() reweights candidates by site plausibility.
        self.som = som
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

    def generate_multistep(
        self,
        sub: str,
        config: "MultiStepConfig",
        threshold: Optional[float] = None,
        max_output: Optional[int] = None,
    ) -> List[str]:
        from .multistep import MetabolicTree

        rule_threshold = threshold if threshold is not None else getattr(self.generator, "calibrated_threshold", None)
        tree = MetabolicTree(self.generator, self.filter, config, rule_threshold=rule_threshold)
        return [smiles for smiles, _ in tree.beam_search(sub, max_output=max_output)]

    def generate(
        self,
        sub: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_threshold: Optional[float] = None,
        max_output: Optional[int] = None,
        multistep: "Optional[MultiStepConfig]" = None,
        gate_by_filter: bool = True,
        som_beta: Optional[float] = None,
        som_aggregation: str = "max",
        filter_candidate_cap: Optional[int] = None,
    ) -> List[str]:
        # Multi-step beam search only when explicitly requested with depth>1; otherwise
        # the exact single-step path below runs unchanged (byte-identical back-compat).
        if multistep is not None and getattr(multistep, "max_depth", 1) > 1:
            return self.generate_multistep(sub, multistep, threshold=threshold, max_output=max_output)
        # Candidates from generate_scored are already standardized, so we normalize
        # through the cached path (idempotent + fast) instead of re-running the
        # expensive uncached tautomer canonicalization on every product.
        from grail_metabolism.utils.preparation import _normalize_smiles_cached
        # Normalize with the SAME mode the generator emits (and was trained on), so the
        # filter scores candidates in the model's distribution and matching stays consistent.
        gen_mode = getattr(self.generator, "gen_normalization", "standardize")
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
        # The pair-filter (MCS-aware graph per candidate) is the dominant cost and scales
        # with candidate count. Since the final rank is filter*generator*som, a candidate the
        # generator already scores low rarely reaches the top max_output -- so cap the filter
        # to the generator's top-N candidates (generate_scored is sorted by score, desc).
        if filter_candidate_cap is not None and filter_candidate_cap > 0:
            scored_candidates = scored_candidates[:filter_candidate_cap]
        normalized_candidates = []
        generator_scores = []
        for candidate, generator_score in scored_candidates:
            try:
                normalized = _normalize_smiles_cached(candidate, gen_mode)
            except Exception:
                normalized = candidate
            normalized_candidates.append(normalized)
            generator_scores.append(float(generator_score))

        # Batch-score all candidates of this substrate in one filter forward pass.
        if hasattr(self.filter, "score_batch"):
            filter_scores = self.filter.score_batch(sub, normalized_candidates)
        else:
            filter_scores = [float(self.filter.score(sub, prod)) for prod in normalized_candidates]

        # Optional site-of-metabolism reweight: combined = filter * generator * som^beta.
        # beta=0 or no SoM model -> multiplier 1 -> exact filter*generator ranking (back-compat).
        # SoM only reshapes the RANK (never the filter gate), honoring the rank-only lesson.
        beta = float(som_beta) if som_beta is not None else 0.0
        som = getattr(self, "som", None)
        use_som = som is not None and beta > 0.0
        sub_mol = None
        som_atoms = None
        if use_som:
            from .som import product_som_score

            sub_mol = Chem.MolFromSmiles(sub)
            som_atoms = som.score_atoms(sub)
            use_som = sub_mol is not None and som_atoms is not None and len(som_atoms) > 0

        evaluated = []
        accepted = []
        for normalized, generator_score, filter_score in zip(normalized_candidates, generator_scores, filter_scores):
            filter_score = float(filter_score)
            som_mult = product_som_score(som_atoms, sub_mol, normalized, som_aggregation) ** beta if use_som else 1.0
            combined = filter_score * generator_score * som_mult
            evaluated.append((normalized, combined, filter_score, generator_score))
            if filter_score >= effective_filter_threshold:
                accepted.append((normalized, combined, filter_score, generator_score))
        sort_key = lambda item: (-item[1], -item[2], -item[3], item[0])
        if not gate_by_filter:
            # rank-only: keep every candidate, ordered by filter*generator score, and let
            # max_output do the truncation. The hard gate discards plausible-but-sub-
            # threshold hits and measurably hurts recall@k; ranking keeps them in reach.
            ranked_candidates = sorted(evaluated, key=sort_key)
        elif accepted:
            ranked_candidates = sorted(accepted, key=sort_key)
        elif evaluated:
            # gated, but nothing cleared the threshold: surface a few best-ranked anyway
            # so a substrate is never silently empty.
            fallback_limit = max(1, min(top_k or 3, 3, len(evaluated)))
            ranked_candidates = sorted(evaluated, key=sort_key)[:fallback_limit]
        else:
            ranked_candidates = []
        # Dedup the output by the SAME tautomer-invariant key the structure metrics match
        # on, not the raw canonical string. Otherwise tautomer/charge variants of one
        # molecule each take a slot of the (small) max_output budget while the metric
        # collapses them to a single hit -- wasting capacity that could hold other distinct
        # metabolites. Bounded to max_output so the tautomer canonicalization stays cheap.
        from grail_metabolism.metrics import _tautomer_inchikey

        seen = set()
        ranked = []
        cap = max_output if (max_output is not None and max_output > 0) else None
        for candidate, _, _, _ in ranked_candidates:
            try:
                key = _tautomer_inchikey(candidate)
            except Exception:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            ranked.append(candidate)
            if cap is not None and len(ranked) >= cap:
                break
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
