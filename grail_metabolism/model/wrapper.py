from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from torch import nn

# The fusion constant the manuscript registers and sweeps: Cormack, Clarke and Buettcher 2009,
# not tuned here. scripts/typed_edit/_rrf.py holds the same value for the analysis path, and the
# two must not drift; grail_metabolism/tests/test_audit_fixes.py holds them to each other.
RRF_K = 60

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
        factorized: "Optional[object]" = None,
    ) -> None:
        self.filter = filter
        # Optional site-of-metabolism prior (model.som.SoMPredictor). When set and used
        # with som_beta>0, generate() reweights candidates by site plausibility.
        self.som = som
        # Optional factorized re-ranker (model.factorized_infer.FactorizedReranker). When set,
        # generate() multiplies its per-candidate P(type|s)*P(site|type,s) factor into the rank
        # (the §10 hybrid re-rank, paired +0.0165). Rank-only: it never gates a candidate out,
        # and with factorized=None the ranking is byte-identical to filter*generator(*som).
        self.factorized = factorized
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

    def _rule_threshold(self, threshold: Optional[float]) -> Optional[float]:
        """The rule gate to run under when the caller did not name one.

        The released generator checkpoint carries ``calibrated_threshold = 0.6`` and this path
        used to fall back to it. Every pool the paper reports was built with ``threshold=None``,
        the top of the score ranking cut at the rule budget and no gate, and on the comparison
        set a median of six rules clear 0.6 against a budget of thirty, binding on 185 of 291
        substrates. So the fallback handed a caller a much narrower selection than any arm that
        was ever measured. The default is now the evaluated configuration; a caller who wants the
        gate passes it, and the checkpoint keeps the value so that choice stays available.
        """
        return threshold

    def generate_multistep(
        self,
        sub: str,
        config: "MultiStepConfig",
        threshold: Optional[float] = None,
        max_output: Optional[int] = None,
    ) -> List[str]:
        from .multistep import MetabolicTree

        rule_threshold = self._rule_threshold(threshold)
        tree = MetabolicTree(self.generator, self.filter, config, rule_threshold=rule_threshold)
        return [smiles for smiles, _ in tree.beam_search(sub, max_output=max_output)]

    # The blend beats the released noisy-or at the head of the list and loses at depth, so a
    # single setting runs the worse one over half the range. The switch point is ten: on the
    # validation draw the blend leads at every budget through thirty, but carried to the
    # comparison set its gain separates from zero only at one, three, five, eight and ten
    # (+0.0286, +0.0391, +0.0526, +0.0391, +0.0286), covers zero at fifteen and twenty, and turns
    # into a loss of 0.0135 at thirty. Ten is the widest budget where the gain is established
    # rather than the widest where validation liked it.
    #
    # results/budget_dependent_schedules.json carries the selection and the transfer, and
    # scripts/budget_dependent_schedules.py is what produced it: chosen on validation, spent once
    # on the comparison set. A caller that names no budget gets the released rule unchanged.
    BLEND_BUDGET = 10

    @contextmanager
    def _aggregation_for(self, max_output: Optional[int]):
        """Swap the candidate aggregation for the budget being asked for, and put it back.

        The aggregation is a generator attribute rather than an argument, so this mutates and
        restores rather than passing a value down. It is not safe to share one wrapper across
        threads while this is open, which is true of the generator's other mutable settings too.
        """
        was = getattr(self.generator, "candidate_aggregation", None)
        if was is None or max_output is None:
            # A generator that does not aggregate over templates has nothing to switch, and a
            # caller that names no budget gets the released rule. Both are left alone rather than
            # given a default, so a test double without the attribute behaves as it did.
            yield was
            return
        rule = "hybrid" if max_output <= self.BLEND_BUDGET else "noisy_or"
        self.generator.candidate_aggregation = rule
        try:
            yield rule
        finally:
            self.generator.candidate_aggregation = was

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
        return_scores: bool = False,
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
        rule_threshold = self._rule_threshold(threshold)
        effective_filter_threshold = (
            float(filter_threshold)
            if filter_threshold is not None
            else float(getattr(self.filter, "calibrated_threshold", 0.5) or 0.5)
        )

        # When a factorized re-ranker is attached, fetch rule provenance (rule_id per candidate)
        # so we can look up each candidate's P(type|s); compute_sites=False skips the costly MCS
        # firing-atom localization (the reranker localizes the site itself from the product SMILES).
        use_factorized = self.factorized is not None and hasattr(self.generator, "generate_scored_with_details")
        detailed = None
        if use_factorized:
            detailed = self.generator.generate_scored_with_details(sub, top_k=top_k, threshold=rule_threshold, compute_sites=False)
            scored_candidates = [(row[0], row[1]) for row in detailed]
        elif hasattr(self.generator, "generate_scored"):
            scored_candidates = self.generator.generate_scored(sub, top_k=top_k, threshold=rule_threshold)
        else:
            scored_candidates = [(candidate, 1.0) for candidate in self.generator.generate(sub, top_k=top_k, threshold=rule_threshold)]
        # The pair-filter (MCS-aware graph per candidate) is the dominant cost and scales
        # with candidate count. Since the final rank is filter*generator*som, a candidate the
        # generator already scores low rarely reaches the top max_output -- so cap the filter
        # to the generator's top-N candidates (generate_scored is sorted by score, desc).
        if filter_candidate_cap is not None and filter_candidate_cap > 0:
            scored_candidates = scored_candidates[:filter_candidate_cap]
            if detailed is not None:
                detailed = detailed[:filter_candidate_cap]
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

        # Factorized re-rank multipliers (P(type|s)*P(site|type,s) per candidate), index-aligned to
        # normalized_candidates. None -> multiplier 1.0 (byte-identical to filter*generator(*som)).
        fac_mults = None
        if use_factorized and detailed:
            fac_sub_mol = sub_mol if sub_mol is not None else Chem.MolFromSmiles(sub)
            if fac_sub_mol is not None:
                try:
                    fac_mults = self.factorized.multipliers(fac_sub_mol, detailed)
                except Exception:
                    fac_mults = None

        evaluated = []
        accepted = []
        for idx, (normalized, generator_score, filter_score) in enumerate(zip(normalized_candidates, generator_scores, filter_scores)):
            filter_score = float(filter_score)
            som_mult = product_som_score(som_atoms, sub_mol, normalized, som_aggregation) ** beta if use_som else 1.0
            fac_mult = float(fac_mults[idx]) if fac_mults is not None and idx < len(fac_mults) else 1.0
            combined = filter_score * generator_score * som_mult * fac_mult
            evaluated.append((normalized, combined, filter_score, generator_score))
            if filter_score >= effective_filter_threshold:
                accepted.append((normalized, combined, filter_score, generator_score))
        # Reciprocal rank fusion, which is what the manuscript says the deployed combination is
        # and what every measurement in it was made under. This path ranked by the product of the
        # two scores instead, so the released software ordered its output by the arrangement the
        # manuscript reports as the WORSE one -- and no reader could have found the difference,
        # because it is not in any number: every reported figure comes from pools re-ranked by
        # scripts/typed_edit/_rrf.py, and no producer of a reported figure calls this method.
        #
        # Only the ORDER changes. The scores handed back stay the filter's, the generator's and
        # their product, because an RRF score is about a fortieth of one and is not a probability;
        # a caller reading `combined` as a confidence would be handed a different kind of thing.
        def _rrf_rank(items):
            """1-based competition ranks fused as sum 1/(K + rank); ties share the lower rank.

            The same rule and the same K as the analysis path, and ranks rather than positions for
            the reason recorded there: a position depends on how the sort broke ties, so two runs
            over one pool can disagree, and on the comparison set they did.
            """
            def ranks(score):
                order = sorted(range(len(items)), key=lambda i: -score(items[i]))
                out, prev, cur = [0] * len(items), None, 1
                for pos, i in enumerate(order, 1):
                    v = score(items[i])
                    if v != prev:
                        prev, cur = v, pos
                    out[i] = cur
                return out
            rf = ranks(lambda it: it[2])   # filter
            rg = ranks(lambda it: it[3])   # generator
            return {id(items[i]): 1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i])
                    for i in range(len(items))}

        def _ordered(items):
            """Fuse the two LEARNED scores by rank, then apply the re-rankers on top of that.

            The fusion replaces the product of the filter's and the generator's opinions, which is
            what the manuscript describes and what the scale argument is about. It does not replace
            the site and factorized multipliers: those are separate re-rankings applied to the
            combination, and a first version of this ordered by the fusion alone, which silently
            took both of them out of the ranking entirely. A uniform multiplier must leave the
            order alone and a non-uniform one must reshape it.
            """
            fused = _rrf_rank(items)
            mult = {id(it): (it[1] / (it[2] * it[3]) if it[2] and it[3] else 1.0) for it in items}
            return sorted(items, key=lambda it: (-(fused[id(it)] * mult[id(it)]),
                                                 -it[1], -it[2], -it[3], it[0]))

        sort_key = lambda item: (-item[1], -item[2], -item[3], item[0])  # noqa: E731
        if not gate_by_filter:
            # rank-only: keep every candidate, ordered by the rank fusion, and let max_output do
            # the truncation. The hard gate discards plausible-but-sub-threshold hits and
            # measurably hurts recall@k; ranking keeps them in reach.
            ranked_candidates = _ordered(evaluated)
        elif accepted:
            ranked_candidates = _ordered(accepted)
        elif evaluated:
            # gated, but nothing cleared the threshold: surface a few best-ranked anyway
            # so a substrate is never silently empty.
            fallback_limit = max(1, min(top_k or 3, 3, len(evaluated)))
            ranked_candidates = _ordered(evaluated)[:fallback_limit]
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
        for candidate, combined, filter_score, generator_score in ranked_candidates:
            try:
                key = _tautomer_inchikey(candidate)
            except Exception:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            # return_scores carries the ranking scores out with the candidates they belong to.
            # It reads off the same list in the same order, so a scored dump cannot drift from
            # the deployed output; the default path is unchanged.
            ranked.append((candidate, float(combined), float(filter_score), float(generator_score))
                          if return_scores else candidate)
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
