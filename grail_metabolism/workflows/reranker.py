"""Stage 2a reranker workflow: pool building, hit labelling, the listwise InfoNCE loss,
a caching trainer, and the val evaluation that decides GO vs DEAD.

The slow part is pool generation (``generate_scored_with_details`` at top_k=200), so each
split's assembled pair-graphs + rule_ids + hit labels are cached to a ``.pt`` and reused
across epochs. Matching is tautomer-InChIKey throughout (``metrics._tautomer_inchikey``).
Selection is on VAL, never test.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from ..metrics import _tautomer_inchikey
from ..model.reranker import BiEncoderReranker, MinimalReranker
from ..utils.seed import seed_everything
from ..utils.transform import from_pair, from_rdmol

PoolEntry = Tuple[str, float, int]  # (smiles, gen_score, rule_id)

# Opt-in per-substrate wall-clock cap for the SERIAL bi-example build. A few substrates trigger
# a candidate-enumeration blow-up (thousands of rule products -> a long Python-level tautomer
# dedup loop) that stalls the build for minutes; SIGALRM fires between candidates and lets the
# builder skip them. Off by default (0); set RERANKER_SUB_TIMEOUT=<seconds> to enable. Only the
# main thread can take SIGALRM, so parallel Pool workers ignore it (they rely on Pool timeouts).
_PER_SUB_TIMEOUT = int(os.environ.get("RERANKER_SUB_TIMEOUT", "0"))


class _SubstrateTimeout(Exception):
    """Raised when one substrate's pool build exceeds ``_PER_SUB_TIMEOUT`` seconds."""


def _raise_substrate_timeout(signum, frame):  # noqa: ARG001 (signal handler signature)
    raise _SubstrateTimeout()


@contextlib.contextmanager
def _substrate_time_budget(idx: int):
    """Arm a SIGALRM budget for one substrate build; no-op when disabled or off-main-thread."""
    if _PER_SUB_TIMEOUT <= 0:
        yield
        return
    try:
        prev = signal.signal(signal.SIGALRM, _raise_substrate_timeout)
    except ValueError:
        yield  # not the main thread (e.g. a Pool worker) -> cannot arm; run uncapped
        return
    signal.alarm(_PER_SUB_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def build_pool(
    generator,
    sub: str,
    top_k: int = 200,
    max_pool: int = 100,
) -> List[PoolEntry]:
    """Generate the candidate pool for one substrate, dedup by tautomer-InChIKey
    (keeping generator order), then truncate to ``max_pool``.

    Returns ``[(smiles, gen_score, rule_id), ...]`` in generator-score order.
    """
    # compute_sites=False: skip the per-product MCS firing-atom localization (the dominant
    # cost at top_k=200). The cross-rule reranker uses rule_id, not firing sites (M0: regio
    # is ~4% of headroom).
    detailed = generator.generate_scored_with_details(sub, top_k=top_k, compute_sites=False)
    pool: List[PoolEntry] = []
    seen: set = set()
    for smiles, gen_score, rule_id, _sites in detailed:
        key = _tautomer_inchikey(smiles)
        if key in seen:
            continue
        seen.add(key)
        pool.append((smiles, float(gen_score), int(rule_id)))
        if len(pool) >= max_pool:
            break
    return pool


def label_hits(pool: Sequence[PoolEntry], true_products: Sequence[str]) -> torch.Tensor:
    """Boolean hit mask over the pool: a candidate is a hit iff its tautomer-InChIKey is
    in the tautomer-InChIKey set of the annotated true products."""
    truth = {_tautomer_inchikey(p) for p in true_products}
    flags = [_tautomer_inchikey(smiles) in truth for smiles, _, _ in pool]
    return torch.tensor(flags, dtype=torch.bool)


def listwise_infonce(logits: torch.Tensor, hit_mask: torch.Tensor) -> torch.Tensor:
    """Per-substrate listwise InfoNCE.

    With pool logits ``s`` (length N) and a boolean ``hit_mask``, treat EACH hit as the
    positive against the WHOLE pool as the softmax denominator::

        L = - (1 / H) * sum_{h in hits} ( s[h] - logsumexp(s) )

    Returns 0 (with grad) when the pool has no hits, so a hit-less substrate is skipped.
    """
    if logits.numel() == 0:
        return logits.sum() * 0.0
    hit_mask = hit_mask.to(logits.device).bool()
    num_hits = int(hit_mask.sum().item())
    if num_hits == 0:
        return logits.sum() * 0.0
    denom = torch.logsumexp(logits, dim=0)  # scalar; full pool is the denominator
    hit_logits = logits[hit_mask]
    per_hit = hit_logits - denom  # log-softmax of each hit over the pool
    return -(per_hit.sum() / num_hits)


# --------------------------------------------------------------------------- #
# Cached per-substrate examples.
# --------------------------------------------------------------------------- #

class _SubstrateExample:
    """One substrate's cached training/eval material: assembled pair graphs, the firing
    rule_id per candidate, the hit mask, the gen_scores (for the generator-alone baseline),
    and the candidate smiles."""

    __slots__ = ("sub", "graphs", "rule_ids", "hit_mask", "gen_scores", "smiles", "true_products")

    def __init__(self, sub, graphs, rule_ids, hit_mask, gen_scores, smiles, true_products):
        self.sub = sub
        self.graphs = graphs
        self.rule_ids = rule_ids
        self.hit_mask = hit_mask
        self.gen_scores = gen_scores
        self.smiles = smiles
        # ALL annotated true products for this substrate (the recall denominator). Some
        # are not in the pool at all -- that gap is exactly why the oracle ceiling < 1.0.
        self.true_products = true_products


def _build_examples(
    generator,
    molframe,
    n_substrates: int,
    top_k: int,
    max_pool: int,
    verbose: bool = True,
) -> List[_SubstrateExample]:
    """Assemble cached examples for up to ``n_substrates`` substrates that have >=1
    parseable candidate. Substrates with no parseable pool are dropped."""
    substrates = list(molframe.map.keys())
    examples: List[_SubstrateExample] = []
    for idx, sub in enumerate(substrates):
        if len(examples) >= n_substrates:
            break
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        pool = build_pool(generator, sub, top_k=top_k, max_pool=max_pool)
        if not pool:
            continue
        graphs: List[Data] = []
        rule_ids: List[int] = []
        gen_scores: List[float] = []
        smiles: List[str] = []
        for cand_smiles, gen_score, rule_id in pool:
            cand_mol = Chem.MolFromSmiles(cand_smiles)
            if cand_mol is None:
                continue
            graph = from_pair(sub_mol, cand_mol)
            if graph is None:
                continue
            graphs.append(graph)
            rule_ids.append(int(rule_id))
            gen_scores.append(float(gen_score))
            smiles.append(cand_smiles)
        if not graphs:
            continue
        # Recompute the hit mask over the SURVIVING (parseable) candidates only.
        surviving_pool = [(s, sc, r) for s, sc, r in zip(smiles, gen_scores, rule_ids)]
        true_products = sorted(molframe.map[sub])
        hit_mask = label_hits(surviving_pool, true_products)
        examples.append(
            _SubstrateExample(
                sub=sub,
                graphs=graphs,
                rule_ids=torch.tensor(rule_ids, dtype=torch.long),
                hit_mask=hit_mask,
                gen_scores=torch.tensor(gen_scores, dtype=torch.float32),
                smiles=smiles,
                true_products=true_products,
            )
        )
        if verbose and (len(examples) % 25 == 0):
            print(
                f"  built {len(examples)} examples (scanned {idx + 1} substrates)",
                flush=True,
            )
    return examples


def _cache_payload(examples: List[_SubstrateExample]) -> List[dict]:
    return [
        {
            "sub": ex.sub,
            "graphs": ex.graphs,
            "rule_ids": ex.rule_ids,
            "hit_mask": ex.hit_mask,
            "gen_scores": ex.gen_scores,
            "smiles": ex.smiles,
            "true_products": ex.true_products,
        }
        for ex in examples
    ]


def _examples_from_payload(payload: List[dict]) -> List[_SubstrateExample]:
    return [
        _SubstrateExample(
            sub=row["sub"],
            graphs=row["graphs"],
            rule_ids=row["rule_ids"],
            hit_mask=row["hit_mask"],
            gen_scores=row["gen_scores"],
            smiles=row["smiles"],
            true_products=row["true_products"],
        )
        for row in payload
    ]


def load_or_build_examples(
    generator,
    molframe,
    n_substrates: int,
    cache_path: Path,
    top_k: int = 200,
    max_pool: int = 100,
    verbose: bool = True,
) -> List[_SubstrateExample]:
    """Load assembled examples from ``cache_path`` if present, else build and cache them.

    Pool generation at top_k=200 is the slow part, so this is the one thing worth caching;
    epochs reuse the cached pair-graphs/rule_ids/hit masks.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        if verbose:
            print(f"  loading cached examples from {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return _examples_from_payload(payload)
    if verbose:
        print(f"  building examples (no cache at {cache_path})", flush=True)
    examples = _build_examples(
        generator, molframe, n_substrates, top_k=top_k, max_pool=max_pool, verbose=verbose
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(_cache_payload(examples), tmp)
    tmp.replace(cache_path)
    if verbose:
        print(f"  cached {len(examples)} examples -> {cache_path}", flush=True)
    return examples


class RerankerTrainer:
    """Trains a ``MinimalReranker`` with the per-substrate listwise InfoNCE objective.

    Loss = ``listwise_infonce`` per substrate (the ONLY loss -- no sibling, no BCE),
    averaged over substrates that have >=1 hit in their pool. Adam, lr 1e-3.
    """

    def __init__(
        self,
        reranker: MinimalReranker,
        lr: float = 1e-3,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.seed = int(seed)
        seed_everything(self.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker = reranker.to(self.device)
        self.optimizer = torch.optim.Adam(self.reranker.parameters(), lr=lr)

    def _batch_for(self, ex: _SubstrateExample) -> Tuple[Batch, torch.Tensor]:
        batch = Batch.from_data_list([g.clone() for g in ex.graphs]).to(self.device)
        rule_ids = ex.rule_ids.to(self.device)
        return batch, rule_ids

    def fit(
        self,
        examples: List[_SubstrateExample],
        epochs: int = 15,
        verbose: bool = True,
    ) -> MinimalReranker:
        # Only substrates with at least one hit in the pool contribute gradient.
        trainable = [ex for ex in examples if bool(ex.hit_mask.any())]
        if verbose:
            print(
                f"  training on {len(trainable)}/{len(examples)} substrates with >=1 pool hit",
                flush=True,
            )
        for epoch in range(epochs):
            seed_everything(self.seed + epoch)  # deterministic per-epoch shuffle
            order = torch.randperm(len(trainable)).tolist()
            self.reranker.train()
            total = 0.0
            count = 0
            for i in order:
                ex = trainable[i]
                batch, rule_ids = self._batch_for(ex)
                self.optimizer.zero_grad()
                logits = self.reranker(batch, rule_ids)
                loss = listwise_infonce(logits, ex.hit_mask.to(self.device))
                loss.backward()
                self.optimizer.step()
                total += float(loss.item())
                count += 1
            if verbose:
                avg = total / max(count, 1)
                print(f"  epoch {epoch + 1}/{epochs} listwise_infonce={avg:.4f}", flush=True)
        return self.reranker


def _recall_at_k(ranked_smiles: Sequence[str], true_products: Sequence[str], k: int) -> float:
    """Tautomer-InChIKey recall@k: fraction of distinct true products whose key appears
    in the top-k of the ranking."""
    truth = {_tautomer_inchikey(p) for p in true_products}
    if not truth:
        return 0.0
    topk_keys = {_tautomer_inchikey(s) for s in ranked_smiles[:k]}
    return len(topk_keys & truth) / len(truth)


@torch.no_grad()
def evaluate(
    reranker: MinimalReranker,
    examples: List[_SubstrateExample],
    ks: Sequence[int] = (5, 10, 12, 15),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Score each substrate's pool with the reranker, rank desc, and compute recall@k
    (tautomer). Also compute generator-alone recall (gen_score order) and oracle recall
    (hits-first) on the SAME pools, for reference.

    Returns a flat dict: ``reranker_recall@{k}``, ``generator_recall@{k}``,
    ``oracle_recall@{k}``, plus ``n_substrates`` and ``mean_pool_size``.
    """
    device = device or next(reranker.parameters()).device
    reranker.eval()
    rer_acc = {k: 0.0 for k in ks}
    gen_acc = {k: 0.0 for k in ks}
    ora_acc = {k: 0.0 for k in ks}
    pool_sizes = 0
    n = 0
    for ex in examples:
        truth = ex.hit_mask  # boolean over the pool, already tautomer-IK derived
        # Recall denominator = ALL annotated true products (incl. those absent from the
        # pool). The oracle ranking (hits-first) therefore tops out below 1.0 exactly by
        # the fraction of true products the generator never enumerated into the pool.
        true_products = ex.true_products
        batch = Batch.from_data_list([g.clone() for g in ex.graphs]).to(device)
        rule_ids = ex.rule_ids.to(device)
        scores = reranker(batch, rule_ids).detach().cpu()

        # reranker ranking: by predicted logit desc. Stable tiebreak on original (pool)
        # order so equal logits keep generator order rather than an arbitrary permutation.
        rer_order = sorted(range(len(ex.smiles)), key=lambda i: (-float(scores[i]), i))
        rer_ranked = [ex.smiles[i] for i in rer_order]
        # generator-alone ranking: the pool is ALREADY in the generator's own score order
        # (generate_scored_with_details sorts by (-gen_score, smiles) and the IK-dedup /
        # parse-survival steps preserve it), so the candidate list as-is IS that ranking.
        gen_ranked = list(ex.smiles)
        # oracle ranking: hits first (best achievable on this pool), generator order within.
        ora_order = sorted(range(len(ex.smiles)), key=lambda i: (0 if truth[i] else 1, i))
        ora_ranked = [ex.smiles[i] for i in ora_order]

        for k in ks:
            rer_acc[k] += _recall_at_k(rer_ranked, true_products, k)
            gen_acc[k] += _recall_at_k(gen_ranked, true_products, k)
            ora_acc[k] += _recall_at_k(ora_ranked, true_products, k)
        pool_sizes += len(ex.smiles)
        n += 1

    out: Dict[str, float] = {"n_substrates": float(n), "mean_pool_size": pool_sizes / max(n, 1)}
    for k in ks:
        out[f"reranker_recall@{k}"] = rer_acc[k] / max(n, 1)
        out[f"generator_recall@{k}"] = gen_acc[k] / max(n, 1)
        out[f"oracle_recall@{k}"] = ora_acc[k] / max(n, 1)
    return out


# =========================================================================== #
# Stage 2a (FAIR): no-MCS bi-encoder example path + trainer + eval.
#
# Same listwise InfoNCE loss, same recall@k eval as the pair path above. What
# changes: examples store SINGLE graphs (from_rdmol, NO from_pair / rdFMCS) for
# the substrate and each product, plus the empirical rule-prior log-odds scalar
# per candidate (generator.rule_prior_logits[rule_id]) -- NOT a learned rule
# embedding. The example build is ~seconds/substrate instead of the MCS path's
# ~23 s/example.
# =========================================================================== #


class _BiExample:
    """One substrate's cached bi-encoder material: the substrate single-graph, the per-
    candidate product single-graphs, the rule-prior scalar + gen_score per candidate, the
    hit mask, the candidate smiles, and ALL annotated true products (recall denominator)."""

    __slots__ = (
        "sub", "sub_graph", "prod_graphs", "rule_priors", "gen_scores",
        "hit_mask", "smiles", "true_products",
    )

    def __init__(self, sub, sub_graph, prod_graphs, rule_priors, gen_scores, hit_mask, smiles, true_products):
        self.sub = sub
        self.sub_graph = sub_graph
        self.prod_graphs = prod_graphs
        self.rule_priors = rule_priors
        self.gen_scores = gen_scores
        self.hit_mask = hit_mask
        self.smiles = smiles
        self.true_products = true_products


def _bi_example_from_pool(
    sub: str,
    pool: Sequence[Tuple[str, float, int]],
    true_products: Sequence[str],
    prior: torch.Tensor,
    num_rules: int,
) -> Optional[_BiExample]:
    """Build a bi-encoder example from an ALREADY-generated candidate pool: ``from_rdmol`` the
    substrate + each candidate (single graphs) and label hits. The parallel path runs this in
    the MAIN process so NO torch tensors cross the multiprocessing boundary -- passing graph
    tensors back from workers exhausts shared-memory file descriptors (the mmap ENOMEM stall).
    """
    sub_mol = Chem.MolFromSmiles(sub)
    if sub_mol is None:
        return None
    sub_graph = from_rdmol(sub_mol)
    if sub_graph is None:
        return None
    if not pool:
        return None
    prod_graphs: List[Data] = []
    rule_priors: List[float] = []
    gen_scores: List[float] = []
    smiles: List[str] = []
    for cand_smiles, gen_score, rule_id in pool:
        cand_mol = Chem.MolFromSmiles(cand_smiles)
        if cand_mol is None:
            continue
        graph = from_rdmol(cand_mol)
        if graph is None:
            continue
        prod_graphs.append(graph)
        # Empirical per-rule prior log-odds; clamp the id defensively.
        rid = int(rule_id) if 0 <= int(rule_id) < num_rules else 0
        rule_priors.append(float(prior[rid]) if num_rules else 0.0)
        gen_scores.append(float(gen_score))
        smiles.append(cand_smiles)
    if not prod_graphs:
        return None
    surviving_pool = [(s, sc, 0) for s, sc in zip(smiles, gen_scores)]
    true_products = sorted(true_products)
    hit_mask = label_hits(surviving_pool, true_products)
    return _BiExample(
        sub=sub,
        sub_graph=sub_graph,
        prod_graphs=prod_graphs,
        rule_priors=torch.tensor(rule_priors, dtype=torch.float32),
        gen_scores=torch.tensor(gen_scores, dtype=torch.float32),
        hit_mask=hit_mask,
        smiles=smiles,
        true_products=true_products,
    )


def _build_one_bi(
    generator,
    sub: str,
    true_products: Sequence[str],
    top_k: int,
    max_pool: int,
    prior: torch.Tensor,
    num_rules: int,
) -> Optional[_BiExample]:
    """Serial per-substrate body: generate the pool (rule application) then build the example
    in one process. The parallel path splits these -- workers generate pools, the main process
    builds the graphs -- but both produce identical examples for the same substrate."""
    pool = build_pool(generator, sub, top_k=top_k, max_pool=max_pool)
    if not pool:
        return None
    return _bi_example_from_pool(sub, pool, true_products, prior, num_rules)


def build_examples_bi(
    generator,
    molframe,
    n_substrates: int,
    top_k: int,
    max_pool: int,
    verbose: bool = True,
) -> List[_BiExample]:
    """Assemble no-MCS bi-encoder examples for up to ``n_substrates`` substrates.

    For each substrate: ``build_pool(compute_sites=False)`` (no MCS), then ``from_rdmol`` the
    substrate ONCE and ``from_rdmol`` each candidate (single graphs only, no merged pair
    graph). The rule signal is the scalar ``rule_prior = float(generator.rule_prior_logits[rule_id])``.
    """
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())
    substrates = list(molframe.map.keys())
    examples: List[_BiExample] = []
    skipped = 0
    idx = -1
    for idx, sub in enumerate(substrates):
        if len(examples) >= n_substrates:
            break
        # A handful of substrates trigger a candidate-enumeration blow-up (thousands of
        # rule products -> a long Python-level tautomer-dedup loop) that stalls the serial
        # build for minutes; SIGALRM fires between candidates and skips them. Opt-in via
        # RERANKER_SUB_TIMEOUT (seconds; 0 = off) so tests/normal callers are unaffected.
        with _substrate_time_budget(idx):
            try:
                ex = _build_one_bi(
                    generator, sub, molframe.map[sub], top_k, max_pool, prior, num_rules
                )
            except _SubstrateTimeout:
                skipped += 1
                print(
                    f"  [timeout] substrate idx={idx} exceeded {_PER_SUB_TIMEOUT}s "
                    f"(pool blow-up) -> skipped ({skipped} total)",
                    flush=True,
                )
                continue
        if ex is None:
            continue
        examples.append(ex)
        if verbose and (len(examples) % 25 == 0):
            print(
                f"  built {len(examples)} bi-examples (scanned {idx + 1} substrates)",
                flush=True,
            )
    if skipped:
        print(f"  [timeout] skipped {skipped} pool-blow-up substrate(s) of {idx + 1} scanned", flush=True)
    return examples


# --------------------------------------------------------------------------- #
# Parallel (process-Pool) bi-example builder.
#
# Pool generation (RDKit rule application inside generate_scored_with_details) is the
# single-threaded wall (~12-21 s/substrate). It parallelizes cleanly across CPU cores:
# each worker loads its OWN generator from the checkpoint once (in the initializer), pins
# torch to 1 thread to avoid oversubscription, and builds one _BiExample per call. The
# example content is identical to the serial path -- only the throughput changes.
#
# START METHOD = "spawn" (NOT fork). Once the parent process has loaded a generator, its
# PyTorch/RDKit native runtime is no longer fork-safe: a forked child CRASHES inside the
# first GNN forward of generate_scored_with_details (the Pool then silently respawns it,
# spinning forever). Spawn gives each worker a fresh interpreter that re-imports torch and
# re-loads the generator cleanly. The trade-off is a per-worker startup of ~80 s (SMARTS
# compile + generator load); this amortizes fully at the 1000+ substrate build scale.
# --------------------------------------------------------------------------- #

# Per-worker module global: the generator loaded once in the initializer and reused for
# every substrate that worker handles (avoids re-paying the ~80 s SMARTS-compile per call).
# Spawn workers each reload the full generator (~80s + a generator-sized RAM copy), so the
# parallel build hard-caps the worker count: more than this OOMs (the 48-worker failure mode).
_MAX_SPAWN_WORKERS = 8
_BI_WORKER_GEN = None


def _bi_worker_init(gen_ckpt_path: str, prior_strength: float, rule_emb=None) -> None:
    """Pool initializer: pin torch to 1 thread, silence rdkit, load the generator once into
    a module global. ``rule_emb`` (warmed ONCE in the parent) is injected as the rule-bank
    embedding cache so this worker SKIPS the slow single-threaded 7581-rule GNN encoding --
    the per-worker stall that made the parallel build appear hung.
    """
    # CRITICAL: without this, N workers each spawn many BLAS/torch threads and oversubscribe
    # the box, making the parallel path SLOWER than serial.
    torch.set_num_threads(1)
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Reuse the exact load logic of scripts/run_reranker_gate.py:_load_generator so the
    # worker generator is identical to the serial one.
    from ..config import GeneratorConfig
    from ..model.grail import _read_checkpoint
    from .factory import build_generator

    state = _read_checkpoint(gen_ckpt_path)
    if state is None or "arch" not in state or "rules" not in state:
        raise RuntimeError(f"Generator checkpoint missing arch/rules: {gen_ckpt_path}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    generator.eval()
    generator.prior_strength = float(prior_strength)
    if rule_emb is not None:
        generator._rule_embedding_cache = rule_emb  # skip the 7581-rule encoding
    global _BI_WORKER_GEN
    _BI_WORKER_GEN = generator


def _bi_pool_worker(args: Tuple[str, Sequence[str], int, int]):
    """Pool worker fn: do ONLY the expensive ``build_pool`` (rule application) for one
    substrate and return the PLAIN pool ``(sub, [(smiles, gen_score, rule_id), ...],
    true_products)``. NO torch tensors cross the multiprocessing boundary -- the main process
    builds the graphs (``_bi_example_from_pool``), which avoids the shared-memory FD
    exhaustion (mmap ENOMEM) that returning graph tensors from workers caused.
    """
    sub, true_products, top_k, max_pool = args
    gen = _BI_WORKER_GEN
    if gen is None:  # pragma: no cover - initializer always runs first
        raise RuntimeError("worker generator not initialized")
    pool = build_pool(gen, sub, top_k=top_k, max_pool=max_pool)
    if not pool:
        return None
    return (sub, pool, list(true_products))


def build_examples_bi_parallel(
    generator,
    molframe,
    n_substrates: int,
    top_k: int,
    max_pool: int,
    workers: int,
    prior_strength: float,
    gen_ckpt: Optional[str] = None,
    verbose: bool = True,
) -> List[_BiExample]:
    """Parallel analogue of ``build_examples_bi``.

    On Linux (Colab) uses a FORK Pool: the generator is loaded ONCE in the parent and inherited
    by every worker via copy-on-write -- NO per-worker reload, NO N x generator memory (the fix
    for high worker counts, where the old spawn path reloaded the 7581-rule generator per
    worker and thrashed RAM at e.g. 48 workers). On macOS/Windows (fork+torch is unsafe there)
    falls back to a SPAWN Pool whose workers reload from ``gen_ckpt`` so local tests still run.
    UNORDERED imap yields each pool as its worker finishes (no head-of-line blocking on a slow
    substrate); stops once ``n_substrates`` are collected. Example order is not deterministic
    across runs, but the trainer is order-invariant.
    """
    # SPAWN is fork-safe everywhere (fork + torch/OpenMP deadlocks on Linux); each worker
    # reloads the generator from gen_ckpt ONCE. The caller MUST cap ``workers`` low (~8) so
    # N generator copies fit in RAM -- a high count (e.g. 48) OOMs from N reloads. This is the
    # reliable path; fork-COW (no reload) is not used because it is unsafe with torch.
    workers = max(1, min(int(workers), os.cpu_count() or 1, _MAX_SPAWN_WORKERS))
    if not gen_ckpt:
        raise ValueError("parallel build requires gen_ckpt for the spawn workers")
    generator.prior_strength = float(prior_strength)
    substrates = list(molframe.map.keys())
    args = [(sub, list(molframe.map[sub]), top_k, max_pool) for sub in substrates]
    # Warm the rule-bank embeddings ONCE here (parent, full threads) so each spawn worker can
    # SKIP the slow single-threaded 7581-rule GNN encoding -- without this the workers stall at
    # 100% CPU for minutes re-encoding the bank before the first example.
    with torch.no_grad():
        rule_emb = generator._rule_embeddings(torch.device("cpu")).detach().cpu().contiguous()
    examples: List[_BiExample] = []
    pool = mp.get_context("spawn").Pool(
        processes=workers, initializer=_bi_worker_init,
        initargs=(str(gen_ckpt), float(prior_strength), rule_emb),
    )
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())
    try:
        # Workers return PLAIN pools (no tensors); the graphs are built here in the main
        # process so nothing torch crosses the mp boundary. imap_UNORDERED yields each pool as
        # its worker finishes -- a single pathological substrate (a big drug + 200 rules) no
        # longer head-of-line-blocks the whole stream (the "stuck at 100" symptom).
        for res in pool.imap_unordered(_bi_pool_worker, args, chunksize=2):
            if res is None:
                continue
            sub_r, pool_r, true_r = res
            ex = _bi_example_from_pool(sub_r, pool_r, true_r, prior, num_rules)
            if ex is None:
                continue
            examples.append(ex)
            if verbose and (len(examples) % 25 == 0):
                print(f"  built {len(examples)} bi-examples (parallel)", flush=True)
            if len(examples) >= n_substrates:
                break
    finally:
        pool.terminate()
        pool.join()
    return examples


def _bi_cache_payload(examples: List[_BiExample]) -> List[dict]:
    return [
        {
            "sub": ex.sub,
            "sub_graph": ex.sub_graph,
            "prod_graphs": ex.prod_graphs,
            "rule_priors": ex.rule_priors,
            "gen_scores": ex.gen_scores,
            "hit_mask": ex.hit_mask,
            "smiles": ex.smiles,
            "true_products": ex.true_products,
        }
        for ex in examples
    ]


def _bi_examples_from_payload(payload: List[dict]) -> List[_BiExample]:
    return [
        _BiExample(
            sub=row["sub"],
            sub_graph=row["sub_graph"],
            prod_graphs=row["prod_graphs"],
            rule_priors=row["rule_priors"],
            gen_scores=row["gen_scores"],
            hit_mask=row["hit_mask"],
            smiles=row["smiles"],
            true_products=row["true_products"],
        )
        for row in payload
    ]


def load_or_build_examples_bi(
    generator,
    molframe,
    n_substrates: int,
    cache_path: Path,
    top_k: int = 200,
    max_pool: int = 100,
    verbose: bool = True,
    workers: int = 1,
    gen_ckpt: Optional[str] = None,
) -> List[_BiExample]:
    """Load assembled bi-encoder examples from ``cache_path`` if present, else build+cache.

    When ``workers > 1`` and ``gen_ckpt`` is given, pool generation runs in a fork Pool
    (each worker loads its own generator from ``gen_ckpt`` once). The parallel path produces
    IDENTICAL example content to the serial path -- only the build throughput changes. The
    passed ``generator``'s ``prior_strength`` is propagated to the workers so the candidate
    pools match exactly. When ``workers <= 1`` (or no ``gen_ckpt``), the serial path runs.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        if verbose:
            print(f"  loading cached bi-examples from {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return _bi_examples_from_payload(payload)
    if int(workers) > 1 and gen_ckpt:
        if verbose:
            print(
                f"  building bi-examples in PARALLEL ({min(int(workers), os.cpu_count() or 1)} "
                f"workers; no cache at {cache_path})",
                flush=True,
            )
        examples = build_examples_bi_parallel(
            generator, molframe, n_substrates, top_k=top_k, max_pool=max_pool,
            workers=int(workers),
            prior_strength=float(getattr(generator, "prior_strength", 0.4)),
            gen_ckpt=str(gen_ckpt), verbose=verbose,
        )
    else:
        if verbose:
            print(f"  building bi-examples (no cache at {cache_path})", flush=True)
        examples = build_examples_bi(
            generator, molframe, n_substrates, top_k=top_k, max_pool=max_pool, verbose=verbose
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(_bi_cache_payload(examples), tmp)
    tmp.replace(cache_path)
    if verbose:
        print(f"  cached {len(examples)} bi-examples -> {cache_path}", flush=True)
    return examples


class BiRerankerTrainer:
    """Trains a ``BiEncoderReranker`` with the same per-substrate listwise InfoNCE objective
    as ``RerankerTrainer`` (the ONLY loss), averaged over substrates with >=1 pool hit."""

    def __init__(
        self,
        reranker: BiEncoderReranker,
        lr: float = 1e-3,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.seed = int(seed)
        seed_everything(self.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker = reranker.to(self.device)
        self.optimizer = torch.optim.Adam(self.reranker.parameters(), lr=lr)

    def fit(
        self,
        examples: List[_BiExample],
        epochs: int = 15,
        verbose: bool = True,
    ) -> BiEncoderReranker:
        trainable = [ex for ex in examples if bool(ex.hit_mask.any())]
        if verbose:
            print(
                f"  training (bi) on {len(trainable)}/{len(examples)} substrates with >=1 pool hit",
                flush=True,
            )
        for epoch in range(epochs):
            seed_everything(self.seed + epoch)
            order = torch.randperm(len(trainable)).tolist()
            self.reranker.train()
            total = 0.0
            count = 0
            for i in order:
                ex = trainable[i]
                sub_graph = ex.sub_graph.clone()
                prod_batch = Batch.from_data_list([g.clone() for g in ex.prod_graphs])
                self.optimizer.zero_grad()
                logits = self.reranker(
                    sub_graph, prod_batch,
                    ex.rule_priors.to(self.device), ex.gen_scores.to(self.device),
                )
                loss = listwise_infonce(logits, ex.hit_mask.to(self.device))
                loss.backward()
                self.optimizer.step()
                total += float(loss.item())
                count += 1
            if verbose:
                avg = total / max(count, 1)
                print(f"  epoch {epoch + 1}/{epochs} listwise_infonce={avg:.4f}", flush=True)
        return self.reranker


@torch.no_grad()
def evaluate_bi(
    reranker: BiEncoderReranker,
    examples: List[_BiExample],
    ks: Sequence[int] = (5, 10, 12, 15),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Bi-encoder analogue of ``evaluate``: reranker vs generator-alone vs oracle recall@k
    (tautomer-InChIKey) on the SAME pools."""
    device = device or next(reranker.parameters()).device
    reranker.eval()
    rer_acc = {k: 0.0 for k in ks}
    gen_acc = {k: 0.0 for k in ks}
    ora_acc = {k: 0.0 for k in ks}
    pool_sizes = 0
    n = 0
    for ex in examples:
        truth = ex.hit_mask
        true_products = ex.true_products
        sub_graph = ex.sub_graph.clone()
        prod_batch = Batch.from_data_list([g.clone() for g in ex.prod_graphs])
        scores = reranker(
            sub_graph, prod_batch,
            ex.rule_priors.to(device), ex.gen_scores.to(device),
        ).detach().cpu()

        rer_order = sorted(range(len(ex.smiles)), key=lambda i: (-float(scores[i]), i))
        rer_ranked = [ex.smiles[i] for i in rer_order]
        gen_ranked = list(ex.smiles)
        ora_order = sorted(range(len(ex.smiles)), key=lambda i: (0 if truth[i] else 1, i))
        ora_ranked = [ex.smiles[i] for i in ora_order]

        for k in ks:
            rer_acc[k] += _recall_at_k(rer_ranked, true_products, k)
            gen_acc[k] += _recall_at_k(gen_ranked, true_products, k)
            ora_acc[k] += _recall_at_k(ora_ranked, true_products, k)
        pool_sizes += len(ex.smiles)
        n += 1

    out: Dict[str, float] = {"n_substrates": float(n), "mean_pool_size": pool_sizes / max(n, 1)}
    for k in ks:
        out[f"reranker_recall@{k}"] = rer_acc[k] / max(n, 1)
        out[f"generator_recall@{k}"] = gen_acc[k] / max(n, 1)
        out[f"oracle_recall@{k}"] = ora_acc[k] / max(n, 1)
    return out


def evaluate_bi_per_substrate(
    reranker: BiEncoderReranker,
    examples: List[_BiExample],
    k: int = 15,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float]]:
    """Per-substrate reranker vs generator-alone recall@k (tautomer-InChIKey) on the SAME pools.

    Same ranking as ``evaluate_bi`` (which returns only the aggregate means); here the per-substrate
    values are kept so a PAIRED bootstrap CI on the reranker-minus-generator improvement can be
    computed (Proposition 1). The means of the returned lists equal ``evaluate_bi``'s
    ``reranker_recall@k`` / ``generator_recall@k`` exactly.
    """
    device = device or next(reranker.parameters()).device
    reranker.eval()
    rer_list: List[float] = []
    gen_list: List[float] = []
    for ex in examples:
        true_products = ex.true_products
        sub_graph = ex.sub_graph.clone()
        prod_batch = Batch.from_data_list([g.clone() for g in ex.prod_graphs])
        scores = reranker(
            sub_graph, prod_batch,
            ex.rule_priors.to(device), ex.gen_scores.to(device),
        ).detach().cpu()
        rer_order = sorted(range(len(ex.smiles)), key=lambda i: (-float(scores[i]), i))
        rer_ranked = [ex.smiles[i] for i in rer_order]
        gen_ranked = list(ex.smiles)
        rer_list.append(_recall_at_k(rer_ranked, true_products, k))
        gen_list.append(_recall_at_k(gen_ranked, true_products, k))
    return rer_list, gen_list


# =========================================================================== #
# Task 6: intermediate-node (depth-2) bootstrap pairs.
#
# The reranker above is trained ONLY on (root, product) pairs -- one hop from an
# annotated substrate. But at inference the forest policy (Stage 2b) also has to rank
# candidates rooted at an INTERMEDIATE metabolite m1 that was itself just generated (not
# annotated). Some annotated metabolites m2 are only reachable as root -> m1 -> m2 (depth-2)
# and never appear as a direct depth-1 child of root -- scripts/census_multistep.py's
# census_depth2 measures how common this is. This section reuses that same depth-1/depth-2
# reachability idea (rather than importing scripts.census_multistep, which lives outside the
# installed `grail_metabolism` package and would break for anyone without the repo's scripts/
# dir on sys.path) to COLLECT the (m1, m2) chains instead of just counting them, then builds
# one _BiExample per chain rooted at m1 via the existing _bi_example_from_pool helper -- so
# the reranker also sees calibration examples where the "substrate" is itself a metabolite.
# NOTE: unlike census_depth2 (which pools depth-2 children across all depth-1 intermediates
# before intersecting with the annotated set, so a depth-2-only metabolite counts once no
# matter how many intermediates reach it), build_intermediate_pairs groups per unlocking
# intermediate m1 -- so the number of _BiExample objects produced here is NOT the same
# number as census_depth2's depth2_only count.
# =========================================================================== #


def _children_ik(sub: str, generator, top_k: int, max_pool: int) -> Dict[str, str]:
    """One generator hop from ``sub``: InChIKey (tautomer) -> smiles of each distinct
    rule-child, generator-order-first-wins on duplicate keys. Mirrors
    ``scripts/census_multistep.py:_children_ik`` exactly (same generate_scored_with_details
    call, same dedup rule) -- this per-hop helper is identical to the script's, though the
    caller below groups its results differently than ``census_depth2`` does (see
    ``build_intermediate_pairs``); ``max_pool`` is accepted for interface parity with that
    script but -- like there -- is NOT forwarded to the real generator call (top_k alone
    bounds it there)."""
    out: Dict[str, str] = {}
    for smiles, _gen_score, _rule_id, *_ in generator.generate_scored_with_details(
        sub, top_k=top_k, compute_sites=False
    ):
        ik = _tautomer_inchikey(smiles)
        if ik is not None and ik not in out:
            out[ik] = smiles
    return out


def build_intermediate_pairs(
    generator,
    molframe,
    n_substrates: int,
    top_k: int,
    max_pool: int = 100,
    verbose: bool = True,
) -> List[_BiExample]:
    """Depth-2 bootstrap examples: for each root substrate with an annotated metabolite m2
    that is depth-2-reachable but NOT depth-1 (root -> m1 -> m2), emit one ``_BiExample``
    rooted at the intermediate m1 with m2 (and any other such depth-2-only metabolites
    sharing that m1) as the positive hit(s) among m1's own rule-children.

    Scans up to ``n_substrates`` roots from ``molframe.map`` (same iteration order as the
    other builders); a root contributes zero or more examples (one per distinct m1 that
    unlocks a depth-2-only annotated metabolite). These are meant to be concatenated with
    the normal depth-1 ``_BiExample`` list for the reranker fine-tune (concatenation itself
    is Task 8, not here).
    """
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())
    substrates = list(molframe.map.keys())
    examples: List[_BiExample] = []
    scanned = 0
    for sub in substrates[:n_substrates]:
        scanned += 1
        annotated_ik = {_tautomer_inchikey(p) for p in molframe.map[sub]} - {None}
        if not annotated_ik:
            continue
        d1 = _children_ik(sub, generator, top_k, max_pool)
        depth1_hits = set(d1) & annotated_ik
        if not (annotated_ik - depth1_hits):
            # depth-1 already covers every annotated metabolite -> this root yields no
            # depth-2-only examples. Skip the (expensive) per-child depth-2 expansion; this
            # is exact (a fully-covered root produces zero examples either way).
            continue
        # Group depth-2-only annotated metabolites by the m1 that unlocks them.
        by_m1: Dict[str, List[str]] = {}
        for m1_smiles in d1.values():
            d2 = _children_ik(m1_smiles, generator, top_k, max_pool)
            depth2_only_iks = (set(d2) & annotated_ik) - depth1_hits
            if not depth2_only_iks:
                continue
            by_m1[m1_smiles] = [d2[ik] for ik in depth2_only_iks]
        for m1_smiles, m2_list in by_m1.items():
            pool = build_pool(generator, m1_smiles, top_k=top_k, max_pool=max_pool)
            if not pool:
                continue
            ex = _bi_example_from_pool(m1_smiles, pool, m2_list, prior, num_rules)
            if ex is None:
                continue
            examples.append(ex)
            if verbose and (len(examples) % 25 == 0):
                print(
                    f"  built {len(examples)} intermediate bi-examples "
                    f"(scanned {scanned} roots)",
                    flush=True,
                )
    return examples
