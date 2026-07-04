"""Multi-step Set-GFlowNet over the rule forest. Terminal = a set of metabolites;
reward = PU set-coverage; forward policy = the Stage-2a reranker; backward = analytic
1/#leaves. See docs/superpowers/specs/2026-07-01-set-gflownet-stage2b-design.md."""
from __future__ import annotations
import math
import multiprocessing as _mp
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List


@dataclass(frozen=True)
class ForestState:
    root: str
    max_depth: int
    max_size: int
    parent: Dict[str, str] = field(default_factory=dict)   # child_ik -> parent_ik

    def add(self, parent_ik: str, child_ik: str) -> "ForestState":
        new_parent = dict(self.parent)
        new_parent[child_ik] = parent_ik
        return replace(self, parent=new_parent)

    def terminal_set(self) -> FrozenSet[str]:
        return frozenset(self.parent.keys())

    def depth_of(self, ik: str) -> int:
        d = 0
        while ik in self.parent:
            ik = self.parent[ik]; d += 1
        return d

    def leaves(self) -> List[str]:
        parents = set(self.parent.values())
        return [ik for ik in self.parent if ik not in parents]

    def frontier(self) -> List[str]:
        nodes = [self.root] + list(self.parent.keys())
        return [n for n in nodes
                if self.depth_of(n) < self.max_depth and len(self.parent) < self.max_size]


def set_coverage_logreward(terminal_set, annotated_ik, beta: float, lam: float) -> float:
    """log R(S) = beta * (TP - lam*|S|). PU-aware: non-annotated members cost only lam
    (size), never a false-negative penalty."""
    tp = len(terminal_set & annotated_ik)
    return float(beta) * (tp - float(lam) * len(terminal_set))


def log_pb_trajectory(post_add_states) -> float:
    """Sum of log(1/#leaves) over the states reached AFTER each ADD action. The last-added
    node of a forest must be a current leaf, so P_B(remove leaf)=1/#leaves is the exact
    analytic backward for forest construction."""
    total = 0.0
    for st in post_add_states:
        n_leaves = max(len(st.leaves()), 1)
        total += math.log(1.0 / n_leaves)
    return total


def _expand_state(generator, state_smiles, top_k):
    """Deterministic top_k child expansion of one state: dedup by SMILES, drop unparseable,
    keep detached (smiles, float gscore, int rid). Shared by candidate_children AND the
    parallel pre-warm workers so the two paths are identical by construction."""
    seen, out = set(), []
    for smiles, gscore, rid, *_ in generator.generate_scored_with_details(
        state_smiles, top_k=top_k, compute_sites=False
    ):
        if smiles in seen:
            continue
        seen.add(smiles)
        if Chem.MolFromSmiles(smiles) is None:
            continue
        out.append((smiles, float(gscore), int(rid)))
    return out


# --------------------------------------------------------------------------- #
# Task 2: parallel two-wave cache pre-warm (spawn Pool, mirrors
# workflows/reranker.py's _bi_worker_init / _bi_pool_worker). Workers return PLAIN
# python data (str/float/int/dict) only -- never torch tensors -- across the mp boundary.
# --------------------------------------------------------------------------- #
_MAX_GFN_WORKERS = 8
# Persist the caches every N newly-expanded states DURING a long prewarm so a preempted /
# killed run resumes warm instead of restarting the whole (hours-long) prewarm from scratch.
_PREWARM_CKPT_EVERY = 200
_GFN_WORKER = {"gen": None, "top_k": None}


def _gfn_worker_init(gen_ckpt, top_k, rule_emb=None):
    """Pool initializer: pin torch to 1 thread, silence rdkit, load the generator once into
    a module global (mirrors reranker.py:_bi_worker_init). ``rule_emb`` (warmed ONCE in the
    parent) is injected as the rule-bank embedding cache so this worker skips the slow
    single-threaded rule-bank GNN encoding."""
    import torch as _t
    _t.set_num_threads(1)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from ..config import GeneratorConfig
    from ..model.grail import _read_checkpoint
    from ..workflows.factory import build_generator
    state = _read_checkpoint(gen_ckpt)
    gen = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    gen.load_state_dict(state["state_dict"], strict=False)
    if rule_emb is not None:
        gen._rule_embedding_cache = rule_emb   # skip the ~80s 7581-rule GNN encoding
    _GFN_WORKER["gen"], _GFN_WORKER["top_k"] = gen, top_k


def _gfn_pool_worker(state_smiles):
    """Pool worker fn: expand one state via the worker's own generator and return PLAIN data
    only -- (state, children, {smiles: tautomer_ik}) -- never torch tensors."""
    gen, top_k = _GFN_WORKER["gen"], _GFN_WORKER["top_k"]
    children = _expand_state(gen, state_smiles, top_k)               # [(smiles, float, int)]
    iks = {state_smiles: _tautomer_inchikey(state_smiles)}
    for c, _g, _rid in children:
        iks[c] = _tautomer_inchikey(c)
    return state_smiles, children, iks                              # PLAIN data only


# --------------------------------------------------------------------------- #
# Task 5: SetGFlowNetTrainer -- reranker forward policy P_F, forest rollout, TB loss.
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from rdkit import Chem  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from ..metrics import _tautomer_inchikey  # noqa: E402
from ..utils.transform import from_rdmol  # noqa: E402


class StopHead(nn.Module):
    """Scalar STOP logit from a pooled representation of the current frontier."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, frontier_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(frontier_embed).view(())


class SetGFlowNetTrainer:
    """Set-GFlowNet trainer: the Stage-2a reranker is the forward policy P_F over a forest
    rollout (each ADD action attaches one metabolite to a frontier node), trained with the
    Trajectory-Balance loss against the PU set-coverage log-reward.

    ``P_F`` at each step is a softmax over ``[reranker child logits ...] + [STOP logit]``:
    the reranker scores every candidate ADD across the frontier, and a small ``StopHead``
    scores termination from a pooled frontier embedding. ``P_B`` is the analytic
    ``1/#leaves`` backward for forest construction (``log_pb_trajectory``).
    """

    def __init__(self, generator, reranker, config, annotated_ik_fn, device=None,
                 child_cache_path=None, ik_cache_path=None):
        self.generator = generator
        self.config = config
        self.annotated_ik_fn = annotated_ik_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Unify device: the reranker (forward policy P_F) MUST live on the same device as
        # stop_head / log_z. Otherwise _frontier_embed produces a CPU embedding (from the
        # un-moved reranker encoder) that the on-device StopHead rejects -- invisible on a
        # CPU-only box, a hard `Expected all tensors to be on the same device` crash on GPU.
        self.reranker = reranker.to(self.device)
        self.log_z = nn.Parameter(torch.zeros(1, device=self.device))
        self.stop_head = StopHead(self.reranker.embed_dim).to(self.device)
        # Environment caches. Both map deterministic pure functions -- (state, top_k) ->
        # children via RDKit rule application, and SMILES -> tautomer-InChIKey -- so persisting
        # them across runs is EXACT: the expensive RDKit / tautomer-canonicalization work is
        # done once and reused by every later M1/M2/seed run. child_cache is (generator, top_k)
        # -specific (the caller must key its file by top_k); ik_cache is universal.
        self._child_cache: dict = {}
        self._ik_cache: dict = {}
        self._child_cache_path = child_cache_path
        self._ik_cache_path = ik_cache_path
        self._load_caches()
        self.loss_history_ = []

    def _load_caches(self):
        for path, attr in ((self._child_cache_path, "_child_cache"),
                           (self._ik_cache_path, "_ik_cache")):
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as fh:
                        setattr(self, attr, pickle.load(fh))
                except Exception as exc:  # a corrupt/incompatible cache must never crash training
                    print(f"[set_gflownet] WARNING: ignoring unreadable cache {path}: {exc}", flush=True)

    def save_caches(self):
        """Persist both environment caches to disk (call after training / eval). No-op if the
        paths are unset. Safe to call repeatedly; only the main process writes."""
        for path, obj in ((self._child_cache_path, self._child_cache),
                          (self._ik_cache_path, self._ik_cache)):
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "wb") as fh:
                    pickle.dump(obj, fh)

    def _save_train_ckpt(self, path, next_epoch, optimizer):
        """Atomically persist the trainable state (P_F reranker + stop_head + logZ + Adam
        moments + loss history) so a preempted/crashed run RESUMES from the last completed
        epoch instead of restarting at epoch 0. The env caches (save_caches) only spare
        re-EXPANSION; training -- the multi-hour GNN backprop -- is the dominant cost this
        guards. Written to a .tmp sibling then os.replace'd, so a kill mid-write never leaves
        a half-written checkpoint that would fail to load."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "epoch": int(next_epoch),                       # epoch index to resume FROM
            "reranker": self.reranker.state_dict(),
            "stop_head": self.stop_head.state_dict(),
            "log_z": self.log_z.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "loss_history": list(self.loss_history_),
        }, tmp)
        os.replace(tmp, path)

    def _load_train_ckpt(self, path, optimizer):
        """Restore trainable state; return the epoch to resume FROM (0 if no usable checkpoint).
        A corrupt/incompatible checkpoint is ignored (train from scratch) rather than crashing.
        ``optimizer`` must already be built over the SAME params (state_dict maps by index)."""
        if not os.path.exists(path):
            return 0
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.reranker.load_state_dict(ckpt["reranker"])
            self.stop_head.load_state_dict(ckpt["stop_head"])
            with torch.no_grad():
                self.log_z.copy_(ckpt["log_z"].to(self.device))
            optimizer.load_state_dict(ckpt["optimizer"])
            self.loss_history_ = list(ckpt.get("loss_history", []))
            return int(ckpt.get("epoch", 0))
        except Exception as exc:  # never let a bad checkpoint abort the run
            print(f"[set_gflownet] WARNING: ignoring unreadable train checkpoint {path}: {exc}",
                  flush=True)
            return 0

    def candidate_children(self, state_smiles):
        """Cached ``(child_smiles, gen_score, rule_id)`` list for a state via the generator.

        Candidates with unparseable SMILES are dropped here -- the one cached enumeration
        point -- so every downstream consumer (``_reranker_child_logits``, ``sample_forest``,
        ``policy_logits``) iterates the SAME filtered list and stays aligned between logits
        and actions. Real generators over large SMIRKS banks occasionally emit RDKit-
        unsanitizable products; without this guard ``Chem.MolFromSmiles`` returns ``None``,
        ``from_rdmol(None)`` returns ``None``, and ``Batch.from_data_list([None, ...])`` raises
        ``TypeError`` deep in the rollout.
        """
        if state_smiles not in self._child_cache:
            self._child_cache[state_smiles] = _expand_state(
                self.generator, state_smiles, self.config.top_k
            )
        return self._child_cache[state_smiles]

    def _reranker_child_logits(self, parent_smiles, children):
        """Reranker logits for ADD candidates of ONE parent (gradients flow to the reranker)."""
        mol = Chem.MolFromSmiles(parent_smiles)
        sub_graph = from_rdmol(mol)
        prod_batch = Batch.from_data_list(
            [from_rdmol(Chem.MolFromSmiles(c)) for c, _, _ in children]
        ).to(self.device)
        rule_prior = self.generator.rule_prior_logits.to(self.device)[
            torch.tensor([rid for _, _, rid in children], device=self.device)
        ]
        gen_score = torch.tensor([g for _, g, _ in children], device=self.device)
        return self.reranker(sub_graph.to(self.device), prod_batch, rule_prior, gen_score)  # [N]

    def policy_logits(self, state, cand):
        """[reranker child logits ...] + [STOP logit] for the given frontier candidates.

        ``cand`` is a list of ``(parent_ik, parent_smiles, child_smiles, child_ik, gen, rid)``
        ADD actions (the ``actions`` gathered in ``sample_forest``); ``smiles_of`` mapping is
        carried implicitly via the tuple contents. Returns ``Tensor[N+1]`` (N children + STOP)
        and the flattened action list in the same order as the child logits."""
        smiles_of = {}  # only needed for the frontier pooling; reconstruct from cand tuples
        for parent_ik, p_smiles, c_smiles, c_ik, _g, _rid in cand:
            smiles_of.setdefault(parent_ik, p_smiles)
            smiles_of.setdefault(c_ik, c_smiles)
        stop_logit = self.stop_head(self._frontier_embed(state, smiles_of)).view(1)
        by_parent = {}
        for a in cand:
            by_parent.setdefault(a[1], []).append(a)
        child_logits = []
        for p_smiles, group in by_parent.items():
            child_logits.append(
                self._reranker_child_logits(
                    p_smiles, [(c, g, rid) for _, _, c, _, g, rid in group]
                )
            )
        child_logits = (
            torch.cat(child_logits) if child_logits else torch.zeros(0, device=self.device)
        )
        flat = [a for group in by_parent.values() for a in group]
        return torch.cat([child_logits, stop_logit]), flat

    def sample_forest(self, root):
        cfg = self.config

        def ik(s):
            # Tautomer-canonical InChIKey: keeps state identity, TB reward
            # (terminal_set() ∩ annotated_ik), and eval reconstruction (run_gflownet.py's
            # ``smiles_of`` / ``evaluate_matrix``) in the SAME key space as
            # ``annotated_ik_fn`` (built with ``metrics._tautomer_inchikey``). Plain
            # ``Chem.MolToInchiKey`` would under-count TPs whenever the rule engine emits
            # a different tautomer of a true metabolite. Memoized here because
            # ``_tautomer_inchikey`` does full tautomer canonicalization (slow) and the
            # same SMILES recurs heavily across children/steps/samples; it is itself
            # ``lru_cache``d in ``metrics.py`` but a local memo avoids repeated dict/hash
            # overhead through that layer too.
            cached = self._ik_cache.get(s)
            if cached is None:
                cached = _tautomer_inchikey(s)
                self._ik_cache[s] = cached
            return cached

        root_ik = ik(root)
        state = ForestState(root=root_ik, max_depth=cfg.max_depth, max_size=getattr(cfg, "max_size", 15))
        smiles_of = {root_ik: root}
        sum_log_pf, post_add = [], []
        for _ in range(getattr(cfg, "max_size", 15)):
            # Gather candidate ADD actions across the frontier (parent, child, gscore, rid).
            actions = []
            for parent_ik in state.frontier():
                p_smiles = smiles_of.get(parent_ik, parent_ik)
                kids = self.candidate_children(p_smiles)
                for c_smiles, g, rid in kids:
                    c_ik = ik(c_smiles)
                    if c_ik in state.terminal_set() or c_ik == root_ik:
                        continue
                    actions.append((parent_ik, p_smiles, c_smiles, c_ik, g, rid))
            # Build the logit vector: [reranker child logits...] + [stop logit].
            stop_logit = self.stop_head(self._frontier_embed(state, smiles_of)).view(1)
            if not actions:
                break
            by_parent = {}
            for a in actions:
                by_parent.setdefault(a[1], []).append(a)
            child_logits = []
            for p_smiles, group in by_parent.items():
                child_logits.append(
                    self._reranker_child_logits(
                        p_smiles, [(c, g, rid) for _, _, c, _, g, rid in group]
                    )
                )
            child_logits = (
                torch.cat(child_logits) if child_logits else torch.zeros(0, device=self.device)
            )
            logits = torch.cat([child_logits, stop_logit])
            log_probs = F.log_softmax(logits, dim=0)
            idx = self._sample_index(log_probs.detach())
            sum_log_pf.append(log_probs[idx])
            if idx == len(actions):  # STOP
                break
            flat = [a for group in by_parent.values() for a in group]  # order of child_logits
            parent_ik, _, c_smiles, c_ik, _, _ = flat[idx]
            state = state.add(parent_ik, c_ik)
            smiles_of[c_ik] = c_smiles
            post_add.append(state)
        total = torch.stack(sum_log_pf).sum() if sum_log_pf else torch.zeros((), device=self.device)
        return state, total, post_add

    def _frontier_embed(self, state, smiles_of):
        """Mean of the reranker's substrate encoding over frontier nodes (coarse pooled rep).

        All frontier graphs are batched into ONE ``Batch`` so ``GraphEncoder``'s BatchNorm
        sees them together; ``encode_substrate`` handles the 1-graph edge (first rollout step,
        frontier == [root]) by running its norm layers in eval for that call."""
        graphs = []
        for ik_ in state.frontier():
            mol = Chem.MolFromSmiles(smiles_of.get(ik_, ik_))
            if mol is not None:
                g = from_rdmol(mol)
                if g is not None:
                    graphs.append(g)
        if not graphs:
            return torch.zeros(self.reranker.embed_dim, device=self.device)
        batch = Batch.from_data_list(graphs).to(self.device)
        embs = self.reranker.encode_substrate(batch)  # (k, embed_dim)
        return embs.mean(0)

    def _sample_index(self, log_probs):
        n = int(log_probs.shape[0])
        if self.config.epsilon > 0.0 and float(torch.rand(())) < self.config.epsilon:
            return int(torch.randint(0, n, (1,)))
        return int(torch.multinomial(log_probs.exp(), 1))

    def tb_loss(self, root):
        state, sum_log_pf, post_add = self.sample_forest(root)
        log_r = set_coverage_logreward(
            state.terminal_set(),
            set(self.annotated_ik_fn(root)),
            self.config.beta,
            getattr(self.config, "lam", 0.1),
        )
        log_pb = log_pb_trajectory(post_add)
        return (self.log_z.squeeze() + sum_log_pf - log_pb - log_r) ** 2

    def fit(self, substrates, epochs=None, verbose=False, resume_path=None):
        cfg = self.config
        epochs = epochs if epochs is not None else cfg.epochs
        subs = [s for s in substrates if s]
        params = [p for p in self.reranker.parameters() if p.requires_grad] + list(
            self.stop_head.parameters()
        )
        opt = torch.optim.Adam(
            [
                {"params": params, "lr": cfg.lr},
                {"params": [self.log_z], "lr": cfg.logz_lr},
            ]
        )
        self.loss_history_ = []
        # Resume from a per-run training checkpoint if one persisted (preemption / crash
        # recovery). MUST come AFTER the optimizer is built -- _load_train_ckpt restores the
        # Adam moment buffers into it. start_epoch>0 means the earlier run already completed
        # `start_epoch` epochs; if it equals `epochs` the loop is empty and we fall straight
        # through to eval (recovers a run killed after training, mid-eval).
        start_epoch = 0
        if resume_path:
            start_epoch = self._load_train_ckpt(resume_path, opt)
            if verbose and start_epoch:
                print(f"[gflownet] resumed training from epoch {start_epoch} "
                      f"(logZ={float(self.log_z):.3f})", flush=True)
        for epoch in range(start_epoch, epochs):
            self.reranker.train()
            order = torch.randperm(len(subs)).tolist()
            ep, nb = 0.0, 0
            for start in range(0, len(subs), cfg.batch_substrates):
                batch = [subs[i] for i in order[start:start + cfg.batch_substrates]]
                losses = [self.tb_loss(s) for s in batch]
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep += float(loss.item())
                nb += 1
            self.loss_history_.append(ep / max(nb, 1))
            if verbose:
                print(
                    f"setgfn epoch={epoch + 1} tb_loss={self.loss_history_[-1]:.4f} "
                    f"logZ={float(self.log_z):.3f}",
                    flush=True,
                )
            # Persist the env caches EVERY epoch so a killed run never loses the (expensive,
            # deterministic) pool-gen it already did -- the next run resumes from a warm cache.
            if self._child_cache_path or self._ik_cache_path:
                self.save_caches()
            # Persist trainable state every epoch so a preemption/crash resumes from here.
            if resume_path:
                self._save_train_ckpt(resume_path, epoch + 1, opt)
        return self

    def _expand_many(self, states, workers, gen_ckpt):
        """Expand every not-yet-cached state in ``states`` (dedup, order-preserving), merge
        the results into ``_child_cache``/``_ik_cache``, and return ``{state: children}`` for
        the newly expanded states only. Serial when ``workers<=1`` or no ``gen_ckpt`` (the
        in-method map calls the SAME ``_expand_state`` as ``candidate_children``, so results
        are identical to lazy serial expansion); a spawn ``Pool`` otherwise, with workers
        returning PLAIN python data only (never torch tensors)."""
        todo = [s for s in dict.fromkeys(states) if s and s not in self._child_cache]
        if not todo:
            return {}
        workers = max(1, min(int(workers), os.cpu_count() or 1, _MAX_GFN_WORKERS))
        expanded: dict = {}

        def _merge(s, children, iks):
            # Merge one result into the caches AS IT ARRIVES, and checkpoint every
            # _PREWARM_CKPT_EVERY states so a preemption mid-prewarm resumes warm.
            self._child_cache[s] = children
            for k, v in iks.items():
                self._ik_cache.setdefault(k, v)
            expanded[s] = children
            if self._child_cache_path and len(expanded) % _PREWARM_CKPT_EVERY == 0:
                self.save_caches()

        if workers <= 1 or not gen_ckpt:
            for s in todo:
                children = _expand_state(self.generator, s, self.config.top_k)
                iks = {s: _tautomer_inchikey(s)}
                for c, _g, _rid in children:
                    iks[c] = _tautomer_inchikey(c)
                _merge(s, children, iks)
        else:
            with torch.no_grad():
                rule_emb = self.generator._rule_embeddings(torch.device("cpu")).detach().cpu().contiguous()
            pool = _mp.get_context("spawn").Pool(
                processes=workers, initializer=_gfn_worker_init,
                initargs=(gen_ckpt, self.config.top_k, rule_emb),
            )
            try:
                for s, children, iks in pool.imap_unordered(_gfn_pool_worker, todo, chunksize=2):
                    _merge(s, children, iks)
            finally:
                pool.close(); pool.join()
        return expanded

    def prewarm_caches(self, root_smiles, workers, gen_ckpt=None, verbose=False, waves=2):
        """Populate _child_cache/_ik_cache for the states fit/eval will expand, in parallel.
        Deterministic; identical to lazy serial candidate_children. Safe to call repeatedly
        (only expands uncached states). Pure cache population -- does NOT consume the sampling RNG.

        ``waves`` bounds how deep the prewarm expands:
          - ``waves=1`` -- expand ONLY the roots (depth-0). ``fit``/eval then expand the depth-1
            states LAZILY via ``candidate_children`` as the policy actually visits them. Because a
            rollout only expands nodes it ADDS to the forest (<= max_size per sample), the visited
            depth-1 subset is far smaller than the full depth-1 frontier -- so wave1-only avoids
            the depth-1 OVER-EXPANSION (expanding all ~top_k children of every root, most never
            sampled). Preferred at scale.
          - ``waves>=2`` -- also expand ALL depth-1 children of every root up front (the original
            behavior). Fully parallel, but over-expands the unvisited depth-1 states.
        """
        roots = list(dict.fromkeys(s for s in root_smiles if s))
        wave1 = self._expand_many(roots, workers, gen_ckpt)
        if verbose:
            print(f"[gflownet] prewarm wave1: {len(wave1)} roots expanded", flush=True)
        if int(waves) >= 2 and int(getattr(self.config, "max_depth", 2)) >= 2:
            # Harvest depth-1 states from ALL roots' cached children -- not just wave1's
            # freshly-expanded ones. After wave1 every root is in _child_cache (freshly
            # expanded OR already-cached from a prior/partial run); reading wave1.values()
            # alone would skip wave2 for pre-cached roots (resume / partial cache), leaving
            # their depth-1 children cold.
            depth1 = list(dict.fromkeys(
                c
                for r in roots if r in self._child_cache
                for c, _g, _rid in self._child_cache[r]
            ))
            wave2 = self._expand_many(depth1, workers, gen_ckpt)
            if verbose:
                print(f"[gflownet] prewarm wave2: {len(wave2)} depth-1 states expanded", flush=True)
        if self._child_cache_path or self._ik_cache_path:
            self.save_caches()
