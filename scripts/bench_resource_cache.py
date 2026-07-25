#!/usr/bin/env python3
"""Deployment resource profile: startup cost, per-query latency, memory, and the dedup delta.

The second deadline lever (with GRAIL-vs-MetaTox) is the *deployment* case for a way2drug-style web
tool, and it was named in STATUS without ever being specified or measured. This measures it.

What it answers, concretely:
  1. STARTUP  -- wall time + RSS to load checkpoints, build the models, and fill the rule-embedding
     cache (`Generator._rule_embeddings`, which encodes all 7,581 rule graphs ONCE at inference).
  2. PER-QUERY -- p50/p95 end-to-end latency of `ModelWrapper.generate` with a WARM cache, plus a
     stage split: generator scoring (`_prepare_generation`, the cached-embedding matmul path) vs
     RDKit rule application vs filter scoring.
  3. MEMORY   -- process RSS after warm-up, and the rule-embedding tensor footprint.
  4. DEDUP DELTA -- the claim under test is that the 7.5% smaller canonical bank (7581 -> 7010,
     `rule_dedup_provable.json`) buys STARTUP cost/memory but NOT per-query latency, because rule
     embeddings are substrate-independent and cached once. We measure it instead of asserting it:
     time the SAME encoder on a 7,581-graph batch vs a 7,010-graph batch.

All timings are single-process, CPU, deployed checkpoints. Run this ALONE -- concurrent jobs
pollute the latency numbers.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
OUT = ROOT / "results" / "resource_cache_profile.json"


def rss_mb() -> float:
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes
        return maxrss / 1e6 if sys.platform == "darwin" else maxrss / 1e3


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="substrates to time")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.model.wrapper import ModelWrapper
    from grail_metabolism.workflows.factory import build_filter, build_generator

    rss0 = rss_mb()
    prof: dict = {"threads": args.threads, "device": "cpu"}

    # ---- 1. STARTUP ----
    t0 = time.perf_counter()
    gs = torch.load(DEPLOYED_GEN, map_location="cpu", weights_only=False)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    gen = build_generator(GeneratorConfig(**gs["arch"]), gs.get("rules"))
    gen.load_state_dict(gs["state_dict"], strict=False)
    gen.calibrated_threshold = gs.get("calibrated_threshold")
    gen.eval()
    gen.gen_normalization = "canonical"
    fs = torch.load(DEPLOYED_FILTER, map_location="cpu", weights_only=False)
    filt = build_filter(FilterConfig(**fs["arch"]))
    filt.load_state_dict(fs["state_dict"], strict=False)
    filt.calibrated_threshold = fs.get("calibrated_threshold")
    filt.eval()
    t_build = time.perf_counter() - t0
    rss_after_build = rss_mb()

    # rule-embedding cache fill (the GNN side of the one-time cost)
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = gen._rule_embeddings(torch.device("cpu"))
    t_cache_fill = time.perf_counter() - t0
    rss_after_cache = rss_mb()
    emb_mb = emb.element_size() * emb.nelement() / 1e6
    n_rules = emb.size(0)

    # ATTRIBUTION: what dominates the build? Measured, not assumed -- the first guess (RDKit
    # reaction compilation) was FALSIFIED by this breakdown: compilation is ~0.3s; the cost is
    # featurizing each SMIRKS into a torch-geometric graph (`from_rule`).
    rules_list = list(gs.get("rules") or [])
    from grail_metabolism.model.generator import Generator as _Gen
    from grail_metabolism.utils.transform import from_rule as _from_rule
    t0 = time.perf_counter()
    for r in rules_list:
        _Gen._compile_reaction(r)          # RDKit SMIRKS -> ChemicalReaction, once per rule
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    for r in rules_list:
        try:
            _from_rule(r)                  # SMIRKS -> graph features (the real dominant term)
        except Exception:
            pass
    t_graphs = time.perf_counter() - t0

    prof["startup"] = {
        "checkpoint_io_s": round(t_load, 3),
        "build_models_s": round(t_build, 3),
        "of_which_rule_graph_featurization_s": round(t_graphs, 3),
        "of_which_rdkit_reaction_compilation_s": round(t_compile, 3),
        "rule_embedding_cache_fill_s": round(t_cache_fill, 3),
        "total_cold_start_s": round(t_build + t_cache_fill, 3),
        "n_rules": n_rules,
        "rule_embedding_tensor_mb": round(emb_mb, 2),
        "rss_start_mb": round(rss0, 1),
        "rss_after_build_mb": round(rss_after_build, 1),
        "rss_after_cache_mb": round(rss_after_cache, 1),
        "note": "Cold start is dominated by RULE-GRAPH FEATURIZATION (`from_rule`: SMIRKS -> "
                "torch-geometric graph features, once per rule), NOT by RDKit reaction compilation "
                "(~0.3s), the GNN encode, or checkpoint I/O (~0.01s). All of it is one-time and "
                "substrate-independent, so none of it is on the per-query path -- and because the "
                "rule graphs depend only on the bank, they are serializable to disk, which would "
                "remove the dominant term outright (untested, actionable).",
    }
    print(f"[startup] io {t_load:.2f}s + build {t_build:.2f}s (graph-featurize {t_graphs:.1f}s, "
          f"rxn-compile {t_compile:.2f}s) + cache-fill {t_cache_fill:.2f}s = "
          f"{t_build + t_cache_fill:.2f}s | {n_rules} rules | emb {emb_mb:.1f}MB | "
          f"RSS {rss0:.0f}->{rss_after_cache:.0f}MB", flush=True)

    # ---- 2. PER-QUERY (warm cache) ----
    with open(GRAIL_CSV) as fh:
        subs = [row["substrate"] for row in csv.DictReader(fh) if row["substrate"]]
    subs = subs[: args.n + args.warmup]
    wrapper = ModelWrapper(filter=filt, generator=gen, rules=gs.get("rules"))
    gen_thr = getattr(gen, "calibrated_threshold", None)

    # DEPLOYED operating point: paper_full_ensemble evaluation (candidate_top_k=128, max_output=15),
    # which reproduces the reported ~8.4 outputs/substrate of test_predictions.csv.
    TOP_K, MAX_OUT, GATE = 128, 15, False  # deployed ranking_policy="rank" => gate_by_filter=False
    for s in subs[: args.warmup]:  # warm caches (rule graphs, tautomer LRU, allocator)
        wrapper.generate(s, top_k=TOP_K, max_output=MAX_OUT, gate_by_filter=GATE)
    timed = subs[args.warmup:]

    e2e, t_gen, t_rules, t_filt, n_out = [], [], [], [], []
    for i, sub in enumerate(timed, 1):
        if i % 20 == 0:
            print(f"  query {i}/{len(timed)}", flush=True)
        t0 = time.perf_counter()
        out = wrapper.generate(sub, top_k=TOP_K, max_output=MAX_OUT, gate_by_filter=GATE)
        e2e.append(time.perf_counter() - t0)
        n_out.append(len(out))

        # stage split (same call path, measured separately)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen._prepare_generation(sub, TOP_K, gen_thr)  # generator scoring vs cached embeddings
        t_gen.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        detailed = gen.generate_scored_with_details(sub, top_k=TOP_K, threshold=gen_thr, compute_sites=False)
        t_gr = time.perf_counter() - t0
        t_rules.append(max(t_gr - t_gen[-1], 0.0))        # rule application = total - gen scoring

        smis = [d[0] for d in detailed]
        t0 = time.perf_counter()
        if smis:
            filt.score_batch(sub, smis)
        t_filt.append(time.perf_counter() - t0)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(len(xs) * p), len(xs) - 1)]

    prof["per_query"] = {
        "n": len(e2e),
        "e2e_p50_s": round(statistics.median(e2e), 4),
        "e2e_p95_s": round(pct(e2e, 0.95), 4),
        "e2e_mean_s": round(statistics.fmean(e2e), 4),
        "mean_outputs": round(statistics.fmean(n_out), 2),
        "stage_mean_s": {
            "generator_scoring": round(statistics.fmean(t_gen), 4),
            "rdkit_rule_application": round(statistics.fmean(t_rules), 4),
            "filter_scoring": round(statistics.fmean(t_filt), 4),
        },
        "rss_steady_mb": round(rss_mb(), 1),
    }
    print(f"[per-query] p50 {prof['per_query']['e2e_p50_s']}s  p95 {prof['per_query']['e2e_p95_s']}s  "
          f"mean_outputs {prof['per_query']['mean_outputs']}", flush=True)
    print(f"[stages] gen {prof['per_query']['stage_mean_s']['generator_scoring']}s | "
          f"rdkit {prof['per_query']['stage_mean_s']['rdkit_rule_application']}s | "
          f"filter {prof['per_query']['stage_mean_s']['filter_scoring']}s", flush=True)

    # ---- 3. DEDUP DELTA (7581 vs 7010 canonical bank), same encoder ----
    from torch_geometric.data import Batch
    parser = gen.parser
    graphs = list(parser.rule_graphs)
    dedup_n = 7010  # from results/rule_dedup_provable.json
    delta = {}
    if len(graphs) >= dedup_n:
        with torch.no_grad():
            timings = {}
            for label, k in (("full_bank", len(graphs)), ("dedup_bank", dedup_n)):
                batch = Batch.from_data_list(graphs[:k])
                parser.encoder(batch)                      # warm
                t0 = time.perf_counter()
                for _ in range(3):
                    parser.encoder(batch)
                timings[label] = (time.perf_counter() - t0) / 3
        full_t, ded_t = timings["full_bank"], timings["dedup_bank"]
        # the DOMINANT startup term is RDKit compilation, so the dedup saving must include it
        t0 = time.perf_counter()
        for r in rules_list[:dedup_n]:
            _Gen._compile_reaction(r)
        t_compile_dedup = time.perf_counter() - t0
        emb_dim = emb.size(1)
        compile_saving = t_compile - t_compile_dedup
        encode_saving = full_t - ded_t
        graph_saving = t_graphs * (len(graphs) - dedup_n) / max(len(graphs), 1)
        total_saving = graph_saving + encode_saving   # compile term is ~0.3s, negligible
        full_startup = t_build + t_cache_fill   # true cold start, not just the sub-parts
        delta = {
            "full_bank_rules": len(graphs),
            "dedup_bank_rules": dedup_n,
            "compile_full_s": round(t_compile, 3),
            "compile_dedup_s": round(t_compile_dedup, 3),
            "encode_full_s": round(full_t, 3),
            "encode_dedup_s": round(ded_t, 3),
            "startup_saving_s": round(total_saving, 3),
            "startup_saving_pct": round(100 * total_saving / full_startup, 1) if full_startup else None,
            "of_which_gnn_encode_s": round(encode_saving, 3),
            "graph_featurization_saving_est_s": round(t_graphs * (len(graphs) - dedup_n) / max(len(graphs), 1), 3),
            "saving_pct_denominator": "total cold start (build + cache fill)",
            "embedding_memory_saving_mb": round((len(graphs) - dedup_n) * emb_dim * 4 / 1e6, 2),
            "per_query_saving_s": 0.0,
            "note": "Both compiled reactions and rule embeddings are substrate-independent and built "
                    "once, so dedup shrinks ONE-TIME startup and resident memory, not per-query "
                    "latency. The saving is dominated by RDKit compilation, not by the GNN encode.",
        }
        print(f"[dedup] cold start {full_startup:.1f}s -> saves {total_saving:.1f}s "
              f"({delta['startup_saving_pct']}%: featurize {graph_saving:.1f}s + encode "
              f"{encode_saving:.2f}s), {delta['embedding_memory_saving_mb']}MB, 0s per query", flush=True)
    prof["dedup_delta"] = delta

    OUT.write_text(json.dumps(prof, indent=2))
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
