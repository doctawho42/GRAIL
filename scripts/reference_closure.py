#!/usr/bin/env python3
"""The reference set is a graph, and the metric reads it as a list.

Metabolism is sequential. A drug is converted to a metabolite, that metabolite is converted again,
and a curated corpus records the conversions it has observed as edges: substrate to product. Every
evaluation in this field then takes one substrate at a time, looks up the products annotated for
*that* substrate, and scores a prediction correct iff it is one of them.

Those two facts do not fit together. If the corpus records A to B and, elsewhere, B to C, then it
asserts that C follows from A in two steps -- but unless some curator also wrote the edge A to C, a
method that emits C for A is scored as a false positive. Whether that edge exists is a property of
how the corpus was assembled, not of whether the chemistry happens.

This measures the mismatch, in three parts and in that order:

    1. the corpus alone. How many two-step compositions does the annotation graph contain, and for
       how many of them is the composed edge also annotated? No model is involved, so the number is
       a property of the reference data that every method on it inherits.

    2. the methods. Of the candidates a method emits and the metric scores wrong, how many are nodes
       the corpus itself reaches from that substrate in two steps? A method that enumerates deeper
       into the reaction network collects more of them, so this is a penalty that tracks a property
       of the method rather than of its correctness.

    3. the consequence. Re-score precision and F1 against the depth-d closure of the annotation
       graph instead of its depth-1 edges, and report both the shift and whether the ORDERING of
       methods moves with it. d = 1 is what the field computes; the curve in d says how much of a
       measured difference is annotation depth.

Node identity is the tautomer-aware InChIKey used everywhere else here, so the graph is not an
artifact of how a curator drew a structure. The graph is built from all three clean splits, because
the claim is about the corpus rather than about one split: the corpus is what asserts B to C while
the split asks only about A.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED, K = 10000, 0, 15
CACHE = ROOT / "results" / "closure_keys.json"


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


class Keyer:
    """Tautomer-aware InChIKey with a disk cache; keying the whole corpus is the slow step."""

    def __init__(self) -> None:
        self.cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.dirty = False
        from grail_metabolism.metrics import _tautomer_inchikey
        self._f = _tautomer_inchikey

    def __call__(self, s: str):
        if s not in self.cache:
            try:
                self.cache[s] = self._f(s)
            except Exception:
                self.cache[s] = None
            self.dirty = True
        return self.cache[s]

    def flush(self) -> None:
        if self.dirty:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache))
            self.dirty = False


def load_edges(split: str, key: Keyer) -> set:
    smi = [Chem.MolToSmiles(m) if m is not None else None
           for m in Chem.SDMolSupplier(str(ROOT / f"grail_metabolism/data/{split}.sdf"))]
    edges = set()
    for line in (ROOT / f"grail_metabolism/data/{split}_triples_clean.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[2] != "1":
            continue
        a, b = int(parts[0]), int(parts[1])
        if a >= len(smi) or b >= len(smi) or smi[a] is None or smi[b] is None:
            continue
        ka, kb = key(smi[a]), key(smi[b])
        if ka and kb and ka != kb:
            edges.add((ka, kb))
    return edges


def closure(out: dict, node, depth: int) -> set:
    """Nodes reachable from `node` in at most `depth` edges, excluding the node itself."""
    seen, frontier = set(), {node}
    for _ in range(depth):
        frontier = {c for b in frontier for c in out.get(b, ())} - seen - {node}
        if not frontier:
            break
        seen |= frontier
    return seen


def load_predictions() -> dict:
    out = {}
    sp = ROOT / "results/scored_predictions.json"
    if sp.exists():
        out["GRAIL"] = {r["sub"]: [c["smiles"] for c in r["candidates"][:K]]
                        for r in json.loads(sp.read_text())["rows"]}
    for name, path in (("SyGMa", "results/sygma_fulltest_predictions.json"),
                       ("MetaPredictor", "artifacts/tier2_1170/metapredictor_preds.json")):
        q = ROOT / path
        if not q.exists():
            continue
        d = json.loads(q.read_text())
        d = d.get("predictions", d)
        d = {s: list(v)[:K] for s, v in d.items() if isinstance(v, (list, tuple))}
        if d:
            out[name] = d
    return out


def _prf(hits: int, emitted: int, refs: int):
    p = hits / emitted if emitted else 0.0
    r = hits / refs if refs else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "reference_closure.json"))
    args = ap.parse_args()

    key = Keyer()
    per_split = {s: load_edges(s, key) for s in ("train", "val", "test")}
    key.flush()
    corpus = set().union(*per_split.values())
    out_all: dict = {}
    for a, b in corpus:
        out_all.setdefault(a, set()).add(b)
    out_te: dict = {}
    for a, b in per_split["test"]:
        out_te.setdefault(a, set()).add(b)

    def graph_stats(g: dict) -> dict:
        two = [(a, b, c) for a in g for b in g[a] if b in g for c in g[b] if c != a]
        pairs = {(a, c) for a, _, c in two}
        openp = {p for p in pairs if p[1] not in g.get(p[0], ())}
        return {"nodes": len({x for e in g for x in (e, *g[e])}),
                "edges": sum(len(v) for v in g.values()),
                "two_step_compositions": len(two),
                "distinct_composed_pairs": len(pairs),
                "composed_pairs_not_annotated": len(openp),
                "share_not_annotated": round(len(openp) / max(len(pairs), 1), 4),
                "reversible_two_paths": len([1 for a in g for b in g[a] if a in g.get(b, ())])}

    rep = {"config": {**_code_version(), "n_boot": N_BOOT, "seed": SEED, "k": K,
                      "node_identity": "inchikey_tautomer",
                      "graph": "clean triples of all three splits; the corpus is what asserts the "
                               "second edge, the split only asks about the first"},
           "edges_per_split": {s: len(e) for s, e in per_split.items()},
           "corpus_graph": graph_stats(out_all),
           "test_only_graph": graph_stats(out_te)}

    g = rep["corpus_graph"]
    print(f"corpus graph: {g['edges']} edges, {g['two_step_compositions']} two-step compositions, "
          f"{g['composed_pairs_not_annotated']} of {g['distinct_composed_pairs']} composed pairs "
          f"carry no annotated edge ({g['share_not_annotated']})", flush=True)

    truth = json.loads((ROOT / "results/test_references.json").read_text())
    preds = load_predictions()
    methods = sorted(preds)
    subs = [s for s in truth if all(s in preds[m] for m in methods)]
    print(f"{len(methods)} methods on {len(subs)} shared substrates: {', '.join(methods)}",
          flush=True)

    # per substrate: the annotated references, and what the corpus reaches beyond them
    ref_keys, reach = {}, {}
    for s in subs:
        ks = key(s)
        ref_keys[s] = {k for k in (key(y) for y in truth[s]) if k}
        reach[s] = {d: (closure(out_all, ks, d) - ref_keys[s] - {ks} if ks else set())
                    for d in range(2, args.max_depth + 1)}
    key.flush()

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))

    def ci(v):
        bt = np.asarray(v)[idx].mean(axis=1)
        return [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)]

    rep["scored_wrong_but_corpus_derivable"] = {}
    derivable_share = {}
    for m in methods:
        shares, counts, sizes = [], [], []
        for s in subs:
            emitted = {k for k in (key(x) for x in preds[m][s]) if k}
            wrong = emitted - ref_keys[s]
            der = wrong & reach[s][2]
            sizes.append(len(emitted))
            counts.append(len(der))
            shares.append(len(der) / len(wrong) if wrong else 0.0)
        derivable_share[m] = np.array(shares)
        rep["scored_wrong_but_corpus_derivable"][m] = {
            "mean_emitted": round(float(np.mean(sizes)), 3),
            "mean_derivable_per_substrate": round(float(np.mean(counts)), 4),
            "share_of_wrong_output": round(float(np.mean(shares)), 4),
            "ci95": ci(shares)}
        v = rep["scored_wrong_but_corpus_derivable"][m]
        print(f"  {m:14} emits {v['mean_emitted']:5.2f}  of its scored-wrong output "
              f"{v['share_of_wrong_output']:.4f} {v['ci95']} is reached by the corpus in two steps",
              flush=True)

    rep["derivable_share_differs_by_method"] = {}
    for a, b in itertools.combinations(methods, 2):
        d = derivable_share[a] - derivable_share[b]
        lo, hi = ci(d)
        rep["derivable_share_differs_by_method"][f"{a} vs {b}"] = {
            "delta": round(float(d.mean()), 4), "ci95": [lo, hi], "certified": bool(lo * hi > 0)}

    # The consequence: score against the depth-d closure instead of the annotated edges. Recall is
    # reported against the annotated references throughout, because widening the reference set would
    # change the question; what widens is the set a candidate may be counted against as correct.
    rep["by_closure_depth"] = {}
    key_f1 = {}
    for d in range(1, args.max_depth + 1):
        rep["by_closure_depth"][str(d)] = {}
        for m in methods:
            P, F = [], []
            for s in subs:
                allowed = ref_keys[s] | (set() if d == 1 else reach[s][min(d, args.max_depth)])
                emitted = {k for k in (key(x) for x in preds[m][s]) if k}
                # precision credits the closure, recall stays against the annotated references:
                # widening what recall is measured against would change the question, while
                # widening what a candidate may be counted correct against is the whole point
                p, _, _ = _prf(len(emitted & allowed), len(emitted), len(ref_keys[s]))
                _, r, _ = _prf(len(emitted & ref_keys[s]), len(emitted), len(ref_keys[s]))
                P.append(p)
                F.append(2 * p * r / (p + r) if (p + r) else 0.0)
            key_f1[(d, m)] = np.array(F)
            rep["by_closure_depth"][str(d)][m] = {
                "precision": round(float(np.mean(P)), 4), "precision_ci95": ci(P),
                "f1": round(float(np.mean(F)), 4), "f1_ci95": ci(F)}
        order = sorted(methods, key=lambda m: -rep["by_closure_depth"][str(d)][m]["f1"])
        rep["by_closure_depth"][str(d)]["ordering_by_f1"] = order
        print(f"  depth {d}: " + "  ".join(
            f"{m} P {rep['by_closure_depth'][str(d)][m]['precision']:.4f} "
            f"F1 {rep['by_closure_depth'][str(d)][m]['f1']:.4f}" for m in methods), flush=True)

    orders = {d: rep["by_closure_depth"][str(d)]["ordering_by_f1"] for d in
              range(1, args.max_depth + 1)}
    rep["ordering_changes_with_depth"] = len({tuple(o) for o in orders.values()}) > 1
    rep["pairs_that_change_sign_with_depth"] = {}
    for a, b in itertools.combinations(methods, 2):
        signs = {}
        for d in range(1, args.max_depth + 1):
            diff = key_f1[(d, a)] - key_f1[(d, b)]
            lo, hi = ci(diff)
            signs[str(d)] = {"margin": round(float(diff.mean()), 4), "ci95": [lo, hi],
                             "certified": bool(lo * hi > 0)}
        flips = len({v["margin"] > 0 for v in signs.values()}) > 1
        rep["pairs_that_change_sign_with_depth"][f"{a} vs {b}"] = {"flips": flips, "by_depth": signs}

    print(f"\n  F1 ordering changes with closure depth: {rep['ordering_changes_with_depth']}")
    for k_, v in rep["pairs_that_change_sign_with_depth"].items():
        if v["flips"]:
            print(f"    {k_} changes sign: " + ", ".join(
                f"d={d} {s['margin']:+.4f}" for d, s in v["by_depth"].items()))

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
