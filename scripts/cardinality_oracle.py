#!/usr/bin/env python3
"""How much is a per-substrate output size worth, before building anything to predict one?

Every method here emits a fixed number of candidates, so the output size is a constant chosen by
convention rather than a property of the substrate. A cardinality head would replace that constant
with a per-substrate estimate. Before training one, this measures the ceiling such a head could
reach: for each substrate, F1@k is computable exactly for every k from the ranked prediction list
and the annotated set, so the best achievable cut k* is known.

Three quantities decide whether the direction is worth pursuing.

1. The headroom: macro F1 at oracle k* against macro F1 at the best single global k. If an oracle
   that knows the perfect cut per substrate barely beats one constant, no head can help, and the
   direction closes here rather than after weeks of training.
2. The spread of k*. A head can only learn what varies. If k* is nearly constant the task is
   degenerate, whatever the headroom looks like.
3. Whether k* is predictable at all from something cheap. We regress it on substrate size and on
   the method's own score profile, as a floor on learnability: a head reading the full graph should
   beat these, but if even the ordering is unlearnable from them the prior is poor.

The oracle is defined against ANNOTATED sets, so it is the ceiling on measurable agreement, not on
chemistry. That is the same property the endpoint has, and it is why the head predicts how many to
emit rather than how many exist.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
KMAX = 30
OUT = ROOT / "results" / "cardinality_oracle.json"


def keyed(items, table):
    """Ranked prediction keys, deduplicated in rank order, as the harness does."""
    seen, out = set(), []
    for s in items:
        k = table.get(s)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def f1_at(pred_keys, real, k):
    tp = len(set(pred_keys[:k]) & real)
    n = min(k, len(pred_keys))
    return 2 * tp / (n + len(real)) if (n + len(real)) else 0.0


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())
    grail = {r["sub"]: r["deployed_top15"]
             for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    t2 = ROOT / "artifacts" / "tier2"
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
        "BioTransformer": json.loads((t2 / "biotransformer_preds.json").read_text()),
        "MetaTrans": json.loads((t2 / "metatrans_preds.json").read_text()),
    }

    rep = {"mode": MODE, "k_max": KMAX, "methods": {}}
    for name, preds in methods.items():
        subs = sorted(set(preds) & set(truth))
        subs = [s for s in subs if truth[s]]
        if not subs:
            continue
        rows = []
        for s in subs:
            pk = keyed(preds.get(s, []), table)
            real = {k for k in (table.get(x) for x in truth[s]) if k}
            if not real:
                continue
            curve = [f1_at(pk, real, k) for k in range(1, KMAX + 1)]
            best = int(np.argmax(curve)) + 1
            rows.append({"sub": s, "kstar": best, "f1_star": curve[best - 1],
                         "n_ref": len(real), "n_pred": len(pk), "curve": curve})
        if not rows:
            continue
        kstars = [r["kstar"] for r in rows]
        # best single global k, chosen on this same set -- an optimistic constant baseline
        by_k = [st.mean([r["curve"][k - 1] for r in rows]) for k in range(1, KMAX + 1)]
        k_glob = int(np.argmax(by_k)) + 1
        f1_glob = by_k[k_glob - 1]
        f1_oracle = st.mean([r["f1_star"] for r in rows])
        # is k* predictable from the cheapest signals available without a model?
        nref = np.array([r["n_ref"] for r in rows], dtype=float)
        npred = np.array([r["n_pred"] for r in rows], dtype=float)
        ks = np.array(kstars, dtype=float)
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        rep["methods"][name] = {
            "n_substrates": len(rows),
            "k_global_best": k_glob,
            "macro_f1_global": round(f1_glob, 4),
            "macro_f1_oracle": round(f1_oracle, 4),
            "headroom": round(f1_oracle - f1_glob, 4),
            "headroom_relative": round((f1_oracle - f1_glob) / f1_glob, 3) if f1_glob else None,
            "kstar": {"mean": round(st.mean(kstars), 2), "sd": round(st.pstdev(kstars), 2),
                      "median": int(st.median(kstars)), "min": min(kstars), "max": max(kstars),
                      "frac_at_1": round(sum(1 for k in kstars if k == 1) / len(kstars), 3),
                      "frac_le_3": round(sum(1 for k in kstars if k <= 3) / len(kstars), 3)},
            "kstar_corr_with_n_ref": round(corr(ks, nref), 3),
            "kstar_corr_with_n_pred": round(corr(ks, npred), 3),
        }

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"criterion {MODE}, k swept 1..{KMAX}\n")
    hdr = f"{'method':16}{'n':>6}{'k_glob':>8}{'F1@k_glob':>11}{'F1@k*':>9}{'headroom':>10}{'rel':>8}"
    print(hdr)
    for name, m in rep["methods"].items():
        print(f"{name:16}{m['n_substrates']:6}{m['k_global_best']:8}{m['macro_f1_global']:11.4f}"
              f"{m['macro_f1_oracle']:9.4f}{m['headroom']:+10.4f}{m['headroom_relative']:8.1%}")
    print(f"\n{'method':16}{'k* mean':>9}{'sd':>7}{'med':>5}{'max':>5}{'@1':>8}{'<=3':>8}{'r(n_ref)':>10}")
    for name, m in rep["methods"].items():
        k = m["kstar"]
        print(f"{name:16}{k['mean']:9.2f}{k['sd']:7.2f}{k['median']:5}{k['max']:5}"
              f"{k['frac_at_1']:8.1%}{k['frac_le_3']:8.1%}{m['kstar_corr_with_n_ref']:10.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
