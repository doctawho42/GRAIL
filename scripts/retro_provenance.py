#!/usr/bin/env python3
"""Does the shared-provenance confound appear in retrosynthesis too?

In metabolite prediction the similarity-to-training axis turned out to measure how well studied a
substrate's chemistry is rather than its distance from the training set, because the rule bank and
the training corpus both derive from the literature of well-studied metabolism. Retrosynthesis is
the obvious second instance -- reaction templates are extracted from USPTO and models are trained on
USPTO -- and this checks the mechanism there.

Scope, stated up front. This tests the MECHANISM, not model degradation. Two things the metabolism
study had are unavailable here: the local USPTO corpus carries no atom mapping, so proper templates
cannot be extracted, and no per-reaction predictions from a retrosynthesis model were retained. What
can be done without either is the part that carries the argument, because it needs no model at all.

The design is a corpus split. USPTO reactions are partitioned by patent number into disjoint halves
A and B, so no reaction in B shares a document with any reaction in A. For each reaction in B we
compute its maximum product-Tanimoto similarity to A, and whether its transformation is covered by
A at all -- coverage being membership of its atom-map-free reaction signature in A's signature set.
Both quantities are defined without reference to any model.

If coverage falls with similarity, the two are entangled by a common cause, which is the claim: a
knowledge base derived from a corpus does not reach the chemistry that corpus under-represents, so a
similarity-to-corpus axis cannot separate distance from difficulty. If coverage is flat, the
mechanism is specific to metabolism and the generalisation fails, which is equally worth knowing.

The reaction signature is a hashed difference fingerprint -- the product's Morgan bits minus the
reactants' -- rather than a template, since without atom maps a template cannot be derived. It is a
coarser instrument than a template and is used only to ask whether a transformation of that shape
occurs in A, never to apply anything.
"""
from __future__ import annotations
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "grail_metabolism" / "data" / "USPTO_FULL.csv"
N_A, N_B, SEED = 30000, 4000, 0
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 1.01]
OUT = ROOT / "results" / "retro_provenance.json"

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse(rxn: str):
    """(reactant mols, product mol) from a `reactants>agents>product` record."""
    try:
        left, _, right = rxn.split(">")
    except ValueError:
        return None
    prod = Chem.MolFromSmiles(right)
    if prod is None or prod.GetNumHeavyAtoms() < 3:
        return None
    reacts = [m for m in (Chem.MolFromSmiles(s) for s in left.split(".")) if m is not None]
    if not reacts:
        return None
    return reacts, prod


def formula_signature(reacts, prod) -> tuple:
    """Control signature: the net change in atom counts, which shares no representation with the axis.

    The Morgan-bit signature is derived from the product's fingerprint and the similarity axis is
    computed from the same fingerprint, so part of their association could be definitional rather
    than chemical. A net-formula signature -- atoms gained and lost, plus the change in ring count
    -- is chemically meaningful and independent of Morgan bits, so if coverage still rises with
    similarity under it, the entanglement is not an artifact of a shared representation.
    """
    def counts(mols):
        c: Counter = Counter()
        for m in mols:
            for a in m.GetAtoms():
                c[a.GetSymbol()] += 1
        return c
    delta = counts([prod])
    delta.subtract(counts(reacts))
    rings = prod.GetRingInfo().NumRings() - sum(m.GetRingInfo().NumRings() for m in reacts)
    return tuple(sorted((k, v) for k, v in delta.items() if v)) + (("ring", rings),)


def signature(reacts, prod) -> int:
    """Atom-map-free reaction signature: the product's Morgan bits not present in the reactants.

    Coarser than a template and deliberately so -- without atom maps no reaction centre can be
    derived. It answers only 'does a transformation of this shape occur in the other corpus'.
    """
    pbits = set(gen.GetFingerprint(prod).GetOnBits())
    rbits: set[int] = set()
    for m in reacts:
        rbits |= set(gen.GetFingerprint(m).GetOnBits())
    formed = tuple(sorted(pbits - rbits))
    return hash(formed[:24])


def main() -> int:
    rng = random.Random(SEED)
    rows_a, rows_b = [], []
    with open(CORPUS) as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            pat = rec.get("PatentNumber") or ""
            if not pat:
                continue
            # deterministic disjoint split on the patent, so A and B share no document
            bucket = hash(pat) % 10
            if bucket < 5:
                if len(rows_a) < N_A * 3 and rng.random() < 0.05:
                    rows_a.append(rec["reactions"])
            else:
                if len(rows_b) < N_B * 3 and rng.random() < 0.02:
                    rows_b.append(rec["reactions"])
            if len(rows_a) >= N_A * 3 and len(rows_b) >= N_B * 3:
                break

    A, sig_a, fsig_a, fps_a = [], set(), set(), []
    for r in rows_a:
        p = parse(r)
        if p is None:
            continue
        sig_a.add(signature(*p))
        fsig_a.add(formula_signature(*p))
        fps_a.append(gen.GetFingerprint(p[1]))
        A.append(r)
        if len(A) >= N_A:
            break
    B = []
    for r in rows_b:
        p = parse(r)
        if p is None:
            continue
        B.append(p)
        if len(B) >= N_B:
            break
    print(f"corpus A {len(A)} reactions, {len(sig_a)} distinct signatures; corpus B {len(B)}",
          flush=True)

    sim, cov, fcov = [], [], []
    for reacts, prod in B:
        f = gen.GetFingerprint(prod)
        sim.append(max(DataStructs.BulkTanimotoSimilarity(f, fps_a)))
        cov.append(1.0 if signature(reacts, prod) in sig_a else 0.0)
        fcov.append(1.0 if formula_signature(reacts, prod) in fsig_a else 0.0)
    sim, cov, fcov = np.array(sim), np.array(cov), np.array(fcov)

    rep = {"n_a": len(A), "n_b": len(B), "n_signatures_a": len(sig_a), "bins": BINS,
           "seed": SEED, "strata": []}
    print(f"\nmax product-Tanimoto of B to A: mean {sim.mean():.3f}, median {np.median(sim):.3f}\n")
    print(f"{'stratum':>16}{'n':>7}{'A-coverage':>13}{'formula-coverage':>19}")
    for i in range(len(BINS) - 1):
        m = (sim >= BINS[i]) & (sim < BINS[i + 1])
        if m.sum() < 20:
            continue
        rep["strata"].append({"lo": BINS[i], "hi": BINS[i + 1], "n": int(m.sum()),
                              "coverage": round(float(cov[m].mean()), 4),
                              "formula_coverage": round(float(fcov[m].mean()), 4)})
        print(f"  [{BINS[i]:.2f},{BINS[i+1]:.2f}){m.sum():7}{cov[m].mean():13.4f}{fcov[m].mean():19.4f}")

    slope = float(np.polyfit(sim, cov, 1)[0])
    rng_np = np.random.default_rng(SEED)
    idx = rng_np.integers(0, len(sim), (2000, len(sim)))
    bt = np.array([np.polyfit(sim[i], cov[i], 1)[0] for i in idx])
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    rep["slope"] = {"value": round(slope, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
    rep["corr"] = round(float(np.corrcoef(sim, cov)[0, 1]), 4)
    fslope = float(np.polyfit(sim, fcov, 1)[0])
    fbt = np.array([np.polyfit(sim[i], fcov[i], 1)[0] for i in idx])
    flo, fhi = float(np.quantile(fbt, .025)), float(np.quantile(fbt, .975))
    rep["formula_slope"] = {"value": round(fslope, 4), "ci95": [round(flo, 4), round(fhi, 4)],
                            "excludes_zero": bool(flo > 0 or fhi < 0)}
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nslope of A-coverage on similarity-to-A: {slope:+.4f} [{lo:+.4f},{hi:+.4f}]"
          f"  {'SIG' if (lo > 0 or hi < 0) else 'n.s.'}")
    print(f"corr(similarity, coverage) = {rep['corr']:+.3f}")
    print(f"CONTROL, formula signature (no shared representation with the axis): "
          f"{fslope:+.4f} [{flo:+.4f},{fhi:+.4f}]  {'SIG' if (flo > 0 or fhi < 0) else 'n.s.'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
