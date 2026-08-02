#!/usr/bin/env python3
"""Is a rule bank's reach a property of the knowledge base? Hold the rules fixed, swap the engine.

Appendix C.3 states the principle the cross-method reach column depends on: "A rule bank is a fixed
object: its reach is a property of the knowledge base and does not change with how the base is
queried." results/bank_overlap_sygma.json already puts that under strain -- 152 of SyGMa's 175
published rules sit verbatim in GRAIL's bank, and through GRAIL's primitives those 152 reach 0.189
where SyGMa reaches 0.542 -- but that comparison changes two things at once, the engine and the 23
rules GRAIL does not hold.

This separates them on one population, with one matcher, at one application depth:

    A  GRAIL's apply_rules_to_molecule, the 152 contained rules      (reproduces 0.1887)
    B  SyGMa's engine, the same 152 rules, one step, nothing composed
    C  SyGMa's engine, all 175 rules, one step
    D  SyGMa's engine, all 175 rules, its deployed phase-I-then-phase-II run

B minus A is the engine at fixed rules. C minus B is the 23 rules at fixed engine. D minus C is the
composition step. Everything is micro, matched by the same tautomer-aware criterion the ceiling
uses, on the substrates results/ceiling_by_provenance.json is measured on.
"""
from __future__ import annotations

import json, multiprocessing, os, sys, tempfile, time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.utils.preparation import apply_rules_to_molecule
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
SYGMA_RULES = Path("/Users/nikitapolomosnov/anaconda3/lib/python3.10/site-packages/sygma/rules")
OUT = ROOT / "results" / "reach_engine_vs_bank.json"
_CTX: dict = {}


def rule_lines() -> dict[str, list[str]]:
    """Raw rule lines per phase file, comments and blanks dropped."""
    out = {}
    for f in ("phase1.txt", "phase2.txt"):
        out[f] = [l for l in (SYGMA_RULES / f).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    return out


def _write_subset(lines_by_file, keep: set[str] | None, tmp: Path) -> dict[str, str]:
    """Rule files holding only `keep` (or everything when None), in SyGMa's own format."""
    paths = {}
    for f, lines in lines_by_file.items():
        sel = [l for l in lines if keep is None or l.split("\t")[0].strip() in keep]
        p = tmp / f
        p.write_text("\n".join(sel) + "\n")
        paths[f] = str(p)
    return paths


def _init(cfg):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX.update(cfg)
    _CTX["scenarios"] = None


def _sygma_pool(mol, paths, composed: bool):
    import sygma
    if _CTX["scenarios"] is None:
        _CTX["scenarios"] = {}
    key = (tuple(sorted(paths.items())), composed)
    if key not in _CTX["scenarios"]:
        if composed:
            _CTX["scenarios"][key] = [sygma.Scenario([[paths["phase1.txt"], 1],
                                                      [paths["phase2.txt"], 1]])]
        else:
            _CTX["scenarios"][key] = [sygma.Scenario([[paths["phase1.txt"], 1]]),
                                      sygma.Scenario([[paths["phase2.txt"], 1]])]
    out = []
    for sc in _CTX["scenarios"][key]:
        try:
            t = sc.run(Chem.Mol(mol))
            t.calc_scores()
            out += [e[0] for e in t.to_smiles()]
        except Exception:
            pass
    return out


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return (0, 0, 0, 0, 0)
    pools = {
        "A": list(apply_rules_to_molecule(mol, _CTX["contained"],
                                          normalization_mode="canonical").keys()),
        "B": _sygma_pool(mol, _CTX["paths_152"], False),
        "C": _sygma_pool(mol, _CTX["paths_all"], False),
        "D": _sygma_pool(mol, _CTX["paths_all"], True),
    }
    hits, u = {}, None
    for k, p in pools.items():
        uu, c, _ = _tautomer_recovered(trues, p, audit=False)
        u = int(uu)
        hits[k] = int(c)
    return (u, hits["A"], hits["B"], hits["C"], hits["D"])


def main() -> int:
    bank = {l.strip() for l in open(ROOT / "grail_metabolism/resources/extended_smirks.txt") if l.strip()}
    lines = rule_lines()
    sy = sorted({l.split("\t")[0].strip() for ls in lines.values() for l in ls})
    contained = sorted(r for r in sy if r in bank)
    print(f"SyGMa rules {len(sy)}; contained in GRAIL's bank {len(contained)}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="sygma_subset_"))
    (tmp / "all").mkdir(); (tmp / "c152").mkdir()
    paths_all = _write_subset(lines, None, tmp / "all")
    paths_152 = _write_subset(lines, set(contained), tmp / "c152")

    src = json.loads((ROOT / "results/filter_vs_prior_ci.json").read_text())["per_substrate"]
    refs = json.loads((ROOT / "results/test_references.json").read_text())
    items = [(r["sub"], refs[r["sub"]]) for r in src if refs.get(r["sub"])]
    print(f"substrates: {len(items)}", flush=True)

    cfg = {"contained": contained, "paths_all": paths_all, "paths_152": paths_152}
    ctx = multiprocessing.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(cfg,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 4), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    arr = np.array(rows)
    U, cols = arr[:, 0], {k: arr[:, i] for i, k in enumerate("ABCD", start=1)}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(U), (N_BOOT, len(U)))

    def est(v):
        pt = v.sum() / U.sum()
        bt = np.array([v[j].sum() / U[j].sum() for j in idx])
        return {"point": round(float(pt), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    rep = {
        "match": "inchikey_tautomer (run_benchmark._tautomer_recovered, as in bank_overlap_sygma.py)",
        "n": len(items), "n_boot": N_BOOT, "seed": SEED,
        "substrate_source": "results/filter_vs_prior_ci.json (the ceiling_by_provenance substrates)",
        "arms": {
            "A_grail_engine_152_rules": est(cols["A"]),
            "B_sygma_engine_152_rules_one_step": est(cols["B"]),
            "C_sygma_engine_175_rules_one_step": est(cols["C"]),
            "D_sygma_engine_175_rules_composed": est(cols["D"]),
        },
        "contrasts": {
            "engine_at_fixed_rules_B_minus_A": est(cols["B"] - cols["A"]),
            "the_23_rules_at_fixed_engine_C_minus_B": est(cols["C"] - cols["B"]),
            "composition_D_minus_C": est(cols["D"] - cols["C"]),
        },
        "gates": {"bank_overlap_sygma_committed_A": 0.1887,
                  "grail_full_bank_same_substrates": 0.7284},
        "n_rules": {"sygma_total": len(sy), "contained": len(contained)},
    }
    print(json.dumps(rep["arms"], indent=1), flush=True)
    print(json.dumps(rep["contrasts"], indent=1), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
