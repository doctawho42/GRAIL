#!/usr/bin/env python3
"""Phase 2: GLORYxR run locally over the evaluated population, in both SOM modes.

Writes results/gloryxr_local_preds_{default,strict}.json.

This is the arm the revision asked for and that an earlier record wrongly reported as impossible.
GLORYxR's implementation is public (GPL-3), its featuriser arrives as the `fame3r` dependency, and
the model dumps its documentation calls unpublished were supplied by its author. What blocked the
arm was never the code; it was that nobody had looked for it.

Two modes, differing only in `strict_soms` on the Reactor. The tutorial's own example shows what
the flag does: with promiscuous atom matching, reactions score highly because the rule matched an
atom that is not really the site, and strict SOM annotation restricts the site to the most relevant
atom or atoms. Both are produced here because the revision asked for both, and because reporting one
as though it were the tool's only behaviour is the kind of undeclared choice this paper is about.

Run under the isolated interpreter, NOT the repository's own:

    <scratchpad>/gloryxr_env/bin/python revision/phase2_gloryxr_run.py \
        --models <scratchpad>/phase2/models/multi_models --mode default

GLORYxR requires Python >= 3.13 while this repository runs 3.10, so `import gloryxr` is deliberately
lazy: the module must import under either interpreter or its tests cannot run at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODES = ("default", "strict")
POPULATION_FILE = ROOT / "results" / "test_references.json"
# One phase setting for both modes, matching the service run this column sits beside: the service
# was called with metabolism_phase=phase_1_and_2, and "1+2" is the same choice here.
PHASE = "1+2"
CHUNK = 25


def population() -> list:
    """The evaluated substrates, sorted, as the corpus stores them.

    The corpus drawing is used rather than a re-tautomerised one, because that is what every other
    comparator was handed and what the scoring joins on. It does mean 155 of the 1,170 carry an
    imidic-acid form of an amide; they parse and predict, which was checked before this ran.
    """
    return sorted(json.loads(POPULATION_FILE.read_text()))


def output_path(mode: str) -> Path:
    """Where one mode's predictions go. An unknown mode raises rather than defaulting.

    The names say which system produced them. A file called gloryx_something would be read as the
    service column, and the two are different systems on different code paths.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known modes are {', '.join(MODES)}")
    return ROOT / "results" / f"gloryxr_local_preds_{mode}.json"


def checkpoint_path(mode: str) -> Path:
    """Written after every chunk, so an interrupted run costs a chunk and not the run."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known modes are {', '.join(MODES)}")
    return ROOT / "results" / f".gloryxr_local_partial_{mode}.json"


def pending(subs: list, held_path) -> list:
    """Those substrates an existing output does not already carry, in the order given."""
    held: set = set()
    path = Path(held_path)
    if path.exists():
        blob = json.loads(path.read_text())
        inner = blob.get("predictions") if isinstance(blob, dict) else None
        held = set(inner if isinstance(inner, dict)
                   else (blob if isinstance(blob, dict) else ()))
    return [s for s in subs if s not in held]


def rank(pairs) -> list:
    """Distinct product structures, best score first.

    One prediction is one product structure. A duplicate keeps its highest score rather than
    whichever copy came first, because the column is a ranked list of distinct metabolites and
    ordering by file position is ordering by accident. Ties break on the structure itself so two
    runs over the same data agree.
    """
    best: dict = {}
    for smiles, score in pairs:
        if smiles not in best or score > best[smiles]:
            best[smiles] = score
    return [s for s, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def unprocessed(flat, requested) -> list:
    """Substrates the run never reached, as distinct from ones it answered with nothing.

    The reason this exists. An earlier version closed with `for s in requested:
    flat.setdefault(s, [])`, which filled every substrate the loop had not reached with an empty
    list -- so a run stopped halfway wrote a file carrying all 1,170 keys, and
    `phase1_tmain.coverage_gaps` checks key presence only, so the artifact passed the coverage gate
    while three quarters of it was silence recorded as "no metabolites predicted". That is the
    single most dangerous defect in this runner: it turns an incomplete run into a full column of
    zero-recall substrates, and every recall figure downstream would be wrong in the direction of
    making the comparator look worse.

    A substrate RDKit cannot parse IS answered: it is recorded with an empty list on purpose, and
    it reaches `flat` inside the loop, so it does not appear here.
    """
    return [s for s in requested if s not in flat]


def load_models(models_path):
    """The delivered FAME3R model family, by reaction subset.

    MultiFAME3RModelProvider is the one whose own docstring says predictions "will closely follow
    those generated by the original GLORYx implementation", which is what makes this column
    comparable to the service one. The eight file names are the eight rule subsets the reaction
    table declares, and the provider indexes models by exactly that key.

    The import is here rather than at module scope: this file has to be importable under the
    repository's 3.10 interpreter for its tests, while GLORYxR needs 3.13.
    """
    path = Path(models_path)
    if not path.exists():
        raise ValueError(f"REFUSING: {path} does not exist; the model dumps are not in this "
                         f"repository and their location must be given")
    from gloryxr.models.fame3r import MultiFAME3RModelProvider
    return MultiFAME3RModelProvider.load(path)


def _predictor(models, strict: bool):
    from gloryxr import GLORYxR, Reactor
    return GLORYxR(models=models, reactor=Reactor.load_builtin(phase=PHASE, strict_soms=strict))


def _package_versions() -> dict:
    """The versions the numbers actually came from, read at run time rather than assumed.

    Only the packages that can move a prediction: scikit-learn unpickles the forests, cdpkit and
    fame3r compute the descriptors, rdkit parses and canonicalises. A name that is not installed
    records None rather than being omitted, so a missing entry cannot be mistaken for an absent
    dependency.
    """
    import importlib.metadata as md
    out = {}
    for name in ("scikit-learn", "cdpkit", "fame3r", "rdkit", "numpy", "scipy", "joblib"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def _products(reactions) -> list:
    """(product SMILES, score) for each predicted reaction.

    GLORYxR already deduplicates by product SMILES inside predict_one, keeping the highest score;
    this re-derives the SMILES the same way it does so the pair list and its own dedup agree.
    """
    from rdkit.Chem.rdmolfiles import MolToSmiles
    from gloryxr.utils import mol_without_mappings
    out = []
    for rxn in reactions:
        smiles = MolToSmiles(mol_without_mappings(rxn.GetProductTemplate(0)))
        out.append((smiles, float(rxn.GetDoubleProp("Score"))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="directory of per-subset .joblib dumps, supplied by GLORYxR's author")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    strict = args.mode == "strict"
    out_path, ckpt = output_path(args.mode), checkpoint_path(args.mode)

    requested = population()
    subs = pending(requested, ckpt)
    if args.limit:
        subs = subs[:args.limit]

    flat, detail = {}, {}
    if ckpt.exists():
        prior = json.loads(ckpt.read_text())
        flat, detail = prior["predictions"], prior.get("with_score", {})
        print(f"  resuming: {len(flat)} substrates already held", flush=True)

    print(f"{args.mode}: {len(requested)} substrates, {len(subs)} to predict, "
          f"strict_soms={strict}", flush=True)

    from rdkit import RDLogger
    from rdkit.Chem.rdmolfiles import MolFromSmiles
    RDLogger.DisableLog("rdApp.*")

    models = load_models(args.models)
    predictor = _predictor(models, strict)

    unparseable = []
    t0 = time.perf_counter()
    for i, s in enumerate(subs, 1):
        mol = MolFromSmiles(s)
        if mol is None:
            # Recorded, not silently dropped: a substrate the corpus holds and RDKit will not read
            # is a fact about the drawing, and the population must stay the population.
            unparseable.append(s)
            flat[s], detail[s] = [], []
            continue
        pairs = _products(predictor.predict_one(mol))
        flat[s] = rank(pairs)
        detail[s] = [{"smiles": a, "score": round(b, 6)} for a, b in
                     sorted({k: v for k, v in pairs}.items(), key=lambda kv: (-kv[1], kv[0]))]
        if i % CHUNK == 0 or i == len(subs):
            ckpt.write_text(json.dumps({"predictions": flat, "with_score": detail}))
            print(f"  {i}/{len(subs)} ({time.perf_counter() - t0:.0f}s) "
                  f"mean list {sum(len(v) for v in flat.values()) / max(len(flat), 1):.1f}",
                  flush=True)

    # Refuse to write a full-coverage artifact for a partial run. Filling the gap with empty
    # lists would pass the coverage gate and read as a comparator that predicted nothing for
    # those substrates.
    missing = unprocessed(flat, requested)
    if missing:
        partial = out_path.with_name(out_path.stem + ".INCOMPLETE.json")
        partial.write_text(json.dumps({"predictions": flat, "with_score": detail,
                                       "n_requested": len(requested),
                                       "n_processed": len(flat),
                                       "n_unprocessed": len(missing)}, indent=1))
        print(f"\nREFUSING to write {out_path.name}: {len(missing)} of {len(requested)} "
              f"substrates were never processed. An artifact with all {len(requested)} keys "
              f"would pass the coverage gate while recording silence as zero recall.")
        print(f"  partial state in {partial.relative_to(ROOT)}; re-run to resume from the checkpoint")
        return 1

    report = {
        "what_this_is": (f"GLORYxR's predictions on the evaluated population, {args.mode} SOM "
                         f"mode, as ranked distinct product SMILES"),
        "system": "GLORYxR",
        "mode": args.mode,
        "strict_soms": strict,
        "phase": PHASE,
        "obtained_from": {
            "kind": "local run, this machine",
            "source": "https://github.com/molinfo-vienna/GLORYxR",
            "models": str(args.models),
            "model_provider": "MultiFAME3RModelProvider",
            "why_this_provider": ("its docstring states predictions will closely follow those of "
                                  "the original GLORYx implementation"),
            "interpreter": sys.version.split()[0],
            # The versions actually loaded, because they are the open question about this column
            # and not a formality. The delivered dumps were pickled by scikit-learn 1.9.0 while
            # this environment resolves 1.9.1, and the project's own uv.lock pins 1.9.0; the
            # riskier drift is cdpkit, which recomputes the descriptors at run time and is locked
            # at 1.2.3. Widths match either way, so a value drift would fail silently -- which is
            # precisely why the versions belong in the artifact rather than in a note.
            "packages": _package_versions(),
            "environment_note": ("the project pins its dependencies in uv.lock; an environment "
                                 "built by a fresh resolve rather than `uv sync` may differ, and "
                                 "scikit-learn grants no guarantee for reading a pickle written "
                                 "by an older version -- its own warning says the load may give "
                                 "invalid results"),
        },
        "drawing": "the substrate as the corpus stores it, which is what every other arm was handed",
        "n_substrates": len(flat),
        "n_with_at_least_one_prediction": sum(1 for v in flat.values() if v),
        "n_unparseable": len(unparseable),
        "unparseable": unparseable,
        "mean_returned": round(sum(len(v) for v in flat.values()) / max(len(flat), 1), 2),
        "predictions": flat,
        "with_score": detail,
    }
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _provenance import stamp
        report = {"provenance": stamp(__file__), **report}
    except Exception as e:
        report = {"provenance": {"unavailable": f"{e.__class__.__name__}: {e}"}, **report}
    out_path.write_text(json.dumps(report, indent=1))

    print(f"\n  {report['n_with_at_least_one_prediction']} of {len(flat)} carry a prediction, "
          f"mean list {report['mean_returned']}, unparseable {len(unparseable)}")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
