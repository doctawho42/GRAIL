#!/usr/bin/env python3
"""What a converged full-split retraining measures, read from the runs rather than re-run.

The released checkpoints saw 5,000 of 9,011 training substrates for eight epochs, and their early
stopping never engaged. The manuscript reports that as an underfitting signature and leaves one
question open: if the learned stages are trained to convergence on the whole split, does the
coverage ceiling move? A retraining answers it, and the retraining does not happen here -- it
happens on a platform with a GPU, one seed per session, and comes back as artifact directories.

Nothing in this repository read those directories. `run_multiseed.py` and `multiseed_headline.py`
both TRAIN; neither aggregates finished runs, and a document claiming otherwise was the reason this
file exists. This is the missing reader.

It takes two spreads and puts one number against the other:

    baseline    the released configuration: 5,000 substrates, an eight-epoch budget, three seeds
    treatment   the full split, an epoch budget early stopping can reach, however many seeds landed

and reports mean +/- std for each, the difference, and -- the part the manuscript actually needs --
whether early stopping engaged this time. `epochs_trained` equal to the configured budget with
`early_stopped_epoch` null is the same signature the released run carries; it is a RESULT, not an
error, so it is reported rather than refused.

What is refused is a spread that is not one. Seeds that ran under different RDKit versions built
their graphs under different tautomer canonicalisations and their recalls are not comparable at the
fourth decimal; seeds whose configs differ in anything but the seed are not repetitions of one
experiment; a seed whose training exited non-zero has no result to average. Each of those stops the
script, because a spread quoted as +/- over a set that does not satisfy them is a fabricated
interval.

    python scripts/full_split_retraining.py --seeds 0 1 2 3 4
    python scripts/full_split_retraining.py --seeds 0 --allow-single   # the first seed, alone
    python scripts/full_split_retraining.py --self-check               # no data needed
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

# The released spread the retraining is measured against: the deployed configuration at 5,000
# substrates and eight epochs, over the three seeds the manuscript reports.
BASELINE_DIRS = ["artifacts/expandedlabels_multiseed_full5000_seed0",
                 "artifacts/expandedlabels_multiseed_full5000_seed1",
                 "artifacts/expandedlabels_multiseed_full5000_seed2"]
TREATMENT_FMT = "artifacts/paper_full_converged_seed{seed}"
SESSION_FMT = "artifacts/seed{seed}_session.json"

# The metrics worth carrying across. Recall leads because precision here is a pessimistic lower
# bound against annotated positives only, and mean_output_size is what makes a recall readable.
HEADLINE = ["top_15_recall", "top_10_recall", "top_5_recall", "top_1_recall",
            "recall", "precision", "f1", "jaccard", "mean_output_size"]


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"REFUSING: {path} is absent; the run it belongs to did not finish or "
                         f"was not brought home")
    return json.loads(path.read_text())


def _config_fingerprint(cfg: dict) -> dict:
    """The config with the fields that are SUPPOSED to differ between seeds removed.

    Two seeds of one experiment differ in `seed`, in the run name that carries it, and in the
    directory they were written to. Anything else differing means the two runs are not repetitions,
    and averaging them would quote a spread over two experiments.
    """
    out = {k: v for k, v in cfg.items() if k not in {"seed", "name", "output_dir", "description"}}
    return out


def _read_run(root: Path, run_dir: str, seed: int, session: Path | None) -> dict:
    d = root / run_dir
    metrics = _load(d / "reports" / "metrics.json")
    gen = _load(d / "reports" / "generator_training.json")
    filt = _load(d / "reports" / "filter_training.json")
    cfg_path = d / "config.yaml"
    row = {
        "seed": seed,
        "directory": run_dir,
        "recorded_seed": metrics.get("reproducibility", {}).get("seed"),
        "ensemble": {k: metrics["ensemble"][k] for k in HEADLINE if k in metrics["ensemble"]},
        "ensemble_val": {k: metrics["ensemble_val"][k] for k in HEADLINE
                         if k in metrics.get("ensemble_val", {})},
        "generator": {k: metrics["generator"][k] for k in HEADLINE if k in metrics["generator"]},
        "runtime": metrics.get("runtime", {}),
        "convergence": {
            "generator": {"epochs_trained": gen.get("epochs_trained"),
                          "early_stopped_epoch": gen.get("early_stopped_epoch"),
                          "stop_reason": gen.get("stop_reason"),
                          "best_val_loss": gen.get("best_val_loss"),
                          "val_loss_history_length": len(gen.get("val_loss_history") or [])},
            "filter": {"epochs_trained": filt.get("epochs_trained"),
                       "early_stopped_epoch": filt.get("early_stopped_epoch"),
                       "stop_reason": filt.get("stop_reason"),
                       "best_val_loss": filt.get("best_val_loss"),
                       "val_loss_history_length": len(filt.get("val_loss_history") or [])},
        },
        "config_present": cfg_path.exists(),
    }
    if row["recorded_seed"] is not None and row["recorded_seed"] != seed:
        raise SystemExit(f"REFUSING: {run_dir} was asked for as seed {seed} but its metrics record "
                         f"seed {row['recorded_seed']}; the directory name and the run disagree")
    if session is not None and session.exists():
        s = _load(session)
        row["session"] = {"exit_code": s.get("exit_code"), "seconds": s.get("seconds"),
                          "environment": s.get("environment", {})}
        if s.get("exit_code") not in (0, None):
            raise SystemExit(f"REFUSING: seed {seed} exited {s['exit_code']}; a failed run has no "
                             f"result to average")
    return row


def _spread(rows: list, block: str) -> dict:
    """mean +/- std per metric, with n, over however many seeds landed."""
    out = {}
    keys = [k for k in HEADLINE if any(k in r[block] for r in rows)]
    for k in keys:
        vals = [r[block][k] for r in rows if k in r[block]]
        out[k] = {"mean": round(statistics.fmean(vals), 6),
                  "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else None,
                  "n": len(vals), "values": [round(v, 6) for v in vals]}
    return out


def _converged(rows: list) -> dict:
    """Did early stopping engage, and if not, is that the released run's signature repeating?"""
    per_stage = {}
    for stage in ("generator", "filter"):
        engaged = [r["convergence"][stage]["early_stopped_epoch"] is not None for r in rows]
        epochs = [r["convergence"][stage]["epochs_trained"] for r in rows]
        reasons = sorted({r["convergence"][stage]["stop_reason"] for r in rows})
        per_stage[stage] = {
            "early_stopping_engaged_in": sum(engaged),
            "of_seeds": len(rows),
            "epochs_trained": epochs,
            "stop_reasons": reasons,
            "stopped_at": [r["convergence"][stage]["early_stopped_epoch"] for r in rows],
        }
    return per_stage


def _compare(baseline: dict, treatment: dict) -> dict:
    """The difference the manuscript's open question is about, per metric, with both spreads."""
    out = {}
    for k in treatment:
        if k not in baseline:
            continue
        b, t = baseline[k], treatment[k]
        out[k] = {"baseline_mean": b["mean"], "baseline_std": b["std"], "baseline_n": b["n"],
                  "treatment_mean": t["mean"], "treatment_std": t["std"], "treatment_n": t["n"],
                  "difference": round(t["mean"] - b["mean"], 6)}
        # A difference is only readable against the spread it sits in. With one seed there is no
        # spread on the treatment side and the field says so rather than printing a bare delta.
        pooled = [s for s in (b["std"], t["std"]) if s is not None]
        out[k]["difference_within_one_std"] = (
            None if not pooled else bool(abs(t["mean"] - b["mean"]) <= max(pooled)))
    return out


def _environment_ledger(rows: list) -> dict:
    """Which RDKit built which run's graphs, and whether the seeds agree.

    The stack pins rdkit==2022.09.5 because tautomer canonicalisation moves between releases and
    the matching key every recall here is scored under is a tautomer-canonical InChIKey. The pin is
    not installable on the platform that has the GPU; what the difference costs is measured in
    results/rdkit_version_drift.json rather than assumed. Mixing two versions WITHIN one spread is
    a different thing from running the whole spread under one unpinned version: the first makes the
    seeds incomparable with each other, and that is refused.
    """
    versions = sorted({r.get("session", {}).get("environment", {}).get("rdkit")
                       for r in rows if r.get("session")})
    versions = [v for v in versions if v]
    devices = sorted({r.get("session", {}).get("environment", {}).get("device")
                      for r in rows if r.get("session")})
    gpus = sorted({r.get("session", {}).get("environment", {}).get("gpu")
                   for r in rows if r.get("session") and r["session"]["environment"].get("gpu")})
    if len(versions) > 1:
        raise SystemExit(
            f"REFUSING: the seeds ran under different RDKit versions ({', '.join(versions)}). "
            f"Their graphs were built under different tautomer canonicalisations, so a spread over "
            f"them is a spread over two standardisations rather than over training seeds. Re-run "
            f"the odd seed, or report the two groups separately.")
    return {"rdkit": versions[0] if versions else None,
            "rdkit_versions_seen": versions,
            "devices": [d for d in devices if d],
            "gpus": gpus,
            "seeds_with_a_session_record": sum(1 for r in rows if r.get("session")),
            "note": ("the pinned release is 2022.09.5; what a different one costs is measured in "
                     "results/rdkit_version_drift.json, not assumed")}


def build(root: Path, seeds: list, baseline_dirs: list, allow_single: bool) -> dict:
    treatment_dirs = [TREATMENT_FMT.format(seed=s) for s in seeds]
    if len(seeds) < 2 and not allow_single:
        raise SystemExit("REFUSING: one seed is not a spread. Pass --allow-single to report it as "
                         "a single run with no interval, and say so wherever it is quoted.")

    t_rows = [_read_run(root, d, s, root / SESSION_FMT.format(seed=s))
              for d, s in zip(treatment_dirs, seeds)]
    b_rows = [_read_run(root, d, i, None) for i, d in enumerate(baseline_dirs)]

    # Every treatment seed must be the same experiment. The configs are compared field by field
    # with the fields that are supposed to differ removed, so a changed epoch budget or a changed
    # rule bank between sessions stops the aggregate instead of being averaged into it.
    fingerprints = {}
    for d, s in zip(treatment_dirs, seeds):
        p = root / d / "config.yaml"
        if not p.exists():
            continue
        import yaml
        fingerprints[s] = _config_fingerprint(yaml.safe_load(p.read_text()))
    if len(fingerprints) > 1:
        first_seed = sorted(fingerprints)[0]
        first = fingerprints[first_seed]
        for s, f in fingerprints.items():
            if f != first:
                differing = sorted(k for k in set(first) | set(f) if first.get(k) != f.get(k))
                raise SystemExit(
                    f"REFUSING: seed {s} and seed {first_seed} differ in {', '.join(differing)}. "
                    f"They are not repetitions of one experiment and their mean is not a spread.")

    env = _environment_ledger(t_rows)
    t_test, t_val = _spread(t_rows, "ensemble"), _spread(t_rows, "ensemble_val")
    b_test = _spread(b_rows, "ensemble")

    inputs = []
    for d in treatment_dirs + baseline_dirs:
        for f in ("reports/metrics.json", "reports/generator_training.json",
                  "reports/filter_training.json"):
            p = root / d / f
            if p.exists():
                inputs.append(str(p.relative_to(root)))
    for s in seeds:
        p = root / SESSION_FMT.format(seed=s)
        if p.exists():
            inputs.append(str(p.relative_to(root)))

    return {
        "provenance": stamp(__file__),
        "inputs": record_inputs(inputs),
        "question": ("whether the coverage ceiling moves when the learned stages are trained to "
                     "convergence on the whole training split rather than on a 5,000-substrate "
                     "sample for eight epochs"),
        "baseline": {"what": ("the released configuration: 5,000 substrates, an eight-epoch "
                              "budget, the three seeds the manuscript reports"),
                     "directories": baseline_dirs,
                     "seeds": len(b_rows),
                     "test": b_test,
                     "convergence": _converged(b_rows)},
        "treatment": {"what": ("the whole training split with an epoch budget early stopping can "
                               "reach"),
                      "directories": treatment_dirs,
                      "seeds": len(t_rows),
                      "test": t_test,
                      "validation": t_val,
                      "convergence": _converged(t_rows),
                      "environment": env,
                      "wall_clock_hours": [round((r.get("session", {}).get("seconds") or 0) / 3600, 3)
                                           for r in t_rows]},
        "comparison": _compare(b_test, t_test),
        "per_seed": {"baseline": b_rows, "treatment": t_rows},
        "reading": ("The manuscript's limitation section says this question is open. If the "
                    "difference sits inside the seed spread and early stopping engaged, the answer "
                    "is that convergence does not move the ceiling, and that is the result to "
                    "print. If early stopping still did not engage, the budget was still too "
                    "small and the question stays open under a larger one."),
    }


def _self_check() -> int:
    """Run the whole path over synthesised runs built from a real one, and check the refusals fire.

    The fixture is COPIED from an artifact on disk rather than hand-written, so a key this reader
    expects and the pipeline does not emit cannot pass unnoticed: a dialect invented for the test
    would agree with the test and with nothing else. Counts are asserted, not just success, because
    a reader that silently finds zero seeds also 'passes'.
    """
    src = ROOT / BASELINE_DIRS[0]
    if not (src / "reports" / "metrics.json").exists():
        print(f"REFUSING: the self-check needs a real run to copy its dialect from, and "
              f"{BASELINE_DIRS[0]} is absent", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "artifacts").mkdir()
        for d in BASELINE_DIRS:
            real = ROOT / d
            if not real.exists():
                print(f"REFUSING: the self-check needs all three baseline runs; {d} is absent",
                      file=sys.stderr)
                return 1
            shutil.copytree(real, td / d)
        for seed in (0, 1):
            dst = td / TREATMENT_FMT.format(seed=seed)
            shutil.copytree(src, dst)
            m = json.loads((dst / "reports" / "metrics.json").read_text())
            m["reproducibility"]["seed"] = seed
            # a treatment that moved the ceiling, so the comparison has something to report
            for block in ("ensemble", "ensemble_val", "generator"):
                for k in list(m[block]):
                    if k.endswith("recall"):
                        m[block][k] = m[block][k] + 0.02 + 0.004 * seed
            (dst / "reports" / "metrics.json").write_text(json.dumps(m))
            g = json.loads((dst / "reports" / "generator_training.json").read_text())
            g["epochs_trained"], g["early_stopped_epoch"] = 23 + seed, 23 + seed
            g["stop_reason"] = "early_stopping"
            (dst / "reports" / "generator_training.json").write_text(json.dumps(g))
            (td / SESSION_FMT.format(seed=seed)).write_text(json.dumps(
                {"seed": seed, "exit_code": 0, "seconds": 21600.0,
                 "environment": {"rdkit": "2026.03.6", "device": "cuda", "gpu": "Tesla P100"}}))

        report = build(td, [0, 1], BASELINE_DIRS, allow_single=False)
        checks = []
        t = report["treatment"]["test"]["top_15_recall"]
        checks.append(("two treatment seeds aggregated", t["n"] == 2))
        checks.append(("baseline kept its three", report["baseline"]["test"]["top_15_recall"]["n"] == 3))
        checks.append(("nine metrics carried", len(report["treatment"]["test"]) == len(HEADLINE)))
        # The fixture copies ONE baseline run twice and lifts it, so the treatment values are
        # that run's value plus the per-seed lift -- not the baseline MEAN plus the lift. Asserting
        # against the mean is what a first draft of this check did, and it failed correctly.
        seed0 = json.loads((ROOT / BASELINE_DIRS[0] / "reports" / "metrics.json").read_text())
        want = [round(seed0["ensemble"]["top_15_recall"] + 0.02 + 0.004 * s, 6) for s in (0, 1)]
        checks.append(("each seed carries its own injected lift",
                       report["treatment"]["test"]["top_15_recall"]["values"] == want))
        checks.append(("the difference is treatment mean minus baseline mean",
                       abs(report["comparison"]["top_15_recall"]["difference"]
                           - (statistics.fmean(want)
                              - report["baseline"]["test"]["top_15_recall"]["mean"])) < 1e-6))
        checks.append(("early stopping reported as engaged",
                       report["treatment"]["convergence"]["generator"]["early_stopping_engaged_in"] == 2))
        checks.append(("baseline reported as not converged",
                       report["baseline"]["convergence"]["generator"]["early_stopping_engaged_in"] == 0))
        checks.append(("the environment ledger found the version",
                       report["treatment"]["environment"]["rdkit"] == "2026.03.6"))
        checks.append(("inputs recorded", len(report["inputs"]) >= 15))

        # The refusals have to fire, or they are decoration. Each is provoked separately.
        def _refuses(what, mutate):
            snapshot = (td / SESSION_FMT.format(seed=1)).read_text()
            mutate()
            try:
                build(td, [0, 1], BASELINE_DIRS, allow_single=False)
            except SystemExit:
                return (what, True)
            else:
                return (what, False)
            finally:
                (td / SESSION_FMT.format(seed=1)).write_text(snapshot)

        checks.append(_refuses("mixed RDKit versions refused", lambda: (
            td / SESSION_FMT.format(seed=1)).write_text(json.dumps(
                {"seed": 1, "exit_code": 0, "seconds": 1.0,
                 "environment": {"rdkit": "2022.09.5", "device": "cuda"}}))))
        checks.append(_refuses("a non-zero exit refused", lambda: (
            td / SESSION_FMT.format(seed=1)).write_text(json.dumps(
                {"seed": 1, "exit_code": 137, "seconds": 1.0,
                 "environment": {"rdkit": "2026.03.6", "device": "cuda"}}))))
        try:
            build(td, [0], BASELINE_DIRS, allow_single=False)
            checks.append(("a single seed refused without the flag", False))
        except SystemExit:
            checks.append(("a single seed refused without the flag", True))

        bad = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        if bad:
            print(f"\nREFUSING: {len(bad)} self-check(s) failed", file=sys.stderr)
            return 1
        print(f"\n{len(checks)} checks passed")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--allow-single", action="store_true",
                    help="report a single seed, with no interval, and say so")
    ap.add_argument("--baseline", nargs="+", default=BASELINE_DIRS)
    ap.add_argument("--self-check", action="store_true",
                    help="exercise the reader and its refusals on synthesised runs; needs no GPU data")
    args = ap.parse_args()

    if args.self_check:
        return _self_check()

    report = build(ROOT, args.seeds, args.baseline, args.allow_single)
    out = ROOT / "results" / "full_split_retraining.json"
    out.write_text(json.dumps(report, indent=1))

    b = report["baseline"]["test"]["top_15_recall"]
    t = report["treatment"]["test"]["top_15_recall"]
    c = report["comparison"]["top_15_recall"]
    print(f"\n  recall@15, ensemble, test split")
    print(f"    released   {b['mean']:.4f} +/- {b['std']:.4f}  over {b['n']} seeds, 5,000 substrates, 8 epochs")
    std = f"{t['std']:.4f}" if t["std"] is not None else "   n/a"
    print(f"    converged  {t['mean']:.4f} +/- {std}  over {t['n']} seeds, the full split")
    print(f"    difference {c['difference']:+.4f}"
          + ("  (inside one standard deviation)" if c["difference_within_one_std"] else ""))
    for stage, s in report["treatment"]["convergence"].items():
        print(f"    {stage:10s} early stopping engaged in {s['early_stopping_engaged_in']} of "
              f"{s['of_seeds']} seeds, at epochs {s['stopped_at']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
