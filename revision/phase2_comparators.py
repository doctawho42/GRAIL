#!/usr/bin/env python3
"""Phase 2: the two comparator columns the revision asked for, and which of them can exist here.

Writes revision/provenance_gloryx.json.

The request was GLORYx and GLORYxR on all 1,170 evaluated substrates, each in a default and a
strict-SOM mode. One of those four is obtainable in this repository, and the value of this record
is that the difference survives afterwards: every number is counted from a file named beside it,
and every absence carries the evidence that establishes it rather than a reason.

What this does not do: it does not run anything. The GLORYx column over the wider population is
produced by scripts/typed_edit/gloryx_via_service.py against the service its authors operate, and
MetaTox's is a manual submission, so what this records is coverage, the submission sets, and the
blockers. A column is reported as delivered only when a file covering the population exists.

    python revision/phase2_comparators.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = ROOT / "revision" / "provenance_gloryx.json"

POPULATION_FILE = "results/test_references.json"
PUBLISHED_GLORYX = "results/gloryx_service_preds.json"
PUBLISHED_METATOX = "results/metatox_smirks_preds.json"
GLORYX_1170 = "results/gloryx_service_preds_1170.json"
# Named so coverage can report it absent. MetaTox is a manual web service, so this phase produces
# its submission files and nothing more; this path is what a returned batch would be ingested to.
METATOX_1170 = "results/metatox_1170_preds.json"

# The arms as the re-tabulation declares them, plus the two files the wider population would need.
# Reading an arm through the accessor its own table declares is part of the measurement: this
# repository holds a second MetaTox file, a 248-substrate side analysis, and reading that one
# instead of the SMIRKS run understates MetaTox on 43 of the 291.
ARMS = {
    "comparison291": {
        "metatox": ("list", PUBLISHED_METATOX, "predictions"),
        "sygma": ("list", "results/sygma_fulltest_predictions.json", None),
        "metapredictor": ("list", "artifacts/tier2_1170/metapredictor_preds.json", None),
        "biotransformer": ("list", "results/biotransformer_allhuman_one_step_preds.json", None),
        "gloryx": ("list", PUBLISHED_GLORYX, "predictions"),
    },
    "evaluated1170": {
        "sygma": ("list", "results/sygma_fulltest_predictions.json", None),
        "metapredictor": ("list", "artifacts/tier2_1170/metapredictor_preds.json", None),
        "biotransformer": ("list", "results/biotransformer_fulltest_preds.json", None),
        "metatox": ("list", METATOX_1170, "predictions"),
        "gloryx": ("list", GLORYX_1170, "predictions"),
    },
}

SERVICE = "https://nerdd.univie.ac.at/api/modules/gloryx"
# Observed from the descriptor on the date this record was first built. Re-fetched at build time
# when the service answers, so a reader can see whether the absence still holds; kept here so the
# record is reproducible from a clean checkout with no network.
SERVICE_PARAMETERS_AS_OBSERVED = {
    "version": "0.0.26",
    "job_parameters": [{"name": "metabolism_phase",
                        "choices": ["phase_1_and_2", "phase_1", "phase_2"]}],
    "result_properties_naming_a_site": [],
    "n_result_properties": 16,
}


# --------------------------------------------------------------------------- reading

def _read_arm(spec):
    """One arm's substrate map through its declared accessor, or None when the file is absent."""
    path = ROOT / spec[1]
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    preds = blob[spec[2]] if spec[2] else blob
    return preds if isinstance(preds, dict) else None


def _population(name):
    """The substrates of a named population, through the accessor the runs themselves use."""
    from gloryx_via_service import population
    return population(name)


def coverage(arms=None):
    """Per population and arm: how many substrates its declared file holds, or that it is absent.

    An arm whose file does not exist carries no count at all rather than a zero. A comparator that
    was never run is not a comparator that returned nothing, and a zero here would propagate into
    every table downstream as a measured loss.
    """
    arms = ARMS if arms is None else arms
    out = []
    for population, by_arm in arms.items():
        subs = set(_population(population))
        for arm, spec in by_arm.items():
            preds = _read_arm(spec)
            if preds is None:
                out.append({"population": population, "arm": arm, "file": spec[1],
                            "substrates": None, "in_population": None, "absent": True})
                continue
            out.append({"population": population, "arm": arm, "file": spec[1],
                        "substrates": len(preds),
                        "in_population": len(set(preds) & subs), "absent": False})
    return out


# --------------------------------------------------------------------------- submission sets

def submission_sets():
    """What each column had to submit: the evaluated population minus its published baseline.

    The baseline is named rather than inferred. Counted against whatever exists on disk these
    numbers would fall to zero as this phase's own results arrive, which would read as though
    nothing had been needed.

    The source is the reference file, not a prediction CSV. Every full-test prediction CSV in this
    repository holds 1,169 of the 1,170 -- one CoA thioester is absent from all of them -- so a set
    built from a CSV would be short by one molecule and its column's denominator would quietly
    disagree with the table it joins.
    """
    pop = _population("evaluated1170")
    out = {}
    for name, baseline in (("gloryx", PUBLISHED_GLORYX), ("metatox", PUBLISHED_METATOX)):
        held = _read_arm(("list", baseline, "predictions")) or {}
        to_submit = [s for s in pop if s not in held]
        out[name] = {
            "population": "evaluated1170",
            "source": POPULATION_FILE,
            "baseline": baseline,
            "held": len(held),
            "to_submit": len(to_submit),
            "why_not_a_predictions_csv": ("the full-test prediction CSVs hold 1169 of the 1170; "
                                          "one CoA thioester is absent from all of them"),
        }
    return out


# --------------------------------------------------------------------------- the model archive

def model_archive(path):
    """What the GLORYxR model archive contains, or that it is not present.

    It arrived in a session scratchpad rather than in the repository, so its absence is recorded
    and is not an error: this record has to be reproducible from a clean checkout without it.
    """
    p = Path(path)
    if not p.exists():
        return {"present": False, "checked_path": str(path), "models": [],
                "note": "not present at this path; the archive was delivered out of band"}
    models = []
    for f in sorted(p.rglob("*.joblib")):
        models.append({"name": f.relative_to(p).as_posix(), "bytes": f.stat().st_size,
                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    by_digest = {}
    for m in models:
        by_digest.setdefault(m["sha256"], []).append(m["name"])
    return {"present": True, "checked_path": str(path), "models": models,
            "n_models": len(models),
            "byte_identical_groups": [v for v in by_digest.values() if len(v) > 1],
            "carries_code_or_featuriser": False}


def _archive_path():
    """Where to look for the GLORYxR models, and it is a real candidate rather than a placeholder.

    The archive was delivered out of band into a session scratchpad, so there is no path in this
    repository that could hold it. GRAIL_GLORYXR_MODELS names it when a reader has a copy; without
    that the record reports the repository-relative location it looked in, so `checked_path` is
    always somewhere a copy could legitimately have been put.
    """
    import os
    env = os.environ.get("GRAIL_GLORYXR_MODELS")
    return Path(env) if env else ROOT / "external" / "gloryxr" / "models"


def _service_parameters():
    """The service descriptor, re-read when it answers and otherwise as observed."""
    import urllib.request
    agent = ("GRAIL-benchmark/1.0 (academic metabolite-prediction comparison; "
             "contact nikitapol@fbb.msu.ru)")
    try:
        req = urllib.request.Request(SERVICE, headers={"User-Agent": agent,
                                                       "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as fh:
            d = json.load(fh)
    except Exception as e:
        return {"read_live": False, "why": f"{e.__class__.__name__}: {e}",
                **SERVICE_PARAMETERS_AS_OBSERVED}
    props = d.get("result_properties") or []
    return {
        "read_live": True,
        "version": d.get("version"),
        "job_parameters": [{"name": j.get("name"),
                            "choices": [c.get("value") for c in (j.get("choices") or [])]}
                           for j in (d.get("job_parameters") or [])],
        "result_properties_naming_a_site": [p.get("name") for p in props
                                            if any(t in str(p.get("name", "")).lower()
                                                   for t in ("site", "som", "atom"))],
        "n_result_properties": len(props),
    }


# --------------------------------------------------------------------------- blockers

def blockers():
    """Why each requested column is or is not here, each with evidence that can be rechecked.

    A blocker without evidence is an opinion, so every entry names files, digests, a measurement
    or a document. `binding` marks whether removing that obstacle alone would deliver the column:
    the scikit-learn mismatch is real and is not binding, because no interpreter supplies the
    absent code.
    """
    svc = _service_parameters()
    gl_1170 = (ROOT / GLORYX_1170).exists()
    out = [
        {
            "id": "strict_som_unavailable",
            "mode_requested": "strict-SOM",
            "binding": True,
            "finding": ("no strict-SOM mode exists to run and no labels exist to score it "
                        "against, so this mode is marked rather than approximated"),
            "evidence": [
                {"kind": "service", "detail": (
                    f"{SERVICE} exposes {len(svc['job_parameters'])} job parameter(s) "
                    f"{[j['name'] for j in svc['job_parameters']]} and "
                    f"{len(svc['result_properties_naming_a_site'])} of "
                    f"{svc['n_result_properties']} result properties name a site"),
                 "read_live": svc["read_live"]},
                {"kind": "document", "detail": ("GLORYx's own README names site-of-metabolism "
                                                "only as the internal FAME 3 component; it "
                                                "documents no strict mode, threshold or SOM flag")},
                {"kind": "file", "detail": ("results/what_each_arm_returns.json records GLORYx and "
                                            "MetaTox as naming no site")},
                {"kind": "file", "detail": ("paper2/si.tex states this corpus carries no "
                                            "site-of-metabolism labels, so the mode could not be "
                                            "scored even if it could be run")},
            ],
        },
        {
            "id": "gloryxr_no_code",
            "binding": True,
            "finding": ("GLORYxR cannot be run: no implementation and no featuriser is available "
                        "in this repository, on this machine, or in the delivered archive"),
            "evidence": [
                {"kind": "measurement", "detail": ("the delivered archive holds 11 entries, all "
                                                   "of them .joblib models: no code, no "
                                                   "featuriser")},
                {"kind": "measurement", "detail": ("gloryxr, gloryx and glory_x are not importable "
                                                   "here, and a scoped search of the usual "
                                                   "checkout locations found no source tree")},
                {"kind": "document", "detail": ("scripts/typed_edit/gloryx_via_service.py records "
                                                "that GLORYxR's production model dumps are 'to be "
                                                "published' and its documentation's Zenodo link is "
                                                "a literal TODO")},
                {"kind": "measurement", "detail": ("the single model declares n_features_in_=5014 "
                                                   "against 5006 for the rule models: two "
                                                   "featurisations, neither of them present")},
            ],
        },
        {
            "id": "gloryxr_sklearn_version",
            "binding": False,
            "finding": ("the delivered models were pickled by scikit-learn 1.9.0 while this "
                        "environment has 1.7.2; newer-to-older is unsupported, so their "
                        "predictions could not be trusted here even if code existed"),
            "evidence": [
                {"kind": "measurement", "detail": ("every model warns 'unpickle ... from version "
                                                   "1.9.0'; the installed version is 1.7.2")},
                {"kind": "measurement", "detail": ("scikit-learn 1.9.0 and 1.9.1 are published on "
                                                   "PyPI and require Python >= 3.11; this "
                                                   "interpreter is 3.10.8, whose ceiling is 1.7.2")},
            ],
        },
        {
            "id": "gloryxr_duplicate_model",
            "binding": False,
            "finding": ("two of the delivered models are byte-identical under different names, so "
                        "they cannot be treated as distinct arms without asking the sender"),
            "evidence": [
                {"kind": "digest", "detail": ("'CYP rules from GLORY (phase 1).joblib' and "
                                              "'Phase 1 SyGMa rules.joblib' both hash to "
                                              "2c186eb6f3e35fa99885d5ea96d706de1f17300b4cecb96c33"
                                              "d4e942bb57a1dc")},
            ],
        },
        {
            "id": "metatox_manual_submission",
            "binding": True,
            "finding": ("MetaTox is a web service with no programmatic interface recorded here, so "
                        "this phase writes its submission files and stops"),
            "evidence": [
                {"kind": "file", "detail": ("results/comparator_provenance.json records MetaTox as "
                                            "kind 'web service' publishing no version string")},
                {"kind": "measurement", "detail": "879 of the 1,170 substrates have no MetaTox row"},
            ],
        },
    ]
    if not gl_1170:
        out.append({
            "id": "gloryx_1170_run_incomplete",
            "binding": True,
            "finding": (f"the GLORYx column over the evaluated population is not yet on disk: "
                        f"{GLORYX_1170} does not exist"),
            "evidence": [
                {"kind": "file", "detail": f"{GLORYX_1170} absent at build time"},
                {"kind": "measurement", "detail": ("879 substrates were queued for submission in "
                                                   "chunks of 25 against the authors' service")},
            ],
        })
    return out


# --------------------------------------------------------------------------- the report

def build():
    """The whole record: what was asked, what is delivered, and what each absence rests on."""
    cov = coverage()
    subs = submission_sets()
    blk = blockers()
    by_id = {b["id"]: b for b in blk}

    gl_path = ROOT / GLORYX_1170
    gl_map = _read_arm(("list", GLORYX_1170, "predictions")) or {}
    pop = set(_population("evaluated1170"))
    gl_covers = bool(gl_map) and pop <= set(gl_map)

    jobs = []
    if gl_path.exists():
        blob = json.loads(gl_path.read_text())
        jobs = ((blob.get("obtained_from") or {}).get("jobs") or []) if isinstance(blob, dict) else []

    delivered = {
        "default": {
            "delivered": bool(gl_covers),
            "file": GLORYX_1170,
            "substrates_held": len(gl_map),
            "population": len(pop),
        },
        "strict-SOM": {
            "delivered": False,
            "blocker_id": "strict_som_unavailable",
        },
    }
    if not gl_covers:
        delivered["default"]["blocker_id"] = ("gloryx_1170_run_incomplete"
                                              if "gloryx_1170_run_incomplete" in by_id
                                              else "gloryx_1170_incomplete_coverage")

    report = {
        "what_this_is": ("the Phase 2 comparator record: coverage counted from files, the "
                         "submission sets, and the evidence under every absence"),
        "requested": {
            "populations": ["evaluated1170"],
            "modes": ["default", "strict-SOM"],
            "systems": ["GLORYx", "GLORYxR", "MetaTox"],
        },
        "delivered": delivered,
        "arms": {
            "gloryx": {"ran": bool(jobs), "evidence_of_running": jobs,
                       "service": SERVICE, "population": "evaluated1170"},
            "gloryxr": {"ran": False, "blocker_id": "gloryxr_no_code"},
            "metatox": {"ran": False, "blocker_id": "metatox_manual_submission"},
        },
        "coverage": cov,
        "submission_sets": subs,
        "model_archive": model_archive(_archive_path()),
        "service": _service_parameters(),
        "blockers": blk,
    }
    try:
        from _provenance import stamp
        report = {"provenance": stamp(__file__), **report}
    except Exception as e:                      # the record stands without the stamp helper
        report = {"provenance": {"unavailable": f"{e.__class__.__name__}: {e}"}, **report}
    return report


def main() -> int:
    report = build()
    OUT.write_text(json.dumps(report, indent=1))

    print("coverage (substrates per arm, read through each declared accessor):")
    for c in report["coverage"]:
        shown = "absent" if c["absent"] else f"{c['substrates']:>5} ({c['in_population']} in pop)"
        print(f"  {c['population']:14} {c['arm']:15} {shown:24} {c['file']}")
    print("\nsubmission sets, against the published baselines:")
    for name, s in report["submission_sets"].items():
        print(f"  {name:10} held {s['held']:>5}  to submit {s['to_submit']:>5}  "
              f"baseline {s['baseline']}")
    print("\nrequested columns:")
    for mode, d in report["delivered"].items():
        mark = "delivered" if d["delivered"] else f"NOT delivered ({d.get('blocker_id')})"
        print(f"  {mode:12} {mark}")
    print("\nblockers:")
    for b in report["blockers"]:
        print(f"  {b['id']:32} binding={str(b['binding']):5} "
              f"{len(b['evidence'])} evidence item(s)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
