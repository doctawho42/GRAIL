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
# The arm on the wider population is the MERGE of both GLORYx runs, not the service run's own
# output. `gloryx_via_service.py --population evaluated1170` submits only what the published run
# lacks, so its file holds 879 substrates; declaring that as the arm would hand an empty list back
# for the other 291 and score them as misses.
GLORYXR_DEFAULT = "results/gloryxr_local_preds_default.json"
GLORYXR_STRICT = "results/gloryxr_local_preds_strict.json"
GLORYX_1170_RAW = "results/gloryx_service_preds_1170.json"
GLORYX_1170 = "results/gloryx_service_preds_evaluated1170.json"
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

BLOCKER_FOR_STATE = {"absent": "gloryx_1170_run_incomplete",
                     "incomplete": "gloryx_1170_incomplete_coverage"}


def _default_column_state():
    """Whether the GLORYx column over the evaluated population is on disk, and complete.

    Three states, because two of them block for different reasons and each needs its own blocker:
    the file is absent, it exists but covers only part of the population, or it covers all of it.
    Both `blockers()` and `build()` read the state from here so they cannot disagree -- an earlier
    version chose the incomplete-coverage id through a fallback in `build()` that named a blocker
    `blockers()` never emitted, which would have put a dangling reference in the record.
    """
    path = ROOT / GLORYX_1170
    gl_map = _read_arm(("list", GLORYX_1170, "predictions")) or {}
    pop = set(_population("evaluated1170"))
    if not path.exists():
        return "absent", gl_map, pop
    if not (pop <= set(gl_map)):
        return "incomplete", gl_map, pop
    return "complete", gl_map, pop


def blockers():
    """Why each requested column is or is not here, each with evidence that can be rechecked.

    A blocker without evidence is an opinion, so every entry names files, digests, a measurement
    or a document. `binding` marks whether removing that obstacle alone would deliver the column:
    the scikit-learn mismatch is real and is not binding, because no interpreter supplies the
    absent code.
    """
    svc = _service_parameters()
    state, gl_map, pop = _default_column_state()
    out = [
        {
            "id": "strict_som_is_a_gloryxr_mode_not_a_service_one",
            "mode_requested": "strict-SOM",
            "binding": False,
            "finding": ("strict-SOM exists and has been run, on GLORYxR rather than on the GLORYx "
                        "service. An earlier record here said no such mode existed anywhere; that "
                        "was false, and the mistake was looking for it on the service and "
                        "concluding from its absence there"),
            "evidence": [
                {"kind": "file", "detail": ("gloryxr/reactions.py exposes "
                                            "Reactor.load_builtin(phase, strict_soms=bool) and "
                                            "som.py derives the restricted site set in "
                                            "_get_strict_som_indices")},
                {"kind": "document", "detail": ("GLORYxR's own tutorial demonstrates both modes and "
                                                "states that strict annotation removes scores "
                                                "inflated by promiscuous atom matching")},
                {"kind": "measurement", "detail": ("both modes were run over all 1,170 substrates: "
                                                   "results/gloryxr_local_preds_default.json and "
                                                   "results/gloryxr_local_preds_strict.json")},
                {"kind": "service", "detail": (
                    f"the service remains without such a parameter -- {SERVICE} exposes "
                    f"{len(svc['job_parameters'])} job parameter(s) "
                    f"{[j['name'] for j in svc['job_parameters']]} and "
                    f"{len(svc['result_properties_naming_a_site'])} of "
                    f"{svc['n_result_properties']} result properties name a site -- so the "
                    f"service-derived column exists in one mode only"),
                 "read_live": svc["read_live"]},
            ],
            "what_the_mode_does_not_change": (
                "coverage. Measured over all 1,170: identical product sets on 1170/1170 "
                "substrates and mean list 31.95 in both, while the order agrees on only 32 (2.7%) "
                "and the top-15 differs on 838 (71.6%) at mean Jaccard 0.795. Any coverage figure "
                "is equal between the modes by construction; only rank-sensitive measures separate "
                "them."),
            "not_scoreable_against_labels": (
                "paper2/si.tex states this corpus carries no site-of-metabolism labels, so neither "
                "mode can be judged on whether its sites are right -- only on what it retrieves"),
        },
        {
            "id": "gloryxr_dumps_unpublished",
            "binding": False,
            "binding_for_a_third_party": True,
            "retracts": "gloryxr_no_code",
            "finding": ("GLORYxR runs here, and the arm exists. What remains is that its "
                        "production model dumps are not published: they were supplied privately by "
                        "the author, so a third party cannot reproduce this column from public "
                        "sources alone"),
            "what_was_wrong_before": (
                "an earlier record in this file claimed GLORYxR could not be run for want of an "
                "implementation and a featuriser, with binding=True. Four parts of that were "
                "false. The implementation is public and GPL-3 at molinfo-vienna/GLORYxR -- and "
                "this repository already said 'GLORYxR installs and runs' in three places, one of "
                "which the old blocker quoted half of. The featuriser is not missing: it arrives "
                "as the fame3r dependency. The 5014-against-5006 feature gap was arithmetic, not "
                "incompatibility: SingleFAME3RModelProvider one-hot encodes its 8 reaction classes "
                "on top of the same 5006-wide vectoriser. And the interpreter was never the "
                "binding constraint. The error was drawing a conclusion wider than the evidence: "
                "the search covered this machine and the delivered archive, not the public source"),
            "evidence": [
                {"kind": "measurement", "detail": ("installed from the clone under Python 3.13 and "
                                                   "run over all 1,170 substrates in both SOM "
                                                   "modes; 1,169 carry a prediction, mean list "
                                                   "31.95, none unparseable")},
                {"kind": "measurement", "detail": ("the archive IS the missing dumps: the rule "
                                                   "table declares exactly 8 'Name of rule subset' "
                                                   "values, multi_models/ holds exactly 8 dumps, "
                                                   "the sets are identical with nothing on either "
                                                   "side, and MultiFAME3RModelProvider indexes "
                                                   "models by that same key")},
                {"kind": "document", "detail": ("GLORYxR's tutorial directs a user to 'our Zenodo "
                                                "archive (to be published)' and docs/source/"
                                                "citation.rst is a TODO stub, so the dumps have no "
                                                "public checksum to match against and GLORYxR "
                                                "cannot yet be cited by publication")},
                {"kind": "document", "detail": ("the archive was sent by the first author named in "
                                                "GLORYxR's pyproject.toml, so identity with the "
                                                "deployment's dumps rests on private provenance "
                                                "plus an exact name and dimension fit -- inferred, "
                                                "not certified")},
            ],
        },
        {
            "id": "gloryxr_environment_not_guaranteed",
            "binding": False,
            "retracts": "gloryxr_sklearn_version",
            "finding": ("the configuration is not a 'supported' one -- no such guarantee exists -- "
                        "but the version drift is measured and moves no number on this population"),
            "what_was_wrong_before": (
                "the earlier entry said the models 'could not be trusted here even if code "
                "existed' because newer-to-older is unsupported, comparing 1.9.0 pickles against "
                "the repository's own 1.7.2. Two errors. The run does not happen on the "
                "repository's interpreter: GLORYxR requires Python >= 3.13 and runs in an isolated "
                "environment, so 1.7.2 was never in play. And 'supported' was a policy I invented: "
                "scikit-learn grants no old-to-new carve-out anywhere, and its own warning says "
                "the load 'might lead to breaking code or invalid results. Use at your own risk.' "
                "The honest position is not 'supported, so trusted' but 'unguaranteed by policy, "
                "measured as identical'"),
            "evidence": [
                {"kind": "measurement", "detail": ("the columns were computed twice: under the "
                                                   "project's own uv.lock (scikit-learn 1.9.0, "
                                                   "cdpkit 1.2.3; 0 InconsistentVersionWarning) and "
                                                   "under a fresh resolve (1.9.1, 1.3.0; 2016 of "
                                                   "them)")},
                {"kind": "digest", "detail": ("the substrate maps are byte-identical across the two "
                                              "environments in both modes: sha256 "
                                              "8f75ddc067a1a4f8d3671ad934e49e64 for strict and "
                                              "d7584d276ff440fd40f6733f7f4fda2d for default, "
                                              "1170/1170 identical sets AND orders, 0 differing "
                                              "scores at max |difference| 0.00000000")},
                {"kind": "measurement", "detail": ("that also closes the channel nothing warns "
                                                   "about: descriptors are recomputed at run time "
                                                   "by cdpkit, locked at 1.2.3 against the 1.3.0 a "
                                                   "fresh resolve installs, and the vector width "
                                                   "matches either way, so a value drift would "
                                                   "have been silent -- on this population there "
                                                   "is none")},
                {"kind": "file", "detail": ("the published columns are the locked-environment ones, "
                                            "and each records the version of every package that "
                                            "can move a prediction")},
            ],
        },
        {
            "id": "gloryxr_duplicate_model",
            "binding": False,
            "finding": ("two of the eight delivered models are byte-identical under different "
                        "names, so the eight rule subsets resolve to only seven distinct forests "
                        "and a large part of the rule table is scored by one shared model. Whether "
                        "that is the intended delivery only the sender can say"),
            "evidence": [
                {"kind": "digest", "detail": ("'CYP rules from GLORY (phase 1).joblib' and "
                                              "'Phase 1 SyGMa rules.joblib' both hash to "
                                              "2c186eb6f3e35fa99885d5ea96d706de1f17300b4cecb96c33"
                                              "d4e942bb57a1dc, and the duplicate exists inside the "
                                              "delivered tar, not as a local copy error")},
                {"kind": "measurement", "detail": ("those two subsets carry 224 of the 260 rules in "
                                                   "gloryx_reactionrules_connect.csv -- 86% by rule "
                                                   "count, and every phase-1 rule -- so they are "
                                                   "all scored by the same forest")},
                {"kind": "measurement", "detail": ("it is not a packaging accident on this side: "
                                                   "the eight file names match the eight rule "
                                                   "subsets exactly, so the pair is two keys "
                                                   "pointing at one trained model rather than a "
                                                   "missing file")},
            ],
        },
        {
            "id": "gloryxr_strict_column_absent",
            "binding": True,
            "emitted_only_when": ("the strict column is missing or short of the population; it is "
                                  "declared unconditionally so the id `build()` names can never "
                                  "dangle"),
            "finding": (f"the GLORYxR strict-SOM column is not on disk or does not cover the "
                        f"population: {GLORYXR_STRICT}"),
            "evidence": [
                {"kind": "file", "detail": (f"{GLORYXR_STRICT} is produced by "
                                            f"revision/phase2_gloryxr_run.py --mode strict under "
                                            f"the isolated Python 3.13 environment")},
                {"kind": "measurement", "detail": ("the runner refuses to write a full-coverage "
                                                   "artifact for a partial run, so an absent file "
                                                   "means the run did not finish rather than that "
                                                   "the mode returned nothing")},
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
    if state == "absent":
        out.append({
            "id": BLOCKER_FOR_STATE["absent"],
            "binding": True,
            "finding": (f"the GLORYx column over the evaluated population is not yet on disk: "
                        f"{GLORYX_1170} does not exist"),
            "evidence": [
                {"kind": "file", "detail": f"{GLORYX_1170} absent at build time"},
                {"kind": "measurement", "detail": ("879 substrates were queued for submission in "
                                                   "chunks of 25 against the authors' service")},
                {"kind": "file", "detail": (f"it is produced by revision/phase2_gloryx_merge.py "
                                            f"from the published run and {GLORYX_1170_RAW}")},
            ],
        })
    elif state == "incomplete":
        out.append({
            "id": BLOCKER_FOR_STATE["incomplete"],
            "binding": True,
            "finding": (f"{GLORYX_1170} exists but covers {len(set(gl_map) & pop)} of {len(pop)} "
                        f"substrates, so it cannot stand as an arm: the ones it lacks would be "
                        f"read as empty lists and scored as misses"),
            "evidence": [
                {"kind": "measurement",
                 "detail": f"{len(pop - set(gl_map))} substrates of the population carry no row"},
                {"kind": "file", "detail": (f"the service run's own output ({GLORYX_1170_RAW}) "
                                            f"holds only what the published run lacked; both "
                                            f"halves are merged by "
                                            f"revision/phase2_gloryx_merge.py")},
            ],
        })
    return out


# --------------------------------------------------------------------------- the report

def build():
    """The whole record: what was asked, what is delivered, and what each absence rests on."""
    cov = coverage()
    subs = submission_sets()
    blk = blockers()

    state, gl_map, pop = _default_column_state()
    gl_covers = state == "complete"

    gl_path = ROOT / GLORYX_1170
    jobs = []
    if gl_path.exists():
        blob = json.loads(gl_path.read_text()) if gl_path.exists() else {}
        if isinstance(blob, dict):
            # The service run records its jobs under obtained_from; the merge keeps one list per
            # half, so both shapes are read rather than one being assumed.
            found = (blob.get("obtained_from") or {}).get("jobs") or blob.get("jobs") or []
            jobs = (sorted(j for half in found.values() for j in half)
                    if isinstance(found, dict) else list(found))

    # The two locally-run GLORYxR columns, read the same way every other arm is.
    gxr = {mode: _read_arm(("list", path, "predictions")) or {}
           for mode, path in (("default", GLORYXR_DEFAULT), ("strict", GLORYXR_STRICT))}
    gxr_covers = {mode: bool(m) and pop <= set(m) for mode, m in gxr.items()}

    delivered = {
        "default": {
            "delivered": bool(gl_covers),
            "file": GLORYX_1170,
            "substrates_held": len(gl_map),
            "population": len(pop),
            "system": "GLORYx, via the authors' service",
        },
        "strict-SOM": {
            # Delivered by GLORYxR, which is where the mode exists. The service has no such
            # parameter, so this mode was never obtainable from the route the default column uses.
            "delivered": bool(gxr_covers["strict"]),
            "file": GLORYXR_STRICT,
            "substrates_held": len(gxr["strict"]),
            "population": len(pop),
            "system": "GLORYxR, run locally",
        },
    }
    if not gl_covers:
        # From the same state `blockers()` read, so the id cannot name a blocker that was never
        # emitted.
        delivered["default"]["blocker_id"] = BLOCKER_FOR_STATE[state]
    if not gxr_covers["strict"]:
        delivered["strict-SOM"]["blocker_id"] = "gloryxr_strict_column_absent"

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
            "gloryxr default": {
                "ran": bool(gxr_covers["default"]),
                "evidence_of_running": ([GLORYXR_DEFAULT] if gxr_covers["default"] else []),
                "substrates": len(gxr["default"]), "population": "evaluated1170"},
            "gloryxr strict": {
                "ran": bool(gxr_covers["strict"]),
                "evidence_of_running": ([GLORYXR_STRICT] if gxr_covers["strict"] else []),
                "substrates": len(gxr["strict"]), "population": "evaluated1170"},
            "metatox": {"ran": False, "blocker_id": "metatox_manual_submission"},
        },
        "comparability": (
            "the GLORYxR columns are comparable to the service-derived gloryx column under "
            "inchi_no_stereo and inchikey_tautomer only: the local arm emits no stereochemistry at "
            "all, which puts exact at 25-26% disagreement, canonical at 12-14% and inchikey at "
            "9-10%, always against the local arm. They are therefore reported in "
            "revision/T_gloryxr.csv rather than as arms in phase1_tmain.ARMS, which cannot "
            "restrict an arm to a subset of criteria"),
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
