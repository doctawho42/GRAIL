#!/usr/bin/env python3
"""GLORYx on the comparison set, run through the service its authors operate.

The manuscript carried GLORYx as an absence with three different reasons attached, and the
supporting information said honestly that whether it could have been run was a question about
effort and about its interface that had not been settled. It is settled here, and the answer has
three parts.

The source is public and GPL-3 at github.com/christinadebruynkops/GLORYx, Maven builds it, and its
command line takes an SDF in batch, so "the cost of submission" was never the obstacle for this
system. What blocks a local build is that the FAME 3 models are not bundled: the repository's own
README says they must be obtained from the FAME 3 authors, free of charge for non-profit use. That
is a request only the authors of this paper can make.

The successor, GLORYxR, installs cleanly and has a clean API, but its production model dumps are
"to be published" and the Zenodo link in its documentation is a literal TODO; the only model it
ships is one its tutorial says is not trained on the full dataset.

What remains is the authors' own deployment at nerdd.univie.ac.at, which runs the real thing. Its
backend is open source, its REST interface is documented by that source, and its published
anonymous quota is 100,000 molecules a day against the 291 this needs. So the column is obtained
there, and what it is a column of is stated exactly: GLORYx at the module version this run records,
through that service, on the date this run records. It is not a local build and the artifact says
so, because a comparator whose provenance is vague is the defect this paper is about.

    python scripts/typed_edit/gloryx_via_service.py --smoke     # one molecule, checks the flow
    python scripts/typed_edit/gloryx_via_service.py             # the comparison set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE = "https://nerdd.univie.ac.at/api"
MODULE = "gloryx"
PHASE = "phase_1_and_2"
# Identifying the caller is the courteous minimum when using somebody else's service, and the
# address is the corresponding author's, which is the one a service operator would want.
AGENT = ("GRAIL-benchmark/1.0 (academic metabolite-prediction comparison; "
         "contact nikitapol@fbb.msu.ru)")
# Well inside the operator's published anonymous quota of five active jobs, and small enough that
# one failure costs one chunk rather than the run.
CHUNK = 25
POLL_S = 10
POLL_MAX = 180


def _get(url: str, timeout: int = 90, tries: int = 5):
    """One GET, retried with a widening pause, because the link to this service is not reliable.

    A 404 is an answer and is raised at once: the results endpoint uses it to say there is no such
    page. Everything else that fails is treated as the transport failing rather than the server
    refusing, which is what a run of several hundred molecules over a home connection meets.
    """
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "application/json"})
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.loads(fh.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (404, 422):
                raise
            last = e
        except Exception as e:                      # transport, DNS, timeout
            last = e
        if attempt < tries - 1:
            time.sleep(5 * (attempt + 1))
    raise last


def submit(smiles: list) -> str:
    """Create one job over these inputs and return its id."""
    q = [("inputs", s) for s in smiles] + [("metabolism_phase", PHASE)]
    url = f"{BASE}/{MODULE}/jobs?" + urllib.parse.urlencode(q)
    return _get(url)["id"]


def wait(job_id: str) -> dict:
    """Poll until the job reports completed, or give up and say so."""
    for _ in range(POLL_MAX):
        job = _get(f"{BASE}/{MODULE}/jobs/{job_id}")
        if job.get("status") == "completed":
            return job
        time.sleep(POLL_S)
    raise TimeoutError(f"job {job_id} did not complete within "
                       f"{POLL_MAX * POLL_S} s")


def collect(job_id: str, n_entries: int) -> dict:
    """Every page of one job's results, folded into {input smiles: ranked metabolites}."""
    out, page = {}, 1
    while True:
        try:
            res = _get(f"{BASE}/{MODULE}/jobs/{job_id}/results?page={page}")
        except urllib.error.HTTPError as e:
            if e.code == 404:      # past the last page
                break
            raise
        rows = res.get("data") or []
        if not rows:
            break
        for r in rows:
            sub = r.get("input_smiles")
            met = r.get("metabolite_smiles")
            if not sub:
                continue
            entry = out.setdefault(sub, [])
            # A row with no metabolite is a substrate the tool returned nothing for, which is a
            # real answer and is kept as an empty list rather than dropped.
            if met:
                entry.append((r.get("rank") or 10 ** 6, r.get("priority_score"), met))
        pages = (res.get("pagination") or {}).get("last_page")
        if pages is not None and page >= pages:
            break
        page += 1
    # The service returns a rank per metabolite; the list is ordered by it, and the score is kept
    # beside it so a reader can see what the ordering is on.
    return {sub: [m for _, _, m in sorted(rows)] for sub, rows in out.items()}, \
           {sub: [{"rank": r, "score": sc, "smiles": m} for r, sc, m in sorted(rows)]
            for sub, rows in out.items()}


POPULATIONS = ("comparison291", "evaluated1170")


def population(name: str = "comparison291") -> list:
    """The substrates of one named population, in a deterministic order.

    The comparison set is taken from the accessor every other comparator is read through, so this
    column and theirs cannot drift apart. The evaluated set is the substrates the corpus holds
    references for, which is the denominator the re-tabulation uses.

    An unknown name raises rather than falling back to a default: a population chosen by accident
    is the defect that makes two tables incomparable, and silently defaulting hides it.
    """
    if name == "comparison291":
        from vs_metatox import population as pop
        subs, _, _ = pop()
        return subs
    if name == "evaluated1170":
        truth = json.loads((ROOT / "results" / "test_references.json").read_text())
        return sorted(truth)
    raise ValueError(f"unknown population {name!r}; known populations are "
                     f"{', '.join(POPULATIONS)}")


def pending(subs: list, held_path) -> list:
    """Those substrates an existing output does not already carry, in the order given.

    Phase 2 widens this column from the comparison set to the evaluated set, and the already
    published run covers the narrower one. Asking the operator's service again for molecules whose
    answers are already on disk is a cost to somebody else's machine for no information, so the
    held set is subtracted. A missing or unreadable output means nothing is held, which is the safe
    direction: it submits more rather than silently skipping molecules.
    """
    held: set = set()
    path = Path(held_path)
    if path.exists():
        blob = json.loads(path.read_text())
        inner = blob.get("predictions") if isinstance(blob, dict) else None
        held = set(inner if isinstance(inner, dict)
                   else (blob if isinstance(blob, dict) else ()))
    return [s for s in subs if s not in held]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one molecule from the tool's own documentation, to check the flow")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--population", choices=POPULATIONS, default="comparison291",
                    help="which substrates to run; the wider one is what Phase 2 asks for")
    ap.add_argument("--out", default=None,
                    help="default depends on the population, so the published comparison-set "
                         "artifact cannot be overwritten by a run over the wider one")
    ap.add_argument("--checkpoint", default=None,
                    help="written after every chunk; a re-run resumes from it")
    ap.add_argument("--skip-held", default=None,
                    help="an existing output whose substrates are not asked for again; defaults "
                         "to the published comparison-set run when the population is the wider one")
    args = ap.parse_args()

    # Per population, so neither the output nor the checkpoint of one run can land on another's.
    suffix = "" if args.population == "comparison291" else "_1170"
    if args.out is None:
        args.out = str(ROOT / "results" / f"gloryx_service_preds{suffix}.json")
    if args.checkpoint is None:
        args.checkpoint = str(ROOT / "results" / f".gloryx_service_partial{suffix}.json")
    if args.skip_held is None and args.population == "evaluated1170":
        args.skip_held = str(ROOT / "results" / "gloryx_service_preds.json")

    if args.smoke:
        jid = submit(["CC(=O)Nc1ccc(O)cc1"])
        wait(jid)
        flat, _ = collect(jid, 1)
        print(f"job {jid}: {sum(len(v) for v in flat.values())} metabolites for "
              f"{len(flat)} substrate")
        return 0

    requested = population(args.population)
    subs = pending(requested, args.skip_held) if args.skip_held else list(requested)
    held_elsewhere = len(requested) - len(subs)
    if args.limit:
        subs = subs[:args.limit]
    print(f"{args.population}: {len(requested)} substrates, {held_elsewhere} already held in "
          f"{args.skip_held or 'nothing'}, {len(subs)} to submit, chunks of {CHUNK}", flush=True)

    module = _get(f"{BASE}/modules/{MODULE}")

    # A chunk that has landed is written down before the next is asked for, so a link that drops
    # in the middle of three hundred molecules costs one chunk and not the run. Re-running picks
    # up where it stopped and submits nothing it already holds.
    ckpt = Path(args.checkpoint)
    flat, detail, jobs = {}, {}, []
    if ckpt.exists():
        prior = json.loads(ckpt.read_text())
        flat, detail, jobs = prior["predictions"], prior["with_rank_and_score"], prior["jobs"]
        print(f"  resuming: {len(flat)} substrates already held from "
              f"{len(jobs)} job(s)", flush=True)

    t0 = time.perf_counter()
    for i in range(0, len(subs), CHUNK):
        chunk = [s for s in subs[i:i + CHUNK] if s not in flat]
        if not chunk:
            continue
        jid = submit(chunk)
        wait(jid)
        f, d = collect(jid, len(chunk))
        flat.update(f); detail.update(d); jobs.append(jid)
        # A substrate the service answered for with nothing still counts as answered, or the
        # resume would ask for it again on every restart.
        for s in chunk:
            flat.setdefault(s, [])
        ckpt.write_text(json.dumps({"predictions": flat, "with_rank_and_score": detail,
                                    "jobs": jobs}))
        got = sum(1 for s in chunk if flat.get(s))
        print(f"  {min(i + CHUNK, len(subs))}/{len(subs)} "
              f"({time.perf_counter() - t0:.0f}s) job {jid[:8]} returned for {got} of "
              f"{len(chunk)}", flush=True)

    # A substrate the service returned no row for at all is recorded as an empty list, so the
    # population stays the population and a silent drop cannot look like a short list.
    for s in subs:
        flat.setdefault(s, [])

    from _provenance import stamp
    report = {
        "provenance": stamp(__file__),
        "what_this_is": "GLORYx's predictions on the comparison set, as SMILES, one ranked list "
                        "per substrate",
        "obtained_from": {
            "service": f"{BASE}/{MODULE}",
            "operated_by": "the authors of GLORYx (NERDD, University of Vienna)",
            "module_version": module.get("version"),
            "job_parameters": {"metabolism_phase": PHASE},
            "jobs": jobs,
            "why_not_a_local_build": (
                "the FAME 3 models GLORYx needs are not bundled with its source and its README "
                "directs a user to obtain them from the FAME 3 authors; the successor GLORYxR "
                "ships no production model yet"),
        },
        "drawing": "the substrate as the corpus stores it, which is what every comparator but "
                   "MetaTox was handed",
        "n_substrates": len(subs),
        "n_with_at_least_one_prediction": sum(1 for v in flat.values() if v),
        "mean_returned": round(sum(len(v) for v in flat.values()) / max(len(subs), 1), 2),
        "predictions": flat,
        "with_rank_and_score": detail,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n{report['n_with_at_least_one_prediction']} of {len(subs)} substrates carry a "
          f"prediction, mean list {report['mean_returned']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
