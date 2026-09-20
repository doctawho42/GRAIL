#!/usr/bin/env python3
"""Generate BioTransformer 3.0 predictions for a set of parent SMILES -> a predictions JSON
{parent_smiles: [metabolite_smiles ...]} that drops into the match-sensitivity engine.

Runs the JAR per parent (so every metabolite, across generations up to --steps, maps to its
original parent) from the JAR's own dir (it loads btkb/supportfiles relatively). Output is a
rule-based unranked set (BioTransformer gives no confidence score).

A FAILURE IS NOT A PREDICTION. Every earlier version of this file returned [] from a bare
`except Exception` with no returncode check, so a JVM crash, a 600-second timeout and a parent
BioTransformer genuinely has no metabolite for were written into the artifact as the same thing:
an empty list. That is a property of the code, and it is the whole of what is established here.
Two consequences followed from it. `--resume` skipped any parent already present, so a swallowed
failure was frozen in permanently and could never be retried. And the empties could not be
counted, because nothing distinguished them.

WHAT IS NOT ESTABLISHED, and was asserted here in an earlier draft of this docstring: that the
frozen one-step predictions "do not reproduce, the frozen set being a strict SUBSET of a re-run".
No artifact in this repository records such a re-run, so that sentence was an inference about a
mechanism restated as a measurement, and it had been copied forward unchecked. What IS measured
is weaker and different: the two frozen BioTransformer artifacts disagree WITH EACH OTHER in both
directions over the 291 parents they share -- 80 parents carry a metabolite in
biotransformer_allhuman_one_step_preds.json that biotransformer_fulltest_preds.json lacks, 15 are
empty in the second and not in the first, 1 the other way. They were written by different
producers, and neither records the -s it ran under, so neither can be called the one-step arm.
Whether either reproduces is a question this runner exists to answer, not one it may assume.

Here a parent enters `preds` only on a clean exit; anything else is recorded, by name and reason,
in the run sidecar, and is retried on the next --resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
BT_DIR = Path(os.environ.get("BIOTRANSFORMER_DIR",
                             ROOT / "artifacts" / "tier2" / "biotransformer"))
JAR = Path(os.environ.get("BIOTRANSFORMER_JAR", BT_DIR / "biotransformer-3.0.0.jar"))
# The bundled JNI InChI artefact is cached for MAC-X86_64. On an arm64 machine the jar exits
# before predicting anything under the default JDK, and runs under an x86_64 Java 8 one. Naming
# the runtime here is the difference between "this tool cannot be run" and "this tool needs a
# runtime we have", and the two are not the same claim about a comparator.
JAVA = os.environ.get("BIOTRANSFORMER_JAVA", "java")


def load_parents(spec: str) -> List[str]:
    p = Path(spec)
    raw = p.read_text()
    if spec.endswith(".json"):
        data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw))
        # A predictions file is {parent: [metabolite ...]}; a GLORYx file is a list of records.
        # Taking the keys of the first is how a fresh run covers exactly the parents an earlier
        # one did, in the spelling that run used, without re-deriving the population.
        if isinstance(data, dict):
            return [k for k in data if isinstance(k, str) and k.strip()]
        return [x["smiles"] for x in data if x.get("smiles")]
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def bt_one(smiles: str, steps: int, timeout: int = 900) -> Tuple[List[str], str]:
    """Return (metabolites, status). Status is "ok" ONLY on a clean exit with a readable file.

    The caller must not store a non-ok result as a prediction. An empty list with status "ok" is
    a real BioTransformer answer; an empty list with any other status is a missing measurement.
    """
    out = Path(tempfile.mktemp(suffix=".csv"))
    try:
        r = subprocess.run(
            [JAVA, "-jar", str(JAR), "-k", "pred", "-b", "allHuman", "-cm", "3",
             "-s", str(steps), "-ismi", smiles, "-ocsv", str(out)],
            cwd=str(BT_DIR), capture_output=True, timeout=timeout,
        )
        if r.returncode != 0:
            # The exit code alone sends a reader back to the jar to reproduce the crash by hand,
            # which is what it cost to learn that these are two distinct CDK failures and not one
            # flaky run. The exception line goes into the record with it.
            blob = (r.stderr or b"").decode("utf-8", "replace") + \
                   (r.stdout or b"").decode("utf-8", "replace")
            why = ""
            for ln in blob.splitlines():
                if "Exception" in ln or "Error" in ln:
                    why = ln.strip()[:180]
                    break
            return [], f"exit{r.returncode}" + (f" | {why}" if why else "")
        if not out.exists():
            # BioTransformer writes no file when it parses the input and finds nothing to do.
            # That is a real answer and is reported as one, but separately from a clean run that
            # wrote an empty table, because the two are different behaviours of the tool.
            return [], "ok-nofile"
        mets = []
        with open(out) as f:
            for row in csv.DictReader(f):
                smi = (row.get("SMILES") or "").strip()
                if smi:
                    mets.append(smi)
        return mets, "ok"
    except subprocess.TimeoutExpired:
        return [], f"timeout{timeout}"
    except Exception as e:                                   # noqa: BLE001 - reported, not hidden
        return [], f"error:{type(e).__name__}"
    finally:
        out.unlink(missing_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    """A kill during json.dump leaves a truncated file that the next --resume cannot parse, and
    the old code answered that by discarding the artifact and starting the hours again."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sidecar(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".run.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", required=True,
                    help="GLORYx json, a predictions json whose KEYS are the parents, or a txt "
                         "of one SMILES per line")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parents in flight at once; the jar is a process per parent")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per parent")
    ap.add_argument("--every", type=int, default=5, help="checkpoint every N parents")
    ap.add_argument("--resume", action="store_true",
                    help="keep what --out already holds and predict only the rest, since a run "
                         "over the whole evaluated test set is hours and a lost run is hours. "
                         "Parents that FAILED are not kept and are retried.")
    args = ap.parse_args()

    parents = load_parents(args.parents)
    if args.limit:
        parents = parents[: args.limit]
    preds: Dict[str, List[str]] = {}
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        try:
            preds = json.loads(out_path.read_text())
        except Exception:
            print(f"  {out_path} will not parse; starting this population over", flush=True)
            preds = {}
    todo = [p for p in parents if p not in preds]

    failures: Dict[str, str] = {}
    statuses: Dict[str, int] = {}
    t = time.perf_counter()
    run = {
        "tool": "BioTransformer 3.0.0", "flags": ["-k", "pred", "-b", "allHuman", "-cm", "3",
                                                  "-s", str(args.steps)],
        "steps": args.steps, "java": JAVA, "jar": str(JAR), "host": platform.platform(),
        "parents_source": args.parents, "n_parents": len(parents), "workers": args.workers,
        "timeout_s": args.timeout, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    def checkpoint(done: int, total: int) -> None:
        _atomic_write(out_path, json.dumps(preds, indent=2))
        run.update({"done": done, "todo": total, "elapsed_s": round(time.perf_counter() - t, 1),
                    "statuses": statuses, "failures": failures,
                    "n_ok": len(preds), "n_failed": len(failures)})
        _atomic_write(_sidecar(out_path), json.dumps(run, indent=2))

    def record(p: str, mets: List[str], status: str) -> None:
        # The reason string carries the exception text, which is per-parent; the counter is keyed
        # on the class of failure so the tally stays readable.
        statuses[status.split(" | ")[0]] = statuses.get(status.split(" | ")[0], 0) + 1
        if status.startswith("ok"):
            preds[p] = mets
            failures.pop(p, None)
        else:
            failures[p] = status

    print(f"BioTransformer on {len(todo)} of {len(parents)} parents (steps={args.steps}, "
          f"{args.workers} at a time, timeout {args.timeout}s)", flush=True)
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (p, (mets, st)) in enumerate(
                    zip(todo, pool.map(lambda s: bt_one(s, args.steps, args.timeout), todo)), 1):
                record(p, mets, st)
                if i == 1 or i % args.every == 0 or i == len(todo):
                    checkpoint(i, len(todo))
                    print(f"  {i}/{len(todo)} ({time.perf_counter()-t:.0f}s) "
                          f"last={len(mets)} mets [{st}] failed={len(failures)}", flush=True)
    else:
        for i, p in enumerate(todo, 1):
            mets, st = bt_one(p, args.steps, args.timeout)
            record(p, mets, st)
            # Checkpointing was absent from this branch entirely: a single-worker run that died
            # at hour nine had written nothing at all.
            if i == 1 or i % args.every == 0 or i == len(todo):
                checkpoint(i, len(todo))
                print(f"  {i}/{len(todo)} ({time.perf_counter()-t:.0f}s) "
                      f"last={len(mets)} mets [{st}] failed={len(failures)}", flush=True)

    checkpoint(len(todo), len(todo))
    print(f"wrote {args.out} ({len(preds)} parents, "
          f"{sum(len(v) for v in preds.values())} total metabolites); "
          f"{len(failures)} parents FAILED and are not in it -- see {_sidecar(out_path).name}",
          flush=True)
    if failures:
        print("  failures by reason: "
              + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()) if not k.startswith("ok")),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
