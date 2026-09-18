#!/usr/bin/env python3
"""What identifies the GLORYxR arm: an unpublished tool, unpublished weights, and a permission.

Writes results/gloryxr_identity.json.

GLORYxR carries two columns here, run locally in both site-of-metabolism modes. It is the only arm
whose software AND weights are both unpublished. There is no release, no DOI and no paper, so a
repository URL alone does not identify what ran; and the model dumps are not distributed with the
package, were supplied privately by its first author for this work, and exist on this machine only
under a session scratchpad.

WHAT WAS MISSING. results/gloryxr_local_preds_{default,strict}.json record the repository, the
interpreter and seven package versions. They do not record which commit of GLORYxR ran, and they
identify the weights by a scratchpad path. So the key input to two published columns was named by a
directory that will not outlive the session, and the tool by a URL with no revision. This closes
both: the revision is read out of the checkout, and every dump is hashed.

Hashing them now is not bookkeeping. Their author intends to deposit the checkpoints under an
academic licence. When that happens, these digests are what let anyone check the deposit is the
weights these columns were computed from -- a check nobody can make from a path.

TWO OF THE DUMPS ARE THE SAME FILE. That is measured here, not repeated from anyone: the multi-model
provider loads one dump per rule class, and two of those classes carry byte-identical weights. Its
author has confirmed that this is intended, both being Phase 1 reactions the tool scores the same.
Both halves belong in the record. The measurement, because anyone who hashes a future deposit will
find it; the confirmation, because only its author can say whether a duplicated 59 MB file is a
design or a mistake, and a reader who finds it unexplained would reasonably suspect this work of
having loaded the wrong file.

THE PERMISSION is recorded with the conditions it carries, and each condition is checked against the
manuscript by the test beside this file rather than asserted here. A permission recorded as honoured
while the manuscript quietly fails one of its terms would be worse than no record at all.

    python scripts/gloryxr_identity.py --checkout <path to the GLORYxR clone> \\
        --dumps <path to the multi_models directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

OUT = ROOT / "results" / "gloryxr_identity.json"
COLUMNS = (ROOT / "results" / "gloryxr_local_preds_default.json",
           ROOT / "results" / "gloryxr_local_preds_strict.json")

# The three terms the permission carries, and the string in paper2/si.tex that discharges each. The
# strings are the evidence the test looks for, so an edit that drops one fails there rather than
# leaving a condition recorded as met by a sentence that no longer exists.
CONDITIONS = (
    ("the paper says this is a pre-publication version",
     "GLORYxR is itself unpublished"),
    # Without the scheme: this manuscript writes repositories as \texttt{github.com/...}, the way
    # it already writes its own, and the evidence a condition is checked by has to be the string the
    # prose actually uses rather than the one this file would prefer.
    ("the paper links the repository",
     "github.com/molinfo-vienna/GLORYxR"),
    ("the paper states that GLORYxR succeeds GLORYx",
     "successor"),
)


def _git(checkout: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=checkout, capture_output=True, text=True, timeout=20)
    if r.returncode:
        sys.exit(f"git {' '.join(args)} failed in {checkout}: {r.stderr.strip()}")
    return r.stdout.strip()


def tool_identity(checkout: Path) -> dict:
    """Repository, revision, version and licence, read out of the checkout.

    Read rather than typed. A revision copied by hand into a record is the one field nobody can
    check afterwards, and for a tool with no release it is the whole of the identity.
    """
    if not (checkout / ".git").exists():
        sys.exit(f"{checkout} is not a git checkout; the revision that ran cannot be read from it")
    url = _git(checkout, "config", "--get", "remote.origin.url")
    commit = _git(checkout, "rev-parse", "HEAD")
    when = _git(checkout, "log", "-1", "--format=%cI")
    dirty = bool(_git(checkout, "status", "--porcelain"))
    if dirty:
        sys.exit(f"{checkout} has uncommitted changes, so no revision describes what ran; commit or "
                 f"clean it before recording an identity")

    version = licence = None
    pyproject = checkout / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            s = line.strip()
            if s.startswith("version") and version is None:
                version = s.split("=", 1)[1].strip().strip('"\'')
            elif s.startswith("license") and licence is None:
                licence = s.split("=", 1)[1].strip().strip('"\'')
    return {"name": "GLORYxR",
            "repository": url.rstrip("/").removesuffix(".git"),
            "commit": commit,
            "commit_date": when,
            "version": version,
            "licence": licence,
            "published": False,
            "what_unpublished_means_here": (
                "there is no release, no DOI and no paper for this tool, so the revision above is "
                "what identifies it; its first author states it remains unpublished"),
            "relation_to_gloryx": "successor to GLORYx",
            "read_from": "the checkout, not typed into this file"}


def hash_dumps(dumps_dir: Path) -> list:
    """Every model dump the provider loads, by digest and size.

    Sorted by name so two runs of this produce the same order, and a diff of the artifact shows a
    changed weight rather than a reshuffle.
    """
    if not dumps_dir.is_dir():
        sys.exit(f"{dumps_dir} is not a directory; the weights these columns used cannot be hashed")
    out = []
    for p in sorted(dumps_dir.glob("*.joblib")):
        out.append({"name": p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "bytes": p.stat().st_size})
    if not out:
        sys.exit(f"{dumps_dir} holds no .joblib dump")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True, help="the GLORYxR git clone that was run")
    ap.add_argument("--dumps", required=True,
                    help="the directory of model dumps the run loaded, normally multi_models")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tool = tool_identity(Path(args.checkout))
    dumps = hash_dumps(Path(args.dumps))

    # The prediction columns must name the same repository, or this record describes a different
    # tool from the one that produced them.
    for col in COLUMNS:
        if not col.exists():
            continue
        src = (json.loads(col.read_text()).get("obtained_from") or {}).get("source", "")
        if src.rstrip("/").removesuffix(".git") != tool["repository"]:
            sys.exit(f"{col.name} names {src!r} and the checkout is {tool['repository']!r}; one of "
                     f"them is not the arm in this paper")

    by_digest: dict = {}
    for m in dumps:
        by_digest.setdefault(m["sha256"], []).append(m["name"])
    identical = []
    duplicated = 0
    for h, names in by_digest.items():
        if len(names) < 2:
            continue
        size = next(m["bytes"] for m in dumps if m["sha256"] == h)
        duplicated += size * (len(names) - 1)
        identical.append({
            "sha256": h,
            "names": sorted(names),
            "bytes_each": size,
            "measured_here": True,
            "confirmed_intended_by_the_author": True,
            "what_the_author_said": (
                "that the models are identical, both classes being simply Phase 1 reactions that "
                "should be scored the same, and that this is inelegant and wasteful of disk space "
                "and model size but intended for the current version"),
            "why_this_is_recorded": (
                "anyone who hashes these weights will find the duplicate; without the author's "
                "statement beside the measurement a reader would reasonably suspect this work of "
                "having loaded the same file twice by mistake"),
        })

    art = {
        "what_this_is": ("what identifies the GLORYxR arm: the revision of an unpublished tool, the "
                         "digests of unpublished weights, and the permission under which both are "
                         "used here"),
        "provenance": stamp(__file__),
        "inputs": record_inputs([c for c in COLUMNS if c.exists()]),
        "tool": tool,
        "model_provider": "MultiFAME3RModelProvider",
        "why_this_provider": ("the one the published columns took; its docstring states predictions "
                              "will closely follow those of the original GLORYx implementation"),
        "model_dumps": dumps,
        "n_dumps": len(dumps),
        "n_distinct_weights": len(by_digest),
        "identical_dumps": identical,
        "bytes_duplicated": duplicated or None,
        "where_the_dumps_live": (
            "nowhere permanent. They are not distributed with the package and were supplied "
            "privately for this work, so on this machine they exist only under a session "
            "scratchpad. The digests above are what will identify them after that path is gone, "
            "and what will let anyone check a future deposit is the same weights"),
        "permission": {
            "granted_by": "GLORYxR's first author",
            "scope": ("inclusion of GLORYxR in this work as a pre-publication version, and use of "
                      "the privately supplied model dumps to run it"),
            "conditions": [
                {"condition": cond, "met": True, "evidence_in_the_manuscript": ev}
                for cond, ev in CONDITIONS
            ],
            "availability_note": (
                "its author states an intention to deposit the model checkpoints under an academic "
                "licence, which would remove the availability limitation. That has not happened at "
                "the time of writing and nothing here anticipates it: the limitation stands as "
                "recorded until a deposit exists"),
            "recorded_not_quoted": (
                "the correspondence itself is private and is not reproduced here or in the "
                "manuscript; what is recorded is the substance a reader needs, which is that "
                "permission was given and on what terms"),
        },
    }
    Path(args.out).write_text(json.dumps(art, indent=1))
    print(f"wrote {Path(args.out).relative_to(ROOT)}")
    print(f"  {tool['name']} {tool['version']} at {tool['commit'][:12]} ({tool['licence']})")
    print(f"  {len(dumps)} dumps, {len(by_digest)} distinct weights")
    for g in identical:
        print(f"  identical: {' == '.join(g['names'])}  ({g['bytes_each']:,} bytes each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
