#!/usr/bin/env python3
"""Assemble the anonymous supplementary archive for the leaderboard paper, and refuse if it leaks.

A submission whose supplementary material names an author is desk rejected, and this repository has
already shipped that defect four times: SELF_CLAIMS.md row 7 records absolute home directories in
committed artifacts, a hard-coded path in the two scripts behind the paper's largest finding, and a
full name with a mail address in `pyproject.toml`. Each was found only because the search pattern
widened, and three of the four rounds were run after a bulk `git add`.

So the archive is built by a script rather than by hand, and the script is a gate: it copies an
allowlist, scans every byte of what it copied against a pattern wider than any round that has run
here, and refuses to write the zip if anything matches. Refusing is the point. A build that strips
what it finds would leave the next unanticipated form of the same leak in place.

What it does NOT do is redistribute other people's data. The corpora this paper re-scores are
third-party and are cited rather than shipped; the archive carries the derived per-item scores and
the code that produces them from a local copy.

    python scripts/build_anonymous_archive.py                 # build and verify
    python scripts/build_anonymous_archive.py --check-only    # scan without writing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What the paper's claims are computed from. Globs are relative to the repository root, and a
# glob matching nothing is an error rather than a silent omission: an archive is judged by what a
# reader can run, and a missing directory is invisible until a reviewer tries.
ALLOW = [
    # the instrument and the survey
    "scripts/robust_order.py",
    "scripts/robust_order_metabolite.py",
    "scripts/make_robust_tables.py",
    "scripts/make_share_figure.py",
    "scripts/make_decomp_figure.py",
    "scripts/ordering_stability.py",
    "scripts/packing_predicts_order.py",
    "scripts/docking_multiplicity.py",
    "scripts/retro_leaderboard.py",
    "scripts/wmt_board.py",
    "scripts/wmt23_boards.py",
    "scripts/wmt_esa_boards.py",
    "scripts/wmt_official_clusters.py",
    "scripts/budget_matched_leaderboard.py",
    "scripts/audit_claim_words.py",
    # the audits the paper releases
    "scripts/external_overlap_audit.py",
    # the artifacts every printed number comes from
    "results/robust_order*.json",
    "results/retro_*.json",
    "results/evalretro_*.json",
    "results/wmt_*.json",
    "results/docking_multiplicity.json",
    "results/budget_matched_leaderboard.json",
    "results/emission_leaderboard.json",
    "results/stopping_rule.json",
    # the paper's own pre-submission audit, which the reproducibility statement points at
    "paper/SELF_CLAIMS.md",
]

# Anything that could name a person, a machine or an account. Wider than the four rounds recorded
# in SELF_CLAIMS row 7, because each of those rounds found what the previous pattern could not see.
#
# Two of these patterns were written first in the form that reads most natural and had to be
# narrowed, because the natural form fires on the data itself. `\bClaude\b` matches 517 times in
# the WMT24 artifact: Claude-3.5 was a submitted system in that shared task, and every match is a
# row of the leaderboard being re-scored. `github\.com/\w+` matches the WMT organisers' own
# repository, which is where the predictions come from and which the paper cites.
#
# A pattern that fires on legitimate content is not a strict gate. It is a gate that gets a blanket
# exemption written for it, and the exemption is what lets the next real leak through. So each is
# rewritten to match the SHAPE of a leak rather than a token that appears in both: an attribution
# line rather than a model name, an unexpected repository owner rather than any owner at all.
IDENTITY = [
    (r"doctawho", "the author's account name"),
    (r"[Pp]olomoshnov|[Nn]ikita\s+[A-Z]|[Rr]udik", "an author surname or given name"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "a mail address"),
    (r"/Users/[a-z]|/home/[a-z]|C:\\\\Users\\\\", "an absolute home directory"),
    (r"orcid\.org/\d", "an ORCID"),
    (r"fbb\.msu\.ru|msu\.ru|Lomonosov|Moscow State", "an institution"),
    (r"zenodo\.org/(record|doi)", "a deposit that names its depositor"),
    # An attribution line, not a model name: a system called Claude-3.5 is a row of the WMT24
    # board and belongs in the artifact, while a line crediting a tool for the work does not.
    (r"Co-Authored-By|[Gg]enerated with|\bAnthropic\b|claude\.ai|Claude Code|\U0001F916",
     "a tool credit that identifies the workflow"),
]

# The repository owners the paper legitimately cites, because their groups released the
# predictions being re-scored. Any OTHER owner refuses, which is the check that matters: the
# author's own account would be an owner nobody declared.
THIRD_PARTY_OWNERS = {"wmt-conference"}


def unexpected_owners(text: str):
    """Repository owners that nobody declared, with the line each sits on."""
    out = []
    for m in re.finditer(r"github\.com/([A-Za-z0-9_.-]+)", text):
        if m.group(1) not in THIRD_PARTY_OWNERS:
            out.append((text.count("\n", 0, m.start()) + 1, m.group(0)))
    return out

SKIP_BINARY = {".pt", ".pdf", ".png", ".jpg", ".zip", ".gz", ".pyc", ".sdf"}


def collect() -> list:
    """Every file the archive should carry, with a hard error on a glob that matches nothing."""
    out, empty = [], []
    for pattern in ALLOW:
        hits = sorted(ROOT.glob(pattern))
        if not hits:
            empty.append(pattern)
            continue
        out.extend(p for p in hits if p.is_file())
    if empty:
        raise SystemExit("these entries match nothing, so the archive would silently omit them:\n  "
                         + "\n  ".join(empty))
    return sorted(set(out))


def scan(paths, base: Path) -> list:
    """Every identity match in the given files, as readable lines."""
    hits = []
    for path in paths:
        rel = str(path.relative_to(base))
        if path.suffix.lower() in SKIP_BINARY:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for pattern, what in IDENTITY:
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}  {what}: {m.group(0)[:60]!r}")
        for line, hit in unexpected_owners(text):
            hits.append(f"{rel}:{line}  an undeclared repository owner: {hit!r}")
    return hits


README = """# Supplementary material

This archive holds the code and the derived artifacts behind the paper's measurements. It is
anonymous: it carries no author name, no institution, no mail address and no repository that
identifies one, and the script that assembled it refuses to write the archive if any appears.

## What is here

`scripts/` the instrument and the analyses. `robust_order.py` computes the dominance order of a
leaderboard from its released predictions under a declared grid of evaluation choices, which is
the object the paper reports. The rest run it over each board, build the tables and figures, and
carry the audits.

`results/` the artifact behind every number the paper prints. Each records the script that wrote
it and the commit it ran at. Nothing in the paper is typed by hand except the measurements listed
in the audit, and a checker refuses any other numeric literal in the manuscript.

`SELF_CLAIMS.md` the pre-submission audit: seventeen checks the authors ran against the
manuscript's claims about itself, with what each found. Several failed and are documented,
including three defects this work shipped for weeks.

## What is not here, and why

The corpora re-scored here belong to other groups and are cited rather than redistributed. Every
board's per-item scores are in `results/`, so the paper's numbers regenerate from this archive
alone; reproducing them from the raw submissions instead needs a local copy of the corpora, which
the scripts take by path.

Trained model weights are third-party for two of the compared methods and are cited. The
case-study predictor's own checkpoints are large and are not carried here.

## Reproducing

Each script writes one artifact and prints what it measured. Running one over an artifact already
in `results/` recomputes it; the analyses are deterministic given a seed, which each records.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "supplementary.zip"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    files = collect()
    stage = ROOT / "dist" / "_supplementary"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for path in files:
        rel = path.relative_to(ROOT)
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (stage / "README.md").write_text(README)

    # The scan runs over the staged copy, not over the source tree: what ships is what is checked.
    problems = scan(sorted(stage.rglob("*")), stage)

    # The archive is half of what ships. The other half is the PDF, and its identity can leak
    # somewhere no grep over the sources reaches: pdfTeX writes an /Author and a /Creator into
    # the document information dictionary, and a figure included from disk leaves its path in
    # the file. Both are read here out of the bytes that would be uploaded.
    pdf = ROOT / "paper" / "grail_iclr.pdf"
    if pdf.exists():
        raw = pdf.read_bytes().decode("latin-1")
        text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                              capture_output=True, text=True).stdout
        for label, blob in (("the PDF's metadata", raw), ("the PDF's rendered text", text)):
            for pattern, what in IDENTITY:
                for m in re.finditer(pattern, blob):
                    problems.append(f"{label}  {what}: {m.group(0)[:60]!r}")
            for _, hit in unexpected_owners(blob):
                problems.append(f"{label}  an undeclared repository owner: {hit!r}")
        # \iclrfinalcopy is the switch that un-anonymises the paper. Without it the style prints
        # "Anonymous authors" and ignores \author entirely, which is why an \author line naming
        # someone is invisible in a submission build and a scan of the PDF alone reports clean.
        # The switch is what to check, and it also changes the running head, so the head is the
        # evidence: a submission says "Under review", a final copy says "Published as".
        if "Published as a conference paper" in text:
            problems.append("the PDF's running head  the camera-ready switch is on: "
                            "\\iclrfinalcopy un-anonymises the paper and changes the head from "
                            "\"Under review\" to \"Published as\"")
        elif "Under review as a conference paper" not in text:
            problems.append("the PDF's running head  neither the submission nor the camera-ready "
                            "head is present, so the style may not be in use at all")
        print(f"scanned {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB), its metadata and its head")
    else:
        print(f"NOTE: {pdf} is not built, so only the archive was scanned")
    print(f"{len(files)} files staged, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    if problems:
        print(f"\nREFUSING: {len(problems)} identity matches in what would ship")
        for line in problems[:40]:
            print("   " + line)
        if len(problems) > 40:
            print(f"   ... and {len(problems) - 40} more")
        return 1
    print("identity scan: clean")

    if args.check_only:
        return 0

    manifest = {"files": {}, "note": "sha256 of every file in this archive"}
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(stage))] = hashlib.sha256(
                path.read_bytes()).hexdigest()[:16]
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
