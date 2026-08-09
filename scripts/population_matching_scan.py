#!/usr/bin/env python3
"""Which paragraphs could violate "every comparison is matched on population"?

The ledger's third row is a read-through, and a read-through over a forty-page manuscript is
exactly the check that decays: it passes once, the manuscript grows, and nobody re-runs it because
re-running it means reading everything again. This does not replace the reading. It reduces what
has to be read to the paragraphs where the claim can actually fail, so the row is cheap enough to
re-run after every change.

Two independent passes, because they fail differently:

  prose     a paragraph that makes a comparison and names more than one population. Some of these
            are legitimate and say so -- a replication that moves from one set to another declares
            the move -- so this pass reports rather than judges.

  artifacts an artifact whose blocks disagree about which population produced them. This one is
            not a judgement call: a single file recording n=1170 in one block and n=245 in another
            is wrong however the manuscript quotes it, and it is how the defect entered the
            provenance appendix -- a script that measured its cells on the population it was given
            and read its endpoints from a file named for a different one.

Exit status is non-zero only for the second pass. The first prints its findings and returns them
for a human, which is what the row has always been.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The populations this paper distinguishes, as they are written in the manuscript. A name is
# matched by its size, because that is the form a reader can check against a table caption.
POPULATION_PATTERNS = {
    r"\\nshared\b|\b150\b": "150 shared",
    r"\\nclean\b|1\{,\}170|\b1170\b": "1,170 clean test",
    r"\b245\b": "245 subsample",
    r"\b291\b": "291 four-method",
    r"\b994\b": "994 clean validation",
    r"\\nglory\b|\b37\b": "37 GLORYx",
}
COMPARISON = re.compile(
    r"\b(above|below|leads?|behind|ahead|erases?|reorders?|reordering|exchanges?|beats?|"
    r"outperforms?|higher|lower|more than|less than|against|versus)\b", re.I)

# A file named for a population carries that population in its name; the untagged name is the
# subsample the family started on. This is the only mapping the second pass needs.
TAGGED = re.compile(r"__(\w+?)(?:__\w+)?\.json$")
# A declared population must be a name, not a sentence: free text in that field is
# how a file comes to declare something no other file can be compared against.
POPULATION_NAMES = {"subsample245", "clean_test", "clean_val", "shared150"}


def scan_prose(paths: list[Path]) -> list[dict]:
    """Paragraphs that compare, and name more than one population while doing it."""
    out = []
    for f in paths:
        txt = re.sub(r"(?m)^\s*%.*$", "", f.read_text())
        for para in re.split(r"\n\s*\n", txt):
            one = " ".join(para.split())
            if not COMPARISON.search(one):
                continue
            names = sorted({n for pat, n in POPULATION_PATTERNS.items() if re.search(pat, one)})
            if len(names) > 1:
                out.append({"file": str(f.relative_to(ROOT)), "populations": names,
                            "text": one[:220]})
    return out


def _config_strings(cfg, path="config"):
    """Every string a config records, with where it records it -- provenance lives in these."""
    if isinstance(cfg, str):
        yield path, cfg
    elif isinstance(cfg, dict):
        for k, v in cfg.items():
            yield from _config_strings(v, f"{path}.{k}")
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            yield from _config_strings(v, f"{path}[{i}]")


# The sizes the family's populations have, so a referenced file that records only a count can
# still be placed. Anything else is left unplaced rather than guessed at.
BY_SIZE = {245: "subsample245", 1170: "clean_test", 994: "clean_val", 150: "shared150"}


def _population_of(ref: str) -> str | None:
    """Which population a referenced results file holds.

    Asking the file beats reading its name. A file that declares its population answers directly;
    one that holds several is compatible with any caller; one that records only a size is placed
    by that size. Only when none of those apply is the name's tag used, and an untagged name is
    then the subsample the family started on.
    """
    path = ROOT / ref if not ref.startswith("/") else Path(ref)
    if path.exists():
        try:
            d = json.loads(path.read_text())
        except Exception:
            d = None
        if isinstance(d, dict):
            cfg = d.get("config")
            if isinstance(cfg, dict) and cfg.get("population") in POPULATION_NAMES:
                return cfg["population"]
            if isinstance(d.get("populations"), dict):
                return None          # holds several, so it contradicts no caller
            for k in ("n_substrates", "n"):
                if isinstance(d.get(k), int) and d[k] in BY_SIZE:
                    return BY_SIZE[d[k]]
                if isinstance(cfg, dict) and isinstance(cfg.get(k), int) and cfg[k] in BY_SIZE:
                    return BY_SIZE[cfg[k]]
    name = pathlib.Path(ref).name
    if not name.endswith(".json"):
        return None
    m = TAGGED.search(name)
    return m.group(1) if m else "subsample245"


def scan_artifacts(files: list[Path]) -> list[dict]:
    """A file that declares a population must not source a number from a different one.

    This is the check the count-based version could not make. The defect it exists for left no
    second count to find: the endpoints were copied in as bare floats, and the only trace was a
    provenance string in the config naming a file written for another population. Counting `n`
    keys instead finds every stratified artifact and none of this.
    """
    bad = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        cfg = d.get("config") if isinstance(d, dict) else None
        declared = cfg.get("population") if isinstance(cfg, dict) else None
        if declared not in POPULATION_NAMES:
            continue
        for where, text in _config_strings(cfg):
            for ref in re.findall(r"[\w/]+\.json", text):
                other = _population_of(ref)
                if other and other != declared:
                    bad.append({"file": str(f.relative_to(ROOT)), "declared": declared,
                                "at": where, "source": ref, "source_population": other})
    return bad


def _declares(f: Path) -> bool:
    try:
        cfg = json.loads(f.read_text()).get("config")
    except Exception:
        return False
    return isinstance(cfg, dict) and cfg.get("population") in POPULATION_NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper", help="directory holding the manuscript sources")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    tex = sorted((ROOT / args.paper).glob("*.tex")) + sorted((ROOT / args.paper / "app").glob("*.tex"))
    prose = scan_prose(tex)
    print(f"prose: {len(prose)} paragraph(s) compare across more than one population")
    for r in prose:
        print(f"  [{r['file']}] {', '.join(r['populations'])}")
        print(f"      {r['text'][:150]}")
    print("  each needs a sentence declaring the move, or the numbers brought onto one population\n")

    arts = sorted((ROOT / args.results).glob("*.json"))
    bad = scan_artifacts(arts)
    declared = sum(1 for a in arts if _declares(a))
    print(f"artifacts: {len(bad)} of the {declared} that declare a population source a number "
          f"from a different one")
    for r in bad:
        print(f"  {r['file']}  declares {r['declared']}, {r['at']} reads {r['source']} "
              f"({r['source_population']})")
    if bad:
        print("\nan artifact that holds two populations makes every number read from it ambiguous")
        return 1
    print("  none: every declared artifact sources only its own population")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
