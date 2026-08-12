#!/usr/bin/env python3
r"""How many places does the shared task itself say its leaderboard supports?

Everything else here measures "supported places" with this paper's instrument, which invites the
obvious objection: the instrument may be idiosyncratic, and a number no one else computes is a
number no one else can check. The translation task removes the objection, because it computes its
own version and publishes it. Its clustering script tests every pair with a Wilcoxon signed-rank
test on the paired per-segment scores within each domain, combines the domains by Stouffer's $Z$,
and merges two systems into one cluster whenever that test does not separate them at $0.05$. The
number of clusters is the task's own answer to how many ranks its table supports.

So the task's answer is computed with the task's own code -- ``humeval/tools.py`` is imported, not
paraphrased, and ``get_pvalues`` and ``get_clusters`` are theirs unmodified -- and set beside ours
on the same boards. Two of ours are comparable, because a tier count is what the same claim looks
like in this paper's terms:

    own cell only    what the leaderboard's published cell resolves, which is what they measure
    the whole grid   the same, required to hold in every declared cell

Only one thing is reimplemented, and only for speed: the domain of a segment, which their
``attach_resources`` attaches by iterating the frame row by row and which is read here straight out
of the same ``documents/*.docs`` file, with their alignment assertion kept.

A caveat that belongs next to the numbers rather than in a footnote: neither side corrects for
multiplicity here, so both counts are the optimistic ones. That is the point of the comparison --
the task's own optimistic count against ours, computed the same way on the same data.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMEVAL = ROOT / "data/external/wmt24/humeval"
TXT = ROOT / "data/external/wmt24/txt"

BOARDS = [("en-de", "mqm_generalMT2024_ende.tsv", "results/robust_order_wmt24_en-de.json"),
          ("ja-zh", "mqm_generalMT2024_jazh.tsv", "results/robust_order_wmt24_ja-zh.json")]


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _their_tools():
    """Their module, imported as written. ``ipdb`` is a debugger they left at the top of it."""
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    if str(HUMEVAL) not in sys.path:
        sys.path.insert(0, str(HUMEVAL))
    import tools  # noqa: E402

    return tools


def _domains(lp: str) -> list:
    return [ln.rstrip("\n").split("\t") for ln in (TXT / f"documents/{lp}.docs").read_text()
            .splitlines()]


def board(tools, lp: str, tsv: str) -> dict:
    df = tools.load_data(str(HUMEVAL / tsv), is_mqm=True)
    docs = _domains(lp)
    # attach_resources' offset, and its assertion, on the same file it reads
    seg = df["segment_id"].astype(int) + 1
    assert all(docs[i][1] == d for i, d in zip(seg, df["doc_id"])), "document alignment"
    # only the domain is added; the segment key their test pairs on is the one load_data built
    df = df.assign(domain_name=[docs[i][0] for i in seg])

    per_domain = df.groupby(["system_id", "domain_name"])["overall"].mean().reset_index()
    avg = (per_domain.groupby("system_id")["overall"].mean().reset_index()
           .sort_values("overall", ascending=False))
    pvalues = tools.get_pvalues(df, True)
    clusters = tools.get_clusters(pvalues, avg)
    order = list(avg["system_id"])
    return {"official_order": order,
            "clusters": {s: int(clusters[s]) for s in order},
            "n_clusters": int(max(clusters.values())),
            "n_systems": len(order),
            "alpha": tools.ALPHA_THRESHOLD}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "wmt_official_clusters.json"))
    args = ap.parse_args()

    tools = _their_tools()
    dec = ROOT / "results/places_decomposition.json"
    D = json.loads(dec.read_text())["per_board"] if dec.exists() else {}

    out = {}
    for lp, tsv, ours_path in BOARDS:
        theirs = board(tools, lp, tsv)
        ours = json.loads((ROOT / ours_path).read_text())
        key = [k for k in D if k.startswith(ours["published_cell"])
               and k.endswith("/".join(ours["published_order"][:2]))]
        own = D[key[0]]["own_cell_separates"] if key else None
        theirs["ours_own_cell_only"] = own
        theirs["ours_whole_grid"] = ours["tiers_distinguished"]
        theirs["orders_agree"] = theirs["official_order"] == ours["published_order"]
        out[lp] = theirs
        print(f"  {lp}: {theirs['n_systems']} systems; the task's own clustering supports "
              f"{theirs['n_clusters']}, ours {own} in the published cell and "
              f"{ours['tiers_distinguished']} across the grid; "
              f"the two orders {'agree' if theirs['orders_agree'] else 'DIFFER'}")

    rep = {"config": {**_code_version(),
                      "source": "WMT24 humeval/tools.py get_pvalues and get_clusters, imported",
                      "note": "neither count is corrected for multiplicity; both are the "
                              "optimistic reading, which is what makes them comparable"},
           "boards": out}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
