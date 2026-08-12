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
on the same boards. Comparing the two counts directly settles nothing, because they differ in two
places at once: the test that decides whether a pair is separated, and the construction that turns
pairwise verdicts into a count. Their construction walks the score-sorted table and closes a cluster
only when nothing at or above the current position is tied with anything below it, so a cluster is
an interval of the published order and can hold pairs its own test separates; ours is the longest
chain in the separation relation. On identical verdicts the chain is never shorter than the number
of clusters, so the raw comparison is decided by the construction before either test is consulted.

The two are therefore crossed, which is the only reading that attributes the difference:

    their test, their construction    the count the task itself would publish
    our test, their construction      their clustering fed our published-cell verdicts
    their test, our construction      the longest chain in the relation their p-values induce
    our test, our construction        the places this paper says the board supports

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
            "pvalues": {f"{a}||{b}": float(v) for (a, b), v in pvalues.items()},
            "alpha": tools.ALPHA_THRESHOLD}


def _their_clusters_on_our_verdicts(tools, ours: dict, order: list) -> int:
    """Their construction, fed this paper's separation verdicts in the published cell.

    ``get_clusters`` consults a $p$ only against its own threshold, so a verdict transplants
    exactly: separated becomes a $p$ below it, unresolved a $p$ above. Nothing else changes, which
    is what makes this the test's contribution and not the construction's.
    """
    import pandas as pd

    cell = ours["published_cell"]
    pv = {}
    for name, v in ours["pairs"].items():
        hi, lo = name.split(" over ")
        sep = v["per_cell"][cell]["separated"]
        pv[(hi, lo)] = pv[(lo, hi)] = 0.0 if sep else 1.0
    df = pd.DataFrame({"system_id": order})
    return int(max(tools.get_clusters(pv, df).values()))


def _our_chain_on_their_verdicts(theirs: dict) -> int:
    """This paper's construction, fed their test's verdicts: the longest chain they induce."""
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("_ro", ROOT / "scripts" / "robust_order.py")
    ro = _iu.module_from_spec(spec)
    sys.modules["_ro"] = ro
    spec.loader.exec_module(ro)

    order = theirs["official_order"]
    rank = {s: i for i, s in enumerate(order)}
    edges: dict = {}
    for key, p in theirs["pvalues"].items():
        a, b = key.split("||")
        if p <= theirs["alpha"] and rank[a] < rank[b]:
            edges.setdefault(a, set()).add(b)
    return ro._tiers(order, edges)


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
        theirs["their_construction_our_test"] = _their_clusters_on_our_verdicts(
            tools, ours, theirs["official_order"])
        theirs["our_construction_their_test"] = _our_chain_on_their_verdicts(theirs)
        theirs.pop("pvalues")
        out[lp] = theirs
        print(f"  {lp}: {theirs['n_systems']} systems")
        print(f"      their test, their construction: {theirs['n_clusters']}")
        print(f"      our test,   their construction: {theirs['their_construction_our_test']}")
        print(f"      their test, our construction:   {theirs['our_construction_their_test']}")
        print(f"      our test,   our construction:   {own}")
        print(f"      the two orders "
              f"{'agree' if theirs['orders_agree'] else 'DIFFER'}")

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
