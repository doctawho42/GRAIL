#!/usr/bin/env python3
"""Does the inclusion filter drop papers by publisher rather than by content?

The worry that prompted this was the ``no_text`` branch: OpenAlex takes abstracts from Crossref and
not every publisher deposits them, so a work without one might be dropped whatever it says. That
turned out to be wrong, and the first run of this script is what showed it -- ``passes_inclusion``
scans title AND abstract together, so a present title always saves a work from ``no_text``, and the
count is zero.

The bias channel is real, but it wears a different label. A work with no abstract has only its title
scanned, so it is far likelier to fail on ``no_rule_term`` or ``no_reach_term`` -- and those two
account for almost every exclusion. If abstract coverage correlates with publisher, so does the
frame. That is what this measures: the term-drop rate among works with an abstract against the rate
among works without, and the same split per publisher.

It writes no frame and no sample: running it does not freeze the snapshot, which is the point of
running it first.

    python audit/check_no_text.py --out audit/no_text_report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_frame import (  # noqa: E402
    OPENALEX, RULE_RX, REACH_RX, SEEDS, inverted_index_to_text, passes_inclusion,
)

SELECT = ("id,doi,title,publication_year,type,is_retracted,"
          "abstract_inverted_index,primary_location")


def publisher(work) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("host_organization_name") or "(no publisher recorded)"


def venue(work) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or "(no venue recorded)"


def fetch(seed_doi, session, mailto):
    meta = session.get(f"{OPENALEX}/doi:{seed_doi}", params={"mailto": mailto}, timeout=30)
    meta.raise_for_status()
    seed_id = meta.json()["id"].rsplit("/", 1)[-1]
    out, cursor = [], "*"
    while cursor:
        r = session.get(OPENALEX, params={"filter": f"cites:{seed_id}", "per-page": 200,
                                          "cursor": cursor, "mailto": mailto, "select": SELECT},
                        timeout=60)
        r.raise_for_status()
        page = r.json()
        out.extend(page["results"])
        cursor = page["meta"].get("next_cursor")
        if not page["results"]:
            break
    return out


def title_only_would_enter(work) -> bool:
    """Would this work pass on its title alone, ignoring the missing abstract?"""
    title = work.get("title") or ""
    return bool(RULE_RX.search(title) and REACH_RX.search(title))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "no_text_report.json"))
    ap.add_argument("--mailto", default="", help="left empty on purpose: an address sent here is "
                                                 "sent to a third party")
    args = ap.parse_args()

    import requests
    session = requests.Session()

    seen, drops = {}, []
    for doi, label in SEEDS.items():
        try:
            cits = fetch(doi, session, args.mailto)
        except Exception as e:  # noqa: BLE001
            print(f"! {label}: {e}", file=sys.stderr)
            continue
        print(f"  {label:<44} {len(cits):>5} citing works", flush=True)
        for w in cits:
            seen.setdefault(w["id"], w)

    works = list(seen.values())
    reasons = collections.Counter()
    with_abs, without_abs = [], []
    for w in works:
        _, why = passes_inclusion(w)
        reasons[why] += 1
        if why == "no_text":
            drops.append(w)
        (with_abs if w.get("abstract_inverted_index") else without_abs).append((w, why))

    # who the abstract-less works belong to, and how many of them the filter would still want
    by_pub = collections.Counter(publisher(w) for w in drops)
    by_venue = collections.Counter(venue(w) for w in drops)
    recoverable = [w for w in drops if title_only_would_enter(w)]
    rec_by_pub = collections.Counter(publisher(w) for w in recoverable)

    # the base rate to compare against: how the whole citation graph splits by publisher, so a
    # publisher that is simply large is not mistaken for a publisher that is systematically missing
    all_by_pub = collections.Counter(publisher(w) for w in works)
    skew = []
    for pub, n_drop in by_pub.most_common(15):
        n_all = all_by_pub[pub]
        skew.append({"publisher": pub, "works": n_all, "dropped_no_text": n_drop,
                     "share_of_that_publisher_dropped": round(n_drop / max(n_all, 1), 4)})

    def term_drop_rate(rows):
        if not rows:
            return None
        dropped = sum(1 for _, why in rows if why in ("no_rule_term", "no_reach_term"))
        return round(dropped / len(rows), 4)

    # the real channel: with no abstract only the title is scanned, so the term filter bites harder
    abs_split = {
        "with_abstract": {"n": len(with_abs), "term_drop_rate": term_drop_rate(with_abs),
                          "included": sum(1 for _, why in with_abs if why == "included")},
        "without_abstract": {"n": len(without_abs), "term_drop_rate": term_drop_rate(without_abs),
                             "included": sum(1 for _, why in without_abs if why == "included")},
    }
    per_pub = {}
    pubs = collections.Counter(publisher(w) for w in works)
    for pub, n_all in pubs.most_common(15):
        rows_w = [(w, why) for w, why in with_abs if publisher(w) == pub]
        rows_o = [(w, why) for w, why in without_abs if publisher(w) == pub]
        per_pub[pub] = {
            "works": n_all,
            "share_without_abstract": round(len(rows_o) / max(n_all, 1), 4),
            "term_drop_rate_with_abstract": term_drop_rate(rows_w),
            "term_drop_rate_without_abstract": term_drop_rate(rows_o),
            "included": sum(1 for _, why in rows_w + rows_o if why == "included"),
        }

    rep = {
        "note": "diagnostic only; no frame written and no snapshot frozen",
        "abstract_coverage": abs_split,
        "per_publisher": per_pub,
        "n_unique_citing_works": len(works),
        "exclusion_reasons": dict(reasons),
        "no_text": {
            "n": len(drops),
            "share_of_all_citing_works": round(len(drops) / max(len(works), 1), 4),
            "would_enter_on_title_alone": len(recoverable),
            "by_publisher": dict(by_pub.most_common(20)),
            "by_venue": dict(by_venue.most_common(20)),
            "recoverable_by_publisher": dict(rec_by_pub.most_common(20)),
        },
        "per_publisher_drop_rate": skew,
        "included": reasons["included"],
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n  {len(works)} unique citing works")
    for why, n in reasons.most_common():
        print(f"    {why:<16} {n:>5}")
    print(f"\n  no_text is {rep['no_text']['share_of_all_citing_works']:.1%} of the graph "
          f"({len(drops)} works), so that branch is not the bias channel")
    aw, ao = abs_split["with_abstract"], abs_split["without_abstract"]
    print(f"\n  with an abstract:    {aw['n']:>5} works, dropped on terms {aw['term_drop_rate']:.1%}"
          f", included {aw['included']}")
    print(f"  without an abstract: {ao['n']:>5} works, dropped on terms "
          f"{ao['term_drop_rate'] if ao['term_drop_rate'] is None else format(ao['term_drop_rate'], '.1%')}"
          f", included {ao['included']}")
    print("\n  publishers by how much of their output has no abstract:")
    for pub, r in sorted(per_pub.items(), key=lambda kv: -kv[1]["share_without_abstract"])[:10]:
        print(f"    {pub[:40]:<40} {r['works']:>5} works, "
              f"{r['share_without_abstract']:.0%} without abstract, {r['included']:>3} included")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
