#!/usr/bin/env python3
"""How much of `sparse at rule level' is a notation artefact rather than sparsity?

The H1 stratum reads each test transformation's mined SMIRKS two ways: as a rule label, looked
up in the mined catalog, and as a type label. The rule-level lookup is an EXACT STRING MATCH --
on objects the second paper proves different hands write differently. 972 of 2,536 pairs
(38.3%) find no catalog entry and are recorded with support zero, which is indistinguishable
from a rule that exists and is rare.

That is the comparison this project criticises, applied to the stratum H1 stands on. This
measures the artefact rather than arguing about it: the same join is redone under keys that do
not depend on how a template was written, and the stratum is recomputed under each.

  string     exact SMIRKS equality, as built
  radius 2   the step-0 signature with a two-bond environment and hydrogen dropped from the
             identity of an edit. Fine enough to separate rules -- 5,678 signatures over 7,580
             bank rules -- and blind to the constructs Table 7 names
  radius 1   the same, one bond out: coarser, so it bounds the artefact from above

A signature is an approximation to rule identity, not a proof of it: two genuinely different
templates can share one. The three numbers are therefore reported together, and the spread
between them is the uncertainty in the stratum's denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


from rdkit import RDLogger, Chem  # noqa: E402,F401
from rdkit.Chem import rdChemReactions  # noqa: E402

from grail_metabolism.utils.preparation import load_default_rules  # noqa: E402
from type_curve import signature  # noqa: E402

RDLogger.DisableLog("rdApp.*")

STRATUM = ROOT / "results" / "h1_stratum.json"
CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"
DENSE = 5


def sig_of(smirks, radius):
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        rxn.Initialize()
    except Exception:
        return None
    return signature(rxn, radius, "no_explicit_H")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "h1_join_sensitivity.json"))
    args = ap.parse_args()

    counts = {k: int(v.get("count", 0)) for k, v in json.loads(CATALOG.read_text()).items()}
    rows = json.loads(STRATUM.read_text())["rows"]
    typed = [r for r in rows if r["smirks"]]
    print(f"{len(typed)} typeable pairs, {len(counts)} catalog rules", file=sys.stderr, flush=True)

    # catalog support pooled under each key, over the rules the catalog knows
    pooled = {}
    for radius in (1, 2):
        acc = Counter()
        for smirks, n in counts.items():
            s = sig_of(smirks, radius)
            if s:
                acc[s] += n
        pooled[radius] = acc
        print(f"  radius {radius}: {len(acc)} distinct keys over the catalog",
              file=sys.stderr, flush=True)

    # type-level support is unchanged: it is the radius-0 pooling the stratum already used
    type_pooled = Counter()
    for smirks in load_default_rules():
        if smirks in counts:
            s = sig_of(smirks, 0)
            if s:
                type_pooled[s] += counts[smirks]

    out_rows, stats = [], defaultdict(Counter)
    for r in typed:
        sm = r["smirks"]
        rec = {"substrate": r["substrate"], "metabolite": r["metabolite"],
               "type_support": r["type_support"],
               "rule_support": {"string": r["rule_support"]}}
        for radius in (1, 2):
            s = sig_of(sm, radius)
            rec["rule_support"][f"radius{radius}"] = pooled[radius].get(s, 0) if s else 0
        for key, sup in rec["rule_support"].items():
            inside = sup < DENSE <= rec["type_support"]
            stats[key]["sparse_at_rule"] += int(sup < DENSE)
            stats[key]["in_stratum"] += int(inside)
            stats[key]["found"] += int(sup > 0)
            rec.setdefault("in_stratum", {})[key] = inside
        out_rows.append(rec)

    n = len(typed)
    report = {"provenance": stamp(__file__), "n_typeable_pairs": n, "dense_at": DENSE, "by_key": {}}
    for key in ("string", "radius2", "radius1"):
        c = stats[key]
        subs = {r["substrate"] for r in out_rows if r["in_stratum"][key]}
        report["by_key"][key] = {
            "pairs_with_a_match": c["found"], "share_matched": round(c["found"] / n, 4),
            "sparse_at_rule": c["sparse_at_rule"],
            "in_stratum_pairs": c["in_stratum"],
            "in_stratum_substrates": len(subs),
            "share_pairs": round(c["in_stratum"] / n, 4),
        }
    base = report["by_key"]["string"]
    report["artefact"] = {
        "pairs_unmatched_by_string": n - base["pairs_with_a_match"],
        "of_those_matched_at_radius2":
            report["by_key"]["radius2"]["pairs_with_a_match"] - base["pairs_with_a_match"],
        "stratum_pairs_string_minus_radius2":
            base["in_stratum_pairs"] - report["by_key"]["radius2"]["in_stratum_pairs"],
        "stratum_substrates_string_minus_radius2":
            base["in_stratum_substrates"] - report["by_key"]["radius2"]["in_stratum_substrates"],
    }
    # the primary definition, named in advance: membership confirmed by every key. The
    # conjunction is the test; the three ratios beside it are the robustness report.
    keys = ("string", "radius2", "radius1")
    inter = [r for r in out_rows if all(r["in_stratum"][k] for k in keys)]
    inter_subs = sorted({r["substrate"] for r in inter})
    all_subs = sorted({r["substrate"] for r in out_rows})
    strata = ROOT / "strata"
    strata.mkdir(exist_ok=True)
    (strata / "sparse_at_rule_dense_at_type_intersection.txt").write_text(
        "\n".join(inter_subs) + "\n")
    (strata / "sparse_at_rule_dense_at_type_intersection_complement.txt").write_text(
        "\n".join(s for s in all_subs if s not in set(inter_subs)) + "\n")
    report["primary"] = {
        "definition": "in the stratum under every join key",
        "keys": list(keys),
        "pairs": len(inter),
        "substrates": len(inter_subs),
        "share_pairs": round(len(inter) / n, 4),
        "file": "strata/sparse_at_rule_dense_at_type_intersection.txt",
    }
    # the arithmetic ceiling on the enrichment factor: g_S <= S forces K <= N / G
    report["feasibility"] = {
        "N_typeable_references": n,
        "note": "the requirement share_of_gain >= K * p is satisfiable only while the gain G "
                "obeys K <= N / G, because the gain inside the stratum cannot exceed the "
                "stratum itself. A test that fails because the intervention worked too well is "
                "broken by construction.",
        "max_gain_for_K": {str(K): int(n // K) for K in (2.5, 3.4, 8.9)},
    }
    report["rows"] = out_rows
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'key':<10}{'matched':>9}{'share':>8}{'sparse':>8}{'stratum pairs':>15}{'substrates':>12}")
    for key in ("string", "radius2", "radius1"):
        b = report["by_key"][key]
        print(f"{key:<10}{b['pairs_with_a_match']:>9}{b['share_matched']:>8.3f}"
              f"{b['sparse_at_rule']:>8}{b['in_stratum_pairs']:>15}"
              f"{b['in_stratum_substrates']:>12}")
    a = report["artefact"]
    print(f"\nunmatched by string: {a['pairs_unmatched_by_string']}; of those, "
          f"{a['of_those_matched_at_radius2']} do match a catalog rule at radius 2")
    print(f"the stratum shrinks by {a['stratum_pairs_string_minus_radius2']} pairs and "
          f"{a['stratum_substrates_string_minus_radius2']} substrates when the join stops "
          f"depending on notation")
    pr = report["primary"]
    print(f"\nprimary definition (every key): {pr['pairs']} pairs over {pr['substrates']} "
          f"substrates, {pr['share_pairs']:.1%} of the typeable references")
    print("arithmetic ceiling on the enrichment factor, K <= N/G:")
    for K, g in report["feasibility"]["max_gain_for_K"].items():
        print(f"  K={K:<5} feasible while the gain stays under {g} references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
