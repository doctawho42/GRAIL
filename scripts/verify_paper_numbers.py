#!/usr/bin/env python3
"""Every quantity the manuscript derives from the decomposition, recomputed and compared.

SELF_CLAIMS row 11 says every numeric passage traces to its artifact, and nothing enforced it. The
cost of that showed up when the ceiling was corrected: the macro was updated, and seventeen values
derived from it were not -- a conversion ratio, a count of references lost to truncation, a paired
difference that became arithmetically impossible, an external ceiling measured in the other
convention, and a figure caption. Every one was found by a reader rather than by a check.

This is that check. It does not scan for numbers that happen to appear in both a paper and an
artifact -- with tens of thousands of values in the record, any three-decimal figure matches
something, and a test that cannot fail is worse than none. It recomputes each derived quantity from
the canonical artifact and compares it to what the manuscript prints, by name.

Exit status is non-zero when anything disagrees, so this belongs in front of a submission the way a
test suite belongs in front of a commit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "grail_iclr.tex"
TEX = [PAPER] + sorted((ROOT / "paper" / "app").glob("*.tex"))


def macro(name: str, text: str):
    """The macro's value with LaTeX digit grouping removed, e.g. 1{,}170 -> 1170."""
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.+?)\}\s*(?:%|\n)", text)
    if m is None:
        return None
    return m.group(1).replace("{,}", "").replace(",", "").strip()


def close(a, b, tol=5e-4):
    """Does the manuscript's printed value follow from the artifact's?

    A printed number is correct when it is the artifact rounded to the precision it is printed at,
    so that is what is compared. A fixed tolerance gets the boundary cases wrong in both directions:
    0.1875 printed as 0.188 differs by exactly 5e-4 and fails a 5e-4 tolerance, while a value printed
    at two decimals passes a tolerance far tighter than its own precision. Both showed up here.
    """
    if a is None or b is None:
        return False
    try:
        av, bv = float(a), float(b)
    except (TypeError, ValueError):
        return False
    # Only a value READ FROM THE MANUSCRIPT arrives as a string, and only for those is the printed
    # precision the right yardstick. Artifact-against-artifact comparisons pass floats and keep the
    # tolerance they were given -- applying the rule to them made a deliberate 2e-3 convention check
    # into a four-decimal one, which this check caught on itself.
    text = a.strip() if isinstance(a, str) else ""
    if "." in text and text.replace(".", "").replace("-", "").replace("+", "").isdigit():
        places = len(text.split(".")[1])
        return f"{round(bv, places):.{places}f}" == f"{av:.{places}f}"
    return abs(av - bv) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    art = json.loads((ROOT / "results/recall_factorization.json").read_text())
    f = art["factors"]
    rows = art["per_substrate"]
    S = {k: sum(r[k] for r in rows) for k in ("U", "Cfull", "Cbud", "H")}
    text = PAPER.read_text()
    whole = "".join(p.read_text() for p in TEX)

    checks = []

    def check(name, printed, computed, note=""):
        ok = close(printed, computed, args.tol)
        checks.append((ok, name, printed, computed, note))

    # 1. the macros against the artifact they summarise
    check("\\ceiling", macro("ceiling", text), f["coverage_bank"]["point"])
    check("\\ceilingmacro", macro("ceilingmacro", text), art["macro_coverage_bank"])
    check("\\realised", macro("realised", text), art["micro_recall"])
    check("\\selret", macro("selret", text), f["selection_retention"]["point"])
    check("\\grailmacro", macro("grailmacro", text), art["macro_recall"])
    check("\\ntest", macro("ntest", text), art["n_substrates"])

    # 2. the identity, which is the whole point of the instrument
    prod = (f["coverage_bank"]["point"] * f["selection_retention"]["point"]
            * f["ranking_conversion"]["point"])
    check("identity closes", prod, art["micro_recall"], "coverage x selection x ranking")

    # 3. the factors against the per-substrate record they are pooled from
    check("coverage from rows", f["coverage_bank"]["point"], S["Cfull"] / S["U"])
    check("selection from rows", f["selection_retention"]["point"], S["Cbud"] / S["Cfull"])
    check("ranking from rows", f["ranking_conversion"]["point"], S["H"] / S["Cbud"])

    # 4. quantities the prose derives and prints in words
    conv = re.search(r"converts only \$([\d.]+)\\%\$ of its own ceiling", whole)
    check("conversion percentage", conv and float(conv.group(1)) / 100, S["H"] / S["Cfull"],
          "H / Cfull")
    lost = re.search(r"and all \$(\d+)\$\s*\n?references lost between the budgeted pool", whole)
    check("references lost to truncation", lost and lost.group(1), S["Cbud"] - S["H"],
          "Cbud - H")
    cap = sum(1 for r in rows if len(r.get("deployed_top15") or []) >= 15)
    atcap = re.search(r"binds on \$(\d+)\$ of", re.sub(r"\s+", " ", whole))
    check("substrates at the cap", atcap and atcap.group(1), cap)

    # 5. the nesting the identity needs, which a clamp used to supply silently
    nests = sum(1 for r in rows if r.get("nests", True))
    check("nesting holds on every row", nests, len(rows), "no clamp applied")

    # 6. every OTHER artifact that records a full-bank coverage must be in the same convention as
    # the canonical one, or be named here as a deliberate reference to the other. A checker whose
    # scope is narrower than its claim reports a pass it has not earned, which is how five stale
    # values survived the last sweep while this script printed "all agree".
    EXPANDED_ON_PURPOSE = {
        "recall_factorization_expanded_convention.json",   # kept as the measure of the cost
        "recall_factorization_rerun.json",                 # the same run under its old name
    }
    canonical = f["coverage_bank"]["point"]
    for path in sorted((ROOT / "results").glob("*.json")):
        if path.name in EXPANDED_ON_PURPOSE or path.name == "recall_factorization.json":
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for key in ("coverage", "internal_ceiling", "ceiling"):
            v = d.get(key)
            v = v.get("point") if isinstance(v, dict) else v
            if isinstance(v, (int, float)) and 0.5 < v < 1.0:
                checks.append((close(v, canonical, 2e-3), f"{path.name}:{key}", round(v, 4),
                               round(canonical, 4), "full-bank coverage, must match the convention"))

    # 7. the external ceiling and its split, which live in their own artifacts
    ext = ROOT / "results/ceiling_external_validity.json"
    if ext.exists():
        e = json.loads(ext.read_text())
        for label, printed_pat, value in (
                ("external ceiling", r"the external ceiling is ([\d.]+)",
                 e["external_ceiling_uncapped"]["point"]),):
            m = re.search(printed_pat, whole)
            check(label, m and m.group(1), value)
    spl = ROOT / "results/external_ceiling_split.json"
    if spl.exists():
        sp = json.loads(spl.read_text())
        for label, key, pat in (
                ("external, seen in training", "seen_in_training",
                 r"gives \$([\d.]+)\$ \$\[[\d.,]+\]\$ on the twenty-four seen"),
                ("external, unseen", "unseen", r"against \$([\d.]+)\$ \$\[[\d.,]+\]\$ on the thirteen unseen")):
            m = re.search(pat, whole)
            check(label, m and m.group(1), sp[key]["coverage"])

    # 8. the provenance split, whose own gate only proves it reproduces the ceiling -- nothing until
    # now proved the manuscript prints what it produced. This is the pairing that failed before: a
    # measurement guarded by a frozen literal, and a paper quoting the superseded value it certified.
    prov = ROOT / "results/ceiling_by_provenance__clean_test.json"
    if prov.exists():
        pv = json.loads(prov.read_text())
        for subset in ("curated", "mined", "full"):
            v = pv["subsets"][subset]["coverage"]
            row = re.search(r"^\s*" + ("union" if subset == "full" else subset)
                            + r"\s*&[^&]*&[^&]*&\s*\$([\d.]+)\$", whole, re.M)
            check(f"provenance, {subset}", row and row.group(1), v)
            share = re.search(r"^\s*" + ("union" if subset == "full" else subset)
                              + r"\s*&[^&]*&[^&]*&[^&]*&\s*\$([\d.]+)\\%\$", whole, re.M)
            check(f"provenance, {subset} share of ceiling", share and float(share.group(1)) / 100,
                  v / pv["subsets"]["full"]["coverage"], "subset coverage / union coverage")
        check("provenance gate reproduces the ceiling", pv["ceiling_gate"]["reproduced"],
              pv["ceiling_gate"]["committed"], "read from the factorization, not frozen")
        # Anchored on the sentence's own wording, so a rewrite that drops a figure fails loudly
        # rather than silently matching nothing -- a regex that finds no number reports None, and
        # None never compares equal.
        flat = re.sub(r"\s+", " ", whole)
        for label, key, pat in (
                ("exclusively mined", "mined_only",
                 r"mined templates reach \$([\d.]+)\$ of the references that the curated sets do not"),
                ("exclusively curated", "curated_only",
                 r"the curated sets reach \$([\d.]+)\$ that the mined templates do not"),
                ("reachable by both", "shared", r"and \$([\d.]+)\$ is reachable by both")):
            m = re.search(pat, flat)
            check(label, m and m.group(1), pv["exclusive"][key])

    # 8a. The three populations, and how far apart they are. The appendix printed 0.10 two lines
    # under the word "disjoint", so the number was right and the description was not; both are now
    # tied to the same measurement.
    ing = ROOT / "results/evalretro_ingest.json"
    if ing.exists():
        G = json.loads(ing.read_text())
        flat = re.sub(r"\s+", " ", whole)
        pair = [v for k, v in G["population_overlap"].items() if k != "all"]
        worst = max(v["share_of_smaller"] for v in pair)
        m = re.search(r"the between-cluster\s*overlap is \$([\d.]+)\$", flat)
        check("between-cluster overlap", m and m.group(1), worst,
              "the largest of the pairwise shares, so the printed figure bounds them all")
        check("test sets the eleven sit on", str(G["n_test_sets"]), 3.0, "clusters the ingest found")
        checks.append((all(0.0 < v["share_of_smaller"] < 0.9 for v in pair),
                       "the three populations are neither disjoint nor the same set",
                       f"pairwise shares {[v['share_of_smaller'] for v in pair]}",
                       "strictly between 0 and the clustering threshold",
                       "which is why the manuscript may not call them disjoint"))

    # 8b. The knob attribution: the table that reversed, the endpoints it reversed between, and
    # what carries the reversal. None of this was checked, and the section it backs was the one
    # that turned out to name a population it was not measured on -- so it is anchored on the
    # sentences' own wording, and on the artifact for the split the appendix now reports.
    kn = ROOT / "results/provenance_knob_attribution__clean_test.json"
    if kn.exists():
        K = json.loads(kn.read_text())
        cov, ends, att = K["coverage"], K["committed_endpoints"], K["gap_attribution"]
        flat = re.sub(r"\s+", " ", whole)
        for label, pat, value in (
                ("primitive table, carrying it as deployed",
                 r"carrying the hydrogen atom primitive & \$675\$ *& \$([\d.]+)\$",
                 cov["curated_needs_h|addhs=0"]),
                ("primitive table, carrying it expanded",
                 r"carrying the hydrogen atom primitive & \$675\$ *& \$[\d.]+\$ & \$([\d.]+)\$",
                 cov["curated_needs_h|addhs=1"]),
                ("primitive table, not carrying it as deployed",
                 r"not carrying it & \$1\{,\}040\$ & \$([\d.]+)\$", cov["curated_plain|addhs=0"]),
                ("primitive table, not carrying it expanded",
                 r"not carrying it & \$1\{,\}040\$ & \$[\d.]+\$ & \$([\d.]+)\$",
                 cov["curated_plain|addhs=1"]),
                ("primitive table, mined as deployed",
                 r"mined, for comparison & \$5\{,\}866\$ & \$([\d.]+)\$", ends["deployed"]["mined"]),
                ("primitive table, mined expanded",
                 r"mined, for comparison & \$5\{,\}866\$ & \$[\d.]+\$ & \$([\d.]+)\$",
                 ends["helper"]["mined"]),
                ("the reversal, curated through the helper",
                 r"give curated \$([\d.]+)\$ against mined", ends["helper"]["curated"]),
                ("the reversal, mined through the helper",
                 r"give curated \$[\d.]+\$ against mined \$([\d.]+)\$", ends["helper"]["mined"]),
                ("attribution, gap as deployed",
                 r"between the subsets is \$\+([\d.]+)\$ as deployed", att["gap_deployed"]),
                # "expanded" in the manuscript is the helper convention, expansion and validity
                # floor together -- the same arm the sentence above it quotes as 0.660 against
                # 0.328 -- not the floor-free intermediate the attribution decomposes through.
                ("attribution, gap expanded",
                 r"as deployed and \$-([\d.]+)\$ expanded", -att["gap_helper"]),
                ("attribution, the validity floor",
                 r"validity floor carries \$\+([\d.]+)\$", att["moved_by_the_validity_floor"]),
                ("attribution, expanding hydrogens",
                 r"expanding hydrogens carries the remaining \$-([\d.]+)\$",
                 -att["moved_by_expanding_hydrogens"])):
            m = re.search(pat, flat)
            check(label, m and m.group(1), value)
        # The attribution is an accounting identity, so it has to close on its own numbers before
        # any of them is quoted: the two terms must sum to the distance between the endpoints.
        checks.append((abs((att["moved_by_expanding_hydrogens"] + att["moved_by_the_validity_floor"])
                           - (att["gap_helper"] - att["gap_deployed"])) < 1e-9,
                       "attribution closes: the two terms carry the whole distance",
                       round(att["moved_by_expanding_hydrogens"] + att["moved_by_the_validity_floor"], 4),
                       round(att["gap_helper"] - att["gap_deployed"], 4),
                       "hydrogens plus floor, against helper minus deployed"))

    # 9. the coverage gap and everything the appendix derives from it. These moved when the
    # ceiling's convention was corrected, and a perturbation test found that nothing here was
    # checked -- the in-bank ceiling could be edited to any value and this script stayed green.
    gap = ROOT / "results/coverage_gap_types.json"
    if gap.exists():
        g = json.loads(gap.read_text())
        cov_pairs, unc = g["covered_pairs"], g["uncovered_pairs"]
        total = cov_pairs + unc
        known, novel, untyped = (g["gap"][k] for k in ("known_type", "novel_type", "untypeable"))
        flat = re.sub(r"\s+", " ", whole)
        for label, pat, value in (
                ("uncovered transformations", r"splits the \$([\d,{}]+)\$ uncovered test", unc),
                ("gap, known type", r"into \$(\d+)\$ \(\$\d+\\%\$\) whose reaction type the bank", known),
                ("gap, novel type", r"and \$(\d+)\$ \(\$\d+\\%\$\) whose type is absent", novel),
                ("gap, untypeable", r"the remaining \$(\d+)\$ admit no radius-0 type", untyped),
                ("gap, novel share", r"and \$\d+\$ \(\$(\d+)\\%\$\) whose type is absent",
                 round(100 * novel / max(unc, 1))),
                ("gap, known share", r"into \$\d+\$ \(\$(\d+)\\%\$\) whose reaction type",
                 round(100 * known / max(unc, 1))),
                ("in-bank ceiling", r"= ([\d.]+)\$; closing both groups", (cov_pairs + known) / total),
                ("both groups closed", r"closing both groups would bound it at \$([\d.]+)\$",
                 (cov_pairs + known + novel) / total),
                ("template-generalisation headroom", r"tightly bounded gain of \$([\d.]+)\$",
                 (cov_pairs + known) / total - g["coverage"]),
                ("uncovered share of references", r"default, \$([\d.]+)\\%\$ of reference metabolites",
                 100 * unc / total),
                ("novel share, main body", r"\$(\d+)\\%\$ of uncovered transformations being",
                 round(100 * novel / max(unc, 1)))):
            m = re.search(pat, flat)
            # a percentage carries its tolerance in its own units, not the ratio's
            tol = 0.05 if value > 1.5 else args.tol
            checks.append((close(m and m.group(1).replace("{,}", "").replace(",", ""), value, tol),
                           label, m and m.group(1), value,
                           "from results/coverage_gap_types.json"))
        m = re.search(r"one step: \$([\d,{}]+)\$ of \$([\d,{}]+)\$, the complement", flat)
        check("gap counts printed in the text", m and m.group(1).replace("{,}", ""), unc)
        check("reference total printed in the text", m and m.group(2).replace("{,}", ""), total)

    # 10. the cross-domain split structure. These are counts a reader could not recompute without
    # the released files, which is exactly when a printed number needs a check behind it.
    ing = ROOT / "results/evalretro_ingest.json"
    if ing.exists():
        cl = json.loads(ing.read_text())["clusters"]
        flat = re.sub(r"\s+", " ", whole)
        by_size = sorted(cl.values(), key=lambda c: -len(c["systems"]))
        names = {7: "seven-system", 3: "three-system", 1: "one-system"}
        for c in by_size:
            row = names.get(len(c["systems"]))
            if row is None:
                continue
            m = re.search(row + r"\s*&\s*(\d+)\s*&\s*\$([\d,{}]+)\$\s*&\s*\$([\d.]+)\$", flat)
            check(f"cluster {row}, systems", m and m.group(1), len(c["systems"]),
                  "from results/evalretro_ingest.json")
            check(f"cluster {row}, reactions", m and m.group(2).replace("{,}", ""),
                  c["reactions"], "reactions its members agree on")
            checks.append((close(m and m.group(3), c["share_of_products_in_repo_test_split"], 5e-3),
                           f"cluster {row}, share in our split", m and m.group(3),
                           c["share_of_products_in_repo_test_split"], ""))
        m = re.search(r"ranked outputs of (\w+) single-step retrosynthesis systems", flat)
        WORDS = {"twelve": 12, "eleven": 11}
        check("systems in the released benchmark", m and WORDS.get(m.group(1)), 12,
              "twelve published, eleven as CSV")

    # 10b. the dispatch table and the pre-registered subset arms, on whichever population each ran
    hd = ROOT / "results/hydrogen_dispatch__clean_test.json"
    cur = ROOT / "results/dispatch_paired_ci__clean_test__curated.json"
    if hd.exists():
        H = json.loads(hd.read_text())["banks"]
        flat = re.sub(r"\s+", " ", whole)
        for shown, key in (("SyGMa", "sygma_175"), ("ours", "grail_full"),
                           ("BioTransformer", "biotransformer")):
            m = re.search(shown + r" & \$?[\\a-z{},0-9]+\$? & \$(\d+)\$ & ([\d.]+) \$\[[\d.,]+\]\$ & "
                          r"([\d.]+) & \$\+([\d.]+)\$", flat)
            v = H[key]
            check(f"dispatch table, {shown} dispatched", m and m.group(1), v["dispatched_to_expanded"])
            check(f"dispatch table, {shown} reach", m and m.group(2), v["reach"], "", )
            check(f"dispatch table, {shown} best global", m and m.group(3), v["best_global"])
            check(f"dispatch table, {shown} residual", m and m.group(4),
                  abs(v["residual_convention_dependence"]), "measured in the same run")
    mnd = ROOT / "results/dispatch_paired_ci__mined.json"
    if mnd.exists():
        M = json.loads(mnd.read_text())
        checks.append((M["paired_residual"]["delta"] == 0.0 and M["reach"]["dispatch"]
                       == M["reach"]["all_implicit"],
                       "P1: mined dispatch is the identity", M["paired_residual"]["delta"],
                       0.0, "the registered structural null"))
    if hd.exists() and cur.exists():
        full = json.loads(hd.read_text())["banks"]["grail_full"]["residual_convention_dependence"]
        curr = json.loads(cur.read_text())["paired_residual"]["delta"]
        checks.append((curr > full, "P3 refuted, as the appendix states", f"curated {curr}",
                       f"full {full}", "residual is subadditive across the partition"))

    # 10c. the engine knobs: the paper's largest single finding and the two knobs that move nothing
    ek = ROOT / "results/engine_knobs__clean_test.json"
    if ek.exists():
        K = json.loads(ek.read_text())
        flat = re.sub(r"\s+", " ", whole)
        h = K["one_knob_at_a_time"]["explicit_hydrogens"]
        m = re.search(r"explicit hydrogens & yes \$\\to\$ no & ([\d.]+) & \$\+([\d.]+)\$", flat)
        check("knob table, expanded-off reach", m and m.group(1), h["reach"])
        check("knob table, hydrogen knob", m and m.group(2), h["paired_vs_default"]["delta"])
        for label, pat in (("normalisation moves nothing",
                            r"product normalisation & tautomer \$\\to\$ canonical & ([\d.]+) & \$0.000\$"),
                           ("validity floor moves nothing",
                            r"validity floor & on \$\\to\$ off & ([\d.]+) & \$0.000\$")):
            m2 = re.search(pat, flat)
            check(label, m2 and m2.group(1), K["default_reach"], "must equal the default exactly")
        c = K["against_the_comparator_engine"]
        m3 = re.search(r"The engine term is \$\+([\d.]+)\$, so that one call carries all of it", flat)
        check("engine term, appendix", m3 and m3.group(1), c["committed_engine_term_micro"])

    # 10d. the packing measurement: the empirical half of the reordering condition
    pk = ROOT / "results/packing_vs_differential.json"
    if pk.exists():
        P = json.loads(pk.read_text())
        flat = re.sub(r"\s+", " ", whole)
        t = P["totals"]
        m = re.search(r"\$(\d+)\$ method-pair by criterion-pair comparisons across three domains", flat)
        check("packing, comparisons", m and m.group(1), t["comparisons"])
        m = re.search(r"exceeds the gap in \$(\d+)\$ of the \$(\d+)\$", flat)
        check("packing, move exceeds gap", m and m.group(1), t["closer_than_the_move"])
        check("packing, denominator", m and m.group(2), t["comparisons"])
        m = re.search(r"a reversal follows in \$(\d+)\$ of those", flat)
        check("packing, reversals", m and m.group(1), t["exchanged"])
        NAMES = {"generation, MOSES": "molecular generation, MOSES",
                 "metabolites, GLORYx": "metabolites, external GLORYx",
                 "retrosynthesis, seven": "retrosynthesis, seven-system group",
                 "retrosynthesis, three": "retrosynthesis, three-system group"}
        for shown, key in NAMES.items():
            v = P["per_leaderboard"].get(key)
            if not v:
                continue
            row = re.search(re.escape(shown) + r" & (\d+) & (\d+) & \$([\d.]+)\$ & \$([\d.]+)\$ & \$(\d+)\$", flat)
            for i, (label, val) in enumerate((("methods", v["methods"]),
                                              ("comparisons", v["comparisons"]),
                                              ("median gap", v["median_gap"]),
                                              ("median move", v["median_differential"]),
                                              ("move over gap", v["closer_than_the_move"]))):
                check(f"packing, {shown} {label}", row and row.group(i + 1), val)

    # 10e. The fourth choice, and the only one that did not survive being tested. Both instances
    # are checked, and so is the count that matters most -- the number of interactions surviving
    # correction, which is zero. A checker that only tied the reorderings would let the paper keep
    # claiming an effect the correction removed.
    pa = ROOT / "results/population_axis.json"
    rp = ROOT / "results/retro_population_axis.json"
    if pa.exists() and rp.exists():
        P, R = json.loads(pa.read_text()), json.loads(rp.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"of \$(\d+)\$ comparisons \$(\d+)\$ reorder", flat)
        check("our split, comparisons", m and m.group(1), P["comparisons"])
        check("our split, reordered", m and m.group(2), P["reordered"])
        m = re.search(r"giving \$(\d+)\$\s*comparisons of which \$(\d+)\$ reorder", flat)
        check("their files, comparisons", m and m.group(1), R["comparisons"])
        check("their files, reordered", m and m.group(2), R["reordered"])
        m = re.search(r"share \$(\d+)\$ products, \$(\d+)\$ of which agree", flat)
        check("products in both", m and m.group(1), R["products_in_both"])
        check("agreeing reactions", m and m.group(2), R["agreeing_reactions"])
        m = re.search(r"\$(\d+)\$ of \$380\$ have intervals excluding zero", flat)
        check("their files, marginal intervals", m and m.group(1),
              R["interactions_excluding_zero"])
        m = re.search(r"Not one of the \$(\d+)\$ interactions survives Holm", flat)
        check("the family both instances form", m and m.group(1),
              P["comparisons"] + R["comparisons"])
        # The abstract states the negative once and carries only the family size; the two counts
        # it used to repeat are checked where they are reported, in the appendix above.
        # The claim the paper now rests on: nothing survives. This is a gate, not a comparison --
        # a single survivor in either instance makes the sentence false, whatever it prints.
        checks.append((P["holm_survivors"] == 0 and R["holm_survivors"] == 0,
                       "no population interaction survives Holm in either instance",
                       f"ours {P['holm_survivors']}, theirs {R['holm_survivors']}", "0 and 0",
                       "the manuscript says none does, in the abstract and the appendix"))
        # A null is a result only with a stated detectable size; the manuscript now prints one,
        # so it is tied to the design that produced it.
        flat = re.sub(r"\s+", " ", whole)
        # the derivation's own numbers, so the appendix cannot state a threshold the run did not use
        flat2 = re.sub(r"\s+", " ", whole)
        # the derivation's own quantities are checked in block 10c-h, against the run

    # 10f. The third domain. Its intervals were computed by resampling with replacement, which
    # collides on its own, so every one of the forty sat far below the estimate it belonged to; the
    # gate below is against that returning, and the counts are against the effect being restated
    # without its correction.
    mo = ROOT / "results/moses_uniqueness.json"
    if mo.exists():
        M = json.loads(mo.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"Of \$(\d+)\$ paired\s*interactions", flat)
        check("MOSES, interactions", m and m.group(1), M["n_interactions"])
        m = re.search(r"Of \$112\$ paired\s*interactions[^.]*?\$(\d+)\$ have intervals excluding\s*"
                      r"zero and \$(\d+)\$ survive Holm", flat)
        check("MOSES, excluding zero", m and m.group(1), M["interactions_excluding_zero"])
        check("MOSES, Holm survivors", m and m.group(2), M["holm_survivors"])
        ex = M["interactions"]["inchi_no_stereo"]["combinatorial vs latent_gan"]
        m = re.search(r"an interaction of \$-([\d.]+)\$", flat)
        check("MOSES, the exchange", m and m.group(1), abs(ex["delta"]))
        u = M["uniqueness"]
        for label, model, pat in (
                ("MOSES, hidden Markov under stereo", "hmm",
                 r"falls \$([\d.]+) \\to ([\d.]+)\$ and the \$n\$-gram"),
                ("MOSES, n-gram under stereo", "ngram",
                 r"\$n\$-gram baseline \$([\d.]+) \\to ([\d.]+)\$"),
                ("MOSES, LatentGAN under stereo", "latent_gan",
                 r"LatentGAN falls \$([\d.]+) \\to ([\d.]+)\$")):
            mm = re.search(pat, flat)
            check(label + ", canonical", mm and mm.group(1), u["canonical"][model]["unique@10000"])
            check(label + ", stereo-blind", mm and mm.group(2),
                  u["inchi_no_stereo"][model]["unique@10000"])
        # Every survivor is the stereo step: the sentence says so in bold, and a survivor anywhere
        # else would make it false however the counts print.
        elsewhere = [f"{mode}|{pair}" for mode, pairs in M["interactions"].items()
                     if mode != "inchi_no_stereo"
                     for pair, v in pairs.items() if v["survives_holm"]]
        checks.append((not elsewhere, "MOSES, every survivor is the stereo step",
                       f"{len(elsewhere)} elsewhere", "0 elsewhere",
                       "the appendix claims all nineteen are that one step"))
        # The defect that was there: an interval that does not contain the estimate it belongs to.
        offenders = [(mode, mdl) for mode in u for mdl, v in u[mode].items()
                     if not (v["ci95_at_10000"][0] <= v["unique@10000"] <= v["ci95_at_10000"][1])]
        checks.append((not offenders, "MOSES, every interval contains its own estimate",
                       f"{len(offenders)} do not", "0", "resampling without replacement, at the "
                       "reported size, from three times as many generations"))

    # 11. the cross-domain leaderboard: the run the paper specified in advance and then ran. Its
    # counts are the load-bearing part -- an exchange is visible, a certified interaction is not --
    # so every one of them is recomputed here rather than trusted to a paragraph.
    lb = ROOT / "results/retro_leaderboard_cluster0.json"
    if lb.exists():
        L = json.loads(lb.read_text())
        flat = re.sub(r"\s+", " ", whole)
        acc, sysl = L["accuracy"], L["config"]["systems"]
        for label, pat, value in (
                ("interaction tests", r"Of \$(\d+)\$ paired interaction tests", L["n_interaction_tests"]),
                ("intervals excluding zero", r"\$(\d+)\$ have intervals excluding zero",
                 len(L["certified_interactions"])),
                ("Holm survivors", r"\\textbf\{\$(\d+)\$ survive Holm", len(L["holm_survivors"])),
                ("Holm survivors, main body",
                 r"Of the \$448\$ that remain, \$\d+\$ have intervals excluding zero and "
                 r"\$(\d+)\$ survive Holm", len(L["holm_survivors"])),
                ("intervals excluding zero, main body",
                 r"Of the \$448\$ that remain, \$(\d+)\$ have intervals excluding zero",
                 len(L["certified_interactions"])),
                ("pairs exchanging at top-1", r"Five of the twenty-one pairs exchange",
                 None)):
            if value is None:
                checks.append((bool(re.search(pat, flat)), label, "present", "phrase present", ""))
                continue
            m = re.search(pat, flat)
            check(label, m and m.group(1), value, "from results/retro_leaderboard_cluster0.json")
        # the printed word must match the measured count of exchanging pairs
        WORDS = {"Five": 5, "Four": 4, "Six": 6, "Three": 3, "Seven": 7}
        m = re.search(r"(\w+) of the twenty-one pairs exchange", flat)
        check("exchanging pairs, counted", m and WORDS.get(m.group(1)),
              len(L["pairs_that_exchange"]["top1"]), "pairs that exchange at top-1")
        # every accuracy the table prints, against the artifact that produced it
        NAMES = {"Graph2SMILES": "graph2smiles", "GraphRetro": "graphretro",
                 "Retroformer": "retroformer", "LocalRetro": "localretro", "GLN": "gln",
                 "G2Retro": "g2retro", "RetroXpert": "retroxpert"}
        for shown, key in NAMES.items():
            m = re.search(shown + r"\s*&\s*(?:\\textbf\{)?\$?([\d.]+)\$?\}?\s*&", flat)
            check(f"top-1 canonical, {shown}", m and m.group(1), acc[key]["canonical"]["top1"])
        if set(NAMES.values()) != set(sysl):
            checks.append((False, "leaderboard systems named", sorted(NAMES.values()), sorted(sysl), ""))

    # the replication group, whose value is that it does NOT reorder: the condition the paper states
    # is a comparison between two measured quantities, and both have to be checked
    import itertools as _it
    lb1 = ROOT / "results/retro_leaderboard_cluster1.json"
    if lb.exists() and lb1.exists():
        flat = re.sub(r"\s+", " ", whole)
        rows = {}
        for tag, path in (("seven-system", lb), ("three-system", lb1)):
            L2 = json.loads(path.read_text())
            a2, S2 = L2["accuracy"], L2["config"]["systems"]
            # one row per PAIR, so a pair's gap is compared with that same pair's differential.
            # Sorting the two lists separately silently pairs the smallest gap with the smallest
            # movement, which is a different quantity and gave zero where the answer is seven.
            per_pair = [(abs(a2[x]["canonical"]["top1"] - a2[y]["canonical"]["top1"]),
                         abs((a2[x]["tautomer"]["top1"] - a2[x]["canonical"]["top1"])
                             - (a2[y]["tautomer"]["top1"] - a2[y]["canonical"]["top1"])))
                        for x, y in _it.combinations(S2, 2)]
            gaps = sorted(g for g, _ in per_pair)
            rows[tag] = {"median_gap": gaps[len(gaps) // 2],
                         "max_diff": max(d2 for _, d2 in per_pair),
                         "closer": sum(1 for g, d2 in per_pair if g < d2),
                         "pairs": len(per_pair),
                         "exchanged": len(L2["pairs_that_exchange"]["top1"])}
        # The main body states the condition in terms of the resolution floor; the median margin
        # it replaced is checked in the packing table above, where it is still printed.

    # 10c-b. The engine decomposition in the main text. The figure it previously compared against
    # appeared in no artifact and was covered by no check, which is the defect this file exists for.
    re_b = ROOT / "results/reach_engine_vs_bank__clean_test.json"
    if re_b.exists():
        E = json.loads(re_b.read_text())
        flat = re.sub(r"\s+", " ", whole)
        C, A = E["contrasts"], E["arms"]
        m = re.search(r"an engine term of \$\+([\d.]+)\$", flat)
        check("engine term, main body", m and m.group(1), C["engine_at_fixed_rules_B_minus_A"]["point"])
        m = re.search(r"whole distance between\s*the two configurations is \$\+([\d.]+)\$", flat)
        check("engine decomposition, total", m and m.group(1),
              A["D_sygma_engine_175_rules_composed"]["point"] - A["A_grail_engine_152_rules"]["point"])
        m = re.search(r"\$\+([\d.]+)\$ \$\[\+[\d.]+,\+[\d.]+\]\$\s*from the \$23\$ rules", flat)
        check("engine decomposition, the 23 rules", m and m.group(1),
              C["the_23_rules_at_fixed_engine_C_minus_B"]["point"])
        m = re.search(r"and \$\+([\d.]+)\$ \$\[\+[\d.]+,\+[\d.]+\]\$ from composing steps", flat)
        check("engine decomposition, composition", m and m.group(1), C["composition_D_minus_C"]["point"])
        checks.append((abs(sum(v["point"] for v in C.values())
                           - (A["D_sygma_engine_175_rules_composed"]["point"]
                              - A["A_grail_engine_152_rules"]["point"])) < 5e-4,
                       "the three terms sum to the distance they decompose",
                       round(sum(v["point"] for v in C.values()), 4),
                       round(A["D_sygma_engine_175_rules_composed"]["point"]
                             - A["A_grail_engine_152_rules"]["point"], 4), ""))

    # 10a-a. The main text and the appendix must agree about what was fixed in advance. They did
    # not: one claimed the design's pre-specification was tabulated where the other stated that no
    # family had been pre-specified at all.
    flat = re.sub(r"\s+", " ", whole)
    claims_prespec = bool(re.search(r"pre-specification of the design are given", flat))
    denies = bool(re.search(r"[Nn]o family was pre-specified", flat))
    checks.append((not (claims_prespec and denies),
                   "the main text does not claim a pre-specification the appendix denies",
                   f"claims={claims_prespec}, denies={denies}",
                   "the two may not both hold", ""))

    # 10a-b. The full-split criterion family. The main text says the differential survives Holm
    # over three pairs; that family is a different one from the subset and external families the
    # appendix tabulates, and it was the only correction in the paper with nothing reading it.
    fp = ROOT / "results/match_sensitivity_fulln_paired.json"
    if fp.exists():
        F = json.loads(fp.read_text())
        flat = re.sub(r"\s+", " ", whole)
        sens = F["sensitivity"]
        m = re.search(r"GRAIL gains \$([\d.]+)\$\s*\$\[[\d.,]+\]\$ from canonical to tautomer "
                      r"matching against MetaPredictor's \$([\d.]+)\$", flat)
        check("full-split sensitivity, GRAIL", m and m.group(1), sens["GRAIL"]["gain"])
        check("full-split sensitivity, MetaPredictor", m and m.group(2),
              sens["MetaPredictor"]["gain"])
        m = re.search(r"a differential of \$\+([\d.]+)\$", flat)
        gm = next(r for r in F["pairwise"] if r["pair"] == "GRAIL_vs_MetaPredictor")
        check("full-split differential", m and m.group(1), gm["interaction_b_minus_a"])
        checks.append((len(F["pairwise"]) == 3 and gm["rejected"],
                       "the full-split family is three pairs and this one is rejected in it",
                       f"{len(F['pairwise'])} pairs, rejected={gm['rejected']}",
                       "3 and True", "the family the main text names"))

    # 10b-a. The two harnesses compute the canonical key differently, which the protocol section
    # now states. A check pins each to what is claimed, since a silent change either way would make
    # the same column heading mean two things without saying so.
    mk = ROOT / "grail_metabolism" / "metrics.py"
    if mk.exists():
        src = mk.read_text()
        m = re.search(r"def _canonical_key.*?(?=\n@|\ndef )", src, re.S)
        checks.append((bool(m) and "isomericSmiles=False" in m.group(0),
                       "the metabolite key is computed without stereochemistry",
                       "isomericSmiles=False", "as the protocol states", ""))
    rl = ROOT / "scripts/retro_leaderboard.py"
    if rl.exists():
        src = rl.read_text()
        checks.append(('out["canonical"] = Chem.MolToSmiles(m)' in src
                       and 'out["canonical"] = Chem.MolToSmiles(m, isomericSmiles=False)' not in src,
                       "the cross-domain key retains stereochemistry",
                       "MolToSmiles at its default", "as the protocol states", ""))

    # 10b-b. The external decisions under a test that assumes no distribution, and the degenerate
    # comparisons the approximation mishandles. Both claims are gates, not counts.
    ex = ROOT / "results/external_exact_tests.json"
    mh = ROOT / "results/multiplicity_holm.json"
    if ex.exists() and mh.exists():
        X = json.loads(ex.read_text())
        M = json.loads(mh.read_text())
        paper_four = {r["pair"].replace("_vs_", " vs ")
                      for r in M["external"]["result"] if r["rejected"]}
        signflip = {s.split(" | ", 1)[1] for s in X["holm_signflip"]}
        checks.append((paper_four == signflip and len(paper_four) == 4,
                       "the external rejections survive a sign-flip test",
                       f"{len(signflip)} survive, matching {len(paper_four)}",
                       "the same four", "family of twelve against the declared six"))
        degenerate = [r for r in X["rows"] if r["n_informative"] == 0]
        checks.append((len(degenerate) == 3
                       and all(r["p_signflip"] == 1.0 for r in degenerate)
                       and all(r["p_normal_approx"] == 0.0 for r in degenerate),
                       "the degenerate comparisons are the ones the approximation inverts",
                       f"{len(degenerate)} with no informative substrate",
                       "3, each p=1 by sign-flip and p=0 by the approximation",
                       "none of them enters a declared family"))

    # 10c-e. The census across independent libraries. The claim is that the widely used construct
    # is the inert one, so the check is on the separation and not only on the counts.
    cc = ROOT / "results/convention_census.json"
    if cc.exists():
        C = json.loads(cc.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"Across six\s*independent libraries and \$([\d,{}]+)\$ templates", flat)
        check("census, templates", m and m.group(1).replace("{,}", ""), C["summary"]["templates"])
        checks.append((C["summary"]["independent_libraries"] == 6,
                       "census, independent libraries", C["summary"]["independent_libraries"], 6,
                       ", ".join(C["summary"]["sources"])))
        # The structural claim that survives version drift: the commented-out records hold exactly
        # sixteen templates that appear nowhere else, in the snapshot the paper measured and in the
        # release retrieved for the census alike.
        br = C.get("biotransformer_release")
        if br:
            m = re.search(r"which holds \$(\d+)\$ active\s*templates against the \$(\d+)\$ of the "
                          r"snapshot", flat)
            check("BioTransformer, active in the retrieved release", m and m.group(1), br["active"])
            checks.append((br["commented_out_only"] == 16,
                           "the sixteen commented-out templates reproduce",
                           br["commented_out_only"], 16,
                           "the same gap the earlier snapshot recorded"))

        # The transcription result: within one release, rules copied verbatim keep the source's
        # convention and rules re-typed acquire one the source never uses.
        tr = C.get("transcription")
        if tr:
            m = re.search(r"the \$(\d+)\$ copied verbatim carry the atom primitive not once, while\s*"
                          r"\$(\d+)\$ of the \$(\d+)\$ re-typed do, a construct absent "
                          r"from all \$(\d+)\$ rules", flat)
            check("transcription, copied verbatim", m and m.group(1), tr["identical"])
            check("transcription, re-typed carrying it", m and m.group(2), tr["rewritten_with_atom"])
            check("transcription, re-typed", m and m.group(3), tr["rewritten"])
            check("transcription, source rules", m and m.group(4), tr["source_rules"])
            checks.append((tr["identical_with_atom"] == 0 and tr["source_with_atom"] == 0,
                           "neither the source nor the verbatim copies use the primitive",
                           f"{tr['source_with_atom']} and {tr['identical_with_atom']}", "0 and 0",
                           "which is what makes the re-typed share a change of convention"))
        # The range the main text once printed is now carried by the appendix table, whose rows are
        # each checked below; the claim that survives in the nine pages is the ordering of the
        # three constructs, gated separately.
        IND = {"this work, full bank", "SyGMa phase1", "SyGMa phase2",
               "USPTO extracted templates", "BioTransformer", "GLORY, CYP rules",
               "GLORYx, own rules"}
        atom = [C["banks"][k]["share_atom"] for k in IND if k in C["banks"]]
        cnt = [C["banks"][k]["share_count"] for k in IND
               if k in C["banks"] and C["banks"][k].get("share_count") is not None]
        checks.append((max(cnt) > max(atom),
                       "the inert construct is the most widely used one",
                       f"count reaches {max(cnt)}", f"atom reaches {max(atom)}",
                       "which is what the sentence asserts"))
        # every row of the appendix table, against the census that produced it
        ROWS = {"this work": "this work, full bank", "curated partition": "this work, curated half",
                "mined partition": "this work, mined half", "SyGMa, phase 1": "SyGMa phase1",
                "SyGMa, phase 2": "SyGMa phase2", "USPTO extracted": "USPTO extracted templates",
                "BioTransformer": "BioTransformer", "GLORY, CYP rules": "GLORY, CYP rules",
                "GLORYx, own rules": "GLORYx, own rules",
                "GLORYx, rules attributed to SyGMa": "GLORYx, SyGMa portion",
                "RetroSim, same corpus as USPTO": "RetroSim extracted templates"}
        for shown, key in ROWS.items():
            v = C["banks"][key]
            mm = re.search(re.escape(shown) + r" & \$[\d,{}]+\$ & \$([\d.]+)\$ & \$([\d.]+)\$ & "
                           r"\$([\d.]+)\$", flat)
            check(f"census table, {shown}, atom", mm and mm.group(1), v["share_atom"])
            check(f"census table, {shown}, count", mm and mm.group(2), v["share_count"])
            check(f"census table, {shown}, degree", mm and mm.group(3), v["share_degree"])

        # the separation itself: no library is exposed both ways
        both = [k for k, v in C["banks"].items()
                if (v.get("share_atom") or 0) > 0.05 and (v.get("share_degree") or 0) > 0.05]
        checks.append((not both, "no library is exposed to the step in both directions",
                       f"{len(both)} are", "0", "which is what 'disjoint sets' asserts"))

    # 10c-n. The label-convention audit. The paper's own pipeline labels its rules in one hydrogen
    # convention and fires them in the other; this binds every figure of that audit to the run.
    lc = ROOT / "results/label_convention_audit.json"
    if lc.exists():
        L = json.loads(lc.read_text())
        P, A, C = L["positives"], L["agreement"], L["config"]
        flat = re.sub(r"\s+", " ", whole)
        ml = re.search(r"whole bank both ways to \$(\d+)\$ training substrates, of the \$(\d+)\$ "
                       r"rule--substrate positives that exist in the convention the pipeline fires "
                       r"in, the supervision sees \$(\d+)\$ and misses \$(\d+)\$, and asserts a "
                       r"further \$(\d+)\$ that do not exist at inference; the two matrices agree at "
                       r"a Jaccard of \$([\d.]+)\$, and \$(\d+)\$ of the \$(\d+)\$ substrates carry "
                       r"a different row", flat)
        checks.append((bool(ml), "the label-convention passage parses", "present",
                       "matched" if ml else "not matched", ""))
        if ml:
            src = str(lc.relative_to(ROOT))
            for lbl, said, got in (
                    ("substrates audited", ml.group(1), C["n_substrates_scored"]),
                    ("positives where the pipeline fires", ml.group(2), P["implicit_total"]),
                    ("positives the supervision sees", ml.group(3), P["both"]),
                    ("positives it misses", ml.group(4), P["implicit_only"]),
                    ("positives it asserts and inference lacks", ml.group(5), P["expanded_only"]),
                    ("Jaccard between the two matrices", ml.group(6), A["jaccard"]),
                    ("substrates whose row changes", ml.group(7),
                     A["substrates_whose_label_row_changes"]),
                    ("substrates in the denominator", ml.group(8), A["substrates_scored"])):
                check(f"label convention, {lbl}", said, got, src)
            mp = re.search(r"Over the same \$150\$ substrates, \$(\d+)\$ rules fire productively and "
                           r"\$(\d+)\$ of them are never labelled positive, while \$(\d+)\$ carry a "
                           r"label and never fire; the two priors rank the rules at a Spearman of "
                           r"\$([\d.]+)\$ and share \$(\d+)\$ of their top hundred", flat)
            checks.append((bool(mp), "the frequency-prior passage parses", "present",
                           "matched" if mp else "not matched", ""))
            if mp:
                F = L["frequency_prior"]
                for lbl, said, got in (
                        ("rules that fire productively", mp.group(1),
                         F["rules_positive_where_it_fires"]),
                        ("of those, never labelled", mp.group(2),
                         F["rules_that_fire_but_are_never_labelled"]),
                        ("labelled but never firing", mp.group(3),
                         F["rules_labelled_that_never_fire"]),
                        ("Spearman between the two priors", mp.group(4),
                         F["spearman_between_the_two_priors"]),
                        ("shared in the top hundred", mp.group(5), F["top_k_overlap"]["100"])):
                    check(f"frequency prior, {lbl}", said, got, src)
            # the headline reading in the limitations must not overstate the agreement
            share = P["both"] / max(P["implicit_total"], 1)
            checks.append((0.4 <= share <= 0.6 and "about half the\nrule--substrate positives"
                           .replace("\n", " ") in re.sub(r"\s+", " ", whole),
                           "the limitations say about half", "about half",
                           f"{share:.3f} of the fired-convention positives", src))

    # 10c-m. The curator agreement. The text said notation explains most of the disagreement; two
    # Jaccards of 0.145 and 0.406 leave 0.594 of the union still disagreeing, so notation explains
    # 0.261 of 0.855, about a third. The share is now computed from the two figures the sentence
    # prints, so the wording cannot drift from them again.
    flat = re.sub(r"\s+", " ", whole)
    mc = re.search(r"agree at a Jaccard of \$([\d.]+)\$ under strict \\textsc\{inchikey\} matching and "
                   r"at \$([\d.]+)\$ under the tautomer-aware default, so notation accounts for "
                   r"(a third|a half|most) of what reads as disagreement", flat)
    checks.append((bool(mc), "the curator-agreement sentence parses", "present",
                   "matched" if mc else "not matched", ""))
    if mc:
        j1, j2 = float(mc.group(1)), float(mc.group(2))
        share = (j2 - j1) / (1 - j1)
        band = "a third" if 0.25 <= share < 0.42 else ("a half" if 0.42 <= share < 0.6 else "most")
        checks.append((mc.group(3) == band, "the share notation explains", mc.group(3), band,
                       f"{share:.3f} of the disagreement, from the two printed Jaccards"))

    # 10c-l. The full-split family. It was corrected at m=3 while the rule the paper declares for
    # its other families -- the grid the axes admit -- gives 3 method pairs by 10 criterion steps.
    # The conclusion holds at either size, and the paper now says so rather than quoting the
    # smaller one alone, which is the size a reader would ask about first.
    from math import erf as _erf, sqrt as _sq
    flat = re.sub(r"\s+", " ", whole)
    mf = re.search(r"a differential of \$\+([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$, which survives "
                   r"Holm--Bonferroni over the three method pairs and over the (\w+) cells", flat)
    checks.append((bool(mf), "the full-split family sentence parses", "present",
                   "matched" if mf else "not matched", ""))
    if mf:
        words = {"thirty": 30, "three": 3, "twenty": 20, "twelve": 12}
        m_big = words[mf.group(4).lower()]
        checks.append((m_big == 3 * 10, "the declared rule's family size for this panel",
                       "3 pairs by 10 criterion steps", str(m_big), "the rule stated in the paper"))
        d, lo, hi = (float(mf.group(1)), float(mf.group(2)), float(mf.group(3)))
        se = (hi - lo) / (2 * 1.959964)
        z = d / se if se > 0 else float("inf")
        pv = 2 * (1 - 0.5 * (1 + _erf(z / _sq(2))))
        checks.append((pv <= 0.05 / m_big, "the differential survives the stricter family",
                       f"p <= {0.05 / m_big:.5f}", f"p = {pv:.2e}", "from the reported interval"))

    # 10c-k. The matched-emission arm. Its two intervals sit in the main text under a pointer to
    # an appendix that did not contain them, and no check read them; the numbers were exact and
    # unfindable, which is the same defect as a number that is wrong for anyone checking the paper.
    cw = ROOT / "results/criterion_within_method.json"
    if cw.exists():
        W = json.loads(cw.read_text())["matched_emission"]["paired_differences_at_matched_emission"]
        flat = re.sub(r"\s+", " ", whole)
        mw = re.search(r"below both comparators by \$(-[\d.]+)\$ \$\[(-[\d.]+),(-[\d.]+)\]\$ and "
                       r"\$(-[\d.]+)\$ \$\[(-[\d.]+),(-[\d.]+)\]\$", flat)
        checks.append((bool(mw), "the matched-emission sentence parses", "present",
                       "matched" if mw else "not matched", ""))
        if mw:
            for key, base in (("GRAIL-SyGMa", 1), ("GRAIL-MetaPredictor", 4)):
                g = W[key]
                said = mw.group(base)
                check(f"matched emission, {key}", said,
                      round(g["delta"], len(said.split(".")[1])), str(cw.relative_to(ROOT)))
                check(f"matched emission, {key}, lower", mw.group(base + 1),
                      round(g["ci95"][0], 3), str(cw.relative_to(ROOT)))
                check(f"matched emission, {key}, upper", mw.group(base + 2),
                      round(g["ci95"][1], 3), str(cw.relative_to(ROOT)))

    # 10c-i. The budget axis is the one place a reversal crosses margins certified at both ends,
    # which is the distinction three reviewers said the paper asserts without testing. It is not
    # asserted here: the sweep's pairwise margins carry paired intervals and the count of sign
    # changes that survive them is read from the artifact.
    fm2 = ROOT / "results/four_method_291.json"
    if fm2.exists():
        Q = json.loads(fm2.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mb = re.search(r"(Four|Three|Five|Six) of the six pairs change sign along that sweep and "
                       r"(three|two|four|one) change it between margins that each separate from zero",
                       flat)
        checks.append((bool(mb), "the budget-reversal sentence parses", "present",
                       "matched" if mb else "not matched", ""))
        if mb:
            words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            check("pairs changing sign along the sweep", words[mb.group(1).lower()],
                  Q["n_pairs_changing_sign"], str(fm2.relative_to(ROOT)))
            check("sign changes certified at both ends", words[mb.group(2).lower()],
                  Q["n_certified_both_ends"], str(fm2.relative_to(ROOT)))
            checks.append((Q["n_certified_both_ends"] >= 1,
                           "at least one budget reversal crosses two certified margins",
                           "at least one", str(Q["n_certified_both_ends"]),
                           str(fm2.relative_to(ROOT))))

    # 10c-h. The minimum detectable effect: the quantile, the two summaries, and which one the
    # bound is asserted at. The quantiles stated in the appendix were the pooled family's and a
    # hand-computed value, neither matching the m the sentence names; the code always used the
    # right one, so the artifact is the reference and the prose is what is checked against it.
    from math import erf, sqrt as _sqrt
    def _ppf(pr):
        lo_, hi_ = -10.0, 10.0
        for _ in range(200):
            mid = (lo_ + hi_) / 2
            if 0.5 * (1 + erf(mid / _sqrt(2))) < pr: lo_ = mid
            else: hi_ = mid
        return (lo_ + hi_) / 2
    rp, mp = ROOT / "results/retro_population_axis.json", ROOT / "results/population_axis.json"
    if rp.exists() and mp.exists():
        R = json.loads(rp.read_text()); M = json.loads(mp.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mm = re.search(r"With \$m=(\d+)\$ on the released files the critical quantile is \$z=([\d.]+)\$, "
                       r"and with \$m=(\d+)\$ on our split \$z=([\d.]+)\$\. The median test in each "
                       r"family then detects \$([\d.]+)\$ and \$([\d.]+)\$, and the least sensitive "
                       r"\$([\d.]+)\$ and \$([\d.]+)\$", flat)
        checks.append((bool(mm), "the MDE sentence parses", "present",
                       "matched" if mm else "not matched", ""))
        if mm:
            for lbl, m_str, z_str in (("released files", mm.group(1), mm.group(2)),
                                      ("metabolite split", mm.group(3), mm.group(4))):
                check(f"Holm-threshold quantile, {lbl}", z_str,
                      round(_ppf(1 - 0.05 / (2 * int(m_str))), 3), "closed form")
            for lbl, said, got in (
                    ("released files", mm.group(5), R["minimum_detectable_interaction"]["median"]),
                    ("metabolite split", mm.group(6), M["minimum_detectable_interaction"]["median"]),
                    ("released files, blindest", mm.group(7),
                     R["minimum_detectable_interaction"]["largest"]),
                    ("metabolite split, blindest", mm.group(8),
                     M["minimum_detectable_interaction"]["largest"])):
                check(f"detectable interaction, {lbl}", said, round(got, 3),
                      str((rp if "released" in lbl else mp).relative_to(ROOT)))
            # the family size the sentence names must be the one the run corrected over
            check("family size, released files", mm.group(1), R["n_comparisons"]
                  if "n_comparisons" in R else int(round(0.05 / R["minimum_detectable_interaction"]
                                                         ["alpha_per_test"])), str(rp.relative_to(ROOT)))

    # 10c-g. The abstract's budget sentence. It said the system that emits most moves from last to
    # first; on the only population carrying all four methods the mover emits 33.6 and the system
    # that emits most is a different one that does not move that way. A sentence in the abstract is
    # the most-read claim in the paper and had no gate at all, so it gets one: whoever is last at
    # the tight budgets and first at the field's must be a single method, and must not be the one
    # emitting most.
    fm = ROOT / "results/four_method_291.json"
    if fm.exists():
        P = json.loads(fm.read_text())["per_method"]
        rec = {m: v["recall"] for m, v in P.items()}
        last_tight = {m for m in P if all(rec[m][k] == min(rec[x][k] for x in P) for k in ("1", "5"))}
        first_wide = {m for m in P if rec[m]["15"] == max(rec[x]["15"] for x in P)}
        mover = last_tight & first_wide
        widest = max(P, key=lambda m: P[m]["mean_emitted_uncapped"])
        checks.append((len(mover) == 1, "one method is last at tight budgets and first at 15",
                       "exactly one", f"{sorted(mover) or 'none'}", str(fm.relative_to(ROOT))))
        checks.append((widest not in mover, "the mover is not the method that emits most",
                       "different methods",
                       f"mover {sorted(mover)}, widest {widest} at "
                       f"{P[widest]['mean_emitted_uncapped']}", str(fm.relative_to(ROOT))))
        flat_a = re.sub(r"\s+", " ", whole)
        said = ("one method is last at tight budgets and first at the budget the field reports, "
                "with every method and its ranking held fixed")
        checks.append((said in flat_a, "the abstract says what the sweep shows",
                       "present", "present" if said in flat_a else "not matched", ""))

    # 10c-j. The arms table itself, which no check parsed. The existing engine block compared the
    # main-text prose against the artifact and then verified the artifact's own additivity, which
    # is a self-check: it passes while the appendix table carries three terms from two populations
    # and fails to add up. This reads the four printed arms, ties each to the artifact, and
    # requires the three printed contrasts to sum to the printed endpoint difference.
    rv = ROOT / "results/reach_engine_vs_bank__clean_test.json"
    if rv.exists():
        V = json.loads(rv.read_text())
        arms, con = V["arms"], V["contrasts"]
        flat = re.sub(r"\s+", " ", whole)
        ma = re.search(r"A & \$152\$ shared, expanded & ([\d.]+) .*?"
                       r"B & \$152\$ shared, SyGMa's & ([\d.]+) .*?"
                       r"C & all \$175\$, SyGMa's & ([\d.]+) .*?"
                       r"D & all \$175\$, SyGMa's composed & ([\d.]+) ", flat)
        checks.append((bool(ma), "the arms table parses", "present",
                       "matched" if ma else "not matched", ""))
        if ma:
            for lbl, said, key in (("A", ma.group(1), "A_grail_engine_152_rules"),
                                   ("B", ma.group(2), "B_sygma_engine_152_rules_one_step"),
                                   ("C", ma.group(3), "C_sygma_engine_175_rules_one_step"),
                                   ("D", ma.group(4), "D_sygma_engine_175_rules_composed")):
                check(f"arm {lbl}, on the split", said, round(arms[key]["point"], 3),
                      str(rv.relative_to(ROOT)))
            mt = re.search(r"The engine at fixed rules is worth \$\+([\d.]+)\$ .*?"
                           r"the \$23\$ rules absent from our bank \$\+([\d.]+)\$ .*?"
                           r"and composition \$\+([\d.]+)\$", flat)
            checks.append((bool(mt), "the three printed terms parse", "present",
                           "matched" if mt else "not matched", ""))
            if mt:
                terms = [float(g) for g in mt.groups()]
                endpoints = float(ma.group(4)) - float(ma.group(1))
                checks.append((abs(sum(terms) - endpoints) <= 0.001,
                               "the printed terms sum to the printed endpoints",
                               f"{endpoints:.4f}", f"{sum(terms):.4f}", "the table itself"))
                for lbl, said, key in (("engine", terms[0], "engine_at_fixed_rules_B_minus_A"),
                                       ("the 23 rules", terms[1],
                                        "the_23_rules_at_fixed_engine_C_minus_B"),
                                       ("composition", terms[2], "composition_D_minus_C")):
                    check(f"printed term, {lbl}", said,
                          round(con[key]["point"], 4 if lbl == "composition" else 3),
                          str(rv.relative_to(ROOT)))

    # 10c-f. The mechanism, promoted to the main text without its numbers. A sentence that says
    # "most" and "a half-percent" instead of two integers is still a quantitative claim, and a
    # quantitative claim with no gate is how a number drifts away from what produced it. The words
    # are therefore bound to the measurement: firing within a tenth either way, a majority of
    # fragments unreadable under the expansion, and a rate that rounds to half a percent without it.
    mech = next((ROOT / f"results/{n}" for n in
                 ("explicit_h_mechanism__clean_test.json", "explicit_h_mechanism.json")
                 if (ROOT / f"results/{n}").exists()), None)
    if mech is not None:
        M = json.loads(mech.read_text())["pipeline"]
        exp, imp = M["deployed"], M["without_explicit_h"]
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"the toolkit refuses the result: \$(\d+)\\%\$ of the fragments the expanded products "
                      r"separate into are structures the toolkit will not read back, against "
                      r"\$([\d.]+)\\%\$ without the expansion", flat)
        checks.append((bool(m), "the mechanism sentence parses",
                       "present", "matched" if m else "not matched", ""))
        fire = exp["fired"] / max(imp["fired"], 1)
        checks.append((0.9 <= fire <= 1.1, "templates fire almost equally often either way",
                       "within a tenth", f"ratio {fire:.3f}", str(mech.relative_to(ROOT))))
        if m:
            check("fragments unreadable under the expansion", m.group(1),
                  round(exp["unparseable"] / max(exp["fragments"], 1) * 100),
                  str(mech.relative_to(ROOT)))
            check("fragments unreadable without it", m.group(2),
                  round(imp["unparseable"] / max(imp["fragments"], 1) * 100, 1),
                  str(mech.relative_to(ROOT)))
        m2 = re.search(r"against \$[\d{},]+\$ of \$[\d{},]+\$: \$\d+\\%\$ against \$[\d.]+\\%\$, and\s*"
                       r"\$([\d.]+)\\%\$ of the excess", flat)
        if m2:
            check("the excess that is a valence violation", m2.group(1),
                  round(json.loads(mech.read_text())["where_it_is_lost"]
                        ["share_of_the_drop_that_is_unparseable"] * 100, 1),
                  str(mech.relative_to(ROOT)))
        # the appendix quotes the same measurement as integers, on the same population
        for name, want in (("fired with hydrogens explicit", exp["fired"]),
                           ("fired without them", imp["fired"]),
                           ("fragments under the expansion", exp["fragments"]),
                           ("fragments RDKit refuses", exp["unparseable"]),
                           ("fragments without the expansion", imp["fragments"]),
                           ("fragments refused without it", imp["unparseable"])):
            lit = f"{want:,}".replace(",", "{,}")
            checks.append((f"${lit}$" in flat, name, f"${lit}$",
                           "present" if f"${lit}$" in flat else "absent",
                           str(mech.relative_to(ROOT))))

    # 10c-d. The three-bank reversal, now in the main text: the claim rests on the direction, so
    # the direction is what is checked, not only the six counts.
    # The census of which templates carry the primitive is its own measurement, distinct from how
    # many the dispatch classifier sends anywhere; the two differ and must not be substituted.
    hd_b = ROOT / "results/explicit_h_mechanism.json"
    hy_b = ROOT / "results/hydrogen_dispatch__clean_test.json"
    if hd_b.exists() and hy_b.exists():
        B = json.loads(hd_b.read_text())["hydrogen_convention_by_bank"]
        H = json.loads(hy_b.read_text())["banks"]
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"with hydrogens drawn, BioTransformer recovers \$([\d,{}]+)\$ references and "
                      r"SyGMa\s*\$(\d+)\$; with hydrogens left implicit, SyGMa recovers "
                      r"\$([\d,{}]+)\$ and BioTransformer \$(\d+)\$", flat)
        checks.append((bool(m), "the three-bank sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            num = lambda g: g.replace("{,}", "")
            def recovered(bank, arm):
                v = H[bank]
                return round(v["global_arms"][arm] * v["references"])
            check("BioTransformer, hydrogens drawn", num(m.group(1)),
                  recovered("biotransformer", "all_explicit"))
            check("SyGMa, hydrogens drawn", num(m.group(2)),
                  recovered("sygma_175", "all_explicit"))
            check("SyGMa, hydrogens implicit", num(m.group(3)),
                  recovered("sygma_175", "all_implicit"))
            check("BioTransformer, hydrogens implicit", num(m.group(4)),
                  recovered("biotransformer", "all_implicit"))
            # The counts are recorded rather than re-derived. A check that multiplies a rounded
            # reach by the reference total reproduces the arithmetic the sentence was written with
            # and tests nothing about provenance; where the measurement records the integer, the
            # integer is what the manuscript is held to.
            rec = {b: H[b].get("global_arms_paired") for b in ("biotransformer", "sygma_175")}
            if all(rec.values()):
                for lbl, bank, key, grp in (
                        ("BioTransformer drawn, recorded", "biotransformer", "recovered_explicit", 1),
                        ("SyGMa drawn, recorded", "sygma_175", "recovered_explicit", 2),
                        ("SyGMa implicit, recorded", "sygma_175", "recovered_implicit", 3),
                        ("BioTransformer implicit, recorded", "biotransformer", "recovered_implicit", 4)):
                    check(lbl, num(m.group(grp)), rec[bank][key], str(hy_b.relative_to(ROOT)))
                # the exchange is a paired difference in each bank, and the two run opposite ways
                mi = re.search(r"paired over substrates the step is worth \$([-+][\d.]+)\$ "
                               r"\$\[([-+][\d.]+),([-+][\d.]+)\]\$ to SyGMa and \$([-+][\d.]+)\$ "
                               r"\$\[([-+][\d.]+),([-+][\d.]+)\]\$ to BioTransformer", flat)
                checks.append((bool(mi), "the reversal interval sentence parses",
                               "present", "matched" if mi else "not matched", ""))
                if mi:
                    for bank, base in (("sygma_175", 1), ("biotransformer", 4)):
                        g = rec[bank]
                        check(f"{bank}, explicit minus implicit", mi.group(base),
                              round(g["explicit_minus_implicit"], 3), str(hy_b.relative_to(ROOT)))
                        check(f"{bank}, lower bound", mi.group(base + 1),
                              round(g["ci95"][0], 3), str(hy_b.relative_to(ROOT)))
                        check(f"{bank}, upper bound", mi.group(base + 2),
                              round(g["ci95"][1], 3), str(hy_b.relative_to(ROOT)))
                    checks.append((rec["sygma_175"]["explicit_minus_implicit"] *
                                   rec["biotransformer"]["explicit_minus_implicit"] < 0
                                   and rec["sygma_175"]["excludes_zero"]
                                   and rec["biotransformer"]["excludes_zero"],
                                   "the step moves the two banks in opposite directions",
                                   "opposite signs, both certified",
                                   f"{rec['sygma_175']['explicit_minus_implicit']:+.3f} against "
                                   f"{rec['biotransformer']['explicit_minus_implicit']:+.3f}",
                                   str(hy_b.relative_to(ROOT))))
            # the reversal itself, not merely the four counts
            checks.append((int(num(m.group(1))) > int(num(m.group(2)))
                           and int(num(m.group(3))) > int(num(m.group(4))),
                           "the two banks exchange between the conventions",
                           f"drawn {m.group(1)} vs {m.group(2)}, implicit {m.group(3)} vs {m.group(4)}",
                           "one leads under each", ""))

    # 10c-c. The confirmatory family, reconstructible from the main text: the grid the axes admit,
    # less the cells where two conventions coincide and there is nothing to test.
    lb_f = ROOT / "results/retro_leaderboard_cluster0.json"
    if lb_f.exists():
        Lf = json.loads(lb_f.read_text())
        cfg = Lf["config"]
        n_sys = len(cfg["systems"])
        grid = (n_sys * (n_sys - 1) // 2) * (len(cfg["modes"]) * (len(cfg["modes"]) - 1) // 2) \
               * len(cfg["ks"])
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"which is \$(\d+)\$\s*cells less the \$(\d+)\$ in which", flat)
        check("the grid the two axes admit", m and m.group(1), grid)
        check("cells with nothing to test", m and m.group(2), grid - Lf["n_interaction_tests"])

    # 10d-b. The resolution floor: which published margins this benchmark can separate at all. The
    # paper's condition rests on it, so every figure it states is tied to the measurement.
    rf = ROOT / "results/resolution_floor.json"
    if rf.exists():
        F = json.loads(rf.read_text())["leaderboards"]
        flat = re.sub(r"\s+", " ", whole)
        c0, c1 = F["cluster0"], F["cluster1"]
        m = re.search(r"\\textbf\{Eleven of the twenty-one margins do not separate\s*from zero, and "
                      r"five of the six adjacent margins in the published order do not\.\}", flat)
        checks.append((bool(m) and c0["not_separable"] == 11 and c0["n_pairs"] == 21
                       and c0["adjacent_not_separable"] == 5 and c0["adjacent_pairs"] == 6,
                       "resolution, the sentence matches the artifact",
                       "eleven of twenty-one, five of six",
                       f"{c0['not_separable']} of {c0['n_pairs']}, "
                       f"{c0['adjacent_not_separable']} of {c0['adjacent_pairs']}",
                       "paired bootstrap on each margin, predictions frozen"))
        m = re.search(r"benchmark certifies is \$([\d.]+)\$", flat)
        check("resolution floor, seven-system", m and m.group(1), c0["resolution_floor"])
        m = re.search(r"whose narrowest\s*margin is \$([\d.]+)\$", flat)
        smallest = min(abs(v["margin"]) for v in c1["pairs"].values())
        check("three-system narrowest margin", m and m.group(1), smallest)
        checks.append((abs(smallest - c1["resolution_floor"]) < 1e-9,
                       "the control's floor is its narrowest margin, every pair separating",
                       smallest, c1["resolution_floor"],
                       "which is why the two must not be stated as different numbers"))
        checks.append((c1["not_separable"] == 0, "the control group separates every pair",
                       c1["not_separable"], 0, "which is why no criterion exchanges it"))
        # the two orderings the body prints in words: every number, against the artifact
        L0 = json.loads(lb.read_text())["accuracy"]
        for shown, key, mode in (("GraphRetro", "graphretro", "canonical"),
                                 ("LocalRetro", "localretro", "canonical"),
                                 ("Retroformer", "retroformer", "canonical"),
                                 ("Graph2SMILES", "graph2smiles", "canonical"),
                                 ("Graph2SMILES", "graph2smiles", "nostereo"),
                                 ("Retroformer", "retroformer", "nostereo"),
                                 ("GraphRetro", "graphretro", "nostereo"),
                                 ("LocalRetro", "localretro", "nostereo")):
            want = L0[key][mode]["top1"]
            hits = re.findall(shown + r" \$([\d.]+)\$", flat)
            checks.append((any(close(h, want, 5e-4) for h in hits),
                           f"body ordering, {shown} {mode}", hits or None, round(want, 4),
                           "printed in the two orderings"))
        # The per-group table these checked is superseded by the packing measurement over all
        # four leaderboards. What the body still prints from the two groups is checked above; the
        # count of exchanging pairs is checked here so the phrasing cannot drift from the artifact.
        m = re.search(r"Five of twenty-one pairs exchange", flat)
        checks.append((bool(m) and len(json.loads(lb.read_text())["pairs_that_exchange"]["top1"]) == 5,
                       "exchanging pairs, body phrasing", "Five of twenty-one",
                       len(json.loads(lb.read_text())["pairs_that_exchange"]["top1"]), ""))

    # 12. intervals printed beside the factors
    for label, key in (("ceiling interval", "coverage_bank"), ("ranking interval", "ranking_conversion")):
        lo, hi = f[key]["lo"], f[key]["hi"]
        pat = re.compile(r"\$\[" + f"{lo:.3f}" + r",\s*" + f"{hi:.3f}" + r"\]\$")
        checks.append((bool(pat.search(whole.replace(" ", ""))
                            or re.search(r"\[" + f"{lo:.3f}" + r"," + f"{hi:.3f}" + r"\]",
                                         whole.replace(" ", ""))),
                       label, f"[{lo:.3f},{hi:.3f}]", "present in the manuscript", ""))

    width = max(len(c[1]) for c in checks)
    bad = 0
    for ok, name, printed, computed, note in checks:
        if not ok:
            bad += 1
        mark = "ok  " if ok else "FAIL"
        extra = f"   ({note})" if note else ""
        print(f"  {mark} {name:{width}}  manuscript {str(printed):>10}   artifact {str(computed)[:12]:>12}{extra}")
    print(f"\n  {len(checks) - bad} of {len(checks)} agree")
    if bad:
        print("  a manuscript number does not follow from the artifact it cites")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
