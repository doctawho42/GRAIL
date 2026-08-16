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
import decimal
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
    # A value read from the manuscript arrives as a string, and it arrives with whatever the
    # pattern happened to capture around it: the maths delimiters, a thousands separator, the
    # sentence's own full stop. Those are typography, not the number, and a check that fails on
    # them is a check that reports a defect the manuscript does not have.
    text = a.strip() if isinstance(a, str) else ""
    if text:
        text = text.replace("$", "").replace("{,}", "").replace(",", "").replace("\\,", "")
        text = text.strip().rstrip(".;:,") if text.count(".") > 1 or text.endswith(
            (".", ";", ":", ",")) and not text.rstrip(".;:,").endswith(".") else text.strip()
    try:
        av, bv = float(text if text else a), float(b)
    except (TypeError, ValueError):
        return False
    # Only for a value read from the manuscript is the printed precision the right yardstick.
    # Artifact-against-artifact comparisons pass floats and keep the tolerance they were given --
    # applying the rule to them made a deliberate 2e-3 convention check into a four-decimal one,
    # which this check caught on itself.
    if text and text.replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).isdigit():
        places = len(text.split(".")[1]) if "." in text else 0
        # half-up, the way a reader rounds: 0.5495 printed at three places is 0.550, and binary
        # floating point rounds it the other way
        q = decimal.Decimal(1).scaleb(-places)
        said = decimal.Decimal(text).quantize(q, rounding=decimal.ROUND_HALF_UP)
        # An artifact sitting exactly on the boundary -- 0.8005 printed at three places -- is a
        # number the reader cannot be wrong about either way, and which of the two a script wrote
        # depends on whether its binary representation fell above or below the half. Both are
        # accepted there and only there; anything off the boundary still has one right answer.
        d = decimal.Decimal(repr(bv))
        return said in (d.quantize(q, rounding=decimal.ROUND_HALF_UP),
                        d.quantize(q, rounding=decimal.ROUND_HALF_DOWN))
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
    # These three appear wherever the manuscript states them, main text or appendix, and the gate
    # follows the sentence rather than a section: a claim that moves to an appendix is still a claim
    # and still has to follow from the artifact, while a claim that is cut is not a failure.
    conv = re.search(r"converts \$?([\d.]+)\\%\$? of its own ceiling", whole)
    checks.append((bool(conv), "the conversion sentence is present somewhere", "present",
                   "matched" if conv else "not matched", "the manuscript"))
    if conv:
        check("conversion percentage", float(conv.group(1)) / 100, S["H"] / S["Cfull"], "H / Cfull")
    flat_cap = re.sub(r"\s+", " ", whole)
    lost = re.search(r"every reference the[^.]*?\$(\d+)\$|all \$(\d+)\$ references lost between "
                     r"the budgeted pool", flat_cap)
    if lost:
        check("references lost to truncation", lost.group(1) or lost.group(2),
              S["Cbud"] - S["H"], "Cbud - H")
    cap = sum(1 for r in rows if len(r.get("deployed_top15") or []) >= 15)
    atcap = re.search(r"(?:binds on|is reached on|cap upstream, on) \$(\d+)\$ of", flat_cap)
    checks.append((bool(atcap), "the cap-binding count is stated somewhere", "present",
                   "matched" if atcap else "not matched", "the manuscript"))
    if atcap:
        check("substrates at the cap", atcap.group(1), cap)

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
                          r"([\d.]+) & \$[-+]?([\d.]+)\$", flat)
            if m is None:
                # a bank may be withdrawn from the table; what may not happen is a row whose
                # numbers disagree with the artifact, which is what the checks below enforce
                continue
            v = H[key]
            check(f"dispatch table, {shown} dispatched", m and m.group(1), v["dispatched_to_expanded"])
            check(f"dispatch table, {shown} reach", m and m.group(2), v["reach"], "", )
            check(f"dispatch table, {shown} best global", m and m.group(3), v["best_global"],
                  "over legitimate settings only")
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

    # 10d-0. The contributions list quotes the survey totals, and a headline in a bulleted list is
    # the number a reviewer reads first and the one least likely to be re-checked when the artifacts
    # move. It was stale for a day behind a green suite because nothing held it.
    survey_esa = ROOT / "results/robust_order_wmt24_esa.json"
    if survey_esa.exists():
        E = json.loads(survey_esa.read_text())
        survey_boards = []
        for _fn, _key in (("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
                          ("robust_order_metabolite.json", None),
                          ("robust_order_posebusters.json", None),
                          ("robust_order_wmt24_en-de.json", None),
                          ("robust_order_wmt24_ja-zh.json", None)):
            _q = ROOT / "results" / _fn
            if _q.exists():
                _d = json.loads(_q.read_text())
                survey_boards.append(_d["leaderboards"][_key] if _key else _d)
        survey_boards += list(E["boards"].values())
        _w23 = ROOT / "results/robust_order_wmt23.json"
        if _w23.exists():
            survey_boards += list(json.loads(_w23.read_text())["boards"].values())
        flat = re.sub(r"\s+", " ", whole)
        mc = re.search(r"Measured on \$(\d+)\$ leaderboards in four domains and \$([\d,{}]+)\$ "
                       r"pairs.*?they publish \$(\d+)\$ places and what survives supports "
                       r"\$(\d+)\$", flat)
        checks.append((bool(mc), "the survey-total sentence parses", "present",
                       "matched" if mc else "not matched", ""))
        if mc:
            src = "results/robust_order_*.json"
            check("survey, leaderboards", mc.group(1), len(survey_boards), src)
            check("survey, pairs", mc.group(2).replace("{,}", ""),
                  sum(b["n_pairs"] for b in survey_boards), src)
            check("survey, published places", mc.group(3),
                  sum(b["n_systems"] for b in survey_boards), src)
            check("survey, tiers supported", mc.group(4),
                  sum(b["tiers_distinguished"] for b in survey_boards), src)
        # The same pair of numbers is printed in the abstract, the contributions and Section 5.
        # A gate on one of three printings is what let an arm be printed three ways once already,
        # so every printing is found and checked, and the count of them is held too: dropping one
        # must fail rather than pass quietly.
        places = sum(b["n_systems"] for b in survey_boards)
        tiers = sum(b["tiers_distinguished"] for b in survey_boards)
        printings = re.findall(r"[Pp]ublish \$?(\d+)\$? [Pp]laces\\?\\?\s*and (?:what survives |what no declared "
                               r"choice contradicts )?[Ss]upports? \$?(\d+)\$?|"
                               # the abstract states it as systems ranked and ranks supported
                               r"rank \$(\d+)\$ systems, and the orderings\s*that survive support "
                               r"\$(\d+)\$ ranks|"
                               # and the title, which states the same pair
                               r"Publish \$(\d+)\$ Ranks and Support \$(\d+)\$", flat)
        printings = [tuple(x for x in pr if x) for pr in printings]
        checks.append((len(printings) >= 4, "every printing of the survey total is found",
                       "4 or more, the title among them", str(len(printings)), "the manuscript"))
        for i, (pl, ti) in enumerate(printings, 1):
            check(f"survey printing {i}, places", pl, places, "the manuscript")
            check(f"survey printing {i}, tiers", ti, tiers, "the manuscript")

    # 10d-f. The emission threshold, cross-fitted. The paper must not read the held-out spread as a
    # second interval on the gain -- it is a fifth of the substrates -- so the gate holds both the
    # reproduced point estimate and the fact that the interval is NOT claimed to exclude zero.
    xf = ROOT / "results/emission_crossfit.json"
    if xf.exists():
        XF = json.loads(xf.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mxf = re.search(r"reproduces the point estimate at \$\+([\d.]+)\$ and selects "
                        r"\$\\alpha=0\.5\$ as the training-part argmax in all \$(\d+)\$\. The "
                        r"spread across held-out fifths is wide, \$\[(-[\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(mxf), "the cross-fitting sentence parses", "present",
                       "matched" if mxf else "not matched", ""))
        if mxf:
            src = str(xf.relative_to(ROOT))
            check("cross-fit, held-out gain", mxf.group(1), XF["held_out_gain_mean"], src)
            check("cross-fit, splits", mxf.group(2), XF["config"]["splits"], src)
            check("cross-fit, interval low", "-" + mxf.group(3).lstrip("-"),
                  XF["held_out_gain_ci95"][0], src)
            check("cross-fit, interval high", mxf.group(4), XF["held_out_gain_ci95"][1], src)
            checks.append((XF["share_of_splits_choosing_alpha_0.5"] == 1.0,
                           "the threshold is the training-part argmax in every split", "1.0",
                           str(XF["share_of_splits_choosing_alpha_0.5"]), src))
            checks.append((not XF["separated_from_zero"],
                           "the held-out spread is not claimed to exclude zero", "not separated",
                           str(XF["separated_from_zero"]), src))

    # 10d-s. The absolute-threshold sweep, which is a released negative result the paper had never
    # cited. An uncited artifact that bears on a claim is what makes an objection constructible.
    sr = ROOT / "results/stopping_rule.json"
    if sr.exists():
        SR = json.loads(sr.read_text())["paired_vs_constant"]
        flat = re.sub(r"\s+", " ", whole)
        msr = re.search(r"gains \$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ and calibrating it "
                        r"\$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$, neither\s*separated from "
                        r"zero.*?expected-F1 rule at \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(msr), "the absolute-threshold sentence parses", "present",
                       "matched" if msr else "not matched", ""))
        if msr:
            src = str(sr.relative_to(ROOT))
            check("threshold rule, gain", msr.group(1), SR["threshold"]["delta"], src)
            check("calibrated rule, gain", msr.group(4), SR["calibrated"]["delta"], src)
            check("expected-F1 rule, gain", msr.group(7), SR["expected_f1"]["delta"], src)
            checks.append((not SR["threshold"]["excludes_zero"]
                           and not SR["calibrated"]["excludes_zero"],
                           "the absolute rules are the null the paper says they are",
                           "neither separated",
                           f"threshold {SR['threshold']['excludes_zero']}, "
                           f"calibrated {SR['calibrated']['excludes_zero']}", src))

    # 10d-x. The docking board's second axis moves the prediction, not only its score. The paper
    # says so with numbers rather than leaving a reviewer to find it, so the numbers are held.
    _pb = ROOT / "results/robust_order_posebusters.json"
    if _pb.exists():
        _sg = json.loads(_pb.read_text())["sub_grids"]
        flat = re.sub(r"\s+", " ", whole)
        _mx = re.search(r"\$([\d.]+)\\%\$ of the \$2\{,\}922\$ paired poses differ between the "
                        r"arms, by a median of \$([\d.]+)\$.*?\$([\d.]+)\\%\$ of per-item verdicts "
                        r"change.*?leaves \$(\d+)\$ of the\s*\$21\$ pairs dominating", flat)
        checks.append((bool(_mx), "the minimisation-axis sentence parses", "present",
                       "matched" if _mx else "not matched", ""))
        if _mx:
            check("docking, criteria-only dominating", _mx.group(4),
                  _sg["criteria only, at the published post-processing"]["n_dominating"],
                  str(_pb.relative_to(ROOT)))

    # 10d-c. What a certified reversal reverses. A pair whose published cell never separated it was
    # not an ordering the table asserted, so counting its reversal under the second claim would be
    # counting the first claim twice. Both halves are stated and both are held.
    _cert = _res = 0
    for _fn, _key in (("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
                      ("robust_order_metabolite.json", None),
                      ("robust_order_posebusters.json", None),
                      ("robust_order_wmt24_en-de.json", None),
                      ("robust_order_wmt24_ja-zh.json", None)):
        _q = ROOT / "results" / _fn
        if _q.exists():
            _d = json.loads(_q.read_text())
            _b = _d["leaderboards"][_key] if _key else _d
            for _pair in _b.get("contested_after_correction", []):
                _cert += 1
                _res += bool(_b["pairs"][_pair].get("resolved_in_the_published_cell"))
    for _src in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
        _q = ROOT / "results" / _src
        if _q.exists():
            for _b in json.loads(_q.read_text())["boards"].values():
                for _pair in _b.get("contested_after_correction", []):
                    _cert += 1
                    _res += bool(_b["pairs"][_pair].get("resolved_in_the_published_cell"))
    if _cert:
        flat = re.sub(r"\s+", " ", whole)
        # the prose names a SUBJECT ("translation's nineteen boards"), and the gate must bind the
        # aggregate to that subject rather than to any loop that happens to produce the number --
        # binding it to an all-board loop is what let a translation sentence carry all-board totals
        _tb = []
        for _s in ("robust_order_wmt24_en-de.json", "robust_order_wmt24_ja-zh.json"):
            _q = ROOT / "results" / _s
            if _q.exists():
                _tb.append(json.loads(_q.read_text()))
        for _s in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
            _q = ROOT / "results" / _s
            if _q.exists():
                _tb += list(json.loads(_q.read_text())["boards"].values())
        _mt = re.search(r"translation's (\w+) boards contest \$(\d+)\$ pairs of which \$(\d+)\$ "
                        r"survive their own boards' correction and \$(\d+)\$ the union of all\s*"
                        r"twenty-three\. Not one of the \$(\d+)\$ reverses an ordering its own "
                        r"published cell had separated from zero: all (\w+) such reversals are "
                        r"the budget's, never the", flat)
        checks.append((bool(_mt), "the translation-subject sentence parses", "present",
                       "matched" if _mt else "not matched", ""))
        if _mt and _tb:
            src = "results/robust_order_wmt*.json"
            _w = {"nineteen": 19, "six": 6}
            check("translation, boards", _w.get(_mt.group(1), -1), len(_tb), src)
            check("translation, contested", _mt.group(2),
                  sum(b["n_contested"] for b in _tb), src)
            check("translation, certified", _mt.group(3),
                  sum(b["n_contested_after_correction"] for b in _tb), src)
            _unj2 = ROOT / "results/union_multiplicity.json"
            if _unj2.exists():
                _PBT = json.loads(_unj2.read_text())["per_board"]
                _tru = sum(v["pairs_union"] for k, v in _PBT.items() if "wmt" in k)
                check("translation, certified under the union", _mt.group(4), _tru,
                      "results/union_multiplicity.json")
            _tres = sum(1 for b in _tb for pr in b.get("contested_after_correction", [])
                        if b["pairs"][pr].get("resolved_in_the_published_cell"))
            checks.append((_tres == 0, "no translation certification reverses a resolved ordering",
                           "0", str(_tres), src))
            _wordn = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
            checks.append((_wordn.get(_mt.group(6), -1) == _res,
                           "the sentence names as many as there are", str(_res),
                           _mt.group(6), src))

        # the qualifier moved into the translation sentence, so what is held now is the pair of
        # totals it rests on: how many certifications there are and how many reverse a resolved
        # ordering, wherever the prose puts them
        _wordnum = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                    "eight": 8, "nine": 9}
        checks.append((_cert > 0, "there are certified reversals to qualify", "at least one",
                       str(_cert), "results/robust_order_*.json"))
        checks.append((_cert == 21, "the certified total the body quotes", "21", str(_cert),
                       "results/robust_order_*.json"))
        # how many certifications reverse an ordering the published cell had established is a
        # number the prose quotes in three places, so it is computed here and each printing is
        # held against it rather than against a literal
        _mres = re.findall(r"(\w+)\s*of the twenty-one certified reversals reverse an ordering", flat)
        checks.append((bool(_mres) and all(_wordnum.get(x.lower(), -1) == _res for x in _mres),
                       "the body's count of certifications of a separated ordering", str(_res),
                       ", ".join(_mres) or "not stated", "results/robust_order_*.json"))

    # 10d-m. The MQM boards' orientation. The published cell must be the aggregation the task
    # actually ranks by, and the check of that must not recompute the board's own quantity: it did,
    # and reported a self-comparison as evidence for a page. Both are held here.
    for _lp in ("en-de", "ja-zh"):
        _q = ROOT / f"results/robust_order_wmt24_{_lp}.json"
        if _q.exists():
            _d = json.loads(_q.read_text())["config"]
            checks.append((_d.get("published_cell_reproduces_the_official_ranking") is True,
                           f"wmt24 {_lp}, the published cell is the task's own ranking", "reproduces",
                           str(_d.get("published_cell_reproduces_the_official_ranking")),
                           f"results/robust_order_wmt24_{_lp}.json"))
    # The aggregation is a declared choice like the two in the grid, so the board is run under both
    # readings and the appendix quotes the difference. Both artifacts are held, including the one
    # the paper argues against: a comparison whose losing arm has no artifact is not a comparison.
    _mv = ROOT / "results/robust_order_wmt24_en-de.json"
    _sg = ROOT / "results/robust_order_wmt24_en-de__segment.json"
    if _mv.exists() and _sg.exists():
        _e, _s = json.loads(_mv.read_text()), json.loads(_sg.read_text())
        flat = re.sub(r"\s+", " ", whole)
        _words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                  "eight": 8, "nine": 9, "no": 0}
        _mm = re.search(r"nineteen-system leaderboard has \$(\d+)\$ dominating pairs and "
                        r"(\w+) certified reversals?, against \$(\d+)\$ and (\w+) under the "
                        r"task's own aggregation", flat)
        checks.append((bool(_mm), "the aggregation comparison parses", "present",
                       "matched" if _mm else "not matched", ""))
        if _mm:
            check("aggregation, dominating under the segment mean", _mm.group(1),
                  _s["n_dominating"], "results/robust_order_wmt24_en-de__segment.json")
            check("aggregation, certified under the segment mean",
                  _words.get(_mm.group(2), -1), _s["n_contested_after_correction"],
                  "results/robust_order_wmt24_en-de__segment.json")
            check("aggregation, dominating under the task's own", _mm.group(3),
                  _e["n_dominating"], "results/robust_order_wmt24_en-de.json")
            check("aggregation, certified under the task's own", _words.get(_mm.group(4), -1),
                  _e["n_contested_after_correction"], "results/robust_order_wmt24_en-de.json")
        _jz = ROOT / "results/robust_order_wmt24_ja-zh.json"
        _jzs = ROOT / "results/robust_order_wmt24_ja-zh__segment.json"
        if _jz.exists() and _jzs.exists():
            _j, _js = json.loads(_jz.read_text()), json.loads(_jzs.read_text())
            _mj = re.search(r"fifteen-system board the same reading changes almost nothing --- "
                            r"\$(\d+)\$ dominating pairs against \$(\d+)\$, and (\w+) certified "
                            r"reversals? either way", flat)
            checks.append((bool(_mj), "the second board's aggregation sentence parses", "present",
                           "matched" if _mj else "not matched", ""))
            if _mj:
                check("aggregation, ja-zh under the segment mean", _mj.group(1),
                      _js["n_dominating"], "results/robust_order_wmt24_ja-zh__segment.json")
                check("aggregation, ja-zh under the task's own", _mj.group(2), _j["n_dominating"],
                      "results/robust_order_wmt24_ja-zh.json")
                checks.append((_words.get(_mj.group(3), -1) == _j["n_contested_after_correction"]
                               == _js["n_contested_after_correction"],
                               "aggregation, ja-zh certified either way", _mj.group(3),
                               f"{_j['n_contested_after_correction']} and "
                               f"{_js['n_contested_after_correction']}",
                               "results/robust_order_wmt24_ja-zh*.json"))
        _mp = re.search(r"no longer reproduces the official ranking: (\w+) systems change place",
                        flat)
        _moved = sum(1 for _a, _b in zip(_s["config"]["official_ranking"], _s["published_order"])
                     if _a != _b)
        check("aggregation, systems that change place", _words.get(_mp.group(1), -1) if _mp else None,
              _moved, "results/robust_order_wmt24_en-de__segment.json")
        checks.append((_s["config"]["published_cell_reproduces_the_official_ranking"] is False,
                       "the segment mean is not the task's ranking", "does not reproduce",
                       str(_s["config"]["published_cell_reproduces_the_official_ranking"]),
                       "results/robust_order_wmt24_en-de__segment.json"))
        # The agreement between our cell and the official order cannot be evidence, because the
        # function that checks it charges the annotations with the same transcribed weights. That
        # is a property of the code, so it is read there, and the prose is held to it.
        _wb = (ROOT / "scripts/wmt_board.py").read_text()
        _oo = _wb[_wb.index("def official_order"):]
        checks.append(("w_wmt(" in _oo, "the ranking check reuses the transcribed weighting",
                       "w_wmt", "w_wmt" if "w_wmt(" in _oo else "some other function",
                       "scripts/wmt_board.py"))
        checks.append(("unit test on the weighting and not evidence about the data" in flat
                       and "agreement is evidence" not in flat,
                       "the reproduction is not offered as evidence", "not evidence",
                       "as stated" if "unit test on the weighting and not evidence about the data"
                       in flat else "claimed as evidence", "the manuscript"))

    # 10c-5. The rest of what check_coverage.py reported as unread in the body: figures a reader
    # meets in the introduction, the contributions and the two paragraphs that bound the survey.
    flat = re.sub(r"\s+", " ", whole)
    _hd = ROOT / "results/hydrogen_dispatch__clean_test.json"
    if _hd.exists():
        HD = json.loads(_hd.read_text())
        # the same two arms the appendix quotes as a swing of -0.374; the bullet states its size
        _arm = HD["banks"]["biotransformer"]["global_arms"]
        _sw = abs(_arm["all_explicit_completed"] - _arm["all_implicit"])
        m = re.search(r"the engine that applies it, moving by\s*up to \$([\d.]+)\$ with a "
                      r"preprocessing step", flat)
        checks.append((bool(m), "the contribution's engine figure parses", "present",
                       "matched" if m else "not matched", ""))
        # no "if m and ..." here: a value that cannot be found must fail, not be skipped
        check("the contribution's engine figure", m.group(1) if m else None, _sw,
              "results/hydrogen_dispatch__clean_test.json")

    _pd = ROOT / "results/places_decomposition.json"
    if _pd.exists():
        PDx = json.loads(_pd.read_text())
        m = re.search(r"The \$(\d+)\$ the title quotes is the weaker reading.*?strictest number has "
                      r"\$(\d+)\$, and one who wants what the conventions cost has \$(\d+)\$", flat)
        checks.append((bool(m), "the decomposition trailer parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("decomposition trailer, the title's number", m.group(1),
                  PDx["supported_when_every_cell_agrees_in_sign"], "results/places_decomposition.json")
            check("decomposition trailer, the strictest", m.group(2),
                  PDx["supported_when_every_cell_separates"], "results/places_decomposition.json")
            check("decomposition trailer, the grid's cost", m.group(3), PDx["lost_to_the_grid"],
                  "results/places_decomposition.json")

    # the one language pair scored twice, a year apart, which is the replication claim
    _e24 = ROOT / "results/robust_order_wmt24_esa.json"
    _e23 = ROOT / "results/robust_order_wmt23.json"
    if _e24.exists() and _e23.exists():
        E24 = json.loads(_e24.read_text())["boards"]
        E23 = json.loads(_e23.read_text())["boards"]
        m = re.search(r"\\textsc\{en-zh\} standing at \$([\d.]+)\$ in one edition and \$([\d.]+)\$ "
                      r"in the next", flat)
        checks.append((bool(m), "the replication sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            _a = E24.get("en-zh", {}).get("robustness")
            _b = next((v["robustness"] for k, v in E23.items() if k in ("eng-zho", "en-zh")), None)
            _hi, _lo = max(_a, _b), min(_a, _b)
            check("replication, the higher edition", max(float(m.group(1)), float(m.group(2))), _hi,
                  "results/robust_order_wmt24_esa.json / robust_order_wmt23.json")
            check("replication, the lower edition", min(float(m.group(1)), float(m.group(2))), _lo,
                  "results/robust_order_wmt24_esa.json / robust_order_wmt23.json")

    _pp = ROOT / "results/packing_predicts_order.json"
    if _pp.exists():
        PPo = json.loads(_pp.read_text())
        m = re.search(r"rather than from \$([\d,{}]+)\$ paired bootstraps", flat)
        check("the pairs the arithmetic replaces a bootstrap on",
              m and m.group(1).replace("{,}", ""), PPo["totals"]["n_pairs"],
              "results/packing_predicts_order.json")

    _ro = ROOT / "results/robust_order.json"
    if _ro.exists():
        RO = json.loads(_ro.read_text())["leaderboards"]
        m = re.search(r"seven systems on their \$([\d,{}]+)\$ common reactions", flat)
        check("the retrosynthesis group's reactions", m and m.group(1).replace("{,}", ""),
              RO["cluster0"]["n_items"], "results/robust_order.json")
        # the smallest separable margin against the largest move any criterion makes on that board
        import ast as _a2
        _acc = RO["cluster0"]["system_accuracy_by_cell"]
        _pub = _a2.literal_eval(RO["cluster0"]["published_cell"])
        _cells = [_a2.literal_eval(c) for c in next(iter(_acc.values()))]
        _sib = [c for c in _cells if c[1] == _pub[1] and c[0] != _pub[0]]
        _moves = []
        for _nm in RO["cluster0"]["pairs"]:
            _hi2, _lo2 = _nm.split(" over ")
            _d0 = _acc[_hi2][str(_pub)] - _acc[_lo2][str(_pub)]
            _moves += [abs((_acc[_hi2][str(_c)] - _acc[_lo2][str(_c)]) - _d0) for _c in _sib]
        m = re.search(r"largest amount any criterion moves a\s*margin on it is \$([\d.]+)\$", flat)
        check("the largest criterion movement on that board", m and m.group(1), max(_moves),
              "results/robust_order.json")

    _pa, _rpa = ROOT / "results/population_axis.json", ROOT / "results/retro_population_axis.json"
    if _pa.exists() and _rpa.exists():
        PA, RPA = json.loads(_pa.read_text()), json.loads(_rpa.read_text())
        m = re.search(r"changes an ordering in \$(\d+)\$ of \$(\d+)\$ comparisons and \$(\d+)\$ of "
                      r"\$(\d+)\$ on the metabolite split, and none of the \$(\d+)\$ interactions "
                      r"survives Holm at \$([\d.]+)\$", flat)
        checks.append((bool(m), "the population sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("population, retro reordered", m.group(1), RPA["reordered"],
                  "results/retro_population_axis.json")
            check("population, retro comparisons", m.group(2), RPA["comparisons"],
                  "results/retro_population_axis.json")
            check("population, metabolite reordered", m.group(3), PA["reordered"],
                  "results/population_axis.json")
            check("population, metabolite comparisons", m.group(4), PA["comparisons"],
                  "results/population_axis.json")
            check("population, the interactions tested", m.group(5),
                  RPA["comparisons"] + PA["comparisons"],
                  "results/retro_population_axis.json + population_axis.json")
            checks.append((abs(float(m.group(6)) - 0.05) < 1e-9,
                           "population, the level", "0.05", m.group(6), "the paper's alpha"))
            checks.append((RPA["holm_survivors"] == 0 and PA["holm_survivors"] == 0,
                           "population, nothing survives", "0 and 0",
                           f"{RPA['holm_survivors']} and {PA['holm_survivors']}",
                           "results/*population_axis.json"))

    _d2 = ROOT / "results/benchmark_report_depth2.json"
    if _d2.exists():
        D2 = json.loads(_d2.read_text())
        m = re.search(r"lifts the ceiling by \$\{\\sim\}([\d.]+)\$ at \$([\d.]+)\$ times the cost",
                      flat)
        checks.append((bool(m), "the depth-2 sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("depth-2, the lift", m.group(1), D2["lift_over_depth1"],
                  "results/benchmark_report_depth2.json")
            check("depth-2, the cost", m.group(2),
                  D2["depth2_ceiling_lower_bound"]["mean_candidates_per_substrate"]
                  / D2["depth1_ceiling"]["mean_candidates_per_substrate"],
                  "results/benchmark_report_depth2.json")

    _be = ROOT / "results/bank_engine_replication.json"
    if _be.exists():
        BE = json.loads(_be.read_text())
        m = re.search(r"is computed from the \$(\d+)\$ SMIRKS shipped with that tool", flat)
        check("the comparator's rule count", m and m.group(1),
              BE["banks"]["biotransformer"]["n_rules"], "results/bank_engine_replication.json")

    _ov = ROOT / "results/external_overlap_audit.json"
    if _ov.exists():
        OV = json.loads(_ov.read_text())
        m = re.search(r"The GLORYx set overlaps on \$(\d+)\$ of its \$(\d+)\$ drugs.*?reports GRAIL "
                      r"on the \$(\d+)\$ that do not overlap.*?overlaps on none of its \$(\d+)\$",
                      flat)
        checks.append((bool(m), "the overlap sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            _g = OV["GLORYx external set"]
            _s = OV["shared 150-substrate subset"]
            check("overlap, GLORYx drugs inside the split", m.group(1), _g["in_train_or_val"],
                  "results/external_overlap_audit.json")
            check("overlap, GLORYx size", m.group(2), _g["n"],
                  "results/external_overlap_audit.json")
            check("overlap, the unseen remainder", m.group(3), _g["n"] - _g["in_train_or_val"],
                  "results/external_overlap_audit.json")
            check("overlap, the shared subset", m.group(4), _s["n"],
                  "results/external_overlap_audit.json")
            checks.append((_s["in_train_or_val"] == 0, "overlap, the shared subset is clean",
                           "0", str(_s["in_train_or_val"]),
                           "results/external_overlap_audit.json"))

    # 10c-6. The docking control paragraph, which a check reached at exactly one number. Its
    # figures are the ones a reader meets while deciding whether to believe the instrument, and
    # they were the largest unread block in the body.
    _pb = ROOT / "results/robust_order_posebusters.json"
    _ax = ROOT / "results/robust_order_posebusters_astex.json"
    if _pb.exists():
        PB = json.loads(_pb.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"contests exactly two of the \$(\d+)\$ pairs", flat)
        check("docking, pairs the source's finding is read from", m and m.group(1), PB["n_pairs"],
              "results/robust_order_posebusters.json")
        m = re.search(r"read out of \$(\d+)\$ pairs by \$(\d+)\$ cells, so it is granted inside a "
                      r"family of \$(\d+)\$", flat)
        checks.append((bool(m), "the docking family sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("docking, pairs", m.group(1), PB["n_pairs"],
                  "results/robust_order_posebusters.json")
            check("docking, cells", m.group(2), PB["n_cells"],
                  "results/robust_order_posebusters.json")
            check("docking, family", m.group(3), PB["multiplicity"]["family_size"],
                  "results/robust_order_posebusters.json")
        m = re.search(r"Two exchanges out of \$(\d+)\$ looks", flat)
        check("docking, the family the exchanges came from", m and m.group(1),
              PB["multiplicity"]["family_size"], "results/robust_order_posebusters.json")
        m = re.search(r"Of its \$(\d+)\$ pairs, \$(\d+)\$ dominate, \$(\d+)\$ are contested and "
                      r"\$(\d+)\$ the grid leaves unresolved", flat)
        checks.append((bool(m), "the docking breakdown parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            for _i, (_lab, _val) in enumerate((("pairs", PB["n_pairs"]),
                                               ("dominating", PB["n_dominating"]),
                                               ("contested", PB["n_contested"]),
                                               ("unresolved", PB["n_unresolved"]))):
                check(f"docking breakdown, {_lab}", m.group(_i + 1), _val,
                      "results/robust_order_posebusters.json")
        m = re.search(r"on these \$([\d,{}]+)\$ items all ten such pairs dominate", flat)
        check("docking, items", m and m.group(1).replace("{,}", ""), PB["n_items"],
              "results/robust_order_posebusters.json")
        # the convention the paper says costs nothing, which was prose with no artifact behind it
        _fl = PB["config"].get("flatness_checks_included_instead", {})
        m = re.search(r"including them changes the published cell by \$([\d.]+)\$", flat)
        check("docking, what including the flatness checks costs", m and m.group(1),
              _fl.get("largest_change_to_a_system_in_the_published_cell"),
              "results/robust_order_posebusters.json")
        m = re.search(r"which leaves \$(\d+)\$ of the \$(\d+)\$ pairs dominating rather than "
                      r"\$(\d+)\$", flat)
        checks.append((bool(m), "the conservative-reading sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            _crit_only = PB["sub_grids"]["criteria only, at the published post-processing"]
            check("docking, dominating on the criteria alone", m.group(1),
                  _crit_only["n_dominating"], "results/robust_order_posebusters.json")
            check("docking, pairs in that reading", m.group(2), PB["n_pairs"],
                  "results/robust_order_posebusters.json")
            check("docking, dominating across the whole grid", m.group(3), PB["n_dominating"],
                  "results/robust_order_posebusters.json")
    if _ax.exists():
        AX = json.loads(_ax.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"run unchanged on the \$(\d+)\$ astex items, the same eight cells contest "
                      r".*?and leave \$(\d+)\$ of \$(\d+)\$\s*dominating", flat)
        checks.append((bool(m), "the astex sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("astex, items", m.group(1), AX["n_items"],
                  "results/robust_order_posebusters_astex.json")
            check("astex, dominating", m.group(2), AX["n_dominating"],
                  "results/robust_order_posebusters_astex.json")
            check("astex, pairs", m.group(3), AX["n_pairs"],
                  "results/robust_order_posebusters_astex.json")
            # "along with two other pairs" -- three contested in total on that board
            checks.append((AX["n_contested"] == 3, "astex, three pairs are contested",
                           "3", str(AX["n_contested"]),
                           "results/robust_order_posebusters_astex.json"))

    # 10c-7. Table 1, every row and every column. Six of its eight rows were bound one figure at
    # a time by checks written for the prose that quotes them; the two rows standing for nine and
    # eight boards were bound by nothing, which is how the same table in the appendix stopped
    # summing to its own totals.
    def _load_boards():
        out = {}
        for _fn, _key in (("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
                          ("robust_order_metabolite.json", None),
                          ("robust_order_posebusters.json", None),
                          ("robust_order_wmt24_en-de.json", None),
                          ("robust_order_wmt24_ja-zh.json", None)):
            _q = ROOT / "results" / _fn
            if _q.exists():
                _d = json.loads(_q.read_text())
                out[f"{_fn[:-5]}:{_key}" if _key else _fn[:-5]] = (_d["leaderboards"][_key]
                                                                   if _key else _d)
        for _src in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
            _q = ROOT / "results" / _src
            if _q.exists():
                for _lp, _b in json.loads(_q.read_text())["boards"].items():
                    out[f"{_src[:-5]}:{_lp}"] = _b
        return out

    _BD = _load_boards()
    if _BD:
        flat = re.sub(r"\s+", " ", whole)
        _ROWS = {
            "retrosynthesis, seven systems": ["robust_order:cluster0"],
            "retrosynthesis, three systems": ["robust_order:cluster1"],
            "metabolites, three methods": ["robust_order_metabolite"],
            "docking, seven programs": ["robust_order_posebusters"],
            "translation, nineteen systems": ["robust_order_wmt24_en-de"],
            "translation, fifteen systems": ["robust_order_wmt24_ja-zh"],
            "translation, nine further pairs": [k for k in _BD if "esa:" in k],
            "translation, the year before": [k for k in _BD if "wmt23:" in k],
        }
        _seen = 0
        for _label, _keys in _ROWS.items():
            _bs = [_BD[k] for k in _keys if k in _BD]
            if not _bs:
                continue
            # places | items (or a range) | dominate of pairs | contested (certified) |
            # unresolved | places supported
            _row = re.search(re.escape(_label) + r"\s*&\s*\$(\d+)\$\s*&\s*\$?([\d,{}]+)\$?"
                             r"(?:--\$([\d,{}]+)\$)?\s*&\s*\$?(\d+)\$? of \$([\d,{}]+)\$"
                             r"\s*&\s*\$?(\d+)\$? \((\d+)\)\s*&\s*\$?(\d+)\$?\s*&\s*"
                             r"\$(\d+)\$", flat)
            checks.append((bool(_row), f"table 1, the row for {_label}", "present",
                           "matched" if _row else "not matched", "results/robust_order_*.json"))
            if not _row:
                continue
            _seen += 1
            _g = [x.replace("{,}", "") if x else x for x in _row.groups()]
            check(f"table 1, {_label}, places", _g[0], sum(b["n_systems"] for b in _bs),
                  "results/robust_order_*.json")
            if len(_bs) == 1:
                check(f"table 1, {_label}, items", _g[1], _bs[0]["n_items"],
                      "results/robust_order_*.json")
            else:  # an aggregated row gives the range of item counts across its boards
                _it = [b["n_items"] for b in _bs]
                check(f"table 1, {_label}, smallest board", _g[1], min(_it),
                      "results/robust_order_*.json")
                check(f"table 1, {_label}, largest board", _g[2], max(_it),
                      "results/robust_order_*.json")
            check(f"table 1, {_label}, dominating", _g[3],
                  sum(b["n_dominating"] for b in _bs), "results/robust_order_*.json")
            check(f"table 1, {_label}, pairs", _g[4], sum(b["n_pairs"] for b in _bs),
                  "results/robust_order_*.json")
            check(f"table 1, {_label}, contested", _g[5], sum(b["n_contested"] for b in _bs),
                  "results/robust_order_*.json")
            check(f"table 1, {_label}, certified", _g[6],
                  sum(b["n_contested_after_correction"] for b in _bs),
                  "results/robust_order_*.json")
            check(f"table 1, {_label}, unresolved", _g[7], sum(b["n_unresolved"] for b in _bs),
                  "results/robust_order_*.json")
            check(f"table 1, {_label}, supported", _g[8],
                  sum(b["tiers_distinguished"] for b in _bs), "results/robust_order_*.json")
        # the caption quotes the union count beside the per-board one, and both are the artifact's
        _cap = re.search(r"over all twenty-three grids at once, \$(\d+)\$ of the\s*\$(\d+)\$ survive",
                         flat)
        _unj = ROOT / "results/union_multiplicity.json"
        if _cap and _unj.exists():
            _UNC = json.loads(_unj.read_text())["certified_pairs"]
            check("table 1 caption, surviving the union", _cap.group(1), _UNC["union"],
                  "results/union_multiplicity.json")
            check("table 1 caption, certified per board", _cap.group(2), _UNC["per_board"],
                  "results/union_multiplicity.json")
        else:
            checks.append((False, "table 1 caption, the union sentence", "present",
                           "not matched", "results/union_multiplicity.json"))
        checks.append((_seen == len(_ROWS), "table 1, every row is bound", str(len(_ROWS)),
                       str(_seen), "results/robust_order_*.json"))
        # the totals row is the paper's title, so it is held to the sum over every board and not
        # to the sum of the rows above it
        _all = list(_BD.values())
        _tot = re.search(r"all twenty-three & \$\\mathbf\{(\d+)\}\$ & --- & \$([\d,{}]+)\$ of "
                         r"\$([\d,{}]+)\$ & \$(\d+)\$ \((\d+)\) & \$(\d+)\$ & "
                         r"\$\\mathbf\{(\d+)\}\$", flat)
        checks.append((bool(_tot), "table 1, the totals row is present", "present",
                       "matched" if _tot else "not matched", "results/robust_order_*.json"))
        if _tot:
            for _i, (_lab, _val) in enumerate((
                    ("places", sum(b["n_systems"] for b in _all)),
                    ("dominating", sum(b["n_dominating"] for b in _all)),
                    ("pairs", sum(b["n_pairs"] for b in _all)),
                    ("contested", sum(b["n_contested"] for b in _all)),
                    ("certified", sum(b["n_contested_after_correction"] for b in _all)),
                    ("unresolved", sum(b["n_unresolved"] for b in _all)),
                    ("supported", sum(b["tiers_distinguished"] for b in _all)))):
                check(f"table 1, totals, {_lab}", _tot.group(_i + 1).replace("{,}", ""), _val,
                      "results/robust_order_*.json")

    # 10c-4. The two figures the body repeats most often: how many leaderboards there are, and
    # the level every correction is taken at. Both are constants, which is exactly why nothing was
    # checking them -- and a constant that changes silently is the cheapest error to ship.
    flat = re.sub(r"\s+", " ", whole)
    _nb = len(_BD)
    # "$11$ boards it covers" is the translation task's own clustering, not this paper's survey,
    # and it is checked against that artifact rather than against the board count
    _saidb = [x for x in re.findall(r"\$(\d+)\$ (?:leaderboards|boards)(?! it covers)", flat)]
    checks.append((len(_saidb) >= 3 and all(int(x) == _nb for x in _saidb),
                   "every count of leaderboards is the number there are", str(_nb),
                   ", ".join(sorted(set(_saidb))) or "none found", "results/robust_order_*.json"))
    _alpha = re.findall(r"Holm at \$?(?:\\alpha=)?\$?([\d.]+)\$", flat)
    _ro_alpha = float(re.search(r"^ALPHA\s*=\s*([\d.]+)",
                                (ROOT / "scripts/robust_order.py").read_text(),
                                re.M).group(1))
    checks.append((bool(_alpha) and all(abs(float(x) - _ro_alpha) < 1e-12 for x in _alpha),
                   "every level quoted is the level the code corrects at", str(_ro_alpha),
                   ", ".join(sorted(set(_alpha))) or "none found", "scripts/robust_order.py"))

    # 10c-8. The share's median across the twenty-three boards. It was printed as 0.85 with no
    # check while the figure drawn from the same artifacts labelled its own dashed line 0.83 --
    # a disagreement inside one document, and the figure was right. The median is recomputed here
    # over the same boards the survey counts.
    _sh = []
    for _fn, _key in (("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
                      ("robust_order_metabolite.json", None),
                      ("robust_order_posebusters.json", None),
                      ("robust_order_wmt24_en-de.json", None),
                      ("robust_order_wmt24_ja-zh.json", None)):
        _q = ROOT / "results" / _fn
        if _q.exists():
            _d = json.loads(_q.read_text())
            _sh.append((_d["leaderboards"][_key] if _key else _d)["robustness"])
    for _src in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
        _q = ROOT / "results" / _src
        if _q.exists():
            _sh += [_b["robustness"] for _b in json.loads(_q.read_text())["boards"].values()]
    if _sh:
        import statistics as _st
        flat = re.sub(r"\s+", " ", whole)
        checks.append((len(_sh) == 23, "the share median is taken over every board", "23",
                       str(len(_sh)), "results/robust_order_*.json"))
        _med = _st.median(_sh)
        m = re.search(r"none of three to nine tenths with a median of \$([\d.]+)\$", flat)
        check("the share median", m and m.group(1), _med, "results/robust_order_*.json")
        checks.append((min(_sh) == 0.0 and 0.9 <= max(_sh) < 1.0,
                       "the range the prose gives in words", "0 to nine tenths",
                       f"{min(_sh)} to {max(_sh)}", "results/robust_order_*.json"))
        # and the figure's own label, which is generated from the same numbers
        _fig = (ROOT / "paper/app/share_figure.tex")
        if _fig.exists():
            _fm = re.search(r"median \$([\d.]+)\$", _fig.read_text())
            check("the figure's median label", _fm and _fm.group(1), _med,
                  "paper/app/share_figure.tex")

    # 10c-9b. The shape-free check on the boards whose scores are continuous.
    pcj = ROOT / "results/permutation_check.json"
    if pcj.exists():
        PC = json.loads(pcj.read_text())
        flat = re.sub(r"\s+", " ", whole)
        m = re.search(r"cutoff, \$(\d+)\$ tests of which \$(\d+)\$ are certified, it\s*never returns "
                      r"a larger \$p\$ than the analytic one: the ratio runs from \$([\d.]+)\$ to "
                      r"\$([\d.]+)\$", flat)
        checks.append((bool(m), "the permutation paragraph parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("permutation, tests checked", m.group(1), PC["n_tests_checked"],
                  "results/permutation_check.json")
            check("permutation, certified among them", m.group(2), PC["n_certified_checked"],
                  "results/permutation_check.json")
            _rat = [r["permutation_p"] / r["analytic_p"] for r in PC["tests"] if r["analytic_p"]]
            check("permutation, the smallest ratio", m.group(3), min(_rat),
                  "results/permutation_check.json")
            check("permutation, the largest ratio", m.group(4), max(_rat),
                  "results/permutation_check.json")
            checks.append((max(_rat) <= 1.0,
                           "the analytic p is never the smaller of the two", "never above 1",
                           f"{max(_rat):.4f}", "results/permutation_check.json"))
        # the two claims the paragraph rests on: nothing certified is lost, and one pair is gained
        _lost = [r for r in PC["tests"]
                 if r["certified_analytic"] and not r["certified_permutation"]]
        _gained = {(r["board"], r["pair"]) for r in PC["tests"]
                   if r["certified_permutation"] and not r["certified_analytic"]}
        checks.append((not _lost, "no certified reversal is lost to the permutation test", "0",
                       str(len(_lost)), "results/permutation_check.json"))
        checks.append((len(_gained) == 1 and all(b == "en-de" for b, _ in _gained),
                       "one further pair on the nineteen-system board would be certified",
                       "1 on en-de", str(sorted(_gained)), "results/permutation_check.json"))
        m = re.search(r"On the \$(\d+)\$ tests whose tail a resample can\s*resolve at all, the "
                      r"saddlepoint and \$([\d,{}]+)\$ draws", flat)
        checks.append((bool(m), "the resample-validation sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("permutation, tests a resample can resolve", m.group(1),
                  PC["n_validated_against_a_resample"], "results/permutation_check.json")
            check("permutation, the draw count", m.group(2).replace("{,}", ""),
                  PC["config"]["draws"], "results/permutation_check.json")

    # 10c-9. The correction across boards. The paragraph's whole point is that a per-board Holm
    # is not the family the paper's claim ranges over, so every one of its numbers is bound, and
    # the reproduction of the quoted 23 is bound too: if the union script stopped agreeing with
    # the artifacts under the per-board reading it would be measuring something else.
    un = ROOT / "results/union_multiplicity.json"
    if un.exists():
        UN = json.loads(un.read_text())
        flat = re.sub(r"\s+", " ", whole)
        checks.append((UN["certified_pairs"]["per_board"]
                       == UN["certified_pairs"]["quoted_in_the_paper"],
                       "the union script reproduces the per-board count",
                       str(UN["certified_pairs"]["quoted_in_the_paper"]),
                       str(UN["certified_pairs"]["per_board"]),
                       "results/union_multiplicity.json"))
        m = re.search(r"union of the grids: \$([\d,{}]+)\$ cell-level tests", flat)
        check("union, family size", m and m.group(1).replace("{,}", ""), UN["union_family_size"],
              "results/union_multiplicity.json")
        m = re.search(r"rejects \$([\d,{}]+)\$ of them at a cutoff of \$p=([\d.]+)"
                      r"\\times10\^\{-(\d+)\}\$", flat)
        checks.append((bool(m), "the union sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("union, tests rejected", m.group(1).replace("{,}", ""),
                  UN["union_tests_rejected"], "results/union_multiplicity.json")
            # an absolute tolerance is meaningless at 1e-6: every wrong exponent is within it,
            # and moving the printed exponent by one was not caught until this was relative
            _cut = float(m.group(2)) * 10 ** (-int(m.group(3)))
            checks.append((abs(_cut - UN["union_cutoff_p"]) <= 0.01 * UN["union_cutoff_p"],
                           "union, cutoff", f"{_cut:.3g}", f"{UN['union_cutoff_p']:.3g}",
                           "results/union_multiplicity.json"))
        m = re.search(r"certified pairs falls from \$(\d+)\$ to \$(\d+)\$", flat)
        checks.append((bool(m), "the union fall parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("union, before", m.group(1), UN["certified_pairs"]["per_board"],
                  "results/union_multiplicity.json")
            check("union, after", m.group(2), UN["certified_pairs"]["union"],
                  "results/union_multiplicity.json")
        # the abstract and the contributions quote the same two numbers; both are bound here, so
        # a correction that moves cannot be corrected in the appendix alone
        m2 = re.search(r"survive\s*correction, under each board's own family and under a single "
                       r"correction over all \$23\$ grids at\s*once", flat)
        S2 = UN["certified_pairs_the_published_cell_had_separated"]
        checks.append((bool(m2) and S2["per_board"] == S2["union"],
                       "the abstract says the count holds under both corrections it names",
                       f"{S2['per_board']} under each", f"{S2['per_board']} and {S2['union']}",
                       "results/union_multiplicity.json"))
        m2 = re.search(r"and \$(\d+)\$ of the \$(\d+)\$ certified reversals of every kind surviving "
                       r"one correction over the union", flat)
        checks.append((bool(m2), "the contribution states the union correction", "present",
                       "matched" if m2 else "not matched", ""))
        if m2:
            check("union, the contribution's survivors", m2.group(1),
                  UN["certified_pairs"]["union"], "results/union_multiplicity.json")
            check("union, the contribution's total", m2.group(2),
                  UN["certified_pairs"]["per_board"], "results/union_multiplicity.json")
        m = re.search(r"at\s*\$\\alpha/23\$ within each grid, gives \$(\d+)\$", flat)
        check("union, the alpha split", m and m.group(1), UN["certified_pairs"]["alpha_split"],
              "results/union_multiplicity.json")
        m = re.search(r"all \$(\d+)\$ certified reversals of an ordering a published cell had "
                      r"separated survive every one of\s*the three", flat)
        checks.append((bool(m), "the separated-ordering survival parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            S = UN["certified_pairs_the_published_cell_had_separated"]
            check("union, separated reversals", m.group(1), S["per_board"],
                  "results/union_multiplicity.json")
            checks.append((S["per_board"] == S["alpha_split"] == S["union"],
                           "they survive all three corrections",
                           f"{S['per_board']} throughout",
                           f"{S['per_board']}, {S['alpha_split']}, {S['union']}",
                           "results/union_multiplicity.json"))
        # the sentence naming which boards pay, against the per-board table
        PBU = UN["per_board"]
        m = re.search(r"the nineteen-system translation board keeps \$(\d+)\$ of \$(\d+)\$, two more "
                      r"go on \\textsc\{en-hi\} and one on \\textsc\{zho-eng\}, and the "
                      r"retrosynthesis boards keep all (\w+)", flat)
        checks.append((bool(m), "the sentence naming who pays parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            _e = PBU["robust_order_wmt24_en-de"]
            check("union, en-de after", m.group(1), _e["pairs_union"],
                  "results/union_multiplicity.json")
            check("union, en-de before", m.group(2), _e["pairs_per_board"],
                  "results/union_multiplicity.json")
            _r = sum(PBU[k]["pairs_union"] for k in ("robust_order:cluster0",
                                                     "robust_order:cluster1"))
            _r0 = sum(PBU[k]["pairs_per_board"] for k in ("robust_order:cluster0",
                                                          "robust_order:cluster1"))
            _w = {"ten": 10, "eight": 8, "nine": 9, "eleven": 11}
            checks.append((_w.get(m.group(3), -1) == _r == _r0,
                           "union, the retrosynthesis boards keep all of theirs",
                           m.group(3), f"{_r} of {_r0}", "results/union_multiplicity.json"))
        # the paragraph names which of the six goes, so the naming is held to the board, not to
        # the total: the docking board must be the only one losing a separated-ordering reversal
        _survivors = {k for k, v in PBU.items() if v["separated_pairs_union"]}
        checks.append((_survivors == {"robust_order:cluster0", "robust_order:cluster1"},
                       "what survives every correction is retrosynthesis, the budget axis",
                       "the two retrosynthesis boards", ", ".join(sorted(_survivors)) or "none",
                       "results/union_multiplicity.json"))
        # the enumeration is meant to be exhaustive, so the boards it names must account for the
        # whole fall; a sentence that lists some of the losses reads as if it listed them all
        _named_loss = sum(PBU[k]["pairs_per_board"] - PBU[k]["pairs_union"]
                          for k in ("robust_order_wmt24_en-de", "robust_order_wmt24_esa:en-hi",
                                    "robust_order_wmt23:zho-eng")
                          if k in PBU)
        _all_loss = UN["certified_pairs"]["per_board"] - UN["certified_pairs"]["union"]
        checks.append((_named_loss == _all_loss,
                       "the boards the paragraph names are the whole fall", str(_all_loss),
                       str(_named_loss), "results/union_multiplicity.json"))
        # the paragraph now blames the family size and says the stopping rule cost nothing; that
        # is an empirical claim about the pooled vector, so it is computed rather than asserted
        import importlib.util as _iu3
        _sp3 = _iu3.spec_from_file_location("_um3", ROOT / "scripts/union_multiplicity.py")
        _um3 = _iu3.module_from_spec(_sp3)
        sys.modules["_um3"] = _um3
        _sp3.loader.exec_module(_um3)
        _BB = _um3.boards()
        _pool = sorted(pv for _, _b in _BB for pv in _b["multiplicity"]["p_values"])
        _M, _K = len(_pool), _um3.holm(_pool, 0.05)
        _late = sum(1 for _i, _pv in enumerate(_pool, 1)
                    if _i > _K and _pv <= 0.05 / (_M - _i + 1))
        checks.append((_late == 0 and "would have passed its own threshold" in flat,
                       "the stopping rule is said to have cost nothing, and did", "0",
                       str(_late), "results/union_multiplicity.json"))
        m2 = re.search(r"Those three boards are the whole fall\s*of (\w+)", flat)
        _w2 = {"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11}
        checks.append((bool(m2) and _w2.get(m2.group(1), -1) == _all_loss,
                       "the size of the fall the paragraph names",
                       str(_all_loss), m2.group(1) if m2 else "not stated",
                       "results/union_multiplicity.json"))
        # every pair the union removes is a reversal of something the board never established,
        # which is what lets the second claim stand unchanged under all three corrections
        _lost_sep = sum(v["separated_pairs_per_board"] - v["separated_pairs_union"]
                        for v in PBU.values())
        checks.append((_lost_sep == 0 and "never established" in flat,
                       "nothing the union removes was an established ordering", "0",
                       str(_lost_sep), "results/union_multiplicity.json"))
        checks.append((PBU["robust_order_posebusters"]["pairs_per_board"] == 0,
                       "the docking board certifies nothing before the union is reached", "0",
                       str(PBU["robust_order_posebusters"]["pairs_per_board"]),
                       "results/union_multiplicity.json"))

    # 10c-10. The shared task's own clustering, which is the external check on the tier count.
    oc = ROOT / "results/wmt_official_clusters.json"
    if oc.exists():
        OC = json.loads(oc.read_text())["boards"]
        flat = re.sub(r"\s+", " ", whole)
        # the crossed table: four rows, two columns, each cell against the artifact it comes from
        _CELLS = (("their test, their construction", "n_clusters"),
                  ("our test, their construction", "their_construction_our_test"),
                  ("their test, our construction", "our_construction_their_test"),
                  ("our test, our construction", "ours_own_cell_only"))
        _rows = 0
        for _label, _key in _CELLS:
            _r = re.search(re.escape(_label) + r"\s*&\s*\$(\d+)\$\s*&\s*\$(\d+)\$", flat)
            checks.append((bool(_r), f"clusters, the row for {_label}", "present",
                           "matched" if _r else "not matched",
                           "results/wmt_official_clusters.json"))
            if not _r:
                continue
            _rows += 1
            check(f"clusters, {_label}, nineteen systems", _r.group(1), OC["en-de"][_key],
                  "results/wmt_official_clusters.json")
            check(f"clusters, {_label}, fifteen systems", _r.group(2), OC["ja-zh"][_key],
                  "results/wmt_official_clusters.json")
        checks.append((_rows == len(_CELLS), "clusters, the crossed table is complete",
                       str(len(_CELLS)), str(_rows), "results/wmt_official_clusters.json"))
        # the two readings the prose draws from the table, in the direction it draws them
        S = json.loads((ROOT / "results/wmt_official_clusters.json").read_text())["summary"]
        m = re.search(r"run on all \$(\d+)\$ boards it covers", flat)
        check("clusters, the boards their code covers", m and m.group(1), S["n_boards"],
              "results/wmt_official_clusters.json")
        m = re.search(r"our test is stricter on\s*\$(\d+)\$ and never more permissive, and our "
                      r"construction is more generous on \$(\d+)\$ and never less", flat)
        checks.append((bool(m), "the two directions parse", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("clusters, boards where our test is stricter", m.group(1),
                  S["n_boards_our_test_is_stricter"], "results/wmt_official_clusters.json")
            check("clusters, boards where our construction is more generous", m.group(2),
                  S["n_boards_our_construction_is_more_generous"],
                  "results/wmt_official_clusters.json")
        checks.append((S["our_test_never_more_permissive"],
                       "at a fixed construction, our test is never the more permissive",
                       "never", str(S["our_test_never_more_permissive"]),
                       "results/wmt_official_clusters.json"))
        checks.append((S["our_construction_never_less_generous"],
                       "at a fixed test, our construction is never the less generous",
                       "never", str(S["our_construction_never_less_generous"]),
                       "results/wmt_official_clusters.json"))
        m = re.search(r"agree system for system on \$(\d+)\$ of the \$(\d+)\$\. On the other "
                      r"\w+ they\s*disagree about \$(\d+)\$ pairs in total, and our published cell "
                      r"separates none of them", flat)
        checks.append((bool(m), "the ordering-agreement sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("clusters, boards whose orders agree", m.group(1), S["n_orders_agreeing"],
                  "results/wmt_official_clusters.json")
            check("clusters, boards compared", m.group(2), S["n_boards"],
                  "results/wmt_official_clusters.json")
            check("clusters, pairs ordered differently", m.group(3),
                  S["pairs_ordered_differently"], "results/wmt_official_clusters.json")
            checks.append((S["of_those_our_cell_separates"] == 0,
                           "none of the discordant pairs is one our cell separates", "0",
                           str(S["of_those_our_cell_separates"]),
                           "results/wmt_official_clusters.json"))

    # 10d-0. The power behind the negative claim. Every figure in the paragraph is one of the
    # artifact's, and the two facts that keep the docking exception honest -- the flip is not
    # certified, and what is certified there needs the other axis -- are read from the board.
    cp = ROOT / "results/criterion_power.json"
    if cp.exists():
        CP = json.loads(cp.read_text())
        flat = re.sub(r"\s+", " ", whole)
        PB = CP["per_board"]
        m = re.search(r"the matching rule erases at most \$([\d.]+)\$ of a margin on the "
                      r"seven-system retrosynthesis board, \$([\d.]+)\$ on the three-system one and "
                      r"\$([\d.]+)\$ on the metabolite board, and on all three the median comparison "
                      r"erases \$([\d.]+)\$", flat)
        checks.append((bool(m), "the power sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("power, seven-system board", m.group(1),
                  PB["robust_order:cluster0"]["max_ratio"], "results/criterion_power.json")
            check("power, three-system board", m.group(2),
                  PB["robust_order:cluster1"]["max_ratio"], "results/criterion_power.json")
            check("power, metabolite board", m.group(3),
                  PB["robust_order_metabolite"]["max_ratio"], "results/criterion_power.json")
            check("power, median comparison", m.group(4),
                  CP["matching_axis"]["median_ratio_where_nothing_flips"],
                  "results/criterion_power.json")
        # the body states the same power in rounded form; both printings are bound
        m = re.search(r"the matching rule erases at most \$([\d.]+)\$ of a margin\s*the published "
                      r"cell had separated, and the median comparison erases none of it", flat)
        checks.append((bool(m), "the body states the size of the null", "present",
                       "matched" if m else "not matched", ""))
        if m:
            check("power in the body, largest share erased", m.group(1),
                  CP["matching_axis"]["largest_ratio_where_nothing_flips"],
                  "results/criterion_power.json")
            checks.append((CP["matching_axis"]["median_ratio_where_nothing_flips"] == 0.0,
                           "power in the body, the median erases nothing", "0.0",
                           str(CP["matching_axis"]["median_ratio_where_nothing_flips"]),
                           "results/criterion_power.json"))
        # the body names the three boards rather than counting them, and the naming has to be the
        # set the artifact found nothing flipping on
        _named = {"robust_order:cluster0", "robust_order:cluster1", "robust_order_metabolite"}
        _quiet = {k for k, v in CP["per_board"].items()
                  if k in CP["config"]["matching_axis_boards"] and v["n_flipping"] == 0}
        checks.append((_quiet == _named, "the body names the boards where nothing flips",
                       "the two retrosynthesis boards and the metabolite one",
                       ", ".join(sorted(_quiet)), "results/criterion_power.json"))
        m = re.search(r"movement would have to be \$([\d.]+)\$ times larger", flat)
        check("power, fold short on the closest board", m and m.group(1),
              CP["matching_axis"]["fold_short_on_that_board"], "results/criterion_power.json")
        m = re.search(r"over \\textsc\{unimol\}, \$([\d.]+)\$ of\s*it, at the published "
                      r"post-processing", flat)
        check("power, the docking flip", m and m.group(1),
              PB["robust_order_posebusters"]["closest_pair"]["ratio"],
              "results/criterion_power.json")
        # the reason the flip certifies nothing is the board's own correction, and after the test
        # changed nothing on that board is certified at all -- which the paragraph has to say
        _pbj = json.loads((ROOT / "results/robust_order_posebusters.json").read_text())
        _pt = {r["cell"]: r["p"] for r in _pbj["multiplicity"]["reversal_tests"]
               if r["pair"] == PB["robust_order_posebusters"]["closest_pair"]["pair"]}
        _pub_cell = _pbj["published_cell"]
        m = re.search(r"that test reaches \$p=([\d.]+)\$ against the board's threshold of "
                      r"\$([\d.]+)\\times10\^\{-(\d+)\}\$", flat)
        checks.append((bool(m), "the docking exception's reason parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            import ast as _a3
            _cc = str((PB["robust_order_posebusters"]["closest_pair"]["criterion"],
                       _a3.literal_eval(_pub_cell)[1]))
            check("docking exception, the p of the flip", m.group(1), _pt.get(_cc),
                  "results/robust_order_posebusters.json")
            import importlib.util as _iu2
            _sp2 = _iu2.spec_from_file_location("_um2", ROOT / "scripts/union_multiplicity.py")
            _um2 = _iu2.module_from_spec(_sp2)
            sys.modules["_um2"] = _um2
            _sp2.loader.exec_module(_um2)
            _thr = _um2.cutoff(_pbj["multiplicity"]["p_values"], 0.05)
            _saidthr = float(m.group(2)) * 10 ** (-int(m.group(3)))
            checks.append((abs(_saidthr - _thr) <= 0.05 * _thr, "docking exception, the threshold",
                           f"{_saidthr:.3g}", f"{_thr:.3g}",
                           "results/robust_order_posebusters.json"))
        checks.append((_pbj["n_contested_after_correction"] == 0
                       and "no reversal on that board survives" in flat,
                       "the docking board certifies nothing, and the paragraph says so", "0",
                       str(_pbj["n_contested_after_correction"]),
                       "results/robust_order_posebusters.json"))

    # 10d-e. The nine ESA boards, whose totals the appendix states in prose. They are the paper's
    # answer to "your share is one number from one table", so the aggregate is held to the run.
    ea = ROOT / "results/robust_order_wmt24_esa.json"
    if ea.exists():
        EA = json.loads(ea.read_text())
        BB = EA["boards"]
        flat = re.sub(r"\s+", " ", whole)
        me = re.search(r"Across the nine, \$(\d+)\$ of \$(\d+)\$ pairs dominate, \$(\d+)\$ are "
                       r"contested of which \$(\d+)\$ survive correction, and \$(\d+)\$ published "
                       r"places support \$(\d+)\$\. The share runs from \$([\d.]+)\$ to "
                       r"\$([\d.]+)\$ with a median of \$([\d.]+)\$", flat)
        checks.append((bool(me), "the nine-board aggregate sentence parses", "present",
                       "matched" if me else "not matched", ""))
        if me:
            src = str(ea.relative_to(ROOT))
            check("esa, dominate", me.group(1), sum(b["n_dominating"] for b in BB.values()), src)
            check("esa, pairs", me.group(2), EA["n_pairs_total"], src)
            check("esa, contested", me.group(3), sum(b["n_contested"] for b in BB.values()), src)
            check("esa, certified", me.group(4), EA["n_certified_total"], src)
            check("esa, places", me.group(5), EA["places_published"], src)
            check("esa, tiers", me.group(6), EA["places_supported"], src)
            check("esa, share low", me.group(7), round(EA["share_min"], 2), src)
            check("esa, share high", me.group(8), round(EA["share_max"], 2), src)
            check("esa, share median", me.group(9), round(EA["share_median"], 2), src)
            checks.append((len(BB) == 9, "there are nine boards", "9", str(len(BB)), src))

    # 10z. A structural check on the sources was attempted here and withdrawn. The defect it was
    # meant to catch is real -- four table rows ended in one backslash and the table collapsed in
    # the compiled PDF while the suite stayed green -- but the check as written disagreed with the
    # same logic run standalone on the same file, and a gate whose verdict cannot be reproduced by
    # hand is worse than none. The four rows were repaired and the compiled table read to confirm
    # it; the general check wants a LaTeX parse rather than a line scan, which is a separate job.

    # 10d-p. The gap between published and supported places, split by cause. The title puts the two
    # numbers side by side under a subtitle about undeclared choices, which invites the reading that
    # the choices cost the difference. Most of it is the benchmark's own power, and the paper says so.
    pdc = ROOT / "results/places_decomposition.json"
    if pdc.exists():
        PD = json.loads(pdc.read_text())
        flat = re.sub(r"\s+", " ", whole)
        # Six reviewers, one defect: the decomposition was printed three times with three different
        # sets of numbers and none matched the artifact, because this gate matched one printing.
        # Every printing is found by its own pattern and each is held; missing one now fails.
        printings_pd = []
        for _pat in (r"takes \$293\$ places to \$(\d+)\$[,:] (?:so )?\$?(\d+)\$? (?:go|ranks go) to a "
                     r"benchmark's own (?:power|resolving power)\s*before any convention is varied "
                     r"and \$(\d+)\$ more to the grid",
                     r"the \$293\$ published places support \$(\d+)\$: \$(\d+)\$ go before any "
                     r"convention is varied.*?the grid costs \$(\d+)\$ on top"):
            printings_pd += re.findall(_pat, flat)
        checks.append((len(printings_pd) >= 3, "every printing of the decomposition is found",
                       "3 or more", str(len(printings_pd)), "the manuscript"))
        for _i, (_o, _pw, _g) in enumerate(printings_pd, 1):
            check(f"decomposition printing {_i}, own cell", _o,
                  PD["supported_when_its_own_cell_separates"], "the manuscript")
            check(f"decomposition printing {_i}, to power", _pw,
                  PD["lost_before_any_choice_is_varied"], "the manuscript")
            check(f"decomposition printing {_i}, to grid", _g, PD["lost_to_the_grid"],
                  "the manuscript")
        mpd = re.search(r"the \$293\$ published places support \$(\d+)\$: \$(\d+)\$ go before any "
                        r"convention is varied.*?the grid costs \$(\d+)\$ on top", flat)
        checks.append((bool(mpd), "the places decomposition parses", "present",
                       "matched" if mpd else "not matched", ""))
        if mpd:
            src = str(pdc.relative_to(ROOT))
            check("places, own cell separates", mpd.group(1),
                  PD["supported_when_its_own_cell_separates"], src)
            check("places, lost before any choice", mpd.group(2),
                  PD["lost_before_any_choice_is_varied"], src)
            check("places, lost to the grid", mpd.group(3), PD["lost_to_the_grid"], src)
            checks.append((PD["lost_before_any_choice_is_varied"] > PD["lost_to_the_grid"],
                           "the paper does not attribute the gap mainly to the grid",
                           "power exceeds grid",
                           f"{PD['lost_before_any_choice_is_varied']} against "
                           f"{PD['lost_to_the_grid']}", src))

    # 10d-ax. Which axis certifies a reversal of an ordering the published cell had separated. The
    # paper now claims the matching criterion certifies none, which is a stronger and more falsifiable
    # sentence than naming the domains, so it is held to every board rather than to the ones checked.
    import ast as _ast
    _crit = _sep = 0
    _srcs = [("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
             ("robust_order_metabolite.json", None), ("robust_order_posebusters.json", None),
             ("robust_order_wmt24_en-de.json", None), ("robust_order_wmt24_ja-zh.json", None)]
    _bs = []
    for _fn, _k in _srcs:
        _q = ROOT / "results" / _fn
        if _q.exists():
            _d = json.loads(_q.read_text())
            _bs.append(_d["leaderboards"][_k] if _k else _d)
    for _fn in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
        _q = ROOT / "results" / _fn
        if _q.exists():
            _bs += list(json.loads(_q.read_text())["boards"].values())
    for _b in _bs:
        _pub = _ast.literal_eval(_b["published_cell"])
        for _n in _b.get("contested_after_correction", []):
            _v = _b["pairs"][_n]
            if not _v.get("resolved_in_the_published_cell"):
                continue
            _sep += 1
            _cells = [_ast.literal_eval(_c)
                      for _c in _v["cells_that_reverse_it_with_an_interval"]]
            if _cells and all(_c[0] != _pub[0] and _c[1] == _pub[1] for _c in _cells):
                _crit += 1
    if _bs:
        checks.append((_crit == 0,
                       "the matching criterion certifies no reversal of a separated ordering",
                       "0", str(_crit), "results/robust_order_*.json"))
        # and the abstract has to print that same number: the gate below asserts a property of
        # the artifacts, which is exactly the kind of gate that stayed green while the prose said
        # something else.
        _m_abs = re.search(r"Of the orderings a table did establish, \$(\d+)\$ are reversed by a "
                           r"declared choice and survive\s*correction", flat)
        checks.append((bool(_m_abs) and int(_m_abs.group(1)) == _sep,
                       "the abstract prints the certified-reversal count",
                       str(_sep), _m_abs.group(1) if _m_abs else "not found", "the manuscript"))
        checks.append((_sep == 5, "five certified reversals are of a separated ordering", "5",
                       str(_sep), "results/robust_order_*.json"))

    # 10d-w23. The previous edition's boards are a reconstruction, not the published cell, and the
    # distance between the two is stated rather than left for a reviewer to find. The filter is the
    # part that must be exact, and the artifact records whether it is.
    w23 = ROOT / "results/robust_order_wmt23.json"
    if w23.exists():
        A = json.loads(w23.read_text())["official_agreement"]
        flat = re.sub(r"\s+", " ", whole)
        # the caveat this checked is withdrawn: with the task's own preprocessing the boards
        # reproduce its ranking exactly, so what is held now is that reproduction
        checks.append((A["rating_counts_match_on_every_board"]
                       and A["pairs_ordered_differently"] == 0
                       and A["largest_score_gap"] < 1e-9,
                       "the previous edition's boards reproduce the published ranking",
                       "exact, no inversion",
                       f"gap {A['largest_score_gap']}, {A['pairs_ordered_differently']} inversions",
                       str(w23.relative_to(ROOT))))

    # 10d-r. The cross-edition replication. It is the paper's answer to "is the share a property of
    # the benchmark or of your grid", so both the aggregate and the spread it rests on are held.
    rr = ROOT / "results/robust_order_wmt23.json"
    if rr.exists():
        RR = json.loads(rr.read_text()); RB = RR["boards"]
        flat = re.sub(r"\s+", " ", whole)
        # the body states the same pair of numbers in its own words; both printings are held, since
        # a figure that appears twice and agrees once is the defect this paper is about
        mbody = re.search(r"it moves by \$([\d.]+)\$ on average and by \$([\d.]+)\$ at the widest", flat)
        checks.append((bool(mbody), "the body's replication sentence parses", "present",
                       "matched" if mbody else "not matched", ""))
        mrr = re.search(r"moves by \$([\d.]+)\$ on average and by as much as \$([\d.]+)\$; "
                        r"\\textsc\{en-zh\} is at \$([\d.]+)\$ in one edition and \$([\d.]+)\$ in "
                        r"the next\..*?Across the eight, \$(\d+)\$ of \$(\d+)\$ pairs dominate, "
                        r"\$(\d+)\$ are contested of which \$(\d+)\$ survive correction, and "
                        r"\$(\d+)\$ published places support \$(\d+)\$", flat)
        checks.append((bool(mrr), "the replication sentence parses", "present",
                       "matched" if mrr else "not matched", ""))
        if mrr:
            src = str(rr.relative_to(ROOT))
            E24 = json.loads((ROOT / "results/robust_order_wmt24_esa.json").read_text())["boards"]
            M24 = {"en-de": json.loads((ROOT / "results/robust_order_wmt24_en-de.json").read_text())}
            ISO = {"eng-deu": "en-de", "eng-ces": "en-cs", "eng-zho": "en-zh",
                   "eng-jpn": "en-ja", "ces-ukr": "cs-uk"}
            deltas = []
            for k23, k24 in ISO.items():
                b24 = M24.get(k24) or E24.get(k24)
                if k23 in RB and b24:
                    deltas.append(abs(b24["robustness"] - RB[k23]["robustness"]))
            check("replication, mean shift", mrr.group(1),
                  round(sum(deltas) / max(len(deltas), 1), 3), src)
            check("replication, largest shift", mrr.group(2), round(max(deltas), 3), src)
            if mbody:
                check("replication in the body, mean shift", mbody.group(1),
                      round(sum(deltas) / max(len(deltas), 1), 3), src)
                check("replication in the body, largest shift", mbody.group(2),
                      round(max(deltas), 3), src)
            check("replication, en-zh before", mrr.group(3),
                  round(RB["eng-zho"]["robustness"], 3), src)
            check("replication, en-zh after", mrr.group(4),
                  round(E24["en-zh"]["robustness"], 3), src)
            check("replication, dominate", mrr.group(5),
                  sum(b["n_dominating"] for b in RB.values()), src)
            check("replication, pairs", mrr.group(6), RR["n_pairs_total"], src)
            check("replication, contested", mrr.group(7),
                  sum(b["n_contested"] for b in RB.values()), src)
            check("replication, certified", mrr.group(8), RR["n_certified_total"], src)
            check("replication, places", mrr.group(9), RR["places_published"], src)
            check("replication, tiers", mrr.group(10), RR["places_supported"], src)
            checks.append((len(deltas) == 5, "five pairs are scored in both editions", "5",
                           str(len(deltas)), src))

    # 10d. the packing measurement: the empirical half of the reordering condition
    pk = ROOT / "results/packing_vs_differential.json"
    if pk.exists():
        P = json.loads(pk.read_text())
        flat = re.sub(r"\s+", " ", whole)
        t = P["totals"]
        m = re.search(r"\$([\d,{}]+)\$ method-pair by criterion-pair comparisons across five domains", flat)
        check("packing, comparisons", m and m.group(1).replace("{,}", ""), t["comparisons"])
        m = re.search(r"exceeds the larger margin in \$(\d+)\$ of the \$([\d,{}]+)\$", flat)
        check("packing, move exceeds the larger margin", m and m.group(1),
              t["closer_than_the_move"])
        check("packing, denominator", m and m.group(2).replace("{,}", ""), t["comparisons"])
        # the equivalence is the claim, so the gate refuses a sentence that leaves either
        # side of it unstated: every flagged comparison exchanges and every exchange is flagged
        checks.append((t["closer_than_the_move"] == t["exchanged"],
                       "the condition and the exchange are the same set",
                       str(t["exchanged"]), str(t["closer_than_the_move"]),
                       str(pk.relative_to(ROOT))))
        m = re.search(r"flags \$([\d,{}]+)\$ instead.*?all \$([\d,{}]+)\$\s*extra flags are "
                      r"movements that widen", flat)
        check("packing, one-sided flags", m and m.group(1).replace("{,}", ""),
              P["screening_test"]["one_sided_for_comparison"]["flagged"])
        check("packing, one-sided false alarms", m and m.group(2).replace("{,}", ""),
              P["screening_test"]["one_sided_for_comparison"]["not_exchanged_and_flagged"])
        # the paper says the other naming order gives a materially different count, which is the
        # reason it rests on nothing; that number is not in any artifact, so it is recomputed here
        m2 = re.search(r"taking the other gives \$([\d,{}]+)\$", flat)
        if m2:
            import itertools as _it
            import importlib.util as _iu
            _sp = _iu.spec_from_file_location("_pvd", ROOT / "scripts/packing_vs_differential.py")
            _pvd = _iu.module_from_spec(_sp)
            sys.modules["_pvd"] = _pvd
            _sp.loader.exec_module(_pvd)
            _n = 0
            for _dom, _sc in _pvd.leaderboards().values():
                _ms = sorted(_sc)
                _cs = sorted(set.intersection(*(set(_sc[x]) for x in _ms)))
                for _a, _b in _it.combinations(_ms, 2):
                    for _c1, _c2 in _it.combinations(_cs, 2):
                        _d1 = _sc[_a][_c1] - _sc[_b][_c1]
                        _d2 = _sc[_a][_c2] - _sc[_b][_c2]
                        if _d1 and abs(_d2 - _d1) > abs(_d2):
                            _n += 1
            check("packing, the other naming order", m2.group(1).replace("{,}", ""), _n)
        # Every row of the printed table, not the four that happened to have a helper. Two rows
        # aggregate nine and eight boards, and those were the ones that went stale when the
        # translation boards were reoriented: the row said 4,132 comparisons and 337 exchanges
        # against 4,131 and 346, and the table stopped summing to its own totals.
        GROUPS = {"docking, PoseBusters": ["docking, PoseBusters"],
                  "generation, MOSES": ["molecular generation, MOSES"],
                  "metabolites, GLORYx": ["metabolites, external GLORYx"],
                  "retrosynthesis, seven": ["retrosynthesis, seven-system group"],
                  "retrosynthesis, three": ["retrosynthesis, three-system group"],
                  "translation, en-de": ["translation, WMT24 en-de"],
                  "translation, ja-zh": ["translation, WMT24 ja-zh"],
                  "translation, nine more": [k for k in P["per_leaderboard"] if "(esa)" in k],
                  "translation, 2023, eight": [k for k in P["per_leaderboard"] if "WMT23" in k]}
        printed_comps, printed_exch, n_rows = 0, 0, 0
        for shown, keys in GROUPS.items():
            vs = [P["per_leaderboard"][k] for k in keys if k in P["per_leaderboard"]]
            if not vs:
                continue
            row = re.search(re.escape(shown) + r" & ([\d-]+) & (\d+) & (\S+) & (\S+) & \$(\d+)\$",
                            flat)
            checks.append((bool(row), f"packing, {shown} is a row of the table",
                           "present", "found" if row else "missing", str(pk.relative_to(ROOT))))
            if not row:
                continue
            n_rows += 1
            printed_comps += int(row.group(2))
            printed_exch += int(row.group(5))
            if len(vs) == 1:
                check(f"packing, {shown} methods", row.group(1), vs[0]["methods"])
                check(f"packing, {shown} median gap", row.group(3).strip("$"),
                      vs[0]["median_gap"])
                check(f"packing, {shown} median move", row.group(4).strip("$"),
                      vs[0]["median_differential"])
            else:  # an aggregated row states the range of method counts and no median
                lo, hi = min(v["methods"] for v in vs), max(v["methods"] for v in vs)
                # a range is not a number, and close() only compares numbers
                checks.append((row.group(1) == f"{lo}--{hi}", f"packing, {shown} method range",
                               row.group(1), f"{lo}--{hi}", str(pk.relative_to(ROOT))))
                checks.append((row.group(3) == "---" and row.group(4) == "---",
                               f"packing, {shown} prints no median for a group of boards",
                               "--- ---", f"{row.group(3)} {row.group(4)}",
                               str(pk.relative_to(ROOT))))
            check(f"packing, {shown} comparisons", row.group(2),
                  sum(v["comparisons"] for v in vs))
            check(f"packing, {shown} move over gap", row.group(5),
                  sum(v["closer_than_the_move"] for v in vs))
        checks.append((n_rows == len(GROUPS), "packing, every leaderboard is a row",
                       str(len(GROUPS)), str(n_rows), str(pk.relative_to(ROOT))))
        # and the rows have to add up to the total the prose quotes, which is the arithmetic a
        # reader does and no gate did
        check("packing, the rows sum to the comparisons quoted", printed_comps, t["comparisons"])
        check("packing, the rows sum to the exchanges quoted", printed_exch,
              t["closer_than_the_move"])
        m = re.search(r"the one-sided form plus a check on the sign recovers\s*the same "
                      r"\$([\d,{}]+)\$", flat)
        check("packing, the sign check recovers the exchanges", m and m.group(1).replace("{,}", ""),
              P["screening_test"]["one_sided_for_comparison"]["exchanged_and_flagged"])

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
                 r"Of its \$448\$ testable cells, \$\d+\$ have intervals excluding zero and "
                 r"\$(\d+)\$ survive Holm", len(L["holm_survivors"])),
                ("intervals excluding zero, main body",
                 r"Of its \$448\$ testable cells, \$(\d+)\$ have intervals excluding zero",
                 len(L["certified_interactions"])),
                ("pairs exchanging at top-1", r"five of twenty-one pairs exchange",
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
        m = re.search(r"the engine term as the two programs stand, \$\+([\d.]+)\$", flat)
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
        m = re.search(r"[Aa]cross six\s*libraries and \$([\d,{}]+)\$ templates", flat)
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
            # the sentence moved to the appendix when the body was cut to nine pages; the gate
            # follows the claim wherever it is made and does not require it to be made
            m = re.search(r"the \$(\d+)\$ copied verbatim carry the atom primitive not once, while\s*"
                          r"\$(\d+)\$ of the \$(\d+)\$ re-typed do, absent "
                          r"from all \$(\d+)\$ rules", flat)
            if m:
                check("transcription, copied verbatim", m.group(1), tr["identical"])
                check("transcription, re-typed carrying it", m.group(2), tr["rewritten_with_atom"])
                check("transcription, re-typed", m.group(3), tr["rewritten"])
                check("transcription, source rules", m.group(4), tr["source_rules"])
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
    mc = re.search(r"agree at a Jaccard of \$([\d.]+)\$ under strict \\textsc\{inchikey\}\s*"
                   r"matching and \$([\d.]+)\$ under the tautomer-aware default", flat)
    checks.append((bool(mc), "the curator-agreement sentence parses", "present",
                   "matched" if mc else "not matched", ""))
    if mc:
        j1, j2 = float(mc.group(1)), float(mc.group(2))
        share = (j2 - j1) / (1 - j1)
        band = "a third" if 0.25 <= share < 0.42 else ("a half" if 0.42 <= share < 0.6 else "most")
        # the wording may sit in either the sentence or the paragraph that follows it, so the gate
        # requires the right band to appear about disagreement and the wrong ones not to
        said = re.search(r"(a third|a half|most) of what reads as disagreement", flat)
        checks.append((bool(said) and said.group(1) == band, "the share notation explains",
                       said.group(1) if said else "not stated", band,
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
        mb = re.search(r"(four|three|five|six) of the six pairs change sign along that\s*sweep and "
                       r"(three|two|four|one) do so between margins that each separate from zero",
                       flat)
        checks.append((bool(mb), "the budget-reversal sentence parses", "present",
                       "matched" if mb else "not matched", ""))
        if mb:
            words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            check("pairs changing sign along the sweep", words[mb.group(1).lower()],
                  Q["n_pairs_changing_sign"], str(fm2.relative_to(ROOT)))
            check("sign changes certified at both ends", words[mb.group(2).lower()],
                  Q["n_certified_both_ends"], str(fm2.relative_to(ROOT)))
            H = Q.get("holm", {})
            if H:
                check("the declared family for the sweep", 54, H["family_size"],
                      str(fm2.relative_to(ROOT)))
                checks.append((H["n_sign_changes_certified_both_ends"] == words[mb.group(2).lower()],
                               "sign changes surviving Holm at both ends",
                               str(words[mb.group(2).lower()]),
                               str(H["n_sign_changes_certified_both_ends"]),
                               str(fm2.relative_to(ROOT))))
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
        # the budget result moved to the case study with the rest of the chemistry; the claim is
        # checked where it now lives rather than dropped
        said = ("MetaTox at $33.6$ candidates is last at $k\\le5$ and first from $k=15$")
        checks.append((said in re.sub(r"\s+", " ", whole),
                       "the case study says what the sweep shows",
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
        ma = re.search(r"A & \$152\$ shared, expanded, loop as it stood & ([\d.]+) .*?"
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

    # the appendix's arms table now carries the completed arm, and it must be the same number the
    # completed-loop run reports rather than a second measurement of the same thing
    cl2 = ROOT / "results/completed_loop_reach__clean_test.json"
    if cl2.exists():
        flat = re.sub(r"\s+", " ", whole)
        mac = re.search(r"A\$'\$ & \$152\$ shared, expanded, loop completed & ([\d.]+) ", flat)
        checks.append((bool(mac), "the completed arm row parses", "present",
                       "matched" if mac else "not matched", ""))
        if mac:
            check("arm A-prime, the completed loop", mac.group(1),
                  round(json.loads(cl2.read_text())["reach"]["completed"]["reach"], 3),
                  str(cl2.relative_to(ROOT)))

    # 10c-z. The tier count. It is the longest chain in the dominance order and therefore the same
    # measurement as the share of surviving pairs, read in the unit a leaderboard is quoted in.
    ro2 = ROOT / "results/robust_order.json"
    if ro2.exists():
        R2 = json.loads(ro2.read_text())["leaderboards"]
        flat = re.sub(r"\s+", " ", whole)
        mt = re.search(r"seven published places support \$(\d+)\$ tiers, resampling to "
                       r"\$\[(\d+),(\d+)\]\$; nineteen support \$(\d+)\$", flat)
        checks.append((bool(mt), "the tier sentence parses", "present",
                       "matched" if mt else "not matched", ""))
        if mt:
            src = str(ro2.relative_to(ROOT))
            check("tiers, seven-system", mt.group(1), R2["cluster0"]["tiers_distinguished"], src)
            check("tiers, seven-system lower", mt.group(2), R2["cluster0"]["tiers_ci95"][0], src)
            check("tiers, seven-system upper", mt.group(3), R2["cluster0"]["tiers_ci95"][1], src)
            RW = json.loads((ROOT / "results/robust_order_wmt24_en-de.json").read_text())
            check("tiers, nineteen-system", mt.group(4), RW["tiers_distinguished"],
                  "results/robust_order_wmt24_en-de.json")
            checks.append((R2["cluster0"]["tiers_distinguished"] < R2["cluster0"]["n_systems"],
                           "the seven-system table supports fewer tiers than places",
                           "fewer", f"{R2['cluster0']['tiers_distinguished']} of "
                           f"{R2['cluster0']['n_systems']}", src))

    # 10c-y. The ceiling's gap, split by whether the bank has the transformation type at all. The
    # two halves call for different work, so the split is the useful form of the number and every
    # part of it is held to the run, including that they sum to the uncovered total.
    cg = ROOT / "results/coverage_gap_types.json"
    if cg.exists():
        CG = json.loads(cg.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mcg = re.search(r"\$(\d+)\$ of the \$(\d+)\$ need a type the bank does not have at all, "
                        r"\$(\d+)\$ need a type it does have[^,]*, and \$(\d+)\$ cannot be typed",
                        flat)
        checks.append((bool(mcg), "the gap-decomposition sentence parses", "present",
                       "matched" if mcg else "not matched", ""))
        if mcg:
            src = str(cg.relative_to(ROOT))
            g = CG["gap"]
            check("gap, novel type", mcg.group(1), g["novel_type"], src)
            check("gap, uncovered total", mcg.group(2), CG["uncovered_pairs"], src)
            check("gap, known type", mcg.group(3), g["known_type"], src)
            check("gap, untypeable", mcg.group(4), g["untypeable"], src)
            checks.append((g["novel_type"] + g["known_type"] + g["untypeable"]
                           == CG["uncovered_pairs"],
                           "the three kinds sum to the uncovered set", "they sum",
                           f"{g['novel_type'] + g['known_type'] + g['untypeable']} against "
                           f"{CG['uncovered_pairs']}", src))

    # 10c-x. The scaffold baseline. If an untrained similarity ranks the same pool as well as the
    # trained filter, the filter has learned the scaffold and little else, and the controls have to
    # be in the gate too: reversing the order and shuffling it must both be certifiably worse, or
    # the comparison says nothing about what similarity contains.
    sb = ROOT / "results/scaffold_baseline.json"
    if sb.exists():
        SB = json.loads(sb.read_text())["arms"]
        flat = re.sub(r"\s+", " ", whole)
        msb = re.search(r"reaches \$([\d.]+)\$ against the filter's \$([\d.]+)\$, "
                        r"\$\+([\d.]+)\$\s*\$\[(-[\d.]+),\+([\d.]+)\]\$, while reversing that order "
                        r"costs \$([\d.]+)\$ and shuffling it \$([\d.]+)\$", flat)
        checks.append((bool(msb), "the scaffold-baseline sentence parses", "present",
                       "matched" if msb else "not matched", ""))
        if msb:
            src = str(sb.relative_to(ROOT))
            check("similarity ranking", msb.group(1), round(SB["similarity"]["recall"], 3), src)
            check("the filter's ranking", msb.group(2), round(SB["deployed"]["recall"], 3), src)
            check("the difference", msb.group(3), round(SB["similarity"]["vs_deployed"], 3), src)
            check("difference, lower", msb.group(4), round(SB["similarity"]["ci95"][0], 3), src)
            check("difference, upper", msb.group(5), round(SB["similarity"]["ci95"][1], 3), src)
            check("cost of reversing", msb.group(6),
                  round(abs(SB["dissimilarity"]["vs_deployed"]), 3), src)
            check("cost of shuffling", msb.group(7),
                  round(abs(SB["random"]["vs_deployed"]), 3), src)
            checks.append((not SB["similarity"]["certified"],
                           "the filter and similarity are not separable", "not certified",
                           str(SB["similarity"]["certified"]), src))
            checks.append((SB["dissimilarity"]["certified"] and SB["random"]["certified"],
                           "both controls are certifiably worse", "certified",
                           f"{SB['dissimilarity']['certified']} and {SB['random']['certified']}", src))

    # 10g. Composition. The claim is that a one-step predictor already contains part of what a
    # one-step evaluation scores as missed, and the whole weight of it rests on the control: a
    # random intermediate must recover nothing, or the effect is set size. The gate holds the
    # control to zero as well as the effect to its interval.
    cr = ROOT / "results/composition_recovery.json"
    if cr.exists():
        CR = json.loads(cr.read_text())
        P_, D_ = CR["per_method"], CR["differs_by_method"]
        flat = re.sub(r"\s+", " ", whole)
        mc2 = re.search(r"leaves \$(\d+)\$ substrates for GRAIL,\s*\$(\d+)\$ for MetaPredictor and "
                        r"\$(\d+)\$ for SyGMa", flat)
        checks.append((bool(mc2), "the composition-eligibility sentence parses", "present",
                       "matched" if mc2 else "not matched", ""))
        src = str(cr.relative_to(ROOT))
        if mc2:
            for i, m_ in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
                check(f"composition, eligible {m_}", mc2.group(i + 1),
                      P_[m_]["substrates_with_a_predicted_intermediate"], src)
        mc3 = re.search(r"recovers \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ for GRAIL, \$([\d.]+)\$ "
                        r"\$\[([\d.]+),([\d.]+)\]\$ for MetaPredictor and\s*\$([\d.]+)\$ "
                        r"\$\[([\d.]+),([\d.]+)\]\$ for SyGMa", flat)
        checks.append((bool(mc3), "the composition-recovery sentence parses", "present",
                       "matched" if mc3 else "not matched", ""))
        if mc3:
            for i, m_ in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
                check(f"composition, {m_}", mc3.group(1 + 3 * i), P_[m_]["over_the_random_control"], src)
                check(f"composition, {m_} lower", mc3.group(2 + 3 * i), P_[m_]["ci95"][0], src)
                check(f"composition, {m_} upper", mc3.group(3 + 3 * i), P_[m_]["ci95"][1], src)
        checks.append((all(P_[m_]["same_through_random_intermediates"] == 0.0 for m_ in P_),
                       "the random-intermediate control recovers nothing", "0.0000",
                       ", ".join(f"{P_[m_]['same_through_random_intermediates']}" for m_ in P_), src))
        checks.append((all(P_[m_]["separated"] for m_ in P_),
                       "every method's recovery is separated from its control", "all separated",
                       ", ".join(f"{m_} {P_[m_]['separated']}" for m_ in P_), src))
        mc4 = re.search(r"costs \$(\d+)\$ additional candidates for MetaPredictor and\s*\$(\d+)\$ "
                        r"for GRAIL", flat)
        if mc4:
            check("composition cost, MetaPredictor", mc4.group(1),
                  round(P_["MetaPredictor"]["candidates_added_per_reference_recovered"]), src)
            check("composition cost, GRAIL", mc4.group(2),
                  round(P_["GRAIL"]["candidates_added_per_reference_recovered"]), src)

    # 10h. The appendix that reports the dispatch pre-registration must not contradict the section
    # that states its outcome. Four reviewers found it doing exactly that, because a table row was
    # updated and the prose beside it was not, so the two are now tied to one artifact and to each
    # other: the residual quoted in the appendix must be the residual the run recorded, and the
    # count of dispatched templates must reconcile with the census that explains it.
    hy3 = ROOT / "results/hydrogen_dispatch__clean_test.json"
    if hy3.exists():
        H3 = json.loads(hy3.read_text())["banks"]
        flat = re.sub(r"\s+", " ", whole)
        mdp = re.search(r"It does not clear it\. It\s*loses by \$-([\d.]+)\$ "
                        r"\$\[-([\d.]+),-([\d.]+)\]\$", flat)
        checks.append((bool(mdp), "the appendix states the dispatch outcome", "present",
                       "matched" if mdp else "not matched", ""))
        if mdp and "biotransformer" in H3:
            src = str(hy3.relative_to(ROOT))
            B3 = H3["biotransformer"]
            check("appendix residual", mdp.group(1),
                  abs(B3["residual_convention_dependence"]), src)
            check("appendix residual, lower", mdp.group(2), abs(B3["residual_ci95"][0]), src)
            check("appendix residual, upper", mdp.group(3), abs(B3["residual_ci95"][1]), src)
            mrec = re.search(r"the \$(\d+)\$ that name a hydrogen\s*atom on their reactant side, and "
                             r"the \$(\d+)\$ recursive patterns", flat)
            if mrec:
                checks.append((int(mrec.group(1)) + int(mrec.group(2))
                               == B3["dispatched_to_expanded"],
                               "the dispatched count reconciles with the census",
                               str(B3["dispatched_to_expanded"]),
                               f"{mrec.group(1)} + {mrec.group(2)}", src))
        stale = re.findall(r"worth having for one bank in three|beats the best single setting", flat)
        checks.append((not stale, "no passage still reports the prediction as holding",
                       "none", f"{len(stale)} found", "the manuscript"))

    # 10i. The exposure distribution. An earlier wording said the widest leaderboard supplies none;
    # the artifact says it supplies four, and the one supplying none is neither the widest nor the
    # tightest. The claim is now that neither quantity decides alone, so the gate holds each of the
    # four shares and each median gap to the run.
    pk2 = ROOT / "results/packing_vs_differential.json"
    if pk2.exists():
        L = json.loads(pk2.read_text())["per_leaderboard"]
        flat = re.sub(r"\s+", " ", whole)
        # the body summarises the spread and the appendix enumerates it; the gate holds the
        # enumeration, because that is where the per-board numbers are actually asserted
        mdi = re.search(r"twenty-four leaderboards this can be asked of.*?run from \$(\d+)\\%\$ of "
                        r"their comparisons at risk to\s*\$([\d.]+)\\%\$", flat)
        # and the count itself, which is the twenty-three plus the one board the grid does not cover
        checks.append((len(L) == 24, "the exposure sentence's leaderboards", "24", str(len(L)),
                       "results/packing_vs_differential.json"))
        checks.append((bool(mdi), "the exposure-distribution sentence parses", "present",
                       "matched" if mdi else "not matched", ""))
        if mdi:
            src = str(pk2.relative_to(ROOT))
            def share(key):
                v = L[key]
                return round(100 * v["closer_than_the_move"] / v["comparisons"], 1)
            shares = {k: share(k) for k in L}
            check("lowest board share", mdi.group(1), min(shares.values()), src)
            check("highest board share", mdi.group(2), max(shares.values()), src)
            # and the claim the summary rests on: the widest-gapped board is not the least flagged
            widest = max(L, key=lambda k: L[k]["median_gap"])
            checks.append((shares[widest] > min(shares.values()),
                           "the widest-gapped board is not the least flagged",
                           f"above {min(shares.values())}%",
                           f"{widest} at {shares[widest]}%", src))
            checks.append((len(L) == 24, "the summary counts every board the artifact has",
                           "24 boards", f"{len(L)} boards", src))
            checks.append((L["retrosynthesis, three-system group"]["closer_than_the_move"] == 0,
                           "the three-system group is flagged on nothing", "0",
                           str(L["retrosynthesis, three-system group"]["closer_than_the_move"]), src))
            widest = max(L.values(), key=lambda v: v["median_gap"])
            checks.append((widest["closer_than_the_move"] > 0,
                           "no passage claims the widest leaderboard is exposure-free",
                           "it is not", str(widest["closer_than_the_move"]), src))

    # 10j. The contraction. Section 4's engine term depends on a step with two one-line
    # implementations that disagree, so the disagreement is measured and gated rather than assumed
    # away; an earlier version of this repository asserted they were equivalent on eight templates.
    cc = ROOT / "results/contraction_choice.json"
    if cc.exists():
        CC = json.loads(cc.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mcc = re.search(r"gives \$([\d,{}]+)\$ firings, of which the one-line version\s*yields "
                        r"\$([\d,{}]+)\$ products that sanitise and the other \$([\d,{}]+)\$; "
                        r"\$([\d,{}]+)\$ of the first carry an\s*unpaired electron and none of the "
                        r"second do; and of the \$([\d,{}]+)\$ both produce, \$([\d,{}]+)\$ differ",
                        flat)
        checks.append((bool(mcc), "the contraction sentence parses", "present",
                       "matched" if mcc else "not matched", ""))
        if mcc:
            src = str(cc.relative_to(ROOT))
            un = lambda s: s.replace("{,}", "")
            check("contraction, firings", un(mcc.group(1)), CC["firings"], src)
            check("contraction, one call parses", un(mcc.group(2)), CC["parseable"]["one_call"], src)
            check("contraction, restored parses", un(mcc.group(3)),
                  CC["parseable"]["restored"], src)
            check("contraction, radicals", un(mcc.group(4)),
                  CC["carrying_an_unpaired_electron"]["one_call"], src)
            check("contraction, both parse", un(mcc.group(5)), CC["both_parsed"], src)
            check("contraction, differ", un(mcc.group(6)), CC["both_parsed_and_differ"], src)
            checks.append((CC["carrying_an_unpaired_electron"]["restored"] == 0,
                           "the contraction this paper uses leaves no radicals", "0",
                           str(CC["carrying_an_unpaired_electron"]["restored"]), src))

    # 10k. Two capability claims replaced by measurements. "Cannot emit a stereocentre" is a claim
    # about a method; what the evidence supports is a claim about these predictions on this corpus,
    # so the gate counts them rather than trusting the adjective.
    sp2 = ROOT / "results/scored_predictions.json"
    sy2 = ROOT / "results/sygma_fulltest_predictions.json"
    flat = re.sub(r"\s+", " ", whole)
    msc = re.search(r"anywhere in the frozen predictions scored here, in \$([\d,{}]+)\$ and "
                    r"\$([\d,{}]+)\$ candidates scanned", flat)
    checks.append((bool(msc), "the stereocentre sentence is a count, not a capability claim",
                   "present", "matched" if msc else "not matched", "the manuscript"))
    bad_cap = re.findall(r"cannot emit a stereocentre", flat)
    checks.append((not bad_cap, "no passage claims a method cannot emit a stereocentre",
                   "none", f"{len(bad_cap)} found", "the manuscript"))
    if msc and sp2.exists() and sy2.exists():
        from rdkit import Chem as _C
        def scan(pairs, cap):
            n = ch = 0
            for cands in pairs:
                for y in cands[:15]:
                    m = _C.MolFromSmiles(y)
                    if m is None:
                        continue
                    n += 1
                    if _C.FindMolChiralCenters(m, includeUnassigned=False,
                                               useLegacyImplementation=False):
                        ch += 1
                if n > cap:
                    break
            return n, ch
        rows = json.loads(sp2.read_text())["rows"]
        n_g, ch_g = scan(([c["smiles"] for c in r["candidates"]] for r in rows), 4000)
        d_ = json.loads(sy2.read_text())
        d_ = d_.get("predictions", d_)
        n_s, ch_s = scan((v for v in d_.values() if isinstance(v, (list, tuple))), 4000)
        check("stereocentre scan, GRAIL", msc.group(1).replace("{,}", ""), n_g, "recomputed here")
        check("stereocentre scan, SyGMa", msc.group(2).replace("{,}", ""), n_s, "recomputed here")
        checks.append((ch_g == 0 and ch_s == 0, "neither emits an assigned stereocentre",
                       "0 and 0", f"{ch_g} and {ch_s}", "recomputed here"))

    # 10l. A merged artifact can outlive the code that produced part of it. Every bank the
    # manuscript quotes must have been measured by the same revision; a bank measured by an older
    # one may sit in the file but may not be cited, which is the mixed-population rule applied to
    # code rather than to substrates.
    hy4 = ROOT / "results/hydrogen_dispatch__clean_test.json"
    if hy4.exists():
        H4 = json.loads(hy4.read_text())["banks"]
        flat = re.sub(r"\s+", " ", whole)
        versions = {k: (v.get("measured_by") or {}).get("git_commit") for k, v in H4.items()}
        quoted = {"sygma_175": "SyGMa", "biotransformer": "BioTransformer", "grail_full": "ours"}
        # the row must be a dispatch row and not merely a line starting with the same word: the
        # census table elsewhere begins "ours & ... & $675$ &" too, and matching that made the
        # check fire on a bank the manuscript had already withdrawn
        cited = {k for k, shown in quoted.items()
                 if re.search(shown + r" & \$?[\\a-z{},0-9]+\$? & \$\d+\$ & [\d.]+ "
                              r"\$\[[\d.,]+\]\$ &", flat)}
        stale = sorted(k for k in cited if versions.get(k) is None
                       or (len({versions[c] for c in cited if versions.get(c)}) > 1
                           and versions[k] != max((versions[c] for c in cited if versions.get(c)),
                                                  key=lambda x: 0)))
        checks.append((bool(cited) and not any(versions.get(k) is None for k in cited),
                       "every bank the manuscript cites records the revision that measured it",
                       "all recorded",
                       ", ".join(f"{k}:{(versions.get(k) or 'none')[:8]}" for k in sorted(cited))
                       or "none cited",
                       str(hy4.relative_to(ROOT))))

    # 10m. The bounded null must quote the bound the design supports. The appendix derives two
    # detectable effects per family, the median test's and the blindest test's, and says the
    # family-wide statement needs the second; the main text quoted the first for one revision.
    flat_b = re.sub(r"\s+", " ", whole)
    mbd = re.search(r"in a design\s*whose blindest test would see \$([\d.]+)\$ and \$([\d.]+)\$ "
                    r"at \$80\\%\$ power", flat_b)
    checks.append((bool(mbd), "the bounded null quotes the blindest test", "present",
                   "matched" if mbd else "not matched", "the manuscript"))
    if mbd:
        mapp = re.search(r"the least sensitive \$([\d.]+)\$ and \$([\d.]+)\$", flat_b)
        checks.append((bool(mapp), "the appendix derives the same two numbers", "present",
                       "matched" if mapp else "not matched", "the manuscript"))
        if mapp:
            check("bounded null, first family", mbd.group(1), mapp.group(1), "the appendix")
            check("bounded null, second family", mbd.group(2), mapp.group(2), "the appendix")
        med = re.search(r"The median test in each family then detects \$([\d.]+)\$ and \$([\d.]+)\$",
                        flat_b)
        if med:
            checks.append((mbd.group(1) != med.group(1) and mbd.group(2) != med.group(2),
                           "the main text does not quote the median where the maximum is needed",
                           "the maximum", f"{mbd.group(1)},{mbd.group(2)} against median "
                           f"{med.group(1)},{med.group(2)}", "the appendix"))
        overclaim = re.findall(r"[Bb]oth are an order of magnitude above the criterion", flat_b)
        checks.append((not overclaim, "the null's margin over the criterion is not overstated",
                       "none", f"{len(overclaim)} found", "the manuscript"))

    # 10n. One arm, one printed value. Three loci quoted the same unexpanded arm as 0.387, 0.389
    # and 0.390, of which two were the same number rounded differently and the third was another
    # engine entirely; Section 4 attributed the third to the wrong program. The gate fixes the
    # rounding and requires the two arms to be distinguished wherever both appear.
    clr2 = ROOT / "results/completed_loop_reach__clean_test.json"
    if clr2.exists():
        C2 = json.loads(clr2.read_text())["reach"]
        flat_a = re.sub(r"\s+", " ", whole)
        implicit = round(C2["implicit"]["reach"], 3)
        printed = set(re.findall(r"unexpanded arm's \$([\d.]+)\$", flat_a))
        checks.append((printed <= {f"{implicit:.3f}"},
                       "the unexpanded arm is printed at one value everywhere",
                       f"{implicit:.3f}", ", ".join(sorted(printed)) or "not printed",
                       str(clr2.relative_to(ROOT))))
        msy = re.search(r"SyGMa's own program on\s*the same rules reaches \$([\d.]+)\$", flat_a)
        checks.append((bool(msy), "the section distinguishes our engine from SyGMa's own",
                       "present", "matched" if msy else "not matched", "the manuscript"))
        if msy:
            arm_b = re.search(r"B & \$152\$ shared, SyGMa's\s*& ([\d.]+) ", flat_a)
            if arm_b:
                check("SyGMa's own program on the shared rules", msy.group(1), arm_b.group(1),
                      "the appendix arm table")

    # 10f. The screen against the measurement. The paper's two instruments describe the same thing
    # from opposite ends, and the claim that one predicts the other is checked rather than asserted:
    # the screen sees only the published cell and the movement to each other cell, the outcome comes
    # from the full grid, and the arithmetic forbids a miss, so a miss here means one of the two
    # quantities is computed wrongly.
    pp = ROOT / "results/packing_predicts_order.json"
    if pp.exists():
        PP = json.loads(pp.read_text())["totals"]
        flat = re.sub(r"\s+", " ", whole)
        mpp = re.search(r"twenty-three tables that leaves \$(\d+)\$ of \$([\d,{}]+)\$ pairs "
                        r"failing to dominate, exactly the pairs some cell reverses", flat)
        checks.append((bool(mpp), "the screen-predicts-order sentence parses", "present",
                       "matched" if mpp else "not matched", ""))
        if mpp:
            src = str(pp.relative_to(ROOT))
            check("screen, flagged", mpp.group(1), PP["flagged"], src)
            check("screen, pairs", mpp.group(2).replace("{,}", ""), PP["n_pairs"], src)
            check("screen, flagged and failed", mpp.group(1), PP["flagged_and_failed"], src)
            checks.append((PP["failed_but_not_flagged"] == 0,
                           "the screen misses nothing, as the arithmetic requires", "0",
                           str(PP["failed_but_not_flagged"]), src))
            checks.append((PP["flagged_and_failed"] == PP["failed"],
                           "every pair the grid removes is flagged", str(PP["failed"]),
                           str(PP["flagged_and_failed"]), src))

    # 10e. The reference set as a graph. The appendix argues that the standard defence of a low
    # precision figure -- incomplete annotation -- is measurable in one component and small, so the
    # gate holds both halves: the corpus structure, which involves no model, and the share of each
    # method's scored-wrong output the corpus itself reaches, which is what makes it a null.
    rc = ROOT / "results/reference_closure.json"
    if rc.exists():
        RC = json.loads(rc.read_text())
        G = RC["corpus_graph"]
        flat = re.sub(r"\s+", " ", whole)
        mrc = re.search(r"It carries \$([\d,{}]+)\$ edges and \$([\d,{}]+)\$ two-step compositions, "
                        r"which resolve to \$([\d,{}]+)\$ distinct composed pairs\. The direct edge is "
                        r"annotated for \$(\d+)\$ of them\. The remaining \$([\d,{}]+)\$", flat)
        checks.append((bool(mrc), "the closure-structure sentence parses", "present",
                       "matched" if mrc else "not matched", ""))
        if mrc:
            src = str(rc.relative_to(ROOT))
            un = lambda s: s.replace("{,}", "")
            check("closure, edges", un(mrc.group(1)), G["edges"], src)
            check("closure, compositions", un(mrc.group(2)), G["two_step_compositions"], src)
            check("closure, composed pairs", un(mrc.group(3)), G["distinct_composed_pairs"], src)
            check("closure, annotated", mrc.group(4),
                  G["distinct_composed_pairs"] - G["composed_pairs_not_annotated"], src)
            check("closure, not annotated", un(mrc.group(5)), G["composed_pairs_not_annotated"], src)
        D = RC["scored_wrong_but_corpus_derivable"]
        msh = re.search(r"share the corpus itself reaches from that substrate in two steps is "
                        r"\$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ for\s*GRAIL, \$([\d.]+)\$ "
                        r"\$\[([\d.]+),([\d.]+)\]\$ for MetaPredictor and \$([\d.]+)\$ "
                        r"\$\[([\d.]+),([\d.]+)\]\$ for SyGMa", flat)
        checks.append((bool(msh), "the closure-share sentence parses", "present",
                       "matched" if msh else "not matched", ""))
        if msh:
            src = str(rc.relative_to(ROOT))
            for i, m_ in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
                check(f"closure share, {m_}", msh.group(1 + 3 * i),
                      D[m_]["share_of_wrong_output"], src)
                check(f"closure share, {m_} lower", msh.group(2 + 3 * i), D[m_]["ci95"][0], src)
                check(f"closure share, {m_} upper", msh.group(3 + 3 * i), D[m_]["ci95"][1], src)
            checks.append((not RC["ordering_changes_with_depth"],
                           "no ordering changes with the closure depth", "unchanged",
                           str(RC["ordering_changes_with_depth"]), src))
            checks.append((max(v["share_of_wrong_output"] for v in D.values()) < 0.01,
                           "the corpus-derivable share is below one percent for every method",
                           "under 0.01",
                           f"{max(v['share_of_wrong_output'] for v in D.values()):.4f}", src))

    # 10d. The packing condition as a screening test. The main text promotes it from a tally to a
    # rule a maintainer can run on their own table, so every rate it quotes is recomputed from the
    # artifact rather than restated, including the sensitivity that the arithmetic forces to one.
    pk = ROOT / "results/packing_vs_differential.json"
    if pk.exists():
        PK = json.loads(pk.read_text())
        S, T = PK["screening_test"], PK["totals"]
        # The identity is no longer sold as an empirical result, so there is no sentence here to
        # bind. What replaced it -- that the dominance verdicts need no pairwise statistic -- is
        # checked by the screen-predicts-order block above, which compares the two computations.
        checks.append((S["exact"], "the two-sided condition is exact, as an identity must be",
                       "no false alarm and no miss",
                       f"{S['not_exchanged_and_flagged']} false, {S['exchanged_and_missed']} missed",
                       str(pk.relative_to(ROOT))))

    # 10c-w. Reach under per-template dispatch, which the paper reports as the primitive, and the
    # worst case over global settings, which it reports as a diagnostic. Both range only over
    # settings someone might choose: the arm that expands a substrate and never contracts the
    # product is a defect by Section 4's own account, and taking a minimum or a baseline over it
    # would credit the instrument with repairing a bug. The gate refuses that arm outright.
    hy2 = ROOT / "results/hydrogen_dispatch__clean_test.json"
    if hy2.exists():
        H2 = json.loads(hy2.read_text())["banks"]
        flat = re.sub(r"\s+", " ", whole)
        mg = re.search(r"whose \$(\d+)\$ of \$(\d+)\$ templates make it\s*the mixed bank by that "
                       r"syntax, it \\emph\{loses\} to the better single setting by \$-([\d.]+)\$\s*"
                       r"\$\[-([\d.]+),-([\d.]+)\]\$\. It clears that setting only on our own bank, by "
                       r"\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$.*?\$([\d.]+)\$ against the "
                       r"\$([\d.]+)\$ its best single setting reaches", flat)
        checks.append((bool(mg), "the dispatch sentence parses", "present",
                       "matched" if mg else "not matched", ""))
        if mg and "biotransformer" in H2:
            B = H2["biotransformer"]
            src = str(hy2.relative_to(ROOT))
            if B["global_arms"].get("all_explicit_completed") is None:
                checks.append((False, "BioTransformer has a completed-loop arm", "present",
                               "absent; rerun hydrogen_dispatch", src))
            else:
                legit = {k: v for k, v in B["global_arms"].items()
                         if v is not None and k != "all_explicit"}
                check("templates dispatched", mg.group(1), B["dispatched_to_expanded"], src)
                check("templates in the bank", mg.group(2), B["n_rules"], src)
                check("dispatch residual", mg.group(3),
                      abs(B["residual_convention_dependence"]), src)
                check("residual, lower", mg.group(4), abs(B["residual_ci95"][0]), src)
                check("residual, upper", mg.group(5), abs(B["residual_ci95"][1]), src)
                G = H2.get("grail_full", {})
                if (G.get("measured_by") or {}).get("git_commit"):
                    check("our own bank, residual", mg.group(6),
                          G["residual_convention_dependence"], src)
                    check("our own bank, lower", mg.group(7), G["residual_ci95"][0], src)
                    check("our own bank, upper", mg.group(8), G["residual_ci95"][1], src)
                    checks.append((G["residual_ci95"][0] > 0,
                                   "dispatch clears the best global setting on our own bank",
                                   "interval above zero", str(G["residual_ci95"]), src))
                check("guaranteed reach", mg.group(9), round(min(legit.values()), 4), src)
                check("best single setting", mg.group(10), round(max(legit.values()), 4), src)
                checks.append((B["residual_ci95"][1] < 0,
                               "the registered repair fails on the one mixed bank", "interval "
                               "below zero", str(B["residual_ci95"]), src))
                ratio = max(legit.values()) / max(min(legit.values()), 1e-9)
                checks.append((3.5 <= ratio <= 4.5, "BioTransformer's published figure is four "
                               "times its guarantee", "about four", f"{ratio:.2f}x", src))
        if "sygma_175" in H2:
            S_ = H2["sygma_175"]
            src = str(hy2.relative_to(ROOT))
            checks.append((S_["dispatched_to_expanded"] == 0
                           and abs(S_["residual_convention_dependence"]) < 1e-9,
                           "on SyGMa dispatch is the identity, as pre-registered", "0 dispatched, "
                           "0 residual", f"{S_['dispatched_to_expanded']} dispatched, "
                           f"{S_['residual_convention_dependence']} residual", src))

    # 10c-v. The population inside one split. The paper's fourth axis was a bounded null about
    # which released test set a name refers to; this is a different question with a certified
    # answer, and the gate keeps the two apart by requiring both to appear.
    pv = ROOT / "results/parent_vs_metabolite.json"
    if pv.exists():
        PV = json.loads(pv.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mp2 = re.search(r"\$(\d+)\$ are parent drugs and \$(\d+)\$ are annotated products"
                        r".*?GRAIL loses \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ of recall between "
                        r"the two while MetaPredictor loses \$([\d.]+)\$ "
                        r"\$\[(-[\d.]+),([\d.]+)\]\$, an "
                        r"interaction of \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$ and against "
                        r"SyGMa \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(mp2), "the two-population sentence parses", "present",
                       "matched" if mp2 else "not matched", ""))
        if mp2:
            src = str(pv.relative_to(ROOT))
            D, I = PV["population_drop"], PV["drop_differs_by_method"]
            check("parents in the split", mp2.group(1), PV["by_population"]["parents"]["n"], src)
            check("metabolites in the split", mp2.group(2),
                  PV["by_population"]["metabolites"]["n"], src)
            check("GRAIL's loss", mp2.group(3),
                  round(D["GRAIL"]["parents_minus_metabolites"], 3), src)
            check("GRAIL's loss, lower", mp2.group(4), round(D["GRAIL"]["ci95"][0], 3), src)
            check("GRAIL's loss, upper", mp2.group(5), round(D["GRAIL"]["ci95"][1], 3), src)
            check("MetaPredictor's loss", mp2.group(6),
                  round(D["MetaPredictor"]["parents_minus_metabolites"], 3), src)
            k1 = "GRAIL vs MetaPredictor"
            check("the interaction", mp2.group(9), round(I[k1]["delta"], 3), src)
            check("the interaction, lower", mp2.group(10), round(I[k1]["ci95"][0], 3), src)
            check("the interaction, upper", mp2.group(11), round(I[k1]["ci95"][1], 3), src)
            k2 = "GRAIL vs SyGMa"
            check("the second interaction", mp2.group(12), round(I[k2]["delta"], 3), src)
            checks.append((I[k1]["certified"] and I[k2]["certified"],
                           "both interactions with GRAIL are certified", "certified",
                           f"{I[k1]['certified']} and {I[k2]['certified']}", src))
            # what this protects is that the two population questions stay distinguishable, not
            # that either keeps a heading: one is certified and one is a bounded null, and a
            # manuscript that merged them into a single claim would be overstating half of it
            null_said = "The other population question is a null" in flat
            cert_said = "differs certifiably by method" in flat or "an interaction of" in flat
            checks.append((null_said and cert_said,
                           "the null and the certified population result stay distinct",
                           "both stated", f"null {null_said}, certified {cert_said}",
                           "the manuscript"))

    # 10c-u. Where the emission rule does not hold. It beats every global constant on parent drugs
    # and does not on substrates that are themselves metabolites, and the paragraph says so; this
    # gate fails if the manuscript ever states the advantage without the population it holds on.
    pa = ROOT / "results/setsize_headroom__tautomer_parents.json"
    me2 = ROOT / "results/setsize_headroom__tautomer_metabolites.json"
    if pa.exists() and me2.exists():
        flat = re.sub(r"\s+", " ", whole)
        mq = re.search(r"The\s*advantage holds on the \$(\d+)\$ parent drugs, \$\+([\d.]+)\$ "
                       r"\$\[\+([\d.]+),\+([\d.]+)\]\$, and not on the \$(\d+)\$\s*substrates that are "
                       r"themselves metabolites, \$-([\d.]+)\$ \$\[-([\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(mq), "the population qualification parses", "present",
                       "matched" if mq else "not matched", ""))
        if mq:
            # the margin against the best constant is what the claim is about on each population,
            # and the arm is the one the sweep names rather than whichever gap rule happens to win
            def margin(path):
                D = json.loads(path.read_text())
                a_ = D["arms"]["gap rule a=0.5"]
                return (D, a_["f1_gain_over_best_constant"], a_["ci95_vs_best_constant"],
                        a_["separated_vs_best_constant"])
            Dp, mp_, cip, sp = margin(pa)
            Dm, mm_, cim, sm = margin(me2)
            check("parents, substrates", mq.group(1), Dp["config"]["n_substrates"], str(pa.relative_to(ROOT)))
            check("parents, margin", mq.group(2), round(mp_, 3), str(pa.relative_to(ROOT)))
            check("parents, lower", mq.group(3), round(cip[0], 3), str(pa.relative_to(ROOT)))
            check("parents, upper", mq.group(4), round(cip[1], 3), str(pa.relative_to(ROOT)))
            check("metabolites, substrates", mq.group(5), Dm["config"]["n_substrates"], str(me2.relative_to(ROOT)))
            check("metabolites, margin", mq.group(6), round(abs(mm_), 3), str(me2.relative_to(ROOT)))
            check("metabolites, lower", mq.group(7), round(abs(cim[0]), 3), str(me2.relative_to(ROOT)))
            check("metabolites, upper", mq.group(8), round(cim[1], 3), str(me2.relative_to(ROOT)))
            checks.append((sp and not sm,
                           "the rule wins on parents and not on metabolites",
                           "separated on one population only",
                           f"parents {sp}, metabolites {sm}", str(pa.relative_to(ROOT))))

    # 10c-t. The volume claim about the method that leads the field's budget. Recall at a fixed
    # cut-off pays for emitting more, so the aggregate that charges for it is where the lead is
    # tested; every figure of that comparison is held to the run.
    fm4 = ROOT / "results/four_method_291.json"
    if fm4.exists():
        F4 = json.loads(fm4.read_text()).get("macro_f1_by_budget", {})
        flat = re.sub(r"\s+", " ", whole)
        mv = re.search(r"it is\s*second at \$([\d.]+)\$ behind MetaPredictor's \$([\d.]+)\$ and last "
                       r"at \$k=1\$", flat)
        checks.append((bool(mv), "the volume sentence parses", "present",
                       "matched" if mv else "not matched", ""))
        if mv and F4:
            src = str(fm4.relative_to(ROOT))
            check("MetaTox F1 at 15", mv.group(1), F4["MetaTox"]["15"], src)
            check("MetaPredictor F1 at 15", mv.group(2), F4["MetaPredictor"]["15"], src)
            at15 = sorted(F4, key=lambda m: -F4[m]["15"])
            at1 = sorted(F4, key=lambda m: -F4[m]["1"])
            checks.append((at15.index("MetaTox") == 1, "MetaTox is second on F1 at fifteen",
                           "second", f"{at15.index('MetaTox') + 1}", src))
            checks.append((at1[-1] == "MetaTox", "and last on F1 at one", "last",
                           at1[-1], src))

    # 10c-s. The design appendix quotes two measured numbers and one dataset fact, and marks the
    # boundary between them and what it proposes. The boundary is what the gate protects: the
    # appendix must state that nothing in it is trained or evaluated here.
    flat_d = re.sub(r"\s+", " ", whole)
    ss2 = ROOT / "results/setsize_headroom.json"
    lc2 = ROOT / "results/label_convention_audit.json"
    if ss2.exists():
        S2 = json.loads(ss2.read_text())["arms"]["gap rule a=0.5"]
        md = re.search(r"beats the best global\s*constant on frozen scores by \$\+([\d.]+)\$ "
                       r"\$\[\+([\d.]+),\+([\d.]+)\]\$ macro F1, which is (four fifths|three "
                       r"quarters|half) of what an", flat_d)
        checks.append((bool(md), "the design appendix's measured claim parses", "present",
                       "matched" if md else "not matched", ""))
        if md:
            src = str(ss2.relative_to(ROOT))
            check("design appendix, the gain", md.group(1),
                  round(S2["f1_gain_over_best_constant"], 4), src)
            check("design appendix, lower", md.group(2),
                  round(S2["ci95_vs_best_constant"][0], 4), src)
            check("design appendix, upper", md.group(3),
                  round(S2["ci95_vs_best_constant"][1], 4), src)
            orac = json.loads(ss2.read_text())["arms"]["oracle count"][
                "f1_gain_over_best_constant"]
            share = S2["f1_gain_over_best_constant"] / orac
            band = ("four fifths" if 0.72 <= share < 0.87
                    else ("three quarters" if 0.62 <= share < 0.72 else "half"))
            checks.append((md.group(4) == band, "design appendix, share of the oracle",
                           md.group(4), band, f"{share:.3f} of the oracle headroom"))
        checks.append(("None of this is trained or evaluated in this paper." in flat_d,
                       "the design appendix marks what it does not claim", "present",
                       "present" if "None of this is trained or evaluated in this paper." in flat_d
                       else "absent", "the manuscript"))

    # 10c-r. The emission policy. Every arm holds the ranking fixed and changes only the number
    # emitted, so a gain is attributable to the size of the set; the gate additionally requires the
    # sibling-relative rule to beat every global constant, which is the claim the paragraph makes.
    ss = ROOT / "results/setsize_headroom.json"
    if ss.exists():
        S = json.loads(ss.read_text())["arms"]
        flat = re.sub(r"\s+", " ", whole)
        me = re.search(r"the deployed \$k=15\$ scores \$([\d.]+)\$ macro F1, the best global "
                       r"constant \$([\d.]+)\$, and truncating each substrate at its own annotated "
                       r"count \$([\d.]+)\$\..*?within half the leader's score gives \$([\d.]+)\$, "
                       r"and the arm.*?split: \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(me), "the emission sentence parses", "present",
                       "matched" if me else "not matched", ""))
        if me:
            src = str(ss.relative_to(ROOT))
            best_const = max((v["f1"], k) for k, v in S.items() if k.startswith("fixed"))
            check("deployed policy, macro F1", me.group(1), round(S["fixed k=15"]["f1"], 3), src)
            check("best global constant", me.group(2), round(best_const[0], 3), src)
            check("oracle count", me.group(3), round(S["oracle count"]["f1"], 3), src)
            check("the sibling rule", me.group(4), round(S["gap rule a=0.5"]["f1"], 3), src)
            # the margin the sentence states is the one against the best constant, which is the
            # arm the claim turns on; the deployed-budget margin is reported only in the appendix
            check("the sibling rule, gain", me.group(5),
                  round(S["gap rule a=0.5"]["f1_gain_over_best_constant"], 4), src)
            check("the sibling rule, lower", me.group(6),
                  round(S["gap rule a=0.5"]["ci95_vs_best_constant"][0], 4), src)
            check("the sibling rule, upper", me.group(7),
                  round(S["gap rule a=0.5"]["ci95_vs_best_constant"][1], 4), src)
            checks.append((S["gap rule a=0.5"]["separated_vs_best_constant"],
                           "the rule is separated from the best constant", "separated",
                           str(S["gap rule a=0.5"]["separated_vs_best_constant"]), src))
            gaps = [v["f1"] for k, v in S.items() if k.startswith("gap")]
            checks.append((max(gaps) > best_const[0],
                           "the sibling rule beats every global constant", "above every constant",
                           f"{max(gaps):.4f} against {best_const[0]:.4f} at {best_const[1]}", src))
            checks.append((S["predicted count"]["f1"] < best_const[0],
                           "forecasting the count from the substrate fails", "below the best constant",
                           f"{S['predicted count']['f1']:.4f}", src))

    # 10c-p. The docking control's reversals, after correcting for the family they were read in.
    # The control is the paper's answer to "your instrument finds nothing everywhere", and what it
    # establishes changed when the test did: on hit data the exact conditional test is available,
    # and under it neither reversal survives the board's own family.
    dm = ROOT / "results/docking_multiplicity.json"
    if dm.exists():
        DM = json.loads(dm.read_text())
        flat = re.sub(r"\s+", " ", whole)
        md = re.search(r"read out of \$(\d+)\$ pairs by \$(\d+)\$ cells, so it is granted inside a "
                       r"family of \$(\d+)\$\s*cell-level tests --- and correcting for them takes it "
                       r"back\. Neither reversal survives Holm.*?the smallest reversal\s*reaches "
                       r"\$([\d.]+)\\times10\^\{-(\d+)\}\$ against a threshold of "
                       r"\$([\d.]+)\\times10\^\{-(\d+)\}\$", flat)
        checks.append((bool(md), "the docking multiplicity sentence parses", "present",
                       "matched" if md else "not matched", ""))
        if md:
            src = str(dm.relative_to(ROOT))
            check("docking family, pairs", md.group(1), DM["config"]["n_pairs"], src)
            check("docking family, cells", md.group(2), DM["config"]["n_cells"], src)
            check("docking family, size", md.group(3), DM["family_size"], src)
            checks.append((DM["n_surviving_reversals"] == 0,
                           "no docking reversal survives the board's own correction", "0",
                           str(DM["n_surviving_reversals"]), src))
            _pbj2 = json.loads((ROOT / "results/robust_order_posebusters.json").read_text())
            _small = min(r["p"] for r in _pbj2["multiplicity"]["reversal_tests"])
            _said = float(md.group(4)) * 10 ** (-int(md.group(5)))
            checks.append((abs(_said - _small) <= 0.05 * _small,
                           "docking, the smallest reversal p", f"{_said:.3g}", f"{_small:.3g}",
                           src))
            import importlib.util as _iu4
            _sp4 = _iu4.spec_from_file_location("_um4", ROOT / "scripts/union_multiplicity.py")
            _um4 = _iu4.module_from_spec(_sp4)
            sys.modules["_um4"] = _um4
            _sp4.loader.exec_module(_um4)
            _cut4 = _um4.cutoff(_pbj2["multiplicity"]["p_values"], 0.05)
            _saidt = float(md.group(6)) * 10 ** (-int(md.group(7)))
            checks.append((abs(_saidt - _cut4) <= 0.05 * _cut4,
                           "docking, the threshold it misses", f"{_saidt:.3g}", f"{_cut4:.3g}",
                           src))

    # 10c-q. The robust order. The share of a leaderboard's pairwise claims that survive every cell
    # of the declared grid is the paper's answer to seven rounds of "no new metric", so every figure
    # of it is held to the run: the count that dominates, the count some cell reverses with an
    # interval, how many pairs the leaderboard's own cell resolves at all, and the share among those.
    ro = ROOT / "results/robust_order.json"
    rm = ROOT / "results/robust_order_metabolite.json"
    rp = ROOT / "results/robust_order_posebusters.json"
    rwe = ROOT / "results/robust_order_wmt24_en-de.json"
    rwj = ROOT / "results/robust_order_wmt24_ja-zh.json"
    if ro.exists() and rm.exists() and rp.exists() and rwe.exists() and rwj.exists():
        RO = json.loads(ro.read_text())["leaderboards"]
        RM = json.loads(rm.read_text())
        RP = json.loads(rp.read_text())
        RWE, RWJ = json.loads(rwe.read_text()), json.loads(rwj.read_text())
        flat = re.sub(r"\s+", " ", whole)
        # places | items | dominate of pairs | contested (certified) | unresolved | supported
        row = (r"& \$(\d+)\$ & \$([\d,{}]+)\$ & \$(\d+)\$ of \$(\d+)\$ & \$(\d+)\$ \((\d+)\) & "
               r"\$(\d+)\$ & \$(\d+)\$")
        mr = re.search(r"retrosynthesis, seven systems " + row + r".*?"
                       r"retrosynthesis, three systems " + row + r".*?"
                       r"metabolites, three methods " + row + r".*?"
                       r"docking, seven programs " + row + r".*?"
                       r"translation, nineteen systems " + row + r".*?"
                       r"translation, fifteen systems " + row, flat)
        checks.append((bool(mr), "the robust-order table parses", "present",
                       "matched" if mr else "not matched", ""))
        if mr:
            src = str(ro.relative_to(ROOT))
            srm = str(rm.relative_to(ROOT))
            boards = ((("seven systems", RO["cluster0"], src), 0),
                      (("three systems", RO["cluster1"], src), 8),
                      (("three methods", RM, srm), 16),
                      (("seven programs", RP, str(rp.relative_to(ROOT))), 24),
                      (("nineteen systems", RWE, str(rwe.relative_to(ROOT))), 32),
                      (("fifteen systems", RWJ, str(rwj.relative_to(ROOT))), 40))
            for (lbl, r_, s), off in boards:
                check(f"robust order, {lbl}, items", mr.group(off + 2).replace("{,}", ""),
                      r_["n_items"], s)
                check(f"robust order, {lbl}, dominate", mr.group(off + 3), r_["n_dominating"], s)
                check(f"robust order, {lbl}, pairs", mr.group(off + 4), r_["n_pairs"], s)
                check(f"robust order, {lbl}, contested", mr.group(off + 5), r_["n_contested"], s)
                check(f"robust order, {lbl}, certified", mr.group(off + 6),
                      r_["n_contested_after_correction"], s)
                check(f"robust order, {lbl}, unresolved", mr.group(off + 7), r_["n_unresolved"], s)
                checks.append((r_["n_dominating"] + r_["n_contested"] + r_["n_unresolved"]
                               == r_["n_pairs"], f"robust order, {lbl}, the verdicts partition",
                               str(r_["n_pairs"]),
                               str(r_["n_dominating"] + r_["n_contested"] + r_["n_unresolved"]), s))
                check(f"robust order, {lbl}, tiers", mr.group(off + 8), r_["tiers_distinguished"], s)
                check(f"robust order, {lbl}, systems", mr.group(off + 1), r_["n_systems"], s)
            for name, r_, s in (("retro", RO["cluster0"], src), ("metabolite", RM, srm)):
                g = (json.loads(ro.read_text())["config"]["grid"] if name == "retro"
                     else RM["config"]["grid"])
                checks.append((len(g["criteria"]) == 4 and len(g["budgets"]) == 4,
                               f"the {name} grid is four criteria by four budgets", "4 by 4",
                               f"{len(g['criteria'])} by {len(g['budgets'])}", s))
            # the claim that most of the table was never resolved, and the two-axis split
            checks.append((RO["cluster0"]["n_pairs"] - RO["cluster0"][
                "n_resolved_in_the_published_cell"] == 11,
                           "eleven of the twenty-one were never resolved", "11",
                           str(RO["cluster0"]["n_pairs"]
                               - RO["cluster0"]["n_resolved_in_the_published_cell"]), src))
            crit = "criteria only, at the published budget"
            bud = "budgets only, at the published criterion"
            # the sentence says the budget takes all three from one table and one from the other,
            # which is a stronger and more falsifiable claim than "the budget axis does not"
            losses = []
            for name, r_, s in (("three systems", RO["cluster1"], src),
                                ("three methods", RM, srm)):
                checks.append((r_["sub_grids"][crit]["n_dominating"] == r_["n_pairs"],
                               f"on {name} the criterion axis leaves the order intact",
                               str(r_["n_pairs"]),
                               str(r_["sub_grids"][crit]["n_dominating"]), s))
                losses.append(r_["n_pairs"] - r_["sub_grids"][bud]["n_dominating"])
            checks.append((sorted(losses) == [1, 3],
                           "the budget takes all three from one table and one from the other",
                           "3 and 1", f"{losses[0]} and {losses[1]}", src))
            # the published leader is not in the top tier of the surviving order, which is the
            # instrument's sharpest consequence and is recomputed here rather than asserted
            pub = RO["cluster0"]["published_order"]
            e_: dict = {s: set() for s in pub}
            for p_, v_ in RO["cluster0"]["pairs"].items():
                if v_["dominates"]:
                    hi_, lo_ = p_.split(" over ")
                    e_[hi_].add(lo_)
            memo_: dict = {}

            def _chain(n):
                if n not in memo_:
                    memo_[n] = 1 + max((_chain(c) for c in e_[n]), default=0)
                return memo_[n]

            depth_ = {s: _chain(s) for s in pub}
            top_ = max(depth_.values())
            first_in_top = depth_[pub[0]] == top_
            second_in_top = depth_[pub[1]] == top_
            alone = sum(1 for s in pub if depth_[s] == top_) == 1
            checks.append((not first_in_top and second_in_top and alone,
                           "the published leader is not in the top tier and the runner-up alone is",
                           "first out, second alone in",
                           f"first tier {top_ - depth_[pub[0]] + 1}, "
                           f"second tier {top_ - depth_[pub[1]] + 1}, "
                           f"{sum(1 for s in pub if depth_[s] == top_)} in the top tier", src))
            m4 = re.search(r"take \$(\d+)\$ total orders across \$(\d+)\$ cells", flat)
            checks.append((bool(m4), "the distinct-orderings sentence parses", "present",
                           "matched" if m4 else "not matched", src))
            if m4:
                check("distinct orderings", m4.group(1),
                      RO["cluster0"]["distinct_orderings_across_the_grid"], src)
                check("cells in the grid", m4.group(2), RO["cluster0"]["n_cells"], src)

    # 10c-p. The count of choices that reorder, everywhere it appears. It was three in the
    # introduction and two in the abstract and conclusion for one revision, because the engine axis
    # was removed from the count in some places and not others. A count stated in four places is a
    # number like any other and gets a gate: no passage may say three choices reorder.
    flat_c = re.sub(r"\s+", " ", whole)
    bad = re.findall(r"[Tt]hree of the four choices[^.]*(?:reorder|move an ordering)", flat_c)
    checks.append((not bad, "no passage says three choices reorder", "none",
                   f"{len(bad)} found: {bad[:1]}" if bad else "none", "the manuscript"))
    two = re.findall(r"[Tt]wo of the four choices[^.]*reorder|[Tt]wo of them change a published "
                     r"ordering|[Tt]wo move a published ordering|[Tt]wo of the four choices change a "
                     r"published ordering", flat_c)
    # the claim used to be staked in the abstract, the introduction and the conclusion; after the
    # restructure the body states it once and the case study carries the measurements, so what is
    # required is that it be stated and that its complement never be
    checks.append((len(two) >= 1, "the count of two is stated where the claim is staked",
                   "at least once, and three never", f"{len(two)} passages", "the manuscript"))

    # 10c-o. The engine term split. Three quarters of what section 4 called a convention is an
    # application loop that expands a substrate and never contracts the product; the paper now
    # reports both parts and this holds each to the run that measured them.
    cl = ROOT / "results/completed_loop_reach__clean_test.json"
    if cl.exists():
        CL = json.loads(cl.read_text())
        flat = re.sub(r"\s+", " ", whole)
        mcl = re.search(r"contracting it lifts the expanded arm to \$([\d.]+)\$\s*"
                        r"\$\[([\d.]+),([\d.]+)\]\$, worth \$\+([\d.]+)\$ "
                        r"\$\[\+([\d.]+),\+([\d.]+)\]\$, and what survives against the unexpanded "
                        r"arm is\s*\$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", flat)
        checks.append((bool(mcl), "the completed-loop sentence parses", "present",
                       "matched" if mcl else "not matched", ""))
        if mcl:
            src = str(cl.relative_to(ROOT))
            comp, miss, conv = CL["reach"]["completed"], CL["the_missing_call"], CL["what_survives_as_convention"]
            check("completed arm, reach", mcl.group(1), round(comp["reach"], 3), src)
            check("completed arm, lower", mcl.group(2), round(comp["ci95"][0], 3), src)
            check("completed arm, upper", mcl.group(3), round(comp["ci95"][1], 3), src)
            check("the missing call", mcl.group(4), round(miss["delta"], 3), src)
            check("the missing call, lower", mcl.group(5), round(miss["ci95"][0], 3), src)
            check("the missing call, upper", mcl.group(6), round(miss["ci95"][1], 3), src)
            check("what survives as convention", mcl.group(7), round(conv["delta"], 3), src)
            check("convention, lower", mcl.group(8), round(conv["ci95"][0], 3), src)
            check("convention, upper", mcl.group(9), round(conv["ci95"][1], 3), src)
            checks.append((miss["delta"] > conv["delta"] * 2,
                           "the missing call is the larger part",
                           "call exceeds convention", f"{miss['delta']:.4f} against {conv['delta']:.4f}", src))

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
        m = re.search(r"the toolkit then refuses \$(\d+)\\%\$ of the fragments those products\s*"
                      r"separate into against \$([\d.]+)\\%\$ without the expansion", flat)
        checks.append((bool(m), "the mechanism sentence parses",
                       "present", "matched" if m else "not matched", ""))
        fire = exp["fired"] / max(imp["fired"], 1)
        checks.append((0.9 <= fire <= 1.1, "templates fire almost equally often either way",
                       "within a tenth", f"ratio {fire:.3f}", str(mech.relative_to(ROOT))))
        if m:
            check("fragments unreadable under the incomplete loop", m.group(1),
                  round(exp["unparseable"] / max(exp["fragments"], 1) * 100),
                  str(mech.relative_to(ROOT)))
            check("fragments unreadable without the expansion", m.group(2),
                  round(imp["unparseable"] / max(imp["fragments"], 1) * 100, 1),
                  str(mech.relative_to(ROOT)))
        # completing the loop must remove every one of them, which is the claim that inverts the
        # section: the larger part is a defect and only the remainder is a convention
        comp = json.loads(mech.read_text())["pipeline"].get("loop_completed")
        if comp:
            checks.append((comp["unparseable"] == 0,
                           "the completed loop leaves no unreadable fragment", "zero",
                           str(comp["unparseable"]), str(mech.relative_to(ROOT))))
            mfr = re.search(r"loop yielding \$([\d,{}]+)\$ fragments against the unexpanded arm's "
                            r"\$([\d,{}]+)\$", flat)
            checks.append((bool(mfr), "the surviving-matching sentence parses", "present",
                           "matched" if mfr else "not matched", ""))
            if mfr:
                num = lambda g: int(g.replace("{,}", ""))
                check("fragments from the completed loop", num(mfr.group(1)), comp["fragments"],
                      str(mech.relative_to(ROOT)))
                check("fragments without the expansion", num(mfr.group(2)), imp["fragments"],
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
        m = re.search(r"BioTransformer recovers\s*\$([\d,{}]+)\$ references with hydrogens drawn "
                      r"against \$(\d+)\$ with them implicit, a swing of \$(-[\d.]+)\$,\s*while SyGMa "
                      r"recovers \$([\d,{}]+)\$ against \$([\d,{}]+)\$, a swing of \$\+([\d.]+)\$", flat)
        checks.append((bool(m), "the two-bank sentence parses", "present",
                       "matched" if m else "not matched", ""))
        if m:
            num = lambda g: int(g.replace("{,}", ""))
            R = H["sygma_175"]["references"]
            def recovered(bank, arm):
                return round(H[bank]["global_arms"][arm] * R)
            check("BioTransformer, completed, drawn", num(m.group(1)),
                  recovered("biotransformer", "all_explicit_completed"), str(hy_b.relative_to(ROOT)))
            check("BioTransformer, implicit", num(m.group(2)),
                  recovered("biotransformer", "all_implicit"), str(hy_b.relative_to(ROOT)))
            check("SyGMa, completed and drawn", num(m.group(4)),
                  recovered("sygma_175", "all_explicit_completed"), str(hy_b.relative_to(ROOT)))
            check("SyGMa, implicit", num(m.group(5)),
                  recovered("sygma_175", "all_implicit"), str(hy_b.relative_to(ROOT)))
            for lbl, said, bank in (("BioTransformer", m.group(3), "biotransformer"),
                                    ("SyGMa", m.group(6), "sygma_175")):
                a = H[bank]["global_arms"]
                # the sentence reads from the drawn arm to the implicit one, so the swing is
                # implicit minus drawn: negative where a bank loses by dropping the hydrogens
                got = round(a["all_implicit"] - a["all_explicit_completed"], 3)
                checks.append((abs(float(said) - got) < 5e-4, f"{lbl}, the swing",
                               said, f"{got:+.3f}", str(hy_b.relative_to(ROOT))))
            # the claim the section now makes is that there is NO exchange once the loop is complete
            sc = H["sygma_175"]["global_arms"]; bc = H["biotransformer"]["global_arms"]
            checks.append((sc["all_explicit_completed"] > bc["all_explicit_completed"]
                           and sc["all_implicit"] > bc["all_implicit"],
                           "SyGMa leads in both completed arms", "no exchange",
                           f"{sc['all_explicit_completed']} vs {bc['all_explicit_completed']} drawn, "
                           f"{sc['all_implicit']} vs {bc['all_implicit']} implicit",
                           str(hy_b.relative_to(ROOT))))

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
        m = re.search(r"benchmark separates is \$([\d.]+)\$", flat)
        check("resolution floor, seven-system", m and m.group(1), c0["resolution_floor"])
        m = re.search(r"three-system group's narrowest is \$([\d.]+)\$", flat)
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
        m = re.search(r"[Ff]ive of twenty-one pairs exchange", flat)
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

    # 13. the per-appendix modules. They are separate files so that checks for one appendix can be
    # written without touching the checks for another, and so the coverage question -- which of an
    # appendix's numerals no check reads -- is answerable file by file.
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from gates import Ctx, register_all  # noqa: E402

        _cache: dict = {}

        def _art(name):
            if name not in _cache:
                _q = ROOT / "results" / name
                _cache[name] = json.loads(_q.read_text()) if _q.exists() else None
            return _cache[name]

        _ctx = Ctx(flat=re.sub(r"\s+", " ", whole), root=ROOT, check=check, checks=checks,
                   art=_art, tex={p.name: p.read_text() for p in TEX})
        _ran = register_all(_ctx)
        if _ran:
            checks.append((True, "the per-appendix modules that ran", ", ".join(_ran),
                           f"{len(_ran)} modules", "scripts/gates/"))
    except Exception as _e:  # a broken module must fail loudly, not be skipped
        checks.append((False, "the per-appendix modules load", "all load", repr(_e)[:60],
                       "scripts/gates/"))

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
