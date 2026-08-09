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
    atcap = re.search(r"emits at the cap on \$(\d+)\$ of", whole)
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
                 r"\$([\d.]+)\$[^.]{0,80}?on the twenty-four seen"),
                ("external, unseen", "unseen", r"against \$([\d.]+)\$ on the thirteen unseen")):
            m = re.search(pat, whole)
            check(label, m and m.group(1), sp[key]["coverage"])

    # 8. the provenance split, whose own gate only proves it reproduces the ceiling -- nothing until
    # now proved the manuscript prints what it produced. This is the pairing that failed before: a
    # measurement guarded by a frozen literal, and a paper quoting the superseded value it certified.
    prov = ROOT / "results/ceiling_by_provenance.json"
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
        m = re.search(r"--- \$([\d,{}]+)\$ of \$([\d,{}]+)\$, the complement", flat)
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
                ("Holm survivors, main body", r"\$(\d+)\$ of \$\d+\$ paired interactions surviving Holm",
                 len(L["holm_survivors"])),
                ("interaction tests, main body", r"paired interactions surviving Holm", None),
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
        # the main body states the same condition in words; both of its numbers are checked too
        mb = re.search(r"within a median \$([\d.]+)\$ of each other", flat)
        checks.append((close(mb and mb.group(1), rows["seven-system"]["median_gap"], 5e-4),
                       "median gap, main body", mb and mb.group(1),
                       round(rows["seven-system"]["median_gap"], 4), ""))
        mb2 = re.search(r"three systems? \$([\d.]+)\$\s*apart", flat)
        checks.append((close(mb2 and mb2.group(1), rows["three-system"]["median_gap"], 5e-4),
                       "three-system gap, main body", mb2 and mb2.group(1),
                       round(rows["three-system"]["median_gap"], 4), ""))
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
        for tag, v in rows.items():
            m = re.search(tag + r" & \$([\d.]+)\$ & \$([\d.]+)\$ & \$(\d+)\$ of \$(\d+)\$ & \$(\d+)\$", flat)
            for i, (label, val) in enumerate((("median gap", v["median_gap"]),
                                              ("differential move", v["max_diff"]),
                                              ("pairs closer than the move", v["closer"]),
                                              ("pairs in the group", v["pairs"]),
                                              ("pairs exchanged", v["exchanged"]))):
                checks.append((close(m and m.group(i + 1), val, 5e-4 if i < 2 else 0),
                               f"{tag}, {label}", m and m.group(i + 1), round(val, 4),
                               "from the two leaderboard artifacts"))

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
