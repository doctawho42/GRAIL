"""Checks for app/robust.tex, app/negative.tex and app/closure.tex.

Three appendices that argue from numbers: the robustness instrument's grid and the translation
boards it is run on, the three refuted hypotheses about the rule bank plus the per-template
hydrogen dispatch, and the annotation graph's closure under composition. Between them they printed
fifty-nine numerals that no check read, which is to say fifty-nine numbers that could have gone
stale in silence -- and robust.tex is edited more often than any other appendix here.

Two rules are followed throughout, because both have already cost this paper something:

* a figure is bound to ONE leaf of ONE artifact. Where a value is also derivable from counts, the
  counts are used, since an artifact's own rounded summary and the manuscript's rounding of the
  unrounded quantity disagree at the last place often enough to matter (``merge_stability`` at
  0.9745 against 1453/1491, which prints as 97.5 and not 97.4).
* a table is bound cell by cell, every row and every column, and where a column is a function of
  two others -- residual = dispatch - best global -- that identity is held as well, so a column
  that stops summing to itself fails rather than passes.
"""
from __future__ import annotations

import ast
import itertools
import re

SRC_RO = "results/robust_order.json"
SRC_ENDE = "results/robust_order_wmt24_en-de.json"
SRC_JAZH = "results/robust_order_wmt24_ja-zh.json"
SRC_CLU = "results/wmt_official_clusters.json"
SRC_HD = "results/hydrogen_dispatch__clean_test.json"
SRC_EH = "results/explicit_h_mechanism__clean_test.json"
SRC_RC = "results/reference_closure.json"
SRC_CR = "results/composition_recovery.json"


def _find(ctx, name, pattern, note=""):
    """Match a passage, or leave a failing check behind. A passage that moved is not a pass."""
    m = re.search(pattern, ctx.flat)
    if m is None:
        ctx.checks.append((False, name, "present", "not matched", note))
    return m


def _num(s: str) -> str:
    """The manuscript's digit grouping removed: 1{,}368 -> 1368."""
    return s.replace("{,}", "").replace(",", "")


def _ok(ctx, name, cond, printed, computed, note=""):
    ctx.checks.append((bool(cond), name, printed, computed, note))


# ---------------------------------------------------------------- app/robust.tex


def _robust(ctx) -> None:
    RO = ctx.art("robust_order.json")
    if RO is not None:
        c0 = RO["leaderboards"]["cluster0"]
        acc = c0["system_accuracy_by_cell"]
        cells = list(next(iter(acc.values())).keys())
        pairs_of_cells = list(itertools.combinations(cells, 2))
        identical = [(a, b) for a, b in pairs_of_cells
                     if all(abs(acc[s][a] - acc[s][b]) < 1e-12 for s in acc)]
        m = _find(ctx, "robust, the cell-pair sentence parses",
                  r"exactly one of the \$([\d,{}]+)\$ pairs of cells gives identical hit vectors "
                  r"for every system, canonical and InChIKey at top-1", SRC_RO)
        if m:
            ctx.check("robust, pairs of cells on the seven-system board", _num(m.group(1)),
                      len(pairs_of_cells), SRC_RO)
        _ok(ctx, "robust, exactly one pair of cells is redundant", len(identical) == 1,
            "exactly one", str(len(identical)), SRC_RO)
        _ok(ctx, "robust, the redundant pair is canonical against inchikey at top-1",
            len(identical) == 1
            and {ast.literal_eval(x) for x in identical[0]} == {("canonical", 1), ("inchikey", 1)},
            "canonical/inchikey at 1",
            ", ".join(identical[0]) if identical else "none", SRC_RO)

    # The criterion the grid excludes, whose parameters are a fact about the code that computes it.
    gen = (ctx.root / "scripts/annotation_agreement.py").read_text()
    g = re.search(r"GetMorganGenerator\(radius=(\d+), fpSize=(\d+)\)", gen)
    m = _find(ctx, "robust, the tanimoto1 exclusion parses",
              r"at radius \$(\d+)\$ over \$(\d+)\$ bits it returns \$([\d.]+)\$ for decane",
              "scripts/annotation_agreement.py")
    if m and g:
        ctx.check("robust, the fingerprint radius the excluded criterion uses", m.group(1),
                  int(g.group(1)), "scripts/annotation_agreement.py")
        ctx.check("robust, the fingerprint width the excluded criterion uses", m.group(2),
                  int(g.group(2)), "scripts/annotation_agreement.py")
    if m:
        try:
            from rdkit import Chem, DataStructs, RDLogger
            from rdkit.Chem import rdFingerprintGenerator

            RDLogger.DisableLog("rdApp.*")
            fg = rdFingerprintGenerator.GetMorganGenerator(
                radius=int(g.group(1)) if g else 2, fpSize=int(g.group(2)) if g else 2048)
            fp = lambda s: fg.GetFingerprint(Chem.MolFromSmiles(s))  # noqa: E731
            worst = min(DataStructs.TanimotoSimilarity(fp(a), fp(b)) for a, b in
                        (("CCCCCCCCCC", "CCCCCCCCCCC"),
                         ("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O")))
            ctx.check("robust, what that criterion returns for two different compounds",
                      m.group(3), worst, "rdkit, decane/undecane and D-/L-alanine")
        except ImportError:  # rdkit absent: the two parameters above are still held
            pass

    # The two MQM boards, every number of the two sentences that report them, and the identity
    # those sentences rest on: a pair is dominating, unresolved or contested and nothing else.
    E, J = ctx.art("robust_order_wmt24_en-de.json"), ctx.art("robust_order_wmt24_ja-zh.json")
    if E is not None:
        m = _find(ctx, "robust, the nineteen-system grid sentence parses",
                  r"On \$(\d+)\$ pairs of \$(\d+)\$ systems, \$(\d+)\$ dominate, \$(\d+)\$ are "
                  r"unresolved and \$(\d+)\$ are contested of which \$(\d+)\$ survive Holm over "
                  r"the \$([\d,{}]+)\$ cell-level tests that grid creates, so nineteen published "
                  r"places support \$(\d+)\$", SRC_ENDE)
        if m:
            for i, (label, key) in enumerate((
                    ("pairs", "n_pairs"), ("systems", "n_systems"),
                    ("dominating", "n_dominating"), ("unresolved", "n_unresolved"),
                    ("contested", "n_contested"),
                    ("certified", "n_contested_after_correction")), start=1):
                ctx.check(f"robust, en-de {label}", m.group(i), E[key], SRC_ENDE)
            ctx.check("robust, en-de cell-level tests", _num(m.group(7)),
                      E["multiplicity"]["family_size"], SRC_ENDE)
            ctx.check("robust, en-de tiers supported", m.group(8), E["tiers_distinguished"],
                      SRC_ENDE)
        _ok(ctx, "robust, en-de pairs account for themselves",
            E["n_dominating"] + E["n_unresolved"] + E["n_contested"] == E["n_pairs"],
            E["n_pairs"], E["n_dominating"] + E["n_unresolved"] + E["n_contested"],
            "dominating + unresolved + contested")
        _ok(ctx, "robust, en-de family is one test per pair per cell",
            E["n_pairs"] * E["n_cells"] == E["multiplicity"]["family_size"],
            E["multiplicity"]["family_size"], E["n_pairs"] * E["n_cells"], SRC_ENDE)
    if J is not None:
        m = _find(ctx, "robust, the fifteen-system grid sentence parses",
                  r"On \$(\d+)\$ pairs of \$(\d+)\$ systems, \$(\d+)\$ dominate, \$(\d+)\$ are "
                  r"unresolved and \$(\d+)\$ is contested, which does not survive the \$([\d,{}]+)"
                  r"\$ tests of its own, so fifteen places support \$(\d+)\$", SRC_JAZH)
        if m:
            for i, (label, key) in enumerate((
                    ("pairs", "n_pairs"), ("systems", "n_systems"),
                    ("dominating", "n_dominating"), ("unresolved", "n_unresolved"),
                    ("contested", "n_contested")), start=1):
                ctx.check(f"robust, ja-zh {label}", m.group(i), J[key], SRC_JAZH)
            ctx.check("robust, ja-zh cell-level tests", _num(m.group(6)),
                      J["multiplicity"]["family_size"], SRC_JAZH)
            ctx.check("robust, ja-zh tiers supported", m.group(7), J["tiers_distinguished"],
                      SRC_JAZH)
        _ok(ctx, "robust, ja-zh pairs account for themselves",
            J["n_dominating"] + J["n_unresolved"] + J["n_contested"] == J["n_pairs"],
            J["n_pairs"], J["n_dominating"] + J["n_unresolved"] + J["n_contested"],
            "dominating + unresolved + contested")
        _ok(ctx, "robust, ja-zh certifies nothing, which is the null the paragraph claims",
            J["n_contested_after_correction"] == 0, "0",
            str(J["n_contested_after_correction"]), SRC_JAZH)

    # The en-de family size is printed twice, the second time as the reason the two boards differ
    # in kind. One printing gated and the other stale is this paper's most expensive defect, so
    # every printing is found, each is held, and the count of them is held too.
    # The printings are found by their SLOT and not by their value: selecting the printings that
    # already equal the artifact and then checking them against it is a check that cannot fail,
    # and it is how a second printing of this very number went unread.
    fams = re.findall(r"(?:survive Holm over the|carries a reversal past both its own) "
                      r"\$([\d,{}]+)\$ cell-level tests", ctx.flat)
    if E is not None:
        want = E["multiplicity"]["family_size"]
        _ok(ctx, "robust, both printings of the en-de family are found", len(fams) == 2,
            "2", str(len(fams)), "the manuscript")
        for i, f in enumerate(fams, 1):
            ctx.check(f"robust, en-de family printing {i}", _num(f), want, SRC_ENDE)
    # and the union of all twenty-three grids, which is the sum of those families
    boards = _all_boards(ctx)
    m = _find(ctx, "robust, the union-family sentence parses",
              r"The honest family is the union of the grids: \$([\d,{}]+)\$ cell-level tests",
              "sum over results/robust_order*.json")
    if boards and m:
        ctx.check("robust, the union family over the twenty-three grids", _num(m.group(1)),
                  sum(b["multiplicity"]["family_size"] for b in boards),
                  "sum over results/robust_order*.json")
        _ok(ctx, "robust, the union is over twenty-three boards", len(boards) == 23, "23",
            str(len(boards)), "results/robust_order*.json")

    # The task's own clustering, crossed against ours: every cell of a four-by-two table.
    CL = ctx.art("wmt_official_clusters.json")
    if CL is not None:
        m = _find(ctx, "robust, the crossed table parses",
                  r"their test, their construction & \$(\d+)\$ & \$(\d+)\$ \\\\+ "
                  r"our test, their construction & \$(\d+)\$ & \$(\d+)\$ \\\\+ "
                  r"their test, our construction & \$(\d+)\$ & \$(\d+)\$ \\\\+ "
                  r"our test, our construction & \$(\d+)\$ & \$(\d+)\$", SRC_CLU)
        if m:
            for col, lp in ((0, "en-de"), (1, "ja-zh")):
                b = CL["boards"][lp]
                for row, key in enumerate(("n_clusters", "their_construction_our_test",
                                           "our_construction_their_test", "ours_own_cell_only")):
                    ctx.check(f"robust, crossed table [{key}, {lp}]",
                              m.group(1 + 2 * row + col), b[key], SRC_CLU)
        for lp, board in (("en-de", E), ("ja-zh", J)):
            if board is not None and lp in CL["boards"]:
                _ok(ctx, f"robust, {lp} whole-grid tiers agree across the two artifacts",
                    CL["boards"][lp]["ours_whole_grid"] == board["tiers_distinguished"],
                    board["tiers_distinguished"], CL["boards"][lp]["ours_whole_grid"],
                    f"{SRC_CLU} against results/robust_order_wmt24_{lp}.json")
        m = _find(ctx, "robust, the clustering threshold parses",
                  r"merges systems into a cluster whenever that test does not separate them at "
                  r"\$([\d.]+)\$", SRC_CLU)
        if m:
            ctx.check("robust, the clustering threshold", m.group(1),
                      CL["boards"]["en-de"]["alpha"], SRC_CLU)
        _ok(ctx, "robust, every board of the task's own clustering used that threshold",
            len({b["alpha"] for b in CL["boards"].values()}) == 1,
            CL["boards"]["en-de"]["alpha"],
            sorted({b["alpha"] for b in CL["boards"].values()}), SRC_CLU)


def _all_boards(ctx) -> list:
    """The twenty-three grids, in the order the survey counts them."""
    out = []
    for fn, key in (("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
                    ("robust_order_metabolite.json", None),
                    ("robust_order_posebusters.json", None),
                    ("robust_order_wmt24_en-de.json", None),
                    ("robust_order_wmt24_ja-zh.json", None)):
        d = ctx.art(fn)
        if d is not None:
            out.append(d["leaderboards"][key] if key else d)
    for fn in ("robust_order_wmt24_esa.json", "robust_order_wmt23.json"):
        d = ctx.art(fn)
        if d is not None:
            out += list(d["boards"].values())
    return out


# ---------------------------------------------------------------- app/negative.tex


def _negative(ctx) -> None:
    _negative_bank(ctx)
    _negative_embedding(ctx)
    _negative_dispatch(ctx)
    _negative_tail(ctx)


def _negative_bank(ctx) -> None:
    RC = ctx.art("rule_collapse.json")
    if RC is not None:
        m = _find(ctx, "negative, the redundancy sentence parses",
                  r"Applying the full bank to a sample of ([\d,{}]+) substrates, ([\d,{}]+) rules "
                  r"fire at least once, and these fall into ([\d,{}]+) distinct classes",
                  "results/rule_collapse.json")
        if m:
            ctx.check("negative, the probe pool", _num(m.group(1)), RC["pool_size"],
                      "results/rule_collapse.json")
            ctx.check("negative, rules that fire", _num(m.group(2)), RC["n_fired"],
                      "results/rule_collapse.json")
            ctx.check("negative, distinct behavioural classes", _num(m.group(3)),
                      RC["n_distinct_signatures_fired"], "results/rule_collapse.json")
        m = _find(ctx, "negative, the half-split stability sentence parses",
                  r"re-deriving the grouping on each half preserves ([\d.]+)\\% of the "
                  r"equivalences", "results/rule_collapse.json")
        if m:
            st = RC["merge_stability_T1"]
            ctx.check("negative, equivalences preserved on the other half", m.group(1),
                      100.0 * st["A_pairs_preserved_on_B"] / st["A_equivalent_pairs"],
                      "results/rule_collapse.json, from the two counts")
    RD = ctx.art("rule_dedup_provable.json")
    if RD is not None:
        m = _find(ctx, "negative, the canonicalisation sentence parses",
                  r"removes exact duplicates eliminates ([\d.]+)\\% of the whole bank",
                  "results/rule_dedup_provable.json")
        if m:
            ctx.check("negative, share of the bank a provable dedup removes", m.group(1),
                      100.0 * RD["rules_eliminated"] / RD["n_rules_total"],
                      "results/rule_dedup_provable.json, from the two counts")

    TP = ctx.art("rule_train_positives.json")
    if TP is not None:
        m = _find(ctx, "negative, the inert-target sentence parses",
                  r"Across the ([\d,{}]+) training substrates, ([\d.]+)\\% of the bank is never a "
                  r"positive label, and a further ([\d.]+)\\% is positive for exactly one "
                  r"substrate", "results/rule_train_positives.json")
        if m:
            ctx.check("negative, training substrates", _num(m.group(1)), TP["train_substrates"],
                      "results/rule_train_positives.json")
            ctx.check("negative, share of the bank never a positive label", m.group(2),
                      100.0 * TP["never_positive"] / TP["rules"],
                      "results/rule_train_positives.json, from the two counts")
            ctx.check("negative, share positive for exactly one substrate", m.group(3),
                      100.0 * TP["pos_eq_1"] / TP["rules"],
                      "results/rule_train_positives.json, from the two counts")

    PR = ctx.art("prune_and_rerank_val.json")
    if PR is not None:
        src = "results/prune_and_rerank_val.json"
        m = _find(ctx, "negative, the pruning arm parses",
                  r"discarding the ([\d,{}]+) rules that are never a positive label in training, "
                  r"which reduces the bank from ([\d,{}]+) to ([\d,{}]+)", src)
        if m:
            ctx.check("negative, rules discarded", _num(m.group(1)), PR["bank"]["removed"], src)
            ctx.check("negative, bank before pruning", _num(m.group(2)), PR["bank"]["full"], src)
            ctx.check("negative, bank after pruning", _num(m.group(3)), PR["bank"]["pruned"], src)
        _ok(ctx, "negative, the pruned bank is what is left of the full one",
            PR["bank"]["full"] - PR["bank"]["removed"] == PR["bank"]["pruned"],
            PR["bank"]["pruned"], PR["bank"]["full"] - PR["bank"]["removed"], src)
        if TP is not None:
            _ok(ctx, "negative, the discarded rules are the never-positive ones",
                PR["bank"]["removed"] == TP["never_positive"], TP["never_positive"],
                PR["bank"]["removed"], f"{src} against results/rule_train_positives.json")
        m = _find(ctx, "negative, the recall loss parses",
                  r"Recall@15 declines from ([\d.]+) to ([\d.]+) \(paired difference "
                  r"\$(-[\d.]+)\$, 95\\% confidence interval \$\[(-[\d.]+),(-[\d.]+)\]\$\)", src)
        if m:
            R = PR["recall_at_15"]
            ctx.check("negative, recall@15 on the full bank", m.group(1), R["full"], src)
            ctx.check("negative, recall@15 on the pruned bank", m.group(2), R["pruned"], src)
            ctx.check("negative, the paired recall difference", m.group(3), R["delta"], src)
            ctx.check("negative, recall difference interval low", m.group(4), R["ci95"][0], src)
            ctx.check("negative, recall difference interval high", m.group(5), R["ci95"][1], src)
            _ok(ctx, "negative, the recall difference is the difference of the two arms",
                abs((R["pruned"] - R["full"]) - R["delta"]) <= 1e-3, R["delta"],
                round(R["pruned"] - R["full"], 4), src)
        m = _find(ctx, "negative, the pool-coverage loss parses",
                  r"declines from ([\d.]+) to ([\d.]+) \(\$(-[\d.]+)\$, "
                  r"\$\[(-[\d.]+),(-[\d.]+)\]\$\)", src)
        if m:
            P = PR["pool_coverage_recall_inf"]
            ctx.check("negative, pool coverage on the full bank", m.group(1), P["full"], src)
            ctx.check("negative, pool coverage on the pruned bank", m.group(2), P["pruned"], src)
            ctx.check("negative, the paired pool-coverage difference", m.group(3), P["delta"], src)
            ctx.check("negative, pool-coverage interval low", m.group(4), P["ci95"][0], src)
            ctx.check("negative, pool-coverage interval high", m.group(5), P["ci95"][1], src)
        m = _find(ctx, "negative, the pool-size sentence parses",
                  r"The mean pool size falls from ([\d.]+) to ([\d.]+) candidates", src)
        if m:
            ctx.check("negative, mean pool size, full bank", m.group(1),
                      PR["mean_pool_size"]["full"], src)
            ctx.check("negative, mean pool size, pruned bank", m.group(2),
                      PR["mean_pool_size"]["pruned"], src)


def _negative_embedding(ctx) -> None:
    ED = ctx.art("rule_embed_decomp.json")
    if ED is not None:
        src = "results/rule_embed_decomp.json"
        tv = ED["totvar"]
        m = _find(ctx, "negative, the variance decomposition parses",
                  r"attributes (\d+)\\% to the per-rule embedding and (\d+) per cent to the graph "
                  r"encoding", src)
        if m:
            ctx.check("negative, variance carried by the per-rule embedding", m.group(1),
                      round(100.0 * tv["id"] / tv["sum"]), src)
            ctx.check("negative, variance carried by the graph encoding", m.group(2),
                      round(100.0 * tv["graph"] / tv["sum"]), src)
        # the same share is quoted again where the ablation is described
        again = re.findall(r"the component carrying (\d+)\\% of the variance", ctx.flat)
        _ok(ctx, "negative, the second printing of the embedding's variance share is found",
            len(again) == 1, "1", str(len(again)), "the manuscript")
        for i, v in enumerate(again, 1):
            ctx.check(f"negative, embedding variance share, printing {i}", v,
                      round(100.0 * tv["id"] / tv["sum"]), src)

    AT = ctx.art("ablate_id_embedding.json")
    if AT is not None:
        src = "results/ablate_id_embedding.json"
        m = _find(ctx, "negative, the test-split ablation parses",
                  r"changes recall@15 by \$(-[\d.]+)\$ on the test split", src)
        if m:
            ctx.check("negative, test-split ablation at 15", m.group(1), AT["delta"]["@15"], src)
        m = _find(ctx, "negative, the withdrawn test-split effect parses",
                  r"A single test-split run on (\d+) substrates had indicated that removing the "
                  r"embedding improved recall at small \$k\$ \(\$\+([\d.]+)\$ at \$k=1\$, "
                  r"\$\+([\d.]+)\$ at \$k=5\$\)", src)
        if m:
            ctx.check("negative, the withdrawn run's substrates", m.group(1), AT["n"], src)
            ctx.check("negative, the withdrawn effect at 1", m.group(2), AT["delta"]["@1"], src)
            ctx.check("negative, the withdrawn effect at 5", m.group(3), AT["delta"]["@5"], src)

    AV = ctx.art("ablate_id_embedding_val.json")
    if AV is not None:
        src = "results/ablate_id_embedding_val.json"
        m = _find(ctx, "negative, the validation ablation parses",
                  r"paired bootstrap intervals over (\d+) substrates gives no significant change "
                  r"at any cut-off: \$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ at \$k=1\$, "
                  r"\$\+([\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ at \$k=5\$, "
                  r"\$(-[\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ at \$k=15\$, and "
                  r"\$(-[\d.]+)\$ \$\[(-[\d.]+),\+([\d.]+)\]\$ at \$k=30\$", src)
        if m:
            ctx.check("negative, the validation ablation's substrates", m.group(1), AV["n"], src)
            for i, k in enumerate(("@1", "@5", "@15", "@30")):
                ctx.check(f"negative, validation ablation delta {k}", m.group(2 + 3 * i),
                          AV["delta"][k], src)
                ctx.check(f"negative, validation ablation interval low {k}", m.group(3 + 3 * i),
                          AV["paired_bootstrap"][k]["ci95"][0], src)
                ctx.check(f"negative, validation ablation interval high {k}", m.group(4 + 3 * i),
                          AV["paired_bootstrap"][k]["ci95"][1], src)
        _ok(ctx, "negative, no cut-off separates, which is what the sentence claims",
            all(v["ci95"][0] < 0 < v["ci95"][1] for v in AV["paired_bootstrap"].values()),
            "none separated",
            ", ".join(k for k, v in AV["paired_bootstrap"].items()
                      if not v["ci95"][0] < 0 < v["ci95"][1]) or "none", src)


def _negative_dispatch(ctx) -> None:
    HD = ctx.art("hydrogen_dispatch__clean_test.json")
    EH = ctx.art("explicit_h_mechanism__clean_test.json") or ctx.art("explicit_h_mechanism.json")
    if HD is None:
        return
    banks = HD["banks"]
    rows = (("SyGMa", "sygma_175", r"\+"), ("ours", "grail_full", r"\+"),
            ("BioTransformer", "biotransformer", "-"))
    for shown, key, sign in rows:
        b = banks[key]
        m = _find(ctx, f"negative, the dispatch row for {shown} parses",
                  shown + r" & \$([\d,{}]+)\$ & \$(\d+)\$ & ([\d.]+) \$\[([\d.]+),([\d.]+)\]\$ & "
                  r"([\d.]+) & \$(" + sign + r"[\d.]+)\$ \$\[(" + sign + r"?[\d.]+),("
                  + sign + r"?[\d.]+)\]\$", SRC_HD)
        if not m:
            continue
        ctx.check(f"negative, dispatch table [{shown}, rules]", _num(m.group(1)), b["n_rules"],
                  SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, dispatched]", m.group(2),
                  b["dispatched_to_expanded"], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, dispatch]", m.group(3), b["reach"], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, dispatch interval low]", m.group(4),
                  b["ci95"][0], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, dispatch interval high]", m.group(5),
                  b["ci95"][1], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, best global]", m.group(6), b["best_global"],
                  SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, residual]", m.group(7).lstrip("+"),
                  b["residual_convention_dependence"], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, residual interval low]",
                  m.group(8).lstrip("+"), b["residual_ci95"][0], SRC_HD)
        ctx.check(f"negative, dispatch table [{shown}, residual interval high]",
                  m.group(9).lstrip("+"), b["residual_ci95"][1], SRC_HD)
        _ok(ctx, f"negative, {shown}'s residual is its dispatch minus its best global",
            abs((b["reach"] - b["best_global"]) - b["residual_convention_dependence"]) <= 5e-5,
            b["residual_convention_dependence"], round(b["reach"] - b["best_global"], 4), SRC_HD)
        _ok(ctx, f"negative, {shown}'s best global is the better legitimate arm",
            abs(max(b["global_arms"][a] for a in b["legitimate_global_arms"])
                - b["best_global"]) <= 5e-5, b["best_global"],
            max(b["global_arms"][a] for a in b["legitimate_global_arms"]), SRC_HD)

    if EH is not None:
        conv = EH["hydrogen_convention_by_bank"]
        src = "results/explicit_h_mechanism__clean_test.json"
        m = _find(ctx, "negative, the unclassifiable-template count parses",
                  r"with the \$(\d+)\$ templates whose recursive \\textsc\{smarts\} that test "
                  r"cannot see inside", src)
        if m:
            ctx.check("negative, templates the token test cannot classify", m.group(1),
                      sum(v["unclassified_recursive_smarts"] for v in conv.values()),
                      f"{src}, summed over the three banks")
        m = _find(ctx, "negative, the BioTransformer decomposition parses",
                  r"because \$(\d+)\$ of its \$(\d+)\$ templates want the convention a single "
                  r"global arm would deny them: the \$(\d+)\$ that name a hydrogen atom on their "
                  r"reactant side, and the \$(\d+)\$ recursive patterns", src)
        if m:
            bt = conv["biotransformer"]
            ctx.check("negative, BioTransformer templates dispatched", m.group(1),
                      banks["biotransformer"]["dispatched_to_expanded"], SRC_HD)
            ctx.check("negative, BioTransformer templates", m.group(2), bt["rules"], src)
            ctx.check("negative, BioTransformer templates naming a hydrogen atom", m.group(3),
                      bt["with_explicit_hydrogen"], src)
            ctx.check("negative, BioTransformer recursive patterns", m.group(4),
                      bt["unclassified_recursive_smarts"], src)
            _ok(ctx, "negative, BioTransformer's dispatched templates are those two groups",
                bt["with_explicit_hydrogen"] + bt["unclassified_recursive_smarts"]
                == banks["biotransformer"]["dispatched_to_expanded"],
                banks["biotransformer"]["dispatched_to_expanded"],
                bt["with_explicit_hydrogen"] + bt["unclassified_recursive_smarts"],
                f"{src} against {SRC_HD}")
        _ok(ctx, "negative, the two artifacts agree on how many templates each bank has",
            all(conv[k]["rules"] == banks[k]["n_rules"] for k in conv if k in banks),
            [banks[k]["n_rules"] for k in conv if k in banks],
            [conv[k]["rules"] for k in conv if k in banks], f"{src} against {SRC_HD}")

    # our own bank's dispatched templates, quoted twice in the prose after the table
    hand = re.findall(r"where dispatch pays is the one whose \$(\d+)\$ dispatched templates"
                      r"|being mixed but by the \$(\d+)\$ hand-written templates", ctx.flat)
    hand = [a or b for a, b in hand]
    _ok(ctx, "negative, both prose printings of our dispatched templates are found",
        len(hand) == 2, "2", str(len(hand)), "the manuscript")
    for i, v in enumerate(hand, 1):
        ctx.check(f"negative, our dispatched templates, printing {i}", v,
                  banks["grail_full"]["dispatched_to_expanded"], SRC_HD)
    m = _find(ctx, "negative, the two residual verdicts parse",
              r"It loses by \$(-[\d.]+)\$ \$\[(-[\d.]+),(-[\d.]+)\]\$.*?Our own bank does clear "
              r"it, by \$\+([\d.]+)\$ \$\[\+([\d.]+),\+([\d.]+)\]\$", SRC_HD)
    if m:
        bt, gr = banks["biotransformer"], banks["grail_full"]
        ctx.check("negative, BioTransformer's residual in prose", m.group(1),
                  bt["residual_convention_dependence"], SRC_HD)
        ctx.check("negative, BioTransformer's residual interval low in prose", m.group(2),
                  bt["residual_ci95"][0], SRC_HD)
        ctx.check("negative, BioTransformer's residual interval high in prose", m.group(3),
                  bt["residual_ci95"][1], SRC_HD)
        ctx.check("negative, our residual in prose", m.group(4),
                  gr["residual_convention_dependence"], SRC_HD)
        ctx.check("negative, our residual interval low in prose", m.group(5),
                  gr["residual_ci95"][0], SRC_HD)
        ctx.check("negative, our residual interval high in prose", m.group(6),
                  gr["residual_ci95"][1], SRC_HD)
    _ok(ctx, "negative, SyGMa is the structural null the paragraph rests on",
        banks["sygma_175"]["dispatched_to_expanded"] == 0
        and banks["sygma_175"]["residual_convention_dependence"] == 0.0,
        "0 dispatched, residual 0",
        f"{banks['sygma_175']['dispatched_to_expanded']} dispatched, residual "
        f"{banks['sygma_175']['residual_convention_dependence']}", SRC_HD)


def _negative_tail(ctx) -> None:
    FT = ctx.art("fragility_from_table.json")
    if FT is not None:
        src = "results/fragility_from_table.json"
        T = FT["totals"]
        m = _find(ctx, "negative, the fragility-from-the-table sentence parses",
                  r"the threshold rule predicts \$(\d+)\$ fragile pairs against \$(\d+)\$ "
                  r"observed, and replacing the single movement with its own distribution and "
                  r"summing the expected count gives \$([\d.]+)\$", src)
        if m:
            ctx.check("negative, fragile pairs predicted", m.group(1), T["threshold_rule"], src)
            ctx.check("negative, fragile pairs observed", m.group(2), T["observed"], src)
            ctx.check("negative, fragile pairs expected", m.group(3), T["expected_count"], src)
        for label, tkey, bkey in (("predicted", "threshold_rule", "predicted_fragile"),
                                  ("observed", "observed", "observed_fragile"),
                                  ("expected", "expected_count", "expected_fragile")):
            got = sum(b[bkey] for b in FT["leaderboards"].values())
            _ok(ctx, f"negative, the {label} total is the two boards summed",
                abs(got - T[tkey]) <= 5e-9, T[tkey], round(got, 4), src)

    SH = ctx.art("setsize_headroom.json")
    if SH is not None:
        src = "results/setsize_headroom.json"
        m = _find(ctx, "negative, the size-forecast sentence parses",
                  r"Least squares on heavy atoms, rings and rotatable bonds, fitted on the "
                  r"training split and applied to test, reaches \$([\d.]+)\$ macro F1 against the "
                  r"deployed policy's \$([\d.]+)\$ -- and against \$([\d.]+)\$ for simply emitting "
                  r"one candidate to every substrate", src)
        if m:
            ctx.check("negative, the forecast arm's F1", m.group(1),
                      SH["arms"]["predicted count"]["f1"], src)
            ctx.check("negative, the deployed policy's F1", m.group(2),
                      SH["arms"]["fixed k=15"]["f1"], src)
            ctx.check("negative, the best constant's F1", m.group(3),
                      SH["arms"]["fixed k=1"]["f1"], src)
        _ok(ctx, "negative, the constant the forecast is compared with is the best one",
            SH["config"]["best_global_constant"] == "fixed k=1", "fixed k=1",
            SH["config"]["best_global_constant"], src)
        _ok(ctx, "negative, the forecast arm is the regression the sentence describes",
            "least squares on heavy atoms, rings, rotatable bonds"
            in SH["config"]["count_model"], "heavy atoms, rings, rotatable bonds",
            SH["config"]["count_model"][:46], src)


# ---------------------------------------------------------------- app/closure.tex


def _closure(ctx) -> None:
    RC = ctx.art("reference_closure.json")
    if RC is not None:
        _closure_corpus(ctx, RC)
        _closure_worth(ctx, RC)
    CR = ctx.art("composition_recovery.json")
    if CR is not None:
        _closure_recovery(ctx, CR)


def _closure_corpus(ctx, RC) -> None:
    G, T = RC["corpus_graph"], RC["test_only_graph"]
    m = _find(ctx, "closure, the corpus-graph sentence parses",
              r"It carries \$([\d,{}]+)\$ edges and \$([\d,{}]+)\$ two-step compositions, which "
              r"resolve to \$([\d,{}]+)\$ distinct composed pairs\. The direct edge is annotated "
              r"for \$(\d+)\$ of them\. The remaining \$([\d,{}]+)\$", SRC_RC)
    if m:
        ctx.check("closure, corpus edges", _num(m.group(1)), G["edges"], SRC_RC)
        ctx.check("closure, two-step compositions", _num(m.group(2)),
                  G["two_step_compositions"], SRC_RC)
        ctx.check("closure, distinct composed pairs", _num(m.group(3)),
                  G["distinct_composed_pairs"], SRC_RC)
        ctx.check("closure, composed pairs the corpus also annotates", m.group(4),
                  G["distinct_composed_pairs"] - G["composed_pairs_not_annotated"],
                  f"{SRC_RC}, distinct minus unannotated")
        ctx.check("closure, composed pairs it denies in one step", _num(m.group(5)),
                  G["composed_pairs_not_annotated"], SRC_RC)

    # the annotated count and the distinct-pair count are each printed more than once
    ann = re.findall(r"annotated for \$(\d+)\$ of them|turn \$(\d+)\$ annotated compositions",
                     ctx.flat)
    ann = [a or b for a, b in ann]
    _ok(ctx, "closure, both printings of the annotated-composition count are found",
        len(ann) == 2, "2", str(len(ann)), "the manuscript")
    for i, v in enumerate(ann, 1):
        ctx.check(f"closure, annotated compositions, printing {i}", v,
                  G["distinct_composed_pairs"] - G["composed_pairs_not_annotated"],
                  f"{SRC_RC}, distinct minus unannotated")
    dis = re.findall(r"resolve to \$([\d,{}]+)\$ distinct composed pairs"
                     r"|annotated compositions into \$([\d,{}]+)\$"
                     r"|against the corpus's \$([\d,{}]+)\$", ctx.flat)
    dis = [a or b or c for a, b, c in dis]
    _ok(ctx, "closure, all three printings of the distinct-pair count are found",
        len(dis) == 3, "3", str(len(dis)), "the manuscript")
    for i, v in enumerate(dis, 1):
        ctx.check(f"closure, distinct composed pairs, printing {i}", _num(v),
                  G["distinct_composed_pairs"], SRC_RC)

    m = _find(ctx, "closure, the closed-share sentence parses",
              r"The corpus is closed at \$([\d.]+)\\%\$ and the test split alone at "
              r"\$([\d.]+)\\%\$, on \$(\d+)\$ composed pairs", SRC_RC)
    if m:
        ctx.check("closure, the corpus's closed share", m.group(1),
                  100.0 * (G["distinct_composed_pairs"] - G["composed_pairs_not_annotated"])
                  / G["distinct_composed_pairs"], f"{SRC_RC}, from the two counts")
        ctx.check("closure, the test split's closed share", m.group(2),
                  100.0 * (T["distinct_composed_pairs"] - T["composed_pairs_not_annotated"])
                  / T["distinct_composed_pairs"], f"{SRC_RC}, from the two counts")
        ctx.check("closure, the test split's composed pairs", m.group(3),
                  T["distinct_composed_pairs"], SRC_RC)


def _closure_worth(ctx, RC) -> None:
    W = RC["scored_wrong_but_corpus_derivable"]
    D = RC["derivable_share_differs_by_method"]
    m = _find(ctx, "closure, the derivable-share sentence parses",
              r"the share the corpus itself reaches from that substrate in two steps is "
              r"\$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ for GRAIL, \$([\d.]+)\$ "
              r"\$\[([\d.]+),([\d.]+)\]\$ for MetaPredictor and \$([\d.]+)\$ "
              r"\$\[([\d.]+),([\d.]+)\]\$ for SyGMa", SRC_RC)
    if m:
        for i, meth in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
            ctx.check(f"closure, derivable share, {meth}", m.group(1 + 3 * i),
                      W[meth]["share_of_wrong_output"], SRC_RC)
            ctx.check(f"closure, derivable share interval low, {meth}", m.group(2 + 3 * i),
                      W[meth]["ci95"][0], SRC_RC)
            ctx.check(f"closure, derivable share interval high, {meth}", m.group(3 + 3 * i),
                      W[meth]["ci95"][1], SRC_RC)
    m = _find(ctx, "closure, the largest pairwise difference parses",
              r"the largest being MetaPredictor over SyGMa at \$\+([\d.]+)\$ "
              r"\$\[\+([\d.]+),\+([\d.]+)\]\$", SRC_RC)
    if m:
        d = D["MetaPredictor vs SyGMa"]
        ctx.check("closure, MetaPredictor over SyGMa", m.group(1), d["delta"], SRC_RC)
        ctx.check("closure, MetaPredictor over SyGMa, interval low", m.group(2), d["ci95"][0],
                  SRC_RC)
        ctx.check("closure, MetaPredictor over SyGMa, interval high", m.group(3), d["ci95"][1],
                  SRC_RC)
        _ok(ctx, "closure, it is the largest of the certified differences",
            abs(d["delta"]) == max(abs(v["delta"]) for v in D.values() if v["certified"]),
            abs(d["delta"]), max(abs(v["delta"]) for v in D.values() if v["certified"]), SRC_RC)
    _ok(ctx, "closure, two of the three differences are certified",
        sum(1 for v in D.values() if v["certified"]) == 2, "two",
        str(sum(1 for v in D.values() if v["certified"])), SRC_RC)

    B = RC["by_closure_depth"]
    m = _find(ctx, "closure, the credited-precision sentence parses",
              r"raises precision from \$([\d.]+)\$ to \$([\d.]+)\$ for GRAIL, from \$([\d.]+)\$ "
              r"to \$([\d.]+)\$ for MetaPredictor and from \$([\d.]+)\$ to \$([\d.]+)\$ for "
              r"SyGMa; a third step adds a further \$([\d.]+)\$", SRC_RC)
    if m:
        for i, meth in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
            ctx.check(f"closure, precision at depth one, {meth}", m.group(1 + 2 * i),
                      B["1"][meth]["precision"], SRC_RC)
            ctx.check(f"closure, precision at depth two, {meth}", m.group(2 + 2 * i),
                      B["2"][meth]["precision"], SRC_RC)
        meths = ("GRAIL", "MetaPredictor", "SyGMa")
        third = max(B["3"][x]["precision"] - B["2"][x]["precision"] for x in meths)
        ctx.check("closure, what a third step adds at most", m.group(7), third,
                  f"{SRC_RC}, the largest gain from depth two to depth three")
        # "a further" is a diminishing addition, which is a property of the artifact and not of
        # the maximum just taken: asserting the max bounds its own arguments would be vacuous.
        _ok(ctx, "closure, the third step adds less than the second did, for every method",
            all(B["3"][x]["precision"] - B["2"][x]["precision"]
                <= B["2"][x]["precision"] - B["1"][x]["precision"] for x in meths),
            [round(B["2"][x]["precision"] - B["1"][x]["precision"], 4) for x in meths],
            [round(B["3"][x]["precision"] - B["2"][x]["precision"], 4) for x in meths], SRC_RC)
    _ok(ctx, "closure, the ordering by F1 is the same at every depth",
        RC["ordering_changes_with_depth"] is False
        and len({tuple(B[d]["ordering_by_f1"]) for d in B}) == 1, "the same at every depth",
        str(RC["ordering_changes_with_depth"]), SRC_RC)


def _closure_recovery(ctx, CR) -> None:
    P = CR["per_method"]
    m = _find(ctx, "closure, the composable-substrate sentence parses",
              r"which leaves \$(\d+)\$ substrates for GRAIL, \$(\d+)\$ for MetaPredictor and "
              r"\$(\d+)\$ for SyGMa", SRC_CR)
    if m:
        for i, meth in enumerate(("GRAIL", "MetaPredictor", "SyGMa"), start=1):
            ctx.check(f"closure, substrates with a predicted intermediate, {meth}", m.group(i),
                      P[meth]["substrates_with_a_predicted_intermediate"], SRC_CR)
    m = _find(ctx, "closure, the random-intermediate control parses",
              r"That control recovers \$([\d.]+)\$ for all three methods", SRC_CR)
    if m:
        for meth in ("GRAIL", "MetaPredictor", "SyGMa"):
            ctx.check(f"closure, the random-intermediate control, {meth}", m.group(1),
                      P[meth]["same_through_random_intermediates"], SRC_CR)
    m = _find(ctx, "closure, the recovery sentence parses",
              r"composing the method's own predictions recovers \$([\d.]+)\$ "
              r"\$\[([\d.]+),([\d.]+)\]\$ for GRAIL, \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ for "
              r"MetaPredictor and \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ for SyGMa", SRC_CR)
    if m:
        for i, meth in enumerate(("GRAIL", "MetaPredictor", "SyGMa")):
            ctx.check(f"closure, missed references recovered, {meth}", m.group(1 + 3 * i),
                      P[meth]["share_of_missed_references_recovered"], SRC_CR)
            ctx.check(f"closure, recovery interval low, {meth}", m.group(2 + 3 * i),
                      P[meth]["ci95"][0], SRC_CR)
            ctx.check(f"closure, recovery interval high, {meth}", m.group(3 + 3 * i),
                      P[meth]["ci95"][1], SRC_CR)
    _ok(ctx, "closure, every method is separated from its control, as the sentence says",
        all(P[x]["separated"] for x in P), "all separated",
        ", ".join(x for x in P if not P[x]["separated"]) or "all separated", SRC_CR)

    S = CR["differs_by_method"]
    m = _find(ctx, "closure, the recovery differences parse",
              r"MetaPredictor recovers \$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ more than GRAIL and "
              r"\$([\d.]+)\$ \$\[([\d.]+),([\d.]+)\]\$ more than SyGMa, both separated", SRC_CR)
    if m:
        g = S["GRAIL vs MetaPredictor"]
        ctx.check("closure, MetaPredictor over GRAIL", m.group(1), abs(g["delta"]), SRC_CR)
        ctx.check("closure, MetaPredictor over GRAIL, interval low", m.group(2),
                  abs(g["ci95"][1]), SRC_CR)
        ctx.check("closure, MetaPredictor over GRAIL, interval high", m.group(3),
                  abs(g["ci95"][0]), SRC_CR)
        s = S["MetaPredictor vs SyGMa"]
        ctx.check("closure, MetaPredictor over SyGMa in recovery", m.group(4), s["delta"], SRC_CR)
        ctx.check("closure, MetaPredictor over SyGMa in recovery, interval low", m.group(5),
                  s["ci95"][0], SRC_CR)
        ctx.check("closure, MetaPredictor over SyGMa in recovery, interval high", m.group(6),
                  s["ci95"][1], SRC_CR)
        _ok(ctx, "closure, both of those differences are separated",
            g["separated"] and s["separated"], "both separated",
            f"{g['separated']} and {s['separated']}", SRC_CR)
    m = _find(ctx, "closure, the price of composing parses",
              r"costs \$(\d+)\$ additional candidates for MetaPredictor and \$(\d+)\$ for GRAIL",
              SRC_CR)
    if m:
        ctx.check("closure, candidates per reference recovered, MetaPredictor", m.group(1),
                  round(P["MetaPredictor"]["candidates_added_per_reference_recovered"]), SRC_CR)
        ctx.check("closure, candidates per reference recovered, GRAIL", m.group(2),
                  round(P["GRAIL"]["candidates_added_per_reference_recovered"]), SRC_CR)


def register(ctx) -> None:
    _robust(ctx)
    _negative(ctx)
    _closure(ctx)
    register_orphans_fixed(ctx)


def register_orphans_fixed(ctx) -> None:
    """Five figures the appendix printed that no artifact produced, and four of them disagreed.

    They were found by asking of every unread numeral what leaf it comes from, which is the
    question the coverage instrument exists to force. Each is bound here to the leaf it should
    have come from all along.
    """
    import json as _json
    import re as _re

    w = ctx.art("robust_order_wmt23.json")
    if w:
        ann = sorted(b["n_annotators"] for b in w["boards"].values())
        m = _re.search(r"several ratings per segment, and between \$(\d+)\$ and \$(\d+)\$ "
                       r"annotators", ctx.flat)
        ctx.checks.append((bool(m), "wmt23, the annotator range is stated", "present",
                           "matched" if m else "not matched", "results/robust_order_wmt23.json"))
        if m:
            ctx.check("wmt23, fewest annotators on a board", m.group(1), ann[0],
                      "results/robust_order_wmt23.json")
            ctx.check("wmt23, most annotators on a board", m.group(2), ann[-1],
                      "results/robust_order_wmt23.json")

    mx = ctx.art("metatox_smirks_preds.json")
    if mx:
        m = _re.search(r"PASS writes that score only for the \$([\d,{}]+)\$ of \$([\d,{}]+)\$ "
                       r"predictions", ctx.flat)
        ctx.checks.append((bool(m), "PASS, the thresholded share is stated", "present",
                           "matched" if m else "not matched", "results/metatox_smirks_preds.json"))
        if m:
            ctx.check("PASS, predictions carrying a score",
                      m.group(1).replace("{,}", ""),
                      round(mx["mean_output_above_threshold"] * mx["n_substrates"]),
                      "results/metatox_smirks_preds.json")
            ctx.check("PASS, predictions in total", m.group(2).replace("{,}", ""),
                      mx["n_predictions"], "results/metatox_smirks_preds.json")

    cen = ctx.art("convention_census.json")
    if cen:
        sm = cen["nested_not_counted"]["hand-written collection, smallest"]["templates"]
        m = _re.search(r"of (\d+), 656, 500 and 1\{,\}051 rules", ctx.flat)
        ctx.check("arch, the smallest curated bank", m and m.group(1), sm,
                  "results/convention_census.json")

    bm = ctx.art("budget_matched_leaderboard.json")
    if bm:
        ratio = bm["by_method"]["SyGMa"]["mean_output"] / bm["by_method"]["GRAIL"]["mean_output"]
        m = _re.search(r"([\d.]+)-fold difference in output volume it reflects is real", ctx.flat)
        ctx.check("budget, the output-volume ratio", m and m.group(1), ratio,
                  "results/budget_matched_leaderboard.json")

    # the epoch budget is a property of the released preset, so the preset is the artifact
    pre = ctx.root / "grail_metabolism/experiments/presets.py"
    if pre.exists():
        mm = _re.search(r"generator_optim=OptimConfig\(lr=1e-4, epochs=(\d+)", pre.read_text())
        said = _re.findall(r"Adam at a learning rate of \$10\^\{-4\}\$ for at most (\d+) epochs",
                           ctx.flat)
        ctx.checks.append((bool(mm) and len(said) == 2 and all(s == mm.group(1) for s in said),
                           "arch, the epoch budget both stages state",
                           mm.group(1) if mm else "?", ", ".join(said) or "not stated",
                           "grail_metabolism/experiments/presets.py"))
