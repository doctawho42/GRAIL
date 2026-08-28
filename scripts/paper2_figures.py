"""The manuscript's figures, drawn from the artifacts that measured them.

A figure typed from a table falls behind the table, and a table typed from an artifact falls
behind the artifact; both happened in the first draft of this paper. These are regenerated from
results/ on every build, so a figure cannot outlive the run it describes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
W = 3.35          # one NAR column, inches


def art(name):
    return json.loads((ROOT / "results" / name).read_text())


def fig_sweep():
    d = art("deployment_table.json")
    rec, con = d["recall_micro"], d["contrasts"]
    ks = sorted((int(k) for k in rec), key=int)
    style = {"whole bank": ("GRAIL exhaustive", "-", "o", "#1b1b1b"),
             "trained budget": ("GRAIL interactive", "-", "s", "#7a7a7a"),
             "metatox": ("MetaTox", "--", "^", "#b2182b"),
             "sygma": ("SyGMa", "--", "v", "#2166ac"),
             "metapredictor": ("MetaPredictor", "--", "D", "#4d9221")}
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for arm, (lab, ls, mk, col) in style.items():
        if arm not in rec[str(ks[0])]:
            continue
        y = [rec[str(k)][arm] for k in ks]
        ax.plot(ks, y, ls, marker=mk, ms=3, lw=1.1, color=col, label=lab,
                zorder=3 if arm.startswith(("whole", "trained")) else 2)

    # the three regions, from the contrasts rather than from the eye
    ours = ("whole bank", "trained budget")
    others = [a for a in style if a not in ours and a in rec[str(ks[0])]]
    band = []
    for k in ks:
        bo = max(others, key=lambda a: rec[str(k)][a])
        bu = max(ours, key=lambda a: rec[str(k)][a])
        c = con[str(k)][f"{bu} - {bo}"]
        band.append("lose" if (c["gap"] < 0 and c["excludes_zero"])
                    else "win" if (c["gap"] > 0 and c["excludes_zero"]) else "tie")
    for i, k in enumerate(ks):
        lo = ks[i - 1] if i else ks[0] - 0.5
        hi = ks[i + 1] if i < len(ks) - 1 else ks[-1] + 3
        if band[i] == "lose":
            ax.axvspan((lo + k) / 2, (k + hi) / 2, color="#b2182b", alpha=0.06, lw=0, zorder=0)
        elif band[i] == "win":
            ax.axvspan((lo + k) / 2, (k + hi) / 2, color="#2166ac", alpha=0.06, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("output budget $k$")
    ax.set_ylabel("micro recall@$k$")
    ax.legend(loc="lower right", fontsize=6.4, handlelength=1.8)
    ax.set_ylim(0, 0.78)
    fig.savefig(OUT / "fig_sweep.pdf")
    plt.close(fig)
    return band


def fig_ceiling():
    cen = art("novel_type_census.json")
    usp = art("uspto_type_overlap.json")
    cov = art("coverage_gap_types.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W * 2.06, 2.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # left: where the uncovered references go, and what a library supplies
    g = cov["gap"]
    labels = ["type absent\nfrom the bank", "type present,\nrule did not fire", "not typeable"]
    vals = [g["novel_type"], g["known_type"], g["untypeable"]]
    cols = ["#b2182b", "#f4a582", "#cccccc"]
    left = 0
    for v, c, lab in zip(vals, cols, labels):
        a1.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=0.6)
        a1.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=7,
                color="white" if c != "#cccccc" else "#333333")
        left += v
    hit = usp["overlap"]["misses_those_types_carry"]
    a1.barh(-0.75, hit, color="#2166ac", height=0.5)
    a1.text(hit + 8, -0.75, f"{hit} recoverable from "
            f"{usp['uspto']['templates']:,} synthetic templates", va="center", fontsize=6.6)
    a1.set_yticks([0, -0.75])
    a1.set_yticklabels([f"{cov['uncovered_pairs']} uncovered\nreferences", "of which"], fontsize=6.8)
    a1.set_xlim(0, cov["uncovered_pairs"] * 1.02)
    a1.set_xlabel("references")
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)

    # right: the specification curve
    cur = cen["granularity_curve"]
    x = np.arange(len(cur))
    share = [c["share_of_mass_in_singletons"] for c in cur]
    usable = [c["determines_a_product"] for c in cur]
    a2.bar(x, share, color=["#b2182b" if u else "#cccccc" for u in usable], width=0.6)
    for i, c in enumerate(cur):
        a2.text(i, share[i] + 0.02, f"{c['types']}", ha="center", fontsize=6.5)
    a2.set_xticks(x)
    a2.set_xticklabels(["exact\nmultiset", "counts\ndropped", "element\npairs", "bond\ncount"],
                       fontsize=6.5)
    a2.set_ylabel("share of misses in singleton types")
    a2.set_ylim(0, 1.0)
    a2.axhline(0.5, color="#999999", lw=0.5, ls=":")
    a2.text(3.35, 0.93, "type names a\ntransformation", fontsize=6.2, ha="right", color="#b2182b")
    a2.text(3.35, 0.16, "it does not", fontsize=6.2, ha="right", color="#777777")
    fig.savefig(OUT / "fig_ceiling.pdf")
    plt.close(fig)


def fig_cost():
    env = art("cost_envelope.json")["rows"]
    mt = art("mode_timings.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W * 2.06, 2.1),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    fin = [r for r in env if r["finished"]]
    bad = [r for r in env if not r["finished"]]
    a1.scatter([r["heavy"] for r in fin], [r["t_generate"] for r in fin], s=9,
               color="#1b1b1b", lw=0, label="finished")
    dl = art("cost_envelope.json")["deadline_s"]
    a1.scatter([r["heavy"] for r in bad], [dl] * len(bad), s=22, marker="x",
               color="#b2182b", lw=0.9, label=f"did not finish in {int(dl)} s")
    a1.set_yscale("log")
    a1.set_xlabel("heavy atoms in the substrate")
    a1.set_ylabel("generator seconds")
    a1.legend(fontsize=6.4, loc="upper left")

    i, e = mt["interactive"], mt["exhaustive"]
    bars = [("interactive\nmedian", i["median_s"], "#7a7a7a"),
            ("interactive\nmean", i["mean_s"], "#7a7a7a"),
            ("interactive\np90", i["p90_s"], "#7a7a7a"),
            ("exhaustive\nmedian", e["median_s"], "#1b1b1b"),
            ("interactive\nslowest", i["max_s"], "#b2182b")]
    a2.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for j, b in enumerate(bars):
        a2.text(j, b[1] * 1.15, f"{b[1]}", ha="center", fontsize=6.4)
    a2.set_yscale("log")
    a2.set_xticks(range(len(bars)))
    a2.set_xticklabels([b[0] for b in bars], fontsize=6.2)
    a2.set_ylabel("seconds per substrate")
    a2.set_ylim(0.2, 600)
    fig.savefig(OUT / "fig_cost.pdf")
    plt.close(fig)


# The worked example's four annotated metabolites. Colour encodes the SITE, not the metabolite:
# the three phosphorylations fire on the same two substrate atoms under three different rules,
# which is the point the panel has to make, so giving each its own colour would hide it.
DEAMINATION, PHOSPHORYLATION = "#b2182b", "#2166ac"
CASE = [("FIRDBEQIJQERSE-UHFFFAOYSA-N", "dFdU", DEAMINATION),
        ("KNTREFQOVSMROS-UHFFFAOYSA-N", "dFdCMP", PHOSPHORYLATION),
        ("FRQISCZGNNXEMD-UHFFFAOYSA-N", "dFdCDP", PHOSPHORYLATION),
        ("YMOXEIOKAJSRQX-UHFFFAOYSA-N", "dFdCTP", PHOSPHORYLATION)]


def _rgb(h, tint=0.0):
    """The colour, optionally blended toward white.

    MolDraw2DCairo ignores a fourth alpha channel in this RDKit, so a pale highlight has to be a
    pale colour: the atom labels are drawn over the fill and must stay legible through it.
    """
    c = tuple(int(h[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return tuple(v + (1.0 - v) * tint for v in c)


def _substrate_png(smiles, site_groups, width=900, height=520):
    """The substrate with each firing site shaded in its colour.

    Drawn from the atom indices the pipeline reported, not from a hand-marked depiction: the
    panel's claim is that the localisation is a computed field of the prediction.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    highlight, colours = [], {}
    for atoms, rgb in site_groups:
        for a in atoms:
            if a < mol.GetNumAtoms():
                highlight.append(a)
                colours[a] = rgb
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    o = d.drawOptions()
    o.bondLineWidth = 2
    o.highlightRadius = 0.36
    o.fixedFontSize = 24
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlight,
                                       highlightAtomColors=colours)
    d.FinishDrawing()
    return d.GetDrawingText()


def fig_case():
    """The worked example: where the rules fire, and where the two arms put the four answers."""
    import io

    import matplotlib.image as mpimg
    from matplotlib.patches import Circle

    inter, exh = art("case_study.json"), art("case_study_exhaustive.json")
    hits = {arm: {c["key"]: c for c in d["candidates"] if c["is_reference"]}
            for arm, d in (("i", inter), ("e", exh))}

    # the two sites, taken from the exhaustive arm, which produces all four metabolites. The
    # phosphorylation atoms are a set union over three rules and collapse to one site; that
    # collapse is the observation, so it is computed here rather than asserted.
    sites = {DEAMINATION: set(), PHOSPHORYLATION: set()}
    for key, _, col in CASE:
        c = hits["e"].get(key)
        if c:
            sites[col] |= set(c["firing_atoms"])
    n_phos_rules = len({hits["e"][k]["rule_id"] for k, _, col in CASE
                        if col == PHOSPHORYLATION and k in hits["e"]})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W * 2.06, 2.7),
                                 gridspec_kw={"width_ratios": [1.0, 1.3]})
    png = _substrate_png(inter["substrate"],
                         [(sorted(atoms), _rgb(col, 0.68)) for col, atoms in sites.items()])
    a1.imshow(mpimg.imread(io.BytesIO(png), format="png"))
    a1.set_axis_off()
    a1.set_title("substrate, shaded by the atoms the rules fired on", fontsize=7.5, pad=4)
    for i, (col, text) in enumerate((
            (DEAMINATION, "deamination, 1 rule"),
            (PHOSPHORYLATION, f"phosphorylation, {n_phos_rules} rules"))):
        a1.add_patch(Circle((0.055, -0.055 - 0.075 * i), 0.018, transform=a1.transAxes,
                            color=col, clip_on=False))
        a1.text(0.095, -0.055 - 0.075 * i, text, transform=a1.transAxes, fontsize=7,
                va="center", color="#333333")

    # the ranks: every candidate as a tick, the annotated ones marked and named
    rows = [("exhaustive", "e", exh, 1.0), ("interactive", "i", inter, 0.0)]
    for _, _, d, y in rows:
        a2.plot([c["rank"] for c in d["candidates"]], [y] * d["n_candidates"], "|",
                color="#c4c4c4", ms=9, mew=0.8, zorder=2)
    # labels are staggered by rank order so neighbouring ones cannot overprint
    for _, arm, d, y in rows:
        found = sorted((c for c in d["candidates"] if c["is_reference"]),
                       key=lambda c: c["rank"])
        for j, c in enumerate(found):
            col = next(x for k, _, x in CASE if k == c["key"])
            name = next(x for k, x, _ in CASE if k == c["key"])
            a2.plot([c["rank"]], [y], "o", ms=5.5, color=col, mec="white", mew=0.9, zorder=4)
            off = (9, 26, 43)[j % 3] if y else -16
            a2.annotate(f"{name}\nrule {c['rule_id']}", (c["rank"], y),
                        textcoords="offset points", xytext=(0, off),
                        ha="center", va="bottom" if y else "top",
                        fontsize=6.3, color=col, linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=col,
                                        shrinkA=1, shrinkB=3) if y and j % 3 else None)
    for k in (15, 30):
        a2.axvline(k, color="#bdbdbd", lw=0.7, ls=":", zorder=0)
        a2.annotate(f"$k={k}$", (k, -0.62), ha="center", fontsize=6.5, color="#777777")
    a2.set_yticks([0, 1])
    a2.set_yticklabels(
        [f"interactive\n{inter['configuration']['rule_budget']} rules, "
         f"{inter['n_candidates']} returned\n{inter['n_references'] and len(inter['reference_ranks'])} of "
         f"{inter['n_references']} found",
         f"exhaustive\n{exh['configuration']['rule_budget']:,} rules, "
         f"{exh['n_candidates']} returned\n{len(exh['reference_ranks'])} of "
         f"{exh['n_references']} found"], fontsize=6.8)
    a2.set_xlabel("rank in the returned list")
    a2.set_xlim(0.3, max(exh["n_candidates"], 32) + 4)
    a2.set_ylim(-0.9, 1.95)
    a2.set_xscale("symlog", linthresh=30, linscale=1.6)
    a2.set_xticks([1, 5, 10, 15, 20, 30, 50, 100])
    a2.set_xticklabels(["1", "5", "10", "15", "20", "30", "50", "100"])
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    a2.set_title("where the four annotated metabolites land", fontsize=7.5, pad=10)
    fig.savefig(OUT / "fig_case.pdf")
    plt.close(fig)


def digest():
    """A hash of the numbers the figures draw.

    Two runs of matplotlib do not produce byte-identical PDFs, so a test cannot compare the files.
    It can compare what went into them, which is the property that matters: a figure is stale when
    its artifact has moved, not when its timestamp has.
    """
    import hashlib
    d = art("deployment_table.json")
    cen = art("novel_type_census.json")
    usp = art("uspto_type_overlap.json")
    mt = art("mode_timings.json")
    env = art("cost_envelope.json")
    payload = json.dumps({
        "sweep": d["recall_micro"], "contrasts": d["contrasts"],
        "grain": cen["granularity_curve"], "gap": art("coverage_gap_types.json")["gap"],
        "uspto": usp["overlap"], "modes": {"i": mt["interactive"], "e": mt["exhaustive"]},
        "env": [(r["heavy"], r["finished"], r.get("t_generate")) for r in env["rows"]],
        "case": {a: [(c["rank"], c["key"], c["rule_id"], c["firing_atoms"])
                     for c in art(f)["candidates"] if c["is_reference"]]
                 for a, f in (("i", "case_study.json"), ("e", "case_study_exhaustive.json"))},
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    band = fig_sweep()
    fig_ceiling()
    fig_cost()
    fig_case()
    (OUT / "figures.sha256").write_text(digest() + "\n")
    print("  fig_sweep.pdf, fig_ceiling.pdf, fig_cost.pdf, fig_case.pdf")
    print(f"  the sweep's shaded regions, from the contrasts: {band}")
    print(f"  data digest {digest()[:24]}")
