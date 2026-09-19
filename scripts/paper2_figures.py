"""The manuscript's figures, drawn from the artifacts that measured them.

A figure typed from a table falls behind the table, and a table typed from an artifact falls
behind the artifact; both happened in the first draft of this paper. These are regenerated from
results/ on every build, so a figure cannot outlive the run it describes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper2"

# One categorical palette for every figure. It is not chosen by eye: the order below is the one
# that passes all five checks of the validator shipped with the dataviz guidance -- lightness
# band, chroma floor, adjacent-pair separation under simulated colour-vision deficiency, the
# normal-vision floor, and contrast against a light surface. Two orders that looked reasonable
# failed it: the ColorBrewer-style red/green pair separates by DeltaE 2.5 under deuteranopia, and
# swapping slots 2 and 4 puts vermillion beside amber at DeltaE 0.9. Legends therefore follow
# this order rather than any other.
#
# Adjacent luminances are as close as 0.01, so the palette is NOT sufficient in greyscale. Every
# figure carries a second encoding -- distinct markers, line styles and direct labels -- so
# identity is never colour alone, which is also what the guidance requires for a warning-band
# pair.
# Okabe-Ito, colour-blind safe. The seventh entry arrived with the fifth comparator; a
# modulo into a six-colour list had given it the first arm's blue.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#B07A00", "#AD5A87", "#56514C", "#7A5C3E",
           # An eighth, for GLORYxR's two settings, which share it. Chosen away from the mauve at
           # index 4 and the blue at index 0, the two it could be confused with.
           "#5B3FA8"]

# Polarity, for the outcome bands of the sweep. These are regions rather than entities, so
# they must not wear a series hue: the first draft shaded "GRAIL leads" in the same blue as
# the GRAIL exhaustive line, which makes one colour mean two things in one figure. Used at a
# low alpha behind the marks and always with a text label, never as the only signal.
BAND_TRAILS, BAND_LEADS = "#8c6bb1", "#41ab5d"

# Ink, replacing the seven different greys these figures used to mix. Text and rules wear ink;
# only marks wear a series colour.
INK, INK_MUTED, INK_FAINT = "#1a1a1a", "#666666", "#c9c9c9"

plt.rcParams.update({
    # ACS asks for Helvetica or Arial in figure lettering, at no less than 4.5 pt final size and
    # with no rule thinner than 0.5 pt
    # Type 3 is matplotlib's default for PDF text and ACS production returns it: the glyphs are
    # embedded as drawing programs with no ToUnicode map, so a figure label is neither searchable
    # nor extractable. 42 is TrueType, which embeds a real font with a character map.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    # Matplotlib renders anything between dollar signs with its own math font, whose default is
    # DejaVu, so a figure whose lettering is Helvetica still shipped DejaVu for "$k$" and for
    # every sign in the criterion legend. Two typefaces in one figure is a house-style failure
    # and the second one is not the face ACS asks for. Binding the math faces to the text family
    # keeps a budget axis and its label in one typeface.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold", "mathtext.sf": "Helvetica",
    "mathtext.default": "it",
    "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.dpi": 600, "figure.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
})

# ACS: a single-column graphic is at most 240 pt (3.33 in) and a double-column one between 300
# and 504 pt (4.167 and 7 in). W was 3.35 in, which is 241 pt and over the single-column limit.
W = 3.33

# ACS asks for 300 dpi on colour artwork and 1200 on line art. The figures are vector except for
# the rendered structures, which are the one raster element; saving at 600 puts those comfortably
# past the colour bar without making the files unwieldy.
FIG_DPI = 600


def save(fig, stem):
    """The build's PDF and the format ACS accepts.

    ACS takes TIF, JPG, PNG and EPS and does not take PDF, so a submission built only from the
    file pdflatex needs has nothing to upload. These figures are vector apart from the rendered
    structures, so EPS is the format that keeps them vector; the PDF stays because the manuscript
    is compiled with pdflatex.
    """
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.eps")


def art(name):
    return json.loads((ROOT / "results" / name).read_text())


def fig_sweep():
    d = art("deployment_table.json")
    rec, con = d["recall_micro"], d["contrasts"]
    ks = sorted((int(k) for k in rec), key=int)
    style = {"whole bank": ("GRAIL exhaustive", "-", "o", PALETTE[0]),
             "trained budget": ("GRAIL interactive", "-", "s", PALETTE[1]),
             "metatox": ("MetaTox", "--", "^", PALETTE[2]),
             "sygma": ("SyGMa", "--", "v", PALETTE[3]),
             "metapredictor": ("MetaPredictor", "--", "D", PALETTE[4]),
             "biotransformer": ("BioTransformer", "--", "*", PALETTE[5]),
             "gloryx": ("GLORYx", "--", "P", PALETTE[6]),
             # GLORYxR's two site-of-metabolism settings share one hue and differ in dash and
             # marker, because they are one system under one declared knob and not two systems.
             # Giving them separate hues would say the opposite, and the palette is out of hues
             # that stay apart in print at this size anyway.
             "gloryxr_default": ("GLORYxR, default SoM", "--", "X", PALETTE[7]),
             "gloryxr_strict": ("GLORYxR, strict SoM", ":", "<", PALETTE[7])}
    # An arm the sweep computed and this figure has no style for would be a line the reader never
    # sees while the abstract counts the method: that is how the fifth comparator went missing.
    unstyled = [a for a in rec[str(ks[0])] if a not in style]
    if unstyled:
        raise SystemExit(f"deployment_table.json carries arms this figure has no style for: "
                         f"{', '.join(unstyled)}")

    # Taller than the data needs, because the legend sits under the axes. At seven series it
    # fitted inside the lower right; at nine it covered the curves it was naming and the curves
    # ran through its text, and its opaque background also cut a hole in the "leads" band.
    fig, ax = plt.subplots(figsize=(W, 3.15))
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
    # Geometric midpoints, because the axis is logarithmic. Arithmetic ones put a band's edge
    # where the eye does not expect it: with budgets at 20 and 30 the arithmetic midpoint 25 sits
    # past the visual centre, and a referee read the band as reaching a budget it does not cover.
    for i, k in enumerate(ks):
        lo = ks[i - 1] if i else ks[0] * 0.75
        hi = ks[i + 1] if i < len(ks) - 1 else ks[-1] * 1.1
        left, right = (lo * k) ** 0.5, (k * hi) ** 0.5
        if band[i] == "lose":
            ax.axvspan(left, right, color=BAND_TRAILS, alpha=0.10, lw=0, zorder=0)
        elif band[i] == "win":
            ax.axvspan(left, right, color=BAND_LEADS, alpha=0.10, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("output budget $k$")
    ax.set_ylabel("micro recall@$k$")
    # the bands are labelled, so the polarity never rests on colour alone
    for lab, want in (("trails", "lose"), ("leads", "win")):
        xs = [k for k, b in zip(ks, band) if b == want]
        if xs:
            ax.text((min(xs) * max(xs)) ** 0.5, 0.755, lab, ha="center", fontsize=6.2,
                    color=INK_MUTED)
    # The legend is ordered by where the curves finish, so its top-to-bottom order is the order
    # of the lines at the right-hand edge. Declaration order matched no budget on the plot and a
    # reader checking the legend against the curves found neither in the other.
    handles, labels = ax.get_legend_handles_labels()
    by_label = {lab: (arm, h) for (arm, (lab, *_)), h in
                zip([(a, style[a]) for a in style if a in rec[str(ks[0])]], handles)}
    order = sorted(labels, key=lambda lab: -rec[str(ks[-1])][by_label[lab][0]])
    ax.legend([by_label[lab][1] for lab in order], order,
              loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False,
              fontsize=6.2, handlelength=1.8, columnspacing=1.2, handletextpad=0.5)
    ax.set_ylim(0, 0.78)
    # Nine budgets on a logarithmic axis crowd where they are closest together, at 8 and 10.
    ax.tick_params(axis="x", labelsize=6.0)
    save(fig, "fig_sweep")
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
    cols = [PALETTE[1], PALETTE[3], INK_FAINT]
    left = 0
    handles = []
    for v, c, lab in zip(vals, cols, labels):
        b = a1.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=0.6)
        a1.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=7,
                color="white" if c != INK_FAINT else INK)
        handles.append((b, lab.replace("\n", " ")))
        left += v
    # The three segments carried no key. `labels` was written above and never drawn -- the loop
    # bound `lab` and dropped it -- so the panel showed three numbers in three colours and the
    # caption named none of them. Drawn above the bar rather than inside it, because the narrowest
    # segment is forty references wide and no label fits in it.
    a1.legend([h[0] for h in handles], [h[1] for h in handles], loc="lower left",
              bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=6.2,
              handlelength=1.1, handleheight=0.7, columnspacing=1.1, handletextpad=0.4)
    hit = usp["overlap"]["misses_those_types_carry"]
    a1.barh(-0.75, hit, color=PALETTE[0], height=0.5)
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
    # Colour alone carried this distinction, which is the one thing a figure must not ask of a
    # reader who cannot see the difference or is holding a greyscale print. The bars that
    # determine a product are hatched as well as coloured.
    for xi, (s, u) in enumerate(zip(share, usable)):
        a2.bar(xi, s, width=0.6, color=PALETTE[1] if u else INK_FAINT,
               hatch="//" if u else None, edgecolor="white" if u else "none", lw=0.0)
    # The four bars are one sequence and not four categories: the definition of a type widens from
    # left to right and the share falls with it. Two referees in a row read them as independent,
    # so the reading is drawn rather than left to the axis order.
    a2.plot(x, share, "-", color=INK_MUTED, lw=0.8, marker="o", ms=2.2, zorder=4)
    # The label above a bar is the bar's own value. It used to be the number of distinct types at
    # that granularity, which is a different quantity on a different scale sitting in the place
    # the eye reads the height from; the type count now rides under the tick where it names the
    # axis position rather than the height.
    for i, c in enumerate(cur):
        # The line that draws the four bars as one sequence passes through its own markers, and
        # at 0.09 and 0.03 the value label sat on top of it: the marker landed on the digits. The
        # label is lifted clear and carries an opaque patch, so the line runs behind it either way.
        a2.text(i, share[i] + 0.055, f"{share[i]:.2f}", ha="center", fontsize=6.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    a2.set_xticks(x)
    a2.set_xticklabels([f"exact\nmultiset\n{cur[0]['types']} types",
                        f"counts\ndropped\n{cur[1]['types']} types",
                        f"element\npairs\n{cur[2]['types']} types",
                        f"bond\ncount\n{cur[3]['types']} types"],
                       fontsize=6.5)
    a2.set_ylabel("share of misses in singleton types")
    # Headroom for the two annotations. Lifting the value labels clear of the connecting line
    # pushed the tallest of them into "the type definition widens", so the annotations move into a
    # band of their own above the bars instead of sharing the top of the data area with them.
    a2.set_ylim(0, 1.15)
    a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.axhline(0.5, color=INK_FAINT, lw=0.6, ls=":")
    # The line marked half and said so nowhere, in the figure or the caption.
    a2.text(3.45, 0.515, "half the misses", fontsize=5.8, ha="right", va="bottom",
            color=INK_FAINT)
    a2.text(-0.35, 1.14, "the type definition widens, left to right", fontsize=6.2, ha="left",
            va="top", color=INK_MUTED)
    # Back inside the panel, over the two short bars, where it fits on its own. The header band
    # is one line wide at this figure's width and the two annotations ran into each other there.
    a2.text(3.45, 0.95, "hatched: the type names\na transformation", fontsize=6.2, ha="right",
            va="top", color=PALETTE[1])
    a2.text(3.35, 0.16, "plain: it does not", fontsize=6.2, ha="right", color=INK_MUTED)
    save(fig, "fig_ceiling")
    plt.close(fig)


def fig_cost():
    env = art("cost_envelope.json")["rows"]
    mt = art("mode_timings.json")
    # Taller than the other two-panel figures and with the scatter given more of the width: at
    # the previous size the left panel's points were below the resolution a printed page gives a
    # scatter of this density, which is a legibility failure and not a taste one.
    # Placed across both columns rather than in one, so the scatter is read at about twice the
    # width it had. A two-panel figure at a single column's width put this one below the size a
    # printed page can resolve, which a referee reported as illegible.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W * 2.06, 2.35),
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    fin = [r for r in env if r["finished"]]
    bad = [r for r in env if not r["finished"]]
    dl = art("cost_envelope.json")["deadline_s"]
    a1.axhline(dl, color=INK_FAINT, lw=0.6, ls="--", zorder=0)
    a1.scatter([r["heavy"] for r in fin], [r["t_generate"] for r in fin], s=13, alpha=0.75,
               color=PALETTE[0], lw=0, label="finished")
    a1.scatter([r["heavy"] for r in bad], [dl] * len(bad), s=26, marker="x",
               color=PALETTE[1], lw=1.0, label=f"did not finish in {int(dl)} s")
    a1.set_yscale("log")
    # Both axes are logarithmic. On a linear x the substrates run 4 to 291 heavy atoms with one
    # peptide at the top, so nine tenths of the points were squeezed into the left fifth of the
    # panel and the trend they carry could not be read at print size.
    a1.set_xscale("log")
    a1.set_xticks([5, 10, 20, 50, 100, 200])
    a1.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    a1.minorticks_off()
    a1.set_xlabel("heavy atoms in the substrate")
    a1.set_ylabel("generator seconds")
    a1.legend(fontsize=6.4, loc="upper left")

    i, e = mt["interactive"], mt["exhaustive"]
    # the four interactive statistics stay together so the bracket under them is true; the
    # slowest substrate is interactive too, and sat outside that bracket in the first draft
    bars = [("median", i["median_s"], PALETTE[0]),
            ("mean", i["mean_s"], PALETTE[0]),
            ("90th pct", i["p90_s"], PALETTE[0]),
            ("slowest", i["max_s"], PALETTE[1]),
            ("median", e["median_s"], PALETTE[2])]
    a2.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for j, b in enumerate(bars):
        a2.text(j, b[1] * 1.18, f"{b[1]}", ha="center", fontsize=6.2)
    a2.set_yscale("log")
    a2.set_xticks(range(len(bars)))
    # Five ticks in a narrow panel collide at any size that keeps them readable, so they lean.
    a2.set_xticklabels([b[0] for b in bars], fontsize=6.2, rotation=25, ha="right",
                       rotation_mode="anchor")
    # the arm each bar belongs to, named once under the group rather than repeated on every tick
    a2.text(1.5, -0.30, "interactive", ha="center", va="top", fontsize=6.6, color=INK_MUTED,
            transform=a2.get_xaxis_transform())
    a2.text(4, -0.30, "exhaustive", ha="center", va="top", fontsize=6.6, color=INK_MUTED,
            transform=a2.get_xaxis_transform())
    a2.plot([-0.35, 3.35], [-0.26, -0.26], color=INK_FAINT, lw=0.6, clip_on=False,
            transform=a2.get_xaxis_transform())
    a2.plot([3.65, 4.35], [-0.26, -0.26], color=INK_FAINT, lw=0.6, clip_on=False,
            transform=a2.get_xaxis_transform())
    a2.set_ylabel("seconds per substrate")
    a2.set_ylim(0.2, 600)
    save(fig, "fig_cost")
    plt.close(fig)


# The worked example's four annotated metabolites. Colour encodes the SITE, not the metabolite:
# the three phosphorylations fire on the same two substrate atoms under three different rules,
# which is the point the panel has to make, so giving each its own colour would hide it.
DEAMINATION, PHOSPHORYLATION = PALETTE[1], PALETTE[0]
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


def _substrate_png(smiles, site_groups, width=2600, height=1500):
    """The substrate with each firing site shaded in its colour.

    Drawn from the atom indices the pipeline reported, not from a hand-marked depiction: the
    panel's claim is that the localisation is a computed field of the prediction.

    On resolution. Cairo rasterises, and matplotlib then resamples whatever it is given to the
    figure's own dpi. The first version rendered 900 by 520 and saved at matplotlib's default
    100 dpi, so what actually reached the PDF was a 211 by 122 bitmap at 100 ppi -- a third of
    the 300 dpi ACS asks for colour artwork, and visibly soft at print size. The render is now
    large enough that the figure's dpi, not the source, is the binding constraint, and the figure
    is saved at FIG_DPI.
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
    # scaled with the canvas: these were tuned against a 900 px render and are proportions, not
    # absolutes, so they have to grow with it or the structure prints hairline
    o.bondLineWidth = max(2, round(2 * width / 900))
    o.highlightRadius = 0.36
    o.fixedFontSize = max(24, round(24 * width / 900))
    o.clearBackground = False
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
    inter_d, exh_d = art("case_study_drawn.json"), art("case_study_exhaustive_drawn.json")
    hits = {arm: {c["key"]: c for c in d["candidates"] if c["is_reference"]}
            for arm, d in (("i", inter), ("e", exh), ("id", inter_d), ("ed", exh_d))}

    # The structure drawn is the one a chemist draws, which is not the one the corpus stores.
    # An earlier version of this figure took the substrate from the stored artifact and so
    # printed a named drug as its minor 4-imino-2-hydroxy tautomer, with its single interactive
    # hit attributed to a template that cannot fire on the correct structure. The shading comes
    # from the run on that same drawing, so the atoms and the molecule are one molecule.
    sites = {DEAMINATION: set(), PHOSPHORYLATION: set()}
    for key, _, col in CASE:
        c = hits["ed"].get(key)
        if c:
            sites[col] |= set(c["firing_atoms"])
    n_phos_rules = len({hits["ed"][k]["rule_id"] for k, _, col in CASE
                        if col == PHOSPHORYLATION and k in hits["ed"]})

    # The right panel's row labels are three lines long and are drawn to the left of its axes,
    # so the gap between the two panels has to hold them: at the default spacing the structure
    # ran underneath them.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W * 2.12, 3.9),
                                 gridspec_kw={"width_ratios": [0.92, 1.45], "wspace": 0.42})
    png = _substrate_png(exh_d["substrate"],
                         [(sorted(atoms), _rgb(col, 0.68)) for col, atoms in sites.items()])
    a1.imshow(mpimg.imread(io.BytesIO(png), format="png"), interpolation="antialiased")
    a1.set_axis_off()
    a1.set_title("substrate as a chemist draws it,\nshaded by the atoms the rules fired on",
                 fontsize=7.5, pad=4)
    for i, (col, text) in enumerate((
            (DEAMINATION, "deamination, 1 rule"),
            (PHOSPHORYLATION, f"phosphorylation, {n_phos_rules} rules"))):
        a1.add_patch(Circle((0.055, -0.055 - 0.075 * i), 0.018, transform=a1.transAxes,
                            color=col, clip_on=False))
        a1.text(0.095, -0.055 - 0.075 * i, text, transform=a1.transAxes, fontsize=7,
                va="center", color=INK)

    # The ranks, for both arms under both drawings of the substrate. The paper treats the
    # drawing as a declared axis, and this is the axis at one molecule: the same system, the
    # same annotation, two tautomers of one drug, different answers.
    rows = [("exhaustive", exh_d, 3.0, "up"), ("exhaustive", exh, 2.0, "down"),
            ("interactive", inter_d, 1.0, "up"), ("interactive", inter, 0.0, "down")]
    for _, d, y, _ in rows:
        if d["n_candidates"]:
            a2.plot([c["rank"] for c in d["candidates"]], [y] * d["n_candidates"], "|",
                    color=INK_FAINT, ms=8, mew=0.8, zorder=2)
    # Labels point away from the pair they belong to and are lifted onto whichever height is
    # free. Alternating two heights by position was not enough: on the stored exhaustive row the
    # hits at ranks 16 and 20 landed on the same height and their two-line labels overprinted.
    # A level is taken only if the last label on it is far enough away along the axis, measured
    # in the axis's own log-ish coordinate rather than in ranks, since the scale is symlog.
    # Two heights, and a sideways nudge when both are taken. Alternating two heights by position
    # was not enough on its own: on the stored exhaustive row the hits at ranks 16, 18 and 20
    # overprinted each other. A third height is not the answer either, because the rows are one
    # data unit apart and a label lifted that far reads as belonging to the row below.
    LEVELS = (9, 23)
    MIN_SEPARATION = 0.115
    NUDGE = 20

    def axis_position(rank):
        return math.log10(max(rank, 1)) if rank <= 30 else 1.48 + (rank - 30) / 140.0

    for _, d, y, side in rows:
        found = sorted((c for c in d["candidates"] if c["is_reference"]),
                       key=lambda c: c["rank"])
        occupied, nudged = [None] * len(LEVELS), 0
        for c in found:
            col = next(x for k, _, x in CASE if k == c["key"])
            name = next(x for k, x, _ in CASE if k == c["key"])
            a2.plot([c["rank"]], [y], "o", ms=5.0, color=col, mec="white", mew=0.9, zorder=4)
            here = axis_position(c["rank"])
            free = [i for i, last in enumerate(occupied)
                    if last is None or here - last >= MIN_SEPARATION]
            level = free[0] if free else len(LEVELS) - 1
            dx = 0
            if not free:
                nudged += 1
                dx = NUDGE if nudged % 2 else -NUDGE
            occupied[level] = here
            off = LEVELS[level] if side == "up" else -LEVELS[level]
            # The text is opaque and sits above every line in the panel. Without that, two
            # lines ran THROUGH a label: the k=15 guide, which is drawn at zorder 0 but shows
            # through the gaps in the glyphs, and a neighbour's leader, which has to cross the
            # inner label band to reach the outer one. On the stored exhaustive row both crossed
            # "dFdCTP rule 1745" -- the leader struck the P and the 5 -- and a strike-through in
            # a second colour reads as a correction mark rather than as a connector.
            # The leader and the label are TWO artists, and that is the whole of the fix. An
            # annotation draws its text and its arrow together under one zorder, and the arrow is
            # drawn inside the annotation rather than as a child of the axes, so raising the text
            # or lowering ann.arrow_patch moves neither past the other -- both were tried. On the
            # stored exhaustive row dFdU sits at the outer level and dFdCTP at the inner one, so
            # dFdU's leader has to cross dFdCTP's text; it struck the P and the 5 in a second
            # colour, which reads as a correction mark and not as a connector. Nudging cannot
            # help: the leader must reach its point and the point is beyond the inner band. Drawn
            # apart, every leader is under every label and the labels are opaque.
            if level or dx:
                a2.annotate("", (c["rank"], y), textcoords="offset points", xytext=(dx, off),
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=col,
                                            shrinkA=1, shrinkB=3), zorder=3)
            a2.annotate(f"{name}\nrule {c['rule_id']}", (c["rank"], y),
                        textcoords="offset points", xytext=(dx, off),
                        ha="center", va="bottom" if side == "up" else "top",
                        fontsize=6.0, color=col, linespacing=1.15, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))
    for k in (15, 30):
        a2.axvline(k, color=INK_FAINT, lw=0.7, ls=":", zorder=0)
        a2.annotate(f"$k={k}$", (k, -1.02), ha="center", fontsize=6.5, color=INK_MUTED)

    def _label(name, d, drawing):
        found = len(d["reference_ranks"])
        return (f"{name}, {drawing}\n{d['n_candidates']} returned, "
                f"{found} of {d['n_references']} found")

    a2.set_yticks([0, 1, 2, 3])
    a2.set_yticklabels([_label("interactive", inter, "as stored"),
                        _label("interactive", inter_d, "as drawn"),
                        _label("exhaustive", exh, "as stored"),
                        _label("exhaustive", exh_d, "as drawn")], fontsize=6.4)
    a2.set_xlabel("rank in the returned list")
    a2.set_xlim(0.3, max(exh["n_candidates"], 32) + 4)
    a2.set_ylim(-1.25, 3.75)
    a2.set_xscale("symlog", linthresh=30, linscale=1.6)
    a2.set_xticks([1, 5, 10, 15, 20, 30, 50, 100])
    a2.set_xticklabels(["1", "5", "10", "15", "20", "30", "50", "100"])
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    a2.set_title("where the four annotated metabolites land,\nunder each drawing of the substrate",
                 fontsize=7.5, pad=8)
    save(fig, "fig_case")
    plt.close(fig)


def fig_criterion():
    """The verdict grid the manuscript used to print as a table of signs.

    Table and figure carried the same nine by five grid, and the table read as arithmetic while
    the picture reads as a shape: two rows that never leave the trailing colour whatever the
    budget, three that cross once the list is long enough, and one column where the arm being
    compared is not even the same under all five criteria. That shape is the paper's claim.

    Everything here comes from the artifact's own verdicts rather than from a second reading of
    the recall values, so the figure cannot disagree with the sweep it draws.
    """
    d = art("criterion_sweep.json")
    order = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
    label = {"canonical": "canonical SMILES equality", "inchikey": "the full InChIKey",
             "inchi_no_stereo": "InChIKey, no stereo", "tanimoto1": "Tanimoto $=1$",
             "inchikey_tautomer": "tautomer-aware key (default)"}
    missing = [c for c in order if c not in d["by_criterion"]]
    extra = [c for c in d["by_criterion"] if c not in order]
    if missing or extra:
        raise SystemExit(f"criterion_sweep.json and this figure disagree on the criteria: "
                         f"missing {missing}, undrawn {extra}")

    budgets = sorted((int(k) for k in d["by_criterion"][order[0]]["verdict_by_budget"]), key=int)
    CODE = {"trails": -1, "neither": 0, "leads": 1}
    SIGN = {-1: "\u2013", 0: "\u00b7", 1: "+"}
    # One letter per comparator, and a name the figure cannot draw is a refusal rather than a
    # blank cell: the grid is read against a different comparator in almost every column.
    LETTER = {"MetaTox": "M", "SyGMa": "S", "MetaPredictor": "P", "GLORYx": "G",
              "BioTransformer": "B",
              # Two characters, because GLORYxR contributes two columns and a single letter would
              # collapse the setting the cell was read against -- which is the knob this paper
              # asks other papers to declare.
              "GLORYxR default SoM": "Rd", "GLORYxR strict SoM": "Rs"}

    grid, marks = [], []
    for crit in order:
        v = d["by_criterion"][crit]["verdict_by_budget"]
        m = d["by_criterion"][crit]["margin_by_budget"]
        bad = {x for x in v.values()} - set(CODE)
        if bad:
            raise SystemExit(f"criterion_sweep.json records verdicts this figure cannot draw: "
                             f"{sorted(bad)}")
        unknown = {m[str(b)]["theirs"] for b in budgets} - set(LETTER)
        if unknown:
            raise SystemExit(f"criterion_sweep.json names comparators this figure has no letter "
                             f"for: {sorted(unknown)}")
        grid.append([CODE[v[str(b)]] for b in budgets])
        marks.append([LETTER[m[str(b)]["theirs"]] for b in budgets])

    # The arm under comparison, taken per budget across the five criteria. Where they disagree the
    # cell says so rather than picking one, because a sign read against a different arm is a
    # different comparison.
    arms = []
    for b in budgets:
        ours = {d["by_criterion"][c]["margin_by_budget"][str(b)]["ours"] for c in order}
        arms.append("exh." if ours == {"GRAIL exhaustive"}
                    else "int." if ours == {"GRAIL interactive"} else "\u2013")

    import numpy as np
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(W * 2.06, 2.35))
    cmap = ListedColormap(["#C7DCEA", "#F2F2F0", "#EBCDB4"])
    ax.imshow(np.array(grid), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            ax.text(c - 0.10, r, SIGN[v], ha="center", va="center", fontsize=8, color=INK)
            ax.text(c + 0.17, r - 0.20, marks[r][c], ha="center", va="center",
                    fontsize=5.6, color=INK_MUTED)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets], fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label[c] for c in order], fontsize=7)
    ax.set_xlabel("candidates a system may return, $k$", fontsize=7, labelpad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2.0)

    # The arm row sits under the grid rather than inside it: it is not a verdict.
    for c, a in enumerate(arms):
        ax.text(c, len(order) - 0.30, a, ha="center", va="top", fontsize=6.4, color=INK_MUTED,
                transform=ax.transData)
    ax.text(-0.6, len(order) - 0.30, "GRAIL arm", ha="right", va="top", fontsize=6.4,
            color=INK_MUTED)
    ax.set_ylim(len(order) - 0.05, -1.45)

    # The key sits above the grid: below it there is the arm row, the tick labels and the axis
    # label already, and a key placed there printed across the axis label.
    key = [("#C7DCEA", "GRAIL trails"), ("#F2F2F0", "neither separates"), ("#EBCDB4", "GRAIL leads")]
    for i, (col, lab) in enumerate(key):
        ax.add_patch(plt.Rectangle((0.9 + i * 2.6, -1.32), 0.30, 0.26,
                                   facecolor=col, edgecolor="none", clip_on=False))
        ax.text(1.32 + i * 2.6, -1.19, lab, ha="left", va="center",
                fontsize=6.4, color=INK_MUTED, clip_on=False)
    save(fig, "fig_criterion")
    plt.close(fig)
    return {"budgets": budgets, "arms": arms,
            "leads": {c: g.count(1) for c, g in zip(order, grid)}}


def fig_toc():
    """The graphic for the table of contents: the thesis, not the system.

    The specification constrains this more than the figures do. It must give the essence "without
    providing specific results", and it must avoid artwork that already appears in the text. An
    earlier version drew one substrate and two of its metabolites, which is a picture of the
    instrument; under a title about what undeclared choices do to an ordering, the essence is the
    ordering itself changing. So this draws the verdict grid: the same pair of systems compared on
    the same substrates, read under five ways of deciding whether a prediction matches and nine
    budgets, with the sign of the verdict in each cell. No recall value is printed, which is what
    the specification asks, and no artwork from the text is reused, since Table~4 is a table.

    Hard requirements, from the ACS guidelines of 2024-02-28: at most 3.25 by 1.75 inches at the
    size submitted, sans-serif type at 8 pt and never below 6, and TIFF at 300 dpi or EPS with
    fonts embedded, so both are written.
    """
    # The specification, stated once and never used to build anything. The canvas below is sized
    # from it today, but the two must stay separate names: an assertion that compares the output
    # against the same constant that produced it can only catch a process failure, never a wrong
    # constant, and would pass unchanged if someone widened the canvas.
    ACS_MAX_W, ACS_MAX_H, ACS_MIN_DPI, ACS_MIN_PT = 3.25, 1.75, 300, 6.0
    TOC_W, TOC_H, DPI, MIN_PT = 3.25, 1.75, 600, 6.0
    # The guideline prefers 8 pt and forbids below 6. Nothing here now sits at the floor: the
    # smallest type is the key, and it is a full point above it.
    CELL_PT, AXIS_PT, TITLE_PT, KEY_PT = 7.5, 7.0, 8.0, 7.0
    FONTS_PT = (CELL_PT, AXIS_PT, TITLE_PT, KEY_PT)

    d = art("criterion_sweep.json")
    order = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
    label = {"canonical": "canonical SMILES", "inchikey": "InChIKey",
             "inchi_no_stereo": "InChIKey, no stereo", "tanimoto1": "Tanimoto = 1",
             "inchikey_tautomer": "tautomer-aware key"}
    missing = [c for c in order if c not in d["by_criterion"]]
    if missing:
        raise SystemExit(f"criterion_sweep.json has no column for {', '.join(missing)}")

    budgets = sorted((int(k) for k in d["by_criterion"][order[0]]["verdict_by_budget"]), key=int)

    # The verdicts the artifact records, which are the ones Table 4 prints. Recomputing them from
    # the recall values would put a second definition of "leads" in the paper, and the two pictures
    # of one grid would then be free to disagree.
    CODE = {"trails": -1, "neither": 0, "leads": 1}
    grid = []
    for crit in order:
        v = d["by_criterion"][crit]["verdict_by_budget"]
        unknown = {x for x in v.values()} - set(CODE)
        if unknown:
            raise SystemExit(f"criterion_sweep.json records verdicts this figure cannot draw: "
                             f"{', '.join(sorted(unknown))}")
        grid.append([CODE[v[str(b)]] for b in budgets])

    import numpy as np
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(TOC_W, TOC_H))
    # three regions, not three series: a muted diverging triple, readable in grey
    cmap = ListedColormap(["#C7DCEA", "#F2F2F0", "#EBCDB4"])
    ax.imshow(np.array(grid), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            ax.text(c, r, {-1: "\u2013", 0: "\u00b7", 1: "+"}[v], ha="center", va="center",
                    fontsize=CELL_PT, color=INK)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets], fontsize=AXIS_PT)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label[c] for c in order], fontsize=AXIS_PT)
    ax.set_xlabel("candidates a system may return", fontsize=AXIS_PT, labelpad=1.5)
    ax.set_title("one comparison, five ways of judging a match",
                 fontsize=TITLE_PT, pad=3.0, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=1.5)

    # A key, because three colours carrying signs are not self-describing: a reader met this
    # graphic with blue and orange cells marked "$-$" and "$+$" and nothing saying which way round
    # they ran. The labels are the artifact's own vocabulary, so the picture and Table 4 cannot
    # drift apart in what a cell is called.
    from matplotlib.patches import Patch

    key = [Patch(facecolor=cmap(i), edgecolor="none", label=t) for i, t in
           enumerate(("GRAIL trails", "neither", "GRAIL leads"))]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              frameon=False, fontsize=KEY_PT, handlelength=1.0, handleheight=0.9,
              handletextpad=0.35, columnspacing=1.2, borderpad=0.0, borderaxespad=0.0)
    fig.tight_layout(pad=0.25)

    with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
        # tif and eps are what ACS accepts for this graphic; the pdf exists only so pdflatex can
        # place it in the manuscript, since pdflatex reads neither of the other two
        for f, kw in (("fig_toc.tif", {"pil_kwargs": {"compression": "tiff_lzw"}}),
                      ("fig_toc.eps", {}), ("fig_toc.pdf", {})):
            fig.savefig(OUT / f, dpi=DPI, facecolor="white", **kw)
    plt.close(fig)

    # ACS asks for RGB; matplotlib writes RGBA, and an alpha channel in a submitted TIFF is a
    # production problem rather than an error anyone would see here
    from PIL import Image
    with Image.open(OUT / "fig_toc.tif") as im:
        if im.mode != "RGB":
            im.convert("RGB").save(OUT / "fig_toc.tif", compression="tiff_lzw",
                                   dpi=(DPI, DPI))

    # The specification is checked rather than trusted: a graphic over the limit is rejected at
    # submission, and nothing else here would notice.
    with Image.open(OUT / "fig_toc.tif") as im:
        w_in, h_in = im.size[0] / DPI, im.size[1] / DPI
        assert im.mode == "RGB", f"TOC graphic is {im.mode}, not RGB"
        assert w_in <= ACS_MAX_W + 1e-6 and h_in <= ACS_MAX_H + 1e-6, (
            f"TOC graphic is {w_in:.2f} by {h_in:.2f} in, over the ACS maximum of "
            f"{ACS_MAX_W} by {ACS_MAX_H}")
        assert im.info.get("dpi", (0, 0))[0] >= ACS_MIN_DPI, (
            f"TOC graphic is under {ACS_MIN_DPI} dpi")
    assert min(FONTS_PT) >= ACS_MIN_PT, (
        f"a TOC label is set at {min(FONTS_PT)} pt, under the ACS floor of {ACS_MIN_PT}")
    return w_in, h_in, DPI


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
        # The three Supporting Information figures draw from artifacts of their own, and a digest
        # that does not cover them cannot tell a stale one from a current one.
        "si_budget": art("budget_curve.json")["by_budget"],
        "si_rarefaction": art("mining_rarefaction.json")["rarefaction"],
        "si_external": art("external_budget_confound.json")["counted_cells"],
        "grain": cen["granularity_curve"], "gap": art("coverage_gap_types.json")["gap"],
        "uspto": usp["overlap"], "modes": {"i": mt["interactive"], "e": mt["exhaustive"]},
        "env": [(r["heavy"], r["finished"], r.get("t_generate")) for r in env["rows"]],
        # both drawings of the substrate, and the substrate string itself. The digest covered
        # only the stored-dialect runs, so it could not see the staleness that mattered most:
        # the figure went on drawing a named drug as its minor tautomer after the paper had
        # stopped claiming that drawing was the right one, and the gate stayed green.
        "case": {a: {"substrate": art(f)["substrate"],
                     "refs": [(c["rank"], c["key"], c["rule_id"], c["firing_atoms"])
                              for c in art(f)["candidates"] if c["is_reference"]]}
                 for a, f in (("i", "case_study.json"),
                              ("e", "case_study_exhaustive.json"),
                              ("id", "case_study_drawn.json"),
                              ("ed", "case_study_exhaustive_drawn.json"))},
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


# The Supporting Information is single-column at 12 pt, so a figure there has about six inches of
# measure rather than the manuscript's 3.33. It carried 93 pages and no figure at all: twenty-two
# tables, and three quantities whose shape is the finding and which a table cannot show.
W_SI = 5.6


def fig_si_budget():
    """Recall against the rule budget, one line per output budget.

    The manuscript says the two operating modes are two points on one continuous setting and that
    which to run follows from the output budget answered at. The table of this sweep prints thirty
    numbers and the crossing is invisible in it: at a tight output budget the whole bank is worse
    than a budget of thirty, and at a wide one it is far better. That is one line crossing another.
    """
    d = art("budget_curve.json")["by_budget"]
    rbs = sorted((int(b) for b in d), key=int)
    ks = sorted((int(k) for k in d[str(rbs[0])]["recall_micro"]), key=int)
    style = {1: ("-", "o"), 5: ("-", "s"), 10: ("-", "^"), 15: ("-", "v"),
             30: ("-", "D"), 50: ("-", "P")}
    missing = [k for k in ks if k not in style]
    if missing:
        raise SystemExit(f"budget_curve.json carries output budgets this figure has no style for: "
                         f"{missing}")

    fig, ax = plt.subplots(figsize=(W_SI, 3.0))
    for i, k in enumerate(ks):
        ls, mk = style[k]
        y = [d[str(b)]["recall_micro"][str(k)] for b in rbs]
        ax.plot(rbs, y, ls, marker=mk, ms=3.4, lw=1.1, color=PALETTE[i % len(PALETTE)],
                label=f"$k={k}$")
    dep = art("budget_curve.json")["deployed_budget"]
    ax.axvline(dep, color=INK_FAINT, lw=0.6, zorder=0)
    ax.text(dep, ax.get_ylim()[0], " deployed", fontsize=6.2, color=INK_MUTED,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xticks(rbs)
    ax.set_xticklabels([f"{b:,}".replace(",", "\u2009") for b in rbs], fontsize=6.8)
    ax.set_xlabel("templates the generator may apply (rule budget)")
    ax.set_ylabel("micro recall")
    ax.legend(fontsize=6.4, ncol=3, loc="upper left", handlelength=1.8)
    ax.tick_params(labelsize=6.8)
    save(fig, "fig_si_budget")
    plt.close(fig)
    return {k: [d[str(b)]["recall_micro"][str(k)] for b in rbs] for k in (1, 50)}


def fig_si_rarefaction():
    """Templates mined against training pairs, with the spread over the draws at each point.

    The manuscript says template discovery from this corpus is not finished and gives the slope.
    A slope is a claim about a shape, and the shape is the evidence for it.
    """
    r = art("mining_rarefaction.json")
    pts = sorted(r["rarefaction"].values(), key=lambda v: v["pairs"])
    x = [v["pairs"] for v in pts]
    y = [v["templates_mean"] for v in pts]
    sd = [v["templates_sd"] for v in pts]
    fig, ax = plt.subplots(figsize=(W_SI, 2.7))
    ax.fill_between(x, [a - b for a, b in zip(y, sd)], [a + b for a, b in zip(y, sd)],
                    color=PALETTE[0], alpha=0.18, lw=0)
    ax.plot(x, y, "-", marker="o", ms=3.4, lw=1.2, color=PALETTE[0])
    ax.plot([r["distinct_training_pairs"]], [r["mined_templates"]], marker="*", ms=8,
            color=PALETTE[1], lw=0, zorder=4)
    ax.annotate(f"the whole corpus:\n{r['mined_templates']:,} templates".replace(",", "\u2009"),
                xy=(r["distinct_training_pairs"], r["mined_templates"]),
                xytext=(-10, 14), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.4, color=INK_MUTED)
    ax.set_xlabel("annotated substrate-product pairs the bank was mined from")
    ax.set_ylabel("distinct templates")
    ax.tick_params(labelsize=6.8)
    save(fig, "fig_si_rarefaction")
    plt.close(fig)
    return len(pts), r["draws_per_point"]


def fig_si_external():
    """Recall against how much each tool emitted, in a benchmark that is not ours.

    The manuscript's one external check says the arms that emit more score higher. It states a
    rank correlation and a permutation test; the scatter is what those two numbers describe, and
    it also shows the reader that the association is within a drug and not only across drugs.
    """
    e = art("external_budget_confound.json")
    cells = e["counted_cells"]
    tools = sorted({c["tool"] for c in cells})
    MARK = ["o", "s", "^", "v", "D", "P", "X", "*"]
    if len(tools) > len(MARK):
        raise SystemExit(f"external_budget_confound.json carries {len(tools)} tools and this "
                         f"figure has {len(MARK)} markers")
    fig, ax = plt.subplots(figsize=(W_SI, 3.0))
    for i, t in enumerate(tools):
        pts = [c for c in cells if c["tool"] == t]
        ax.scatter([c["counted_emitted"] for c in pts], [c["recall"] for c in pts],
                   s=22, marker=MARK[i], facecolor="none", linewidths=0.9,
                   color=PALETTE[i % len(PALETTE)], label=t)
    ax.set_xscale("log")
    ax.set_xlabel("candidates the tool emitted for that drug, counted from the deposit")
    ax.set_ylabel("recall reported for that drug")
    ax.legend(fontsize=6.2, ncol=2, loc="upper left", handlelength=1.4)
    ax.tick_params(labelsize=6.8)
    rho = e["spearman_recall_against_counted_emission"]
    within = e["association_within_the_arms"][
        "spearman_after_removing_the_arm_effect_and_the_drug_effect"]
    ax.text(0.98, 0.03,
            f"Spearman {rho} over {len(cells)} cells;\n{within} with the arm\n"
            f"and drug effects removed",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=INK_MUTED)
    save(fig, "fig_si_external")
    plt.close(fig)
    return len(cells), len(tools)



if __name__ == "__main__":
    band = fig_sweep()
    fig_ceiling()
    fig_cost()
    fig_case()
    crit = fig_criterion()
    toc = fig_toc()
    sib = fig_si_budget()
    sir = fig_si_rarefaction()
    sie = fig_si_external()
    (OUT / "figures.sha256").write_text(digest() + "\n")
    print("  fig_sweep.pdf, fig_ceiling.pdf, fig_cost.pdf, fig_case.pdf")
    print(f"  fig_toc.tif, fig_toc.eps, fig_toc.pdf  {toc[0]}x{toc[1]} in at {toc[2]} dpi, RGB")
    print(f"  fig_criterion  arms {crit['arms']}, leads per criterion {crit['leads']}")
    print("  fig_si_budget, fig_si_rarefaction, fig_si_external (Supporting Information)")
    print(f"    the budget curve crosses: at k=1 {sib[1][0]:.4f} -> {sib[1][-1]:.4f}, "
          f"at k=50 {sib[50][0]:.4f} -> {sib[50][-1]:.4f}")
    print(f"    rarefaction {sir[0]} points, {sir[1]} draws each; "
          f"external {sie[0]} cells over {sie[1]} tools")
    print(f"  the sweep's shaded regions, from the contrasts: {band}")
    print(f"  data digest {digest()[:24]}")
