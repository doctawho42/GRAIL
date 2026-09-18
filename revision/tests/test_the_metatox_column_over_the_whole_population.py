"""MetaTox as one arm over the evaluated population, and the ways that could quietly be a lie.

Run: python -m pytest revision/tests/test_the_metatox_column_over_the_whole_population.py -q

The second submission came back for all 879 substrates outside the comparison set, nothing the axis
applies excludes the arm, and this is the column that lets it be read there. It is built the way
GLORYx's is -- revision/phase2_gloryx_merge.py, the same shape of arm, a web service run once over
the 291 and again over the rest -- because two arms assembled two different ways cannot be compared
with each other, whatever else is true of them.

A merged column has a small number of ways to be wrong, and every one of them is silent:

  IT CAN FAIL TO COVER. A list arm is read with preds.get(s, []), so a substrate the column lacks
  scores as a miss rather than as missing data. Understating a comparator on part of the population
  is the defect this paper spends its length objecting to, so coverage is exact or the build fails.

  IT CAN RE-RANK THE HALF THAT WAS ALREADY PUBLISHED. The 291 have been scored and reported. If
  merging reorders or re-deduplicates them, the comparison-set numbers in the manuscript stop
  matching the column the manuscript is read from, and nothing downstream would notice. So the 291
  half has to come through identical, list for list, in order.

  IT CAN RANK THE NEW HALF BY SOMETHING ELSE. The first batch is ordered by the method's own Pa
  descending; file order, or alphabetical order, would score MetaTox in an ordering it never
  produced and make every recall@k a statement about the alphabet. So the new half must be ordered
  by its own scores too, and that is checked against the scores the column itself carries.

  IT CAN DISAGREE WITH THE DELIVERY IT CAME FROM. results/metatox_outside_submission.json already
  measured what the delivery holds, de-duplicated the same way. If the column's 879 half does not
  match those counts, one of the two readings of one delivery has drifted, and this test says which
  numbers disagree rather than letting the newer one win by being newer.

  IT CAN CARRY THE SAME STRUCTURE TWICE. The scoring is set-based, so a duplicate is not a wrong
  answer, it is a wasted slot: it inflates the output size that recall@k is read against while
  adding nothing findable. The first batch de-duplicates, so this one must.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

COLUMN = ROOT / "results" / "metatox_smirks_preds_evaluated1170.json"
COMPARISON = ROOT / "results" / "metatox_smirks_preds.json"
WIDER = ROOT / "results" / "metatox_smirks_preds_1170.json"
DELIVERY = ROOT / "results" / "metatox_outside_submission.json"
TRUTH = ROOT / "results" / "test_references.json"


def _col() -> dict:
    assert COLUMN.exists(), (
        f"{COLUMN.relative_to(ROOT)} does not exist. Nothing the population axis applies excludes "
        f"MetaTox, so what keeps it out of the wider population is this column not being built.")
    return json.loads(COLUMN.read_text())


def test_the_column_covers_the_evaluated_population_exactly():
    """Not a substrate more, not a substrate fewer.

    Fewer understates the arm, because an absent substrate scores as a miss. More means the column
    holds substrates the population does not contain, which is a different population wearing this
    one's name.
    """
    d = _col()
    preds = d["predictions"]
    pop = set(json.loads(TRUTH.read_text()))
    missing, extra = pop - set(preds), set(preds) - pop
    assert not missing, (f"{len(missing)} evaluated substrates are absent from the column, and each "
                         f"would score as a miss: e.g. {sorted(missing)[0][:60]}")
    assert not extra, (f"{len(extra)} substrates in the column are outside the evaluated "
                       f"population: e.g. {sorted(extra)[0][:60]}")


def test_the_two_sources_are_disjoint_and_are_named_per_substrate():
    """Which run answered each substrate has to be recoverable from the column.

    If the two runs overlapped, one of them is not the run it reports being, and the overlap would
    be resolved here by whichever happened to be written second. The merge refuses instead, and the
    record says which source each substrate came from so the halves can be read apart afterwards.
    """
    d = _col()
    src = d.get("source_of_each_substrate")
    assert isinstance(src, dict) and src, "the column does not say which run answered each substrate"
    assert set(src) == set(d["predictions"]), "the source map does not cover the column"
    kinds = set(src.values())
    assert len(kinds) == 2, f"expected two sources, found {sorted(kinds)}"
    counts = {k: sum(1 for v in src.values() if v == k) for k in kinds}
    assert sum(counts.values()) == len(d["predictions"])
    comparison = json.loads(COMPARISON.read_text())["predictions"]
    from_comparison = [s for s, v in src.items() if v == "comparison_set"]
    assert set(from_comparison) == set(comparison), (
        "the substrates the column attributes to the comparison-set run are not the ones that run "
        "holds")


def test_the_published_half_comes_through_untouched():
    """The 291 are already reported; merging may not quietly restate them.

    List for list, in order. A reordering here would leave the manuscript's comparison-set numbers
    describing a column that no longer exists, and no gate downstream compares the two.
    """
    d = _col()
    published = json.loads(COMPARISON.read_text())["predictions"]
    col = d["predictions"]
    for s, v in published.items():
        assert s in col, f"a comparison-set substrate is missing from the column: {s[:60]}"
        assert col[s] == v, (
            f"the column's list for a comparison-set substrate is not the published one "
            f"({len(col[s])} entries against {len(v)}); the published half has been restated")


def test_the_new_half_is_ranked_by_the_method_s_own_score():
    """Ordered by Pa descending, checked against the scores the column carries.

    Not asserted from the producer's docstring: the column carries a score per prediction, so the
    order can be read off it. An arm ranked by anything else -- file order, the alphabet -- would
    make every recall@k a statement about that instead of about the method.
    """
    d = _col()
    detail = d.get("predictions_with_scores")
    assert isinstance(detail, dict) and detail, "the column carries no per-prediction scores"
    src = d["source_of_each_substrate"]
    checked = 0
    for s, rows in detail.items():
        if src.get(s) != "wider":
            continue
        scored = [r for r in rows if isinstance(r[1], (int, float)) and r[1] == r[1]]
        if len(scored) < 2:
            continue
        pas = [r[1] for r in scored]
        assert pas == sorted(pas, reverse=True), (
            f"the scored predictions for a wider-run substrate are not in descending Pa order: "
            f"{pas[:6]}")
        checked += 1
    assert checked > 100, f"only {checked} wider-run substrates carried enough scores to check"


def test_no_substrate_carries_the_same_structure_twice():
    """A duplicate is a wasted slot, not a wrong answer, and it inflates the size recall is read at.

    The comparison-set half de-duplicates by canonical SMILES before counting, so the merged column
    has to as well or the two halves are measured differently in the one property this paper is
    about.
    """
    d = _col()
    offenders = {s: len(v) - len(set(v)) for s, v in d["predictions"].items() if len(v) != len(set(v))}
    assert not offenders, (
        f"{len(offenders)} substrates carry a repeated structure, "
        f"{sum(offenders.values())} repeats in total; the worst is "
        f"{max(offenders.values())} on one substrate")


def test_the_column_agrees_with_the_delivery_it_was_built_from():
    """Two readings of one delivery, compared, so neither can drift unnoticed.

    results/metatox_outside_submission.json measured the delivery independently and de-duplicated it
    the same way. The column's wider half must reproduce those counts. If it does not, this says so
    rather than letting whichever artefact was written last stand as the answer.
    """
    d = _col()
    delivery = json.loads(DELIVERY.read_text())
    want = delivery["budget"]
    src = d["source_of_each_substrate"]
    wider = {s: v for s, v in d["predictions"].items() if src.get(s) == "wider"}
    assert len(wider) == want["this_delivery_substrates"], (
        f"the column holds {len(wider)} substrates from the wider run and the delivery measured "
        f"{want['this_delivery_substrates']}")
    got_total = sum(len(v) for v in wider.values())
    assert got_total == want["this_delivery_structures"], (
        f"the column holds {got_total} distinct structures over the wider run and the delivery "
        f"measured {want['this_delivery_structures']}; one of the two readings has drifted")
    got_mean = got_total / len(wider)
    assert abs(got_mean - want["this_delivery_mean_output"]) < 0.005, (
        f"the column's wider-half output size is {got_mean:.4f} against the delivery's "
        f"{want['this_delivery_mean_output']}")


def _fixture_sdf(tmp_path, ids_and_scores):
    """A tiny SDF in the shape both deliveries use, so the ingest can be exercised without them.

    The real files are 702 MB and 32 MB and are not in the repository, which is exactly why this
    exists: the shared ingest was changed to accept a second id shape, and the artefact it produced
    for the comparison set cannot be rebuilt here to prove the change was inert.
    """
    from rdkit import Chem
    blocks = []
    for rec_id, pa, pi, smi in ids_and_scores:
        mol = Chem.MolFromSmiles(smi)
        block = Chem.MolToMolBlock(mol)
        props = [f">  <ID>\n{rec_id}\n"]
        if pa is not None:
            # PASS writes the pair with a comma as the decimal separator, and says in a second tag
            # that it scored the record; the ingest cross-checks the two and refuses if they differ.
            props.append(">  <PASS_ACTIVITY_SPECTRUM>\n"
                         f"{pa:.3f} {pi:.3f} Metabolite\n".replace(".", ","))
            props.append(">  <PASS_RESULT_COUNT>\n1 of 1\n")
        blocks.append(block + "\n".join(props) + "\n$$$$\n")
    path = tmp_path / "fixture.sdf"
    path.write_text("".join(blocks))
    return path


def test_the_shared_ingest_still_ranks_and_dedups_the_way_the_published_half_was_built(tmp_path):
    """The regression guard for a change to code that produced an artefact already in the paper.

    scripts/metatox_smirks_ingest.py was extended to accept the second delivery's ids. It also
    produced results/metatox_smirks_preds.json, whose source SDF is not in this repository, so the
    change cannot be shown inert by rebuilding that artefact. It is shown inert here instead, on a
    fixture that exercises the three behaviours the published half depends on:

      records are ordered by Pa descending, not by the order the file happens to list them;
      a structure seen twice is kept once, at its first (best-scoring) appearance;
      a record PASS declined to score sorts after every record it did score, rather than being
      dropped or sorted as though its score were zero.

    If any of those moved, the comparison-set column in the manuscript would no longer be the column
    the manuscript's numbers came from, and nothing else in this repository would notice.
    """
    import importlib
    ing = importlib.import_module("metatox_smirks_ingest")

    # deliberately out of score order in the file, with one repeat and one unscored record
    recs = [("7_1", 0.300, 0.100, "CCO"),
            ("7_2", 0.900, 0.050, "CCC"),
            ("7_3", None, None, "CCCC"),
            ("7_4", 0.600, 0.200, "CCO"),
            ("7_5", 0.700, 0.100, "CCCCC")]
    got = ing.parse_sdf(_fixture_sdf(tmp_path, recs))
    assert list(got) == ["7"], f"the key is not the id prefix: {list(got)}"
    rows = got["7"]
    pas = [r[1] for r in rows if r[1] == r[1]]
    assert pas == sorted(pas, reverse=True), f"not ordered by Pa descending: {pas}"
    assert pas[0] == 0.9, f"the best-scoring record is not first: {pas}"
    unscored = [r for r in rows if r[1] != r[1]]
    assert len(unscored) == 1, "the record PASS declined to score was dropped or duplicated"
    assert rows.index(unscored[0]) == len(rows) - 1, (
        "the unscored record did not sort last; PASS expresses no preference among records it "
        "declines to score, and placing one above a scored record invents one")

    # de-duplication is the caller's step, on first appearance, which is what the published half did
    seq, seen = [], set()
    for smiles, _pa, _pi in rows:
        if smiles in seen:
            continue
        seen.add(smiles)
        seq.append(smiles)
    assert seq == ["CCC", "CCCCC", "CCO", "CCCC"], (
        f"de-duplication by first appearance no longer keeps the best-scoring copy: {seq}")


def test_each_substrate_separately_looks_like_the_parent_of_its_own_predictions():
    """The check the ingest's gate cannot do, calibrated on the half already accepted.

    scripts/metatox_smirks_ingest.py gates its join on the median over substrates of the median
    Tanimoto to that substrate's own predictions, against the same under a rotated assignment. Its
    docstring offers this as what stops "a whole table of wrong numbers". It cannot: a median over
    substrates is blind to a minority, so a subset of the delivery could be cross-assigned and the
    statistic would not move. That is a property of the instrument, not a suspicion about the data.

    This resolves per substrate instead. For each one, its similarity to its OWN prediction set is
    compared against its similarity to several other substrates' sets, drawn with a fixed seed. A
    substrate whose own predictions do not look more like it than a stranger's do is a candidate
    mis-assignment, and a minority of those shows up here where it cannot show up in a median.

    The threshold is not invented. The comparison-set half was joined positionally, gated, accepted
    and reported long before this column existed, so it is the standard: the new half has to do at
    least as well on the same instrument. Calibrating against the accepted half rather than against
    a number chosen here is the correction this session had to make once already, in the
    output-budget objection that was withdrawn.
    """
    import statistics
    from rdkit import Chem, DataStructs, RDLogger, rdBase  # noqa: F401
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    d = _col()
    preds, src = d["predictions"], d["source_of_each_substrate"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    subs = sorted(preds)
    fp_of = {}
    for s in subs:
        m = Chem.MolFromSmiles(s)
        fp_of[s] = gen.GetFingerprint(m) if m is not None else None

    CAP, RIVALS = 20, 5
    pred_fps = {}
    for s in subs:
        fps = []
        for p in preds[s][:CAP]:
            m = Chem.MolFromSmiles(p)
            if m is not None:
                fps.append(gen.GetFingerprint(m))
        pred_fps[s] = fps

    def against(sub, owner):
        fps = pred_fps[owner]
        if fp_of[sub] is None or not fps:
            return None
        return statistics.median(DataStructs.TanimotoSimilarity(fp_of[sub], f) for f in fps)

    import random
    rng = random.Random(0)
    fails = {"comparison_set": 0, "wider": 0}
    total = {"comparison_set": 0, "wider": 0}
    for i, s in enumerate(subs):
        half = src.get(s)
        own = against(s, s)
        if own is None:
            continue
        total[half] += 1
        beaten = 0
        for _ in range(RIVALS):
            other = subs[rng.randrange(len(subs))]
            if other == s:
                continue
            rival = against(s, other)
            if rival is not None and rival >= own:
                beaten += 1
        if beaten:
            fails[half] += 1

    rate = {h: fails[h] / total[h] for h in total if total[h]}
    assert total["comparison_set"] > 250 and total["wider"] > 800, f"too few substrates: {total}"
    # The accepted half sets the bar. A small absolute allowance rides on top because the new half
    # is three times larger, so a like-for-like rate can land slightly above it by sampling alone.
    bar = rate["comparison_set"] + 0.02
    assert rate["wider"] <= bar, (
        f"on the wider half {fails['wider']} of {total['wider']} substrates "
        f"({100 * rate['wider']:.1f}%) do not look more like their own predictions than a "
        f"stranger's, against {fails['comparison_set']} of {total['comparison_set']} "
        f"({100 * rate['comparison_set']:.1f}%) on the half already accepted. The join on the "
        f"new half is worse than the one this paper already reports.")


def test_the_column_does_not_overstate_what_its_ranking_decides():
    """The column said it was ordered by the method's score. Half of it is not.

    PASS writes a Metabolite spectrum only where the record clears its own threshold, so about half
    of what MetaTox returns carries no score at all. Those records keep the order the delivery gave
    them. Saying "ranked by the method's own Pa, descending" and stopping there describes the half
    that is ranked and lets the reader assume the rest, which is the kind of sentence this paper
    spends its length objecting to in other people's work.

    So the share is recorded, computed, and checked here against the column's own scores rather than
    taken from the producer's word. Both directions bite: an overstated share would hide how much of
    the order is the supplier's file, and an understated one would invent a defect.

    The mitigation is recorded too, because it is real and measured: unscored records sort last, so
    at the budgets the axis reads most the order is mostly the method's, and the file's share grows
    with k. And it is a property of BOTH halves alike, which is what keeps it from making them
    incomparable.
    """
    d = _col()
    share = d.get("ranking_decides_this_share")
    assert isinstance(share, dict) and share, (
        "the column does not record what share of its slots the stated ranking decides")
    assert "why_that_share_matters" in d, "the share is recorded with no account of what it means"

    det = d["predictions_with_scores"]
    src = d["source_of_each_substrate"]

    def scored(row):
        return isinstance(row[1], (int, float)) and row[1] == row[1]

    rows_all = [r for rows in det.values() for r in rows]
    want = sum(map(scored, rows_all)) / len(rows_all)
    assert abs(share["overall"] - want) < 5e-4, (
        f"the recorded overall share {share['overall']} is not the measured {want:.4f}")

    # It has to be recorded per budget, because one overall number would hide that the exposure
    # grows with k -- which is the only part of this that could bias a comparison.
    ks = sorted(int(k.rsplit("_", 1)[1]) for k in share if k.startswith("within_the_first_"))
    assert ks, "the share is recorded only overall, so it cannot be read at the budgets the axis uses"
    prev = 1.1
    for k in ks:
        got = share[f"within_the_first_{k}"]
        slots = [r for rows in det.values() for r in rows[:k]]
        exp = sum(map(scored, slots)) / len(slots)
        assert abs(got - exp) < 5e-4, f"within the first {k}: recorded {got}, measured {exp:.4f}"
        assert got <= prev + 1e-9, (
            f"the scored share rises from {prev} to {got} as k grows to {k}; unscored records sort "
            f"last, so it can only fall, and a rise means the order is not what it claims")
        prev = got

    # Both halves alike, or the column is two things. This is the check that would catch a delivery
    # scored under a different PASS threshold from the one the comparison set was scored under.
    for half in ("comparison_set", "wider"):
        rows = [r for s, v in det.items() if src.get(s) == half for r in v]
        exp = sum(map(scored, rows)) / len(rows)
        assert abs(share[f"overall_{half}"] - exp) < 5e-4, (
            f"{half}: recorded {share[f'overall_{half}']}, measured {exp:.4f}")
    gap = abs(share["overall_comparison_set"] - share["overall_wider"])
    assert gap < 0.05, (
        f"the two halves differ by {gap:.3f} in the share of records PASS scored "
        f"({share['overall_comparison_set']} against {share['overall_wider']}), which would mean "
        f"they were scored under different thresholds and are not one arm")
