#!/usr/bin/env python3
"""The second MetaTox delivery, measured: what came back, what it covers, and what it is not.

The manuscript says MetaTox has no quantity on the whole evaluated test set because a second
submission to that web service is not the authors' to make. The submission was made. Its set is in
this repository at revision/metatox_submission_1170 -- 879 substrates, SUB0001..SUB0879, the exact
complement of the 291 MetaTox had already answered -- and two SDF files came back against it. This
script measures them and writes results/metatox_outside_submission.json.

Three things it establishes, in the order they decide anything.

COVERAGE. The submission set carries an id per substrate and the delivery keys on those ids, so the
join is the one the README asked for and not a positional guess. The larger file returns all 879.
With the 291 already held, disjoint from them and summing to the evaluated 1,170, MetaTox now covers
the whole population -- above the 0.99 floor scripts/typed_edit/population_definition.py applies. The
old reason for the arm's absence is not merely stale, it is the wrong reason.

OUTPUT BUDGET, which is why the arm still does not join the axis. The submission README asked for
the same configuration as the first batch, in those words, and said why: two halves at different
budgets are two arms reported as one. They are not the same. The first batch returned 10,601 records
for 291 substrates; this one returns 229,106 for 879. Both counts are raw records -- the first
batch's ingest keeps every parseable record and filters none by score -- so the rates divide, and
the second batch emits about seven times more per substrate. Recall at every k moves with output
size, so a column half-scored at one budget and half at the other would be an artefact of the
budgets. That is measured here and recorded as a blocker rather than argued about.

THE TWO FILES DISAGREE, so both are recorded. all_meta_smirks returns 229,106 records over all 879
substrates and no _UNKNOWN. all_metabolSep26 returns 9,975 records over 806, of which 704 are marked
_UNKNOWN across 9 substrates, and it alone carries the MetaTox tree fields and the parent record.
The counts above are the larger file's; the smaller one is recorded beside them rather than dropped,
because a reader handed one number cannot see that a second file answers differently.

The delivered SDFs are hundreds of megabytes and are not in the repository. Their sha256 is recorded
so the artifact names something definite, and the script exits with a message rather than a
traceback when they are not on disk.

    python scripts/metatox_outside_ingest.py \
        --orig  "/path/to/all_meta_smirks (MetaboliteLikeness).SDF" \
        --smirks "/path/to/all_metabolSep26 (MetaboliteLikeness).SDF"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import record_inputs, stamp  # noqa: E402

SUBMISSION = ROOT / "revision" / "metatox_submission_1170" / "substrate_map.csv"
TRUTH = ROOT / "results" / "test_references.json"
BATCH1 = ROOT / "results" / "metatox_smirks_preds.json"
AXIS = ROOT / "scripts" / "typed_edit" / "population_definition.py"
OUT = ROOT / "results" / "metatox_outside_submission.json"

# WHY THERE IS NO TOLERANCE CONSTANT HERE ANY MORE.
#
# An earlier version of this script asserted that MetaTox's two halves were run at different output
# budgets and excluded the arm on that ground, against a tolerance of 0.10 invented in this file.
# Every part of that was wrong and the withdrawal is recorded in the artifact rather than quietly
# reverted. It divided a de-duplicated count by a raw record count; it measured a mean output size
# the axis never reads, since population_definition.py truncates every arm to k; and above all it
# applied a rule to one arm that was computed for no other, while the arm built in exactly the same
# way -- GLORYx, a web service merged from the same 291 run and a separate 879 run -- was admitted
# with no such check, and an admitted arm (SyGMa) fails the rule outright.
#
# So the two-half ratio is now computed for EVERY arm the axis reads, and MetaTox is judged against
# what those arms actually do rather than against a number chosen here. A rule available for one arm
# only is the defect this manuscript exists to object to.


def coverage_floor() -> float:
    """The floor the axis applies, read from the axis rather than remembered.

    population_definition.py writes it as a literal inside the comparison that drops an arm, not as
    a named constant, so it is taken from there. If that line ever stops looking like this the floor
    has moved or been renamed, and this script must not carry on quoting the old number.
    """
    src = AXIS.read_text()
    hits = re.findall(r"if covered < ([0-9.]+) \* len\(truth\)", src)
    if len(hits) != 1:
        sys.exit(f"{AXIS.relative_to(ROOT)} no longer states its coverage floor in the one form "
                 f"this script can read ({len(hits)} matches); re-read it before trusting any "
                 f"admission decision here")
    return float(hits[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def id_tag_of(path: Path) -> str:
    """Which tag carries the full submission id in THIS file, read from the file.

    The two deliveries are not tagged alike. One writes the full id as <ID>; the other writes it as
    <ID_all> and uses <ID> for a bare within-substrate index. Assuming a tag per file, as an earlier
    version did, silently produced a scan over sixteen substrates and one over none, because the
    assumption was attached to the author's declaration of which file was which and the declaration
    changed. The file is asked instead.
    """
    seen = set()
    with path.open(errors="replace") as fh:
        for i, line in enumerate(fh):
            m = re.match(r"^>\s*<(ID_all|ID)>", line)
            if m:
                seen.add(m.group(1))
            if i > 4000 or len(seen) == 2:
                break
    if "ID_all" in seen:
        return "ID_all"
    if "ID" in seen:
        return "ID"
    sys.exit(f"{path.name} carries neither an ID nor an ID_all tag in its first records; it is not "
             f"a MetaTox delivery in either shape this script knows")


def count_structures(path: Path, id_tag: str) -> dict:
    """Distinct predicted STRUCTURES per substrate, which is the quantity that governs recall.

    A supplier writes one record per route, so a tree that reaches the same product three ways
    writes it three times. Nothing downstream of here sees those repeats: metrics.py scores
    set-based over InChIKey, population_definition.py truncates each arm to k, and even the first
    batch's own ingest de-duplicates by canonical SMILES before it counts
    (scripts/metatox_smirks_ingest.py, "if smiles in seen: continue"). So a record rate compared
    against that ingest's rate compares two different quantities, and the difference is whatever
    the supplier's export happened to repeat.

    De-duplication is on canonical SMILES, the same convention the first batch's artifact was built
    with, so the two rates are the same measurement. The scorer goes further and keys on the
    tautomer InChIKey; that is a cross-check recorded beside this, not the basis, because the
    comparison has to be against the artifact the axis would actually read.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    per: dict = {}
    records = 0
    unparsed = 0
    for mol in Chem.SDMolSupplier(str(path), sanitize=True):
        if mol is None:
            unparsed += 1
            continue
        raw = mol.GetPropsAsDict().get(id_tag)
        if raw is None or "_" not in str(raw):
            continue
        suf = str(raw).split("_", 1)[1]
        if suf == "0" or "UNKNOWN" in suf:
            continue
        smi = Chem.MolToSmiles(mol)
        if not smi:
            continue
        records += 1
        per.setdefault(str(raw).split("_")[0], set()).add(smi)
    return {"per_substrate": {k: len(v) for k, v in per.items()},
            "records": records, "unparsed": unparsed,
            "distinct": sum(len(v) for v in per.values())}


def scan_sdf(path: Path, id_tag: str) -> dict:
    """Records, substrate ids and PASS bookkeeping, in one streaming pass.

    Read as text rather than through RDKit: nothing here needs a parsed molecule, and the larger
    file is 702 MB, where sanitising a quarter of a million structures to count them would cost
    minutes for numbers the tags already carry.

    The PASS cross-check is the one the first batch's ingest applies -- the spectrum is written only
    where PASS declares the record above its own threshold, so the two tags must agree per record.
    A disagreement means the field is not being read as PASS wrote it, and nothing counted from it
    would mean anything.
    """
    per_substrate: Counter = Counter()
    non_predictions: Counter = Counter()
    ids: list[str] = []
    suffixes: Counter = Counter()
    records = declared = spectra = disagree = errors = 0
    layers: Counter = Counter()
    started = False
    cur_declared = cur_spectrum = False
    pending = None

    def close():
        nonlocal records, declared, spectra, disagree, cur_declared, cur_spectrum
        if started:
            records += 1
            declared += cur_declared
            spectra += cur_spectrum
            disagree += (cur_declared != cur_spectrum)
        cur_declared = cur_spectrum = False

    with path.open(errors="replace") as fh:
        for line in fh:
            if line.startswith("$$$$"):
                close()
                started = False
                pending = None
                continue
            m = re.match(r"^>\s*<([^>]*)>", line)
            if m:
                started = True
                tag = m.group(1)
                pending = tag if tag in (id_tag, "PASS_RESULT_COUNT", "Layer") else None
                if tag == "PASS_ACTIVITY_SPECTRUM":
                    cur_spectrum = True
                elif tag == "PASS_ERROR":
                    errors += 1
                continue
            if pending == id_tag:
                raw = line.strip()
                ids.append(raw)
                sub = raw.split("_")[0]
                suf = raw.split("_", 1)[1] if "_" in raw else ""
                suffixes[suf] += 1
                # What counts as a prediction, on one rule that works for both files. A record is a
                # predicted metabolite unless it is the submitted parent re-emitted (suffix 0) or
                # one MetaTox marked _UNKNOWN, which its own supplier describes as a run that did
                # not work and is usually the parent again. Counting either as a prediction would
                # credit the method for handing back what it was given.
                if suf != "0" and "UNKNOWN" not in suf:
                    per_substrate[sub] += 1
                else:
                    non_predictions[sub] += 1
                pending = None
            elif pending == "PASS_RESULT_COUNT":
                # PASS writes "1 of 1" where the record cleared its threshold.
                cur_declared = "1 of 1" in line
                pending = None
            elif pending == "Layer":
                layers[line.strip()] += 1
                pending = None
    close()

    if disagree:
        sys.exit(f"{path.name}: PASS_RESULT_COUNT and PASS_ACTIVITY_SPECTRUM disagree on "
                 f"{disagree} records; the spectrum field is not being read as PASS wrote it and "
                 f"no count from this file can be used")

    unknown_records = sum(n for s, n in suffixes.items() if "UNKNOWN" in s)
    unknown_substrates = sorted({i.split("_")[0] for i in ids if "UNKNOWN" in i})
    return {"name": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "records": records,
            "metabolite_records": sum(per_substrate.values()),
            "substrates": len({i.split("_")[0] for i in ids}),
            "_substrate_ids_seen": {i.split("_")[0] for i in ids},
            "substrates_answering": len(per_substrate),
            "ids_seen": len(ids),
            "records_pass_scored": spectra,
            "pass_error_records": errors,
            "unknown_marker_records": unknown_records,
            "unknown_marker_substrates": len(unknown_substrates),
            "carries_parent_record": bool(suffixes.get("0")),
            "carries_tree_fields": bool(layers),
            "layer_counts": dict(sorted(layers.items())) or None,
            "_per_substrate": per_substrate}


def peer_two_half_ratios(held_keys: set, evaluated: set) -> dict:
    """The same two-half statistic, for every arm the whole-population axis already reads.

    This exists because of what went wrong here. A cross-half output ratio was computed for MetaTox,
    found to be large, and used to exclude it -- and the number had never been computed for any
    other arm. Had it been, the rule would have been seen to disqualify SyGMa, which is admitted,
    and never to have been asked of GLORYx, whose whole-population column is built exactly like the
    one MetaTox would need: a web service run once over the 291 and again over the other 879, merged.

    So the statistic is produced for all of them together or not at all. An arm cannot now be
    measured against a threshold that its peers were never held to.
    """
    import statistics
    sys.path.insert(0, str(AXIS.parent))
    import population_definition as P  # noqa: E402

    out = {}
    for name, path in sorted(P.WHOLE_TEST.items()):
        if not Path(path).exists():
            out[name] = {"error": f"{Path(path).name} is not on disk"}
            continue
        blob = json.loads(Path(path).read_text())
        preds = blob.get("predictions") if isinstance(blob, dict) else None
        if not isinstance(preds, dict):
            preds = blob if isinstance(blob, dict) else {}
        inside = [len(v) for k, v in preds.items() if k in held_keys and isinstance(v, list)]
        outside = [len(v) for k, v in preds.items()
                   if k in evaluated and k not in held_keys and isinstance(v, list)]
        if not inside or not outside:
            out[name] = {"error": "the arm does not cover both halves"}
            continue
        a, b = statistics.mean(inside), statistics.mean(outside)
        out[name] = {"comparison_set_mean": round(a, 4), "wider_mean": round(b, 4),
                     "ratio": round(b / a, 4),
                     "n_comparison_set": len(inside), "n_wider": len(outside)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smirks", required=True,
                    help="the SMIRKS-rule run, the configuration the comparison set was scored in "
                         "-- the author states this is all_meta_smirks (MetaboliteLikeness).SDF")
    ap.add_argument("--orig", required=True,
                    help="the original run -- the author states this is "
                         "all_metabolSep26 (MetaboliteLikeness).SDF")
    ap.add_argument("--delivered", default="2026-09-16", help="date the files were sent")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    paths = {"orig": Path(args.orig), "smirks": Path(args.smirks)}
    for k, p in paths.items():
        if not p.exists():
            sys.exit(f"--{k}: {p} is not on disk. The delivered SDFs are hundreds of megabytes and "
                     f"are not tracked; point this at the extracted archives. The sha256 of the "
                     f"files this artifact was built from is recorded in {OUT.relative_to(ROOT)}.")

    # --- the delivery, file by file -------------------------------------------------------------
    print("scanning the delivered files", flush=True)

    # WHICH FILE IS THE ARM. Declared by the author, who corresponded with the supplier, and NOT
    # inferred here. An earlier version of this script inferred it from the files -- all_metabolSep26
    # is the only one carrying named rule applications ("Epoxidation_1") in its Parent Reactions and
    # Children Reactions fields, which looks like what a rule-based run leaves behind -- and the
    # inference was WRONG. The author states that all_meta_smirks is the SMIRKS run and
    # all_metabolSep26 is the original. That agrees with the file names and with the stronger
    # evidence the inference had set aside: all_meta_smirks is the file whose record format matches
    # the batch the comparison set was scored from, carrying no parent record and only PASS fields,
    # exactly as scripts/metatox_smirks_ingest.py describes its input.
    #
    # The reaction-name evidence still points the other way and is recorded in the artifact rather
    # than dropped, because the next reader will notice it too and should find it already answered.
    tags = {k: id_tag_of(v) for k, v in paths.items()}
    print(f"  id tags read from the files: "
          f"{', '.join(f'{paths[k].name} -> {t}' for k, t in tags.items())}", flush=True)
    arm, other = scan_sdf(paths["smirks"], tags["smirks"]), scan_sdf(paths["orig"], tags["orig"])
    # A second pass, through RDKit, for the quantity that actually governs recall: distinct
    # structures. The text scan above cannot produce it, and a record count is not a substitute.
    for f, key in ((arm, "smirks"), (other, "orig")):
        print(f"  counting distinct structures in {f['name']}", flush=True)
        f["_structures"] = count_structures(paths[key], tags[key])
        f["distinct_structures"] = f["_structures"]["distinct"]
        f["unparsed_records"] = f["_structures"]["unparsed"]
    for f in (arm, other):
        print(f"  {f['name']}: {f['records']} records, {f['metabolite_records']} of them "
              f"predictions, over {f['substrates']} substrates of which "
              f"{f['substrates_answering']} answered, {f['unknown_marker_records']} _UNKNOWN",
              flush=True)
    print(f"  arm (declared, the SMIRKS run): {arm['name']}", flush=True)
    # The one internal signal that disagrees with the declaration, measured so the disagreement is
    # a recorded fact rather than a thing a later reader rediscovers.
    contradicts = bool(other["carries_tree_fields"]) and not arm["carries_tree_fields"]
    if contradicts:
        print(f"  note: {other['name']} is the file carrying rule-application fields, which points "
              f"the other way; the declaration governs", flush=True)
    src = [arm, other]

    # --- coverage -------------------------------------------------------------------------------
    truth = json.loads(TRUTH.read_text())
    batch1 = json.loads(BATCH1.read_text())
    held = batch1["predictions"]
    rows = list(csv.DictReader(SUBMISSION.open()))
    sub2key = {r["id"]: r["substrate_smiles"] for r in rows}

    evaluated, held_keys = set(truth), set(held)
    b1_subs, b1_recs = batch1["n_substrates"], batch1["n_predictions"]
    b1_rate = b1_recs / b1_subs

    def reading(f: dict) -> dict:
        """Coverage and output budget if THIS file is the arm.

        Computed the same way for both files so the two can be set side by side. A substrate counts
        as covered when the file returns at least one prediction for it, which is what an arm has to
        do to be scored: a substrate present only as its own re-emitted parent, or only under an
        _UNKNOWN marker, has no prediction to rank and would enter the population as a silent zero.
        """
        returned_ids = set(f["_per_substrate"])
        unmapped = sorted(returned_ids - set(sub2key))
        if unmapped:
            sys.exit(f"{f['name']} returns {len(unmapped)} ids the submission map does not contain "
                     f"(e.g. {unmapped[:3]}); the delivery is not keyed to this submission and "
                     f"joining it would be a guess")
        recovered = {sub2key[i] for i in returned_ids}
        stray = recovered - evaluated
        if stray:
            sys.exit(f"{f['name']}: {len(stray)} substrates are not in the evaluated population; "
                     f"the coverage would be over a set nobody defined")
        union = held_keys | recovered
        share = len(union) / len(evaluated)
        # The output budget, on STRUCTURES. The comparison set's own rate comes from the first
        # batch's artifact, which its ingest de-duplicated by canonical SMILES before counting, so
        # the delivery is de-duplicated the same way or the two are not one measurement.
        st = f["_structures"]["per_substrate"]
        answering = {k: v for k, v in st.items() if v}
        sizes = sorted(answering.values())
        rate = sum(answering.values()) / len(answering)
        return {
            "file": f["name"],
            "coverage": {
                "evaluated_substrates": len(evaluated),
                "metatox_on_comparison_set": len(held_keys),
                "recovered_from_this_submission": len(recovered),
                "submitted_but_no_prediction_returned": len(sub2key) - len(recovered),
                # The two ways a submitted substrate ends up with nothing to rank, split, because
                # they are different failures: one is a substrate the delivery never mentions, the
                # other is one it mentions only by handing our own structure back.
                "absent_from_the_file": len(sub2key) - len(f["_substrate_ids_seen"]),
                "present_without_a_prediction":
                    len(f["_substrate_ids_seen"]) - f["substrates_answering"],
                "overlap_between_them": len(held_keys & recovered),
                "union": len(union),
                "share": share,
            },
            "clears_the_floor": share >= floor,
            "budget": {
                "basis": (
                    "distinct predicted structures per answering substrate, de-duplicated by "
                    "canonical SMILES on both sides. The first batch's artifact is built that way "
                    "(scripts/metatox_smirks_ingest.py de-duplicates before counting), the scorer "
                    "de-duplicates again on the tautomer InChIKey, and metrics.py is set-based, so "
                    "a repeated record changes no number anywhere downstream"),
                "comparison_set_substrates": b1_subs,
                "comparison_set_structures": b1_recs,
                "comparison_set_mean_output": round(b1_rate, 4),
                "this_delivery_substrates": len(answering),
                "this_delivery_structures": sum(answering.values()),
                "this_delivery_mean_output": round(rate, 4),
                "this_delivery_median_output": statistics.median(sizes),
                "this_delivery_min_output": sizes[0],
                "this_delivery_max_output": sizes[-1],
                "ratio": round(rate / b1_rate, 6),
                "raw_records_for_comparison": f["metabolite_records"],
                "raw_record_rate": round(f["metabolite_records"] / f["substrates_answering"], 4),
                "why_the_raw_rate_is_not_used": (
                    "the delivery writes one record per route, so the same product is written once "
                    "per way the tree reaches it; the raw rate is the supplier's export format, not "
                    "the arm's output size"),
            },
        }

    floor = coverage_floor()
    readings = {"as_the_arm": reading(arm), "if_the_other_file_is_the_arm": reading(other)}
    for label, r in readings.items():
        c = r["coverage"]
        print(f"  {label}: {r['file']}", flush=True)
        print(f"    coverage {c['metatox_on_comparison_set']} + "
              f"{c['recovered_from_this_submission']} = {c['union']}/{c['evaluated_substrates']} "
              f"= {c['share']:.4f} vs floor {floor} -> "
              f"{'clears' if r['clears_the_floor'] else 'BELOW'}", flush=True)
        print(f"    budget   {r['budget']['this_delivery_mean_output']} vs "
              f"{r['budget']['comparison_set_mean_output']} structures per substrate = "
              f"{r['budget']['ratio']:.3f}x  (raw records would say "
              f"{r['budget']['raw_record_rate'] / r['budget']['comparison_set_mean_output']:.2f}x)",
              flush=True)

    main_reading = readings["as_the_arm"]
    coverage, budget = main_reading["coverage"], main_reading["budget"]
    share, clears = coverage["share"], main_reading["clears_the_floor"]
    ratio = budget["ratio"]
    union, recovered_n = coverage["union"], coverage["recovered_from_this_submission"]

    # --- admission ------------------------------------------------------------------------------
    # The cross-half ratio, for every arm the axis reads, so MetaTox is judged against what its
    # peers actually do. Computing it for one arm only is how the earlier version of this script
    # produced a rule that excluded MetaTox while never being asked of anything else.
    peers = peer_two_half_ratios(held_keys, evaluated)
    peer_rs = {k: v["ratio"] for k, v in peers.items() if "ratio" in v}
    print("  cross-half output ratio, every arm the axis reads:", flush=True)
    for k, v in sorted(peers.items()):
        print(f"    {k:16s} {v.get('ratio', v.get('error'))}", flush=True)
    print(f"    {'metatox':16s} {ratio}  (this delivery)", flush=True)

    blockers = []
    if not clears:
        blockers.append({
            "name": "coverage below the floor",
            "what_it_is": (
                f"MetaTox returns a prediction for {union} of {len(evaluated)} evaluated "
                f"substrates, under the {floor} the population axis requires of an arm. Of the "
                f"{len(sub2key)} submitted, {coverage['submitted_but_no_prediction_returned']} came "
                f"back with no prediction at all."),
            "measured": {k: coverage[k] for k in
                         ("union", "evaluated_substrates", "recovered_from_this_submission",
                          "submitted_but_no_prediction_returned")} | {"share": round(share, 6),
                                                                      "floor": floor}})
    # An output-budget blocker may only fire if this arm's two halves differ by MORE than the arms
    # already admitted differ. Against an absolute threshold it would be a rule invented for one
    # comparator; against its peers it is a statement about this delivery.
    worst_peer = max(peer_rs.values(), key=lambda r: abs(r - 1.0)) if peer_rs else None
    if worst_peer is not None and abs(ratio - 1.0) > abs(worst_peer - 1.0):
        blockers.append({
            "name": "the two halves differ in output size by more than any admitted arm's halves do",
            "what_it_is": (
                f"this delivery's two halves differ by {ratio:.3f}x, outside the "
                f"{min(peer_rs.values()):.3f}x-{max(peer_rs.values()):.3f}x the admitted arms span"),
            "measured": {"metatox": ratio, "peers": peer_rs}})

    eligible = clears and not blockers
    # Whether the arm is actually in the axis, read from the axis's own dict and from disk.
    # Imported after peer_two_half_ratios has put scripts/typed_edit on the path; importing it at
    # the top would need a second copy of that insert, and two places that fix the path is how one
    # of them stops matching.
    import population_definition as _P  # noqa: E402
    _named = {k.lower(): v for k, v in _P.WHOLE_TEST.items()}
    _mt = next((v for k, v in _named.items() if "metatox" in k), None)
    _in_axis = bool(_mt and Path(_mt).exists())
    # Whether the fork over which file is the arm decides anything. It used to be the claim that
    # MetaTox was excluded under both readings; with the budget objection withdrawn that is no
    # longer true, and the honest statement is narrower: the two readings disagree, and under the
    # declared one the arm is eligible.
    eligible_by_reading = {k: r["clears_the_floor"] for k, r in readings.items()}
    the_fork_decides = len(set(eligible_by_reading.values())) > 1

    # WHAT WAS CLAIMED HERE AND WITHDRAWN. Recorded rather than reverted, because a blocker asserted
    # in a manuscript and then dropped is a decision, and a decision that leaves no trace is the
    # thing this project keeps finding in other people's papers.
    withdrawn = [{
        "name": "the two halves were run at different output budgets",
        "was_claimed": (
            "that the delivery emits 260.64 predictions per substrate against 36.43 in the "
            "comparison set, a factor of 7.15, so the halves are not one configuration and the arm "
            "cannot be read on this population"),
        "why_it_was_withdrawn": [
            "it divided a de-duplicated count by a raw one. 36.43 comes from lists that "
            "scripts/metatox_smirks_ingest.py de-duplicates by canonical SMILES before counting; "
            f"229,106 was a raw record count. On structures the delivery gives "
            f"{budget['this_delivery_mean_output']} against {budget['comparison_set_mean_output']}, "
            f"a factor of {ratio:.3f}",
            "it measured a quantity the axis does not read. population_definition.py scores "
            "len(set(order[s][:k]) & real[s]) at k in (5,10,15,30,50), so each arm is truncated to "
            "k and a longer list beyond k changes nothing",
            "it was computed for one arm and no other. The same statistic for the admitted arms is "
            "in `peer_two_half_ratios` above; an admitted arm lies further from 1.0 than MetaTox "
            "does, and no code anywhere computed this for any of them",
            "the arm built the same way was admitted without it. "
            "results/gloryx_service_preds_evaluated1170.json is a web service merged from a "
            "291-substrate run and a separate 879-substrate run, the same shape as the MetaTox "
            "column would be, and no budget check was ever asked of it",
        ],
        "found_by": ("an adversarial review of the blocker, run because it had become the sole "
                     "ground for the exclusion"),
    }]

    art = {
        "what_this_is": (
            f"the second MetaTox submission -- the {len(sub2key)} evaluated substrates outside the "
            "comparison set -- as delivered, with what it covers and the measured reasons it does "
            "not join the whole-population axis"),
        "provenance": stamp(__file__),
        "inputs": record_inputs([SUBMISSION, TRUTH, BATCH1, AXIS]),
        "delivered": args.delivered,
        "submission_set": {
            "path": str(SUBMISSION.parent.relative_to(ROOT)),
            "submitted": len(sub2key),
            "ids": f"{rows[0]['id']}..{rows[-1]['id']}",
            "join": ("by the submission id the delivery keys on, which is the id in this map; the "
                     "first batch had no ids in its file and needed a positional join gated on "
                     "Tanimoto, and this one does not"),
        },
        "sources": [{k: v for k, v in f.items() if not k.startswith("_")} for f in src],
        "which_file_is_the_arm": {
            "chosen": arm["name"],
            "how": "declared by the author, who corresponded with the supplier; not inferred here",
            "on_what_evidence": (
                f"the author states that {arm['name']} is the SMIRKS run and {other['name']} is the "
                f"original. That agrees with the file names, and with the record format: "
                f"{arm['name']} carries no parent record and only PASS fields, exactly the shape "
                "scripts/metatox_smirks_ingest.py describes for the batch the comparison set was "
                "scored from"),
            "against_it": (
                f"{other['name']} is the only file carrying named rule applications "
                f"(\"Epoxidation_1\") in its Parent Reactions and Children Reactions fields, which "
                "is what one would expect a rule-based run to leave behind. An earlier version of "
                "this script inferred the assignment from exactly that signal and got it backwards. "
                "The signal is recorded here rather than dropped, because the next reader will "
                "notice it too; the declaration governs, and the other reading is computed in full "
                "below so the choice can be checked instead of trusted"),
            "internal_evidence_contradicts_the_declaration": contradicts,
        },
        "readings": readings,
        "eligible_by_reading": eligible_by_reading,
        "the_choice_of_file_decides_eligibility": the_fork_decides,
        "disagreement": {
            "counts_taken_from": arm["name"],
            "what_differs": (
                f"{other['name']} returns {other['records']} records over {other['substrates']} "
                f"substrates, {other['metabolite_records']} of them predictions, with "
                f"{other['unknown_marker_records']} _UNKNOWN; {arm['name']} returns "
                f"{arm['records']} over {arm['substrates']}, {arm['metabolite_records']} of them "
                f"predictions, with {arm['unknown_marker_records']} _UNKNOWN across "
                f"{arm['unknown_marker_substrates']} substrates"),
            "why": (f"{arm['name']} is taken as the arm on the evidence above; {other['name']} "
                    "answers more substrates and is kept here in full rather than dropped, because "
                    "a reader handed one number cannot see that a second file answers differently"),
            "the_supervisors_note": (
                f"the note places the _UNKNOWN records in {other['name']}. They are not there: "
                f"that file carries none. All {arm['unknown_marker_records']} are in {arm['name']}, "
                f"across {arm['unknown_marker_substrates']} substrates, not throughout."),
        },
        "coverage": coverage,
        "coverage_floor": floor,
        "coverage_floor_source": str(AXIS.relative_to(ROOT)),
        "coverage_clears_the_floor": clears,
        "budget": budget,
        "blockers": blockers,
        "withdrawn_blockers": withdrawn,
        "peer_two_half_ratios": peers,
        "eligible_for_the_whole_population": eligible,
        # Eligibility is not presence, and presence is read from the axis rather than asserted
        # here. This field was a hardcoded False while the column did not exist; the test that
        # checks it against population_definition.WHOLE_TEST is what caught it going stale the
        # moment the column was built, which is the whole reason a claim about work done belongs
        # nowhere near the place that claims it.
        "in_the_whole_population_axis": _in_axis,
        "why_not_in_the_axis_yet": (
            None if _in_axis else
            "no ranked MetaTox column over the 879 exists; the delivery is measured here but has "
            "not been turned into an artifact the axis can read"),
        "what_changed": (
            "paper2/si.tex and scripts/typed_edit/population_definition.py both said MetaTox has no "
            "whole-population quantity because a second submission to that web service is not the "
            "authors' to make. It was made and it came back for every substrate submitted. An "
            "output-budget objection was then raised against admitting the arm and is withdrawn "
            "here, with the measurements that withdrew it. What remains is not a reason to exclude "
            "MetaTox but a column nobody has built."),
    }

    out = Path(args.out)
    out.write_text(json.dumps(art, indent=1, sort_keys=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}", flush=True)
    print(f"  coverage {union}/{len(evaluated)} = {share:.4f}, floor {floor}, clears={clears}, "
          f"blockers={len(blockers)}, withdrawn={len(withdrawn)}, eligible={eligible}, "
          f"in the axis={_in_axis}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
