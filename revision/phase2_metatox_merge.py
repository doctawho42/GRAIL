#!/usr/bin/env python3
"""MetaTox's two submissions merged into one arm covering the evaluated population.

Writes results/metatox_smirks_preds_evaluated1170.json.

The first submission covers the 291-substrate comparison set. The second asks the service for the
879 that run does not hold. Neither file covers the evaluated population, and that is the trap this
module closes: a list arm is read with `preds.get(s, [])`, so declaring either file as the arm on
evaluated1170 would hand back an empty list for every substrate it lacks and score those as misses.
MetaTox would be understated on three quarters of the population, in the table built to make the
systems comparable.

This is deliberately the same shape as revision/phase2_gloryx_merge.py, down to the order of the
refusals, because GLORYx's whole-population column is the same thing -- one web service run over the
291 and again over the rest -- and two arms assembled two different ways cannot be compared with
each other. Where the vocabulary differs it is because "published" is true of GLORYx's first run and
not of MetaTox's, whose first run appears in no publication; the halves here are named
"comparison_set" and "wider".

WHY THIS COLUMN EXISTS AT ALL is worth stating, because the manuscript said for a long time that it
could not. It said MetaTox had no whole-population quantity because a second submission was not the
authors' to make. It was made. An output-budget objection then stood in its place and was withdrawn:
it compared a raw record count against a de-duplicated one, it measured a quantity the axis never
reads, and it was computed for MetaTox and asked of no other arm -- computed for all of them, an
admitted arm sits further from parity than MetaTox does. results/metatox_outside_submission.json
carries that withdrawal and the per-arm ratios.

Nothing is re-keyed, re-ordered or truncated here. Those are the table's own steps, and doing any of
them twice is how two readings of one column drift apart.

    python revision/phase2_metatox_merge.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

COMPARISON = ROOT / "results" / "metatox_smirks_preds.json"
WIDER = ROOT / "results" / "metatox_smirks_preds_1170.json"
DELIVERY = ROOT / "results" / "metatox_outside_submission.json"
OUT = ROOT / "results" / "metatox_smirks_preds_evaluated1170.json"
POPULATION_FILE = ROOT / "results" / "test_references.json"


def _read(path):
    """One run's predictions and score detail, through the envelope the ingest writes.

    A ValueError rather than SystemExit: SystemExit derives from BaseException and would pass
    straight through a caller's `except Exception`, which is exactly where a refusal must land.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"REFUSING: {path} does not exist; a missing run is not an empty run")
    blob = json.loads(path.read_text())
    preds = blob.get("predictions") if isinstance(blob, dict) else None
    if not isinstance(preds, dict):
        raise ValueError(f"REFUSING: {path} carries no 'predictions' map")
    detail = blob.get("predictions_with_scores")
    cfg = blob.get("config") if isinstance(blob, dict) else {}
    return preds, (detail if isinstance(detail, dict) else {}), (cfg or {})


def _population():
    return sorted(json.loads(POPULATION_FILE.read_text()))


def merge(comparison_path=COMPARISON, wider_path=WIDER, population=None):
    """Both runs as one substrate map covering the population exactly, or a refusal.

    The checks are ordered so the most informative failure wins: a missing file first, then the
    variant the two runs were made in, then an overlap between them, then coverage.

    The variant check is this column's own, and it is the one the submission README asked for in
    writing. The two halves must have been run in the same MetaTox configuration; a column whose
    halves are different configurations is two arms reported as one. It is checked on what the
    ingest recorded rather than on the output sizes, because output size is a consequence and the
    configuration is the thing.
    """
    population = _population() if population is None else list(population)
    comp, comp_detail, comp_cfg = _read(comparison_path)
    wide, wide_detail, wide_cfg = _read(wider_path)

    # The configuration check, on a controlled value rather than on prose. An earlier version of
    # this asked whether the variant string mentioned SMIRKS, which "layer 1, no SMIRKS" passes:
    # a gate that cannot fail in the direction it was written for. The ingest now records a
    # variant_key, and two runs merged into one arm must carry the same one.
    #
    # The comparison-set artefact predates that field and its source SDF is not in this repository,
    # so it cannot be re-run to gain one. Where a key is missing the check falls back to the prose,
    # which can at least be read for a negation, and the report says the check was weaker for that
    # half rather than presenting both as equally established.
    NEG = ("no smirks", "without smirks", "not smirks", "non-smirks")

    def _key_of(cfg):
        k = cfg.get("variant_key")
        if k:
            return str(k), "recorded"
        v = str(cfg.get("variant", ""))
        low = v.lower()
        if any(n in low for n in NEG):
            return None, "prose says it is not the SMIRKS variant"
        if "smirks" in low:
            return "smirks", "inferred from prose, this artefact predates the recorded key"
        return None, f"prose does not identify a variant: {v!r}"

    comp_key, comp_how = _key_of(comp_cfg)
    wide_key, wide_how = _key_of(wide_cfg)
    if comp_key is None or wide_key is None or comp_key != wide_key:
        raise ValueError(
            f"REFUSING: the two runs are not the same configuration. comparison set: "
            f"{comp_key!r} ({comp_how}); wider: {wide_key!r} ({wide_how}). The submission asked for "
            f"the configuration the comparison set was scored in, and halves run in different "
            f"configurations are two arms reported as one.")
    variant_check = {"key": comp_key, "comparison_set": comp_how, "wider": wide_how}

    both = set(comp) & set(wide)
    if both:
        raise ValueError(
            f"REFUSING: {len(both)} substrate(s) appear in both runs, so one of them is not the "
            f"run it reports being; the two were obtained under different submission sets and an "
            f"overlap is a fact about the artefacts rather than a tie to break. "
            f"First: {sorted(both)[0][:60]}")

    merged, detail, source = {}, {}, {}
    for s, v in comp.items():
        merged[s], source[s] = v, "comparison_set"
    for s, v in wide.items():
        merged[s], source[s] = v, "wider"
    detail.update(comp_detail)
    detail.update(wide_detail)

    pop = set(population)
    missing = pop - set(merged)
    extra = set(merged) - pop
    if missing:
        raise ValueError(
            f"REFUSING: the merge does not cover the population: "
            f"{len(pop) - len(missing)} of {len(pop)} substrates, {len(missing)} missing. An arm "
            f"short of its population scores every absent substrate as a miss. "
            f"First missing: {sorted(missing)[0][:60]}")

    # A repeated structure is a wasted slot rather than a wrong answer, and it inflates the output
    # size recall@k is read against. Both ingests de-duplicate; if one stopped, the halves are
    # measured differently in the one property this paper is about.
    repeats = {s: len(v) - len(set(v)) for s, v in merged.items() if len(v) != len(set(v))}
    if repeats:
        raise ValueError(
            f"REFUSING: {len(repeats)} substrate(s) carry a repeated structure, "
            f"{sum(repeats.values())} repeats in all. De-duplication happens in the ingest and must "
            f"not be undone or redone here. First: {sorted(repeats)[0][:60]}")

    return {
        "predictions": {s: merged[s] for s in population},
        "predictions_with_scores": {s: detail[s] for s in population if s in detail},
        "source_of_each_substrate": {s: source[s] for s in population},
        "n_from_comparison_set": sum(1 for s in population if source[s] == "comparison_set"),
        "n_from_wider": sum(1 for s in population if source[s] == "wider"),
        "substrates_outside_the_population": sorted(extra),
        "variants": {"comparison_set": comp_cfg.get("variant"), "wider": wide_cfg.get("variant")},
        "variant_check": variant_check,
        "joins": {"comparison_set": comp_cfg.get("join"), "wider": wide_cfg.get("join")},
    }


def _digest(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> int:
    try:
        got = merge()
    except ValueError as e:
        print(e)
        return 1

    # How much of the column the stated ranking actually decides. The field above used to claim the
    # order outright, and half the entries carry no score to order them by.
    KS = (5, 10, 15, 30, 50)
    det = got["predictions_with_scores"]

    def _scored(row):
        return isinstance(row[1], (int, float)) and row[1] == row[1]

    _all = [r for rows in det.values() for r in rows]
    ranking_share = {"overall": round(sum(map(_scored, _all)) / len(_all), 4)}
    for _k in KS:
        _slots = [r for rows in det.values() for r in rows[:_k]]
        ranking_share[f"within_the_first_{_k}"] = round(
            sum(map(_scored, _slots)) / len(_slots), 4)
    for _half in ("comparison_set", "wider"):
        _h = [r for s, rows in det.items() if got["source_of_each_substrate"].get(s) == _half
              for r in rows]
        ranking_share[f"overall_{_half}"] = round(sum(map(_scored, _h)) / len(_h), 4)

    sizes = [len(v) for v in got["predictions"].values()]
    wider_sizes = [len(v) for s, v in got["predictions"].items()
                   if got["source_of_each_substrate"][s] == "wider"]
    comp_sizes = [len(v) for s, v in got["predictions"].items()
                  if got["source_of_each_substrate"][s] == "comparison_set"]

    report = {
        "what_this_is": ("MetaTox on the evaluated population, assembled from the comparison-set "
                         "submission and the second submission over the rest"),
        "why_a_merge": ("neither run covers the population; a list arm is read with "
                        "preds.get(s, []), so a partial file would score every substrate it lacks "
                        "as a miss"),
        "provenance": stamp(__file__),
        "inputs": record_inputs([COMPARISON, WIDER, POPULATION_FILE]),
        "population": "evaluated1170",
        "sources": {
            "comparison_set": {"path": str(COMPARISON.relative_to(ROOT)),
                               "sha256": _digest(COMPARISON),
                               "substrates": got["n_from_comparison_set"]},
            "wider": {"path": str(WIDER.relative_to(ROOT)), "sha256": _digest(WIDER),
                      "substrates": got["n_from_wider"]},
        },
        "variants": got["variants"],
        "variant_check": got["variant_check"],
        "joins": got["joins"],
        "drawing": ("the substrate as the corpus stores it, which is what both submissions were "
                    "keyed back to and what the scoring joins on"),
        "ranking": (
            "the method's own Pa for the Metabolite class, descending, in both halves; records PASS "
            "declined to score keep the order the delivery gave them and sort after every scored "
            "record. See ranking_decides_this_share -- the score does not decide every slot, and "
            "saying only the first half of this sentence would overstate what orders the column"),
        "ranking_decides_this_share": ranking_share,
        "why_that_share_matters": (
            "PASS writes a Metabolite spectrum only where it clears its own threshold, so about half "
            "of what MetaTox returns carries no score at all and is ordered by the supplier's file. "
            "Because the unscored block sorts last the effect is small at the budgets the axis reads "
            "most and grows with k. It is a property of BOTH halves alike, so it does not make them "
            "incomparable; it is a limit on what recall at large k says about the method rather than "
            "about the file, and it applied to the comparison-set column long before this merge"),
        "n_substrates": len(sizes),
        "n_predictions": int(sum(sizes)),
        "mean_output": round(sum(sizes) / len(sizes), 2),
        "mean_output_comparison_set": round(sum(comp_sizes) / len(comp_sizes), 4),
        "mean_output_wider": round(sum(wider_sizes) / len(wider_sizes), 4),
        "two_half_output_ratio": round(
            (sum(wider_sizes) / len(wider_sizes)) / (sum(comp_sizes) / len(comp_sizes)), 4),
        "what_this_ratio_is_for": (
            "the halves of an arm should be the same configuration, and this is the statistic that "
            "says whether they are. It is recorded for every arm the axis reads in "
            "results/metatox_outside_submission.json under peer_two_half_ratios, because a ratio "
            "computed for one arm only is a rule available wherever it excludes"),
        "predictions": got["predictions"],
        "predictions_with_scores": got["predictions_with_scores"],
        "source_of_each_substrate": got["source_of_each_substrate"],
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {report['n_substrates']} substrates, {report['n_predictions']} predictions, "
          f"mean output {report['mean_output']}")
    print(f"  comparison set {report['mean_output_comparison_set']} | "
          f"wider {report['mean_output_wider']} | ratio {report['two_half_output_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
