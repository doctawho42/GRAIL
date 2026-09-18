#!/usr/bin/env python3
"""The SMIRKS-variant MetaTox predictions, keyed back to the substrates they were run on.

The supplier's earlier delivery was layer 1 without the SMIRKS rules and returned 270 of the 291
submitted parents (results/grail_vs_metatox.json), and on that delivery the paper carried no MetaTox
column at all. This is the SMIRKS variant, it covers all 291, and it is what put MetaTox into the
manuscript: the arm is now a column in fourteen generated tables. This sentence said the opposite
for as long as it went unread, so the count is taken from the files rather than from memory --
grep -l MetaTox paper2/*table*.tex.

The first delivery carries no substrates. Records are identified only as `<substrate index>_<metabolite
index>`, so the substrates have to be recovered from the submission order in
results/metatox_input/substrate_map.csv. A positional join is exactly the kind of assumption that
silently produces a whole table of wrong numbers, so it is gated: a predicted metabolite of a
substrate should look like that substrate, and under an off-by-one or a shuffled order it would not.
The gate is the median Tanimoto between each substrate and its own predictions, against the same
statistic under a deliberately rotated assignment.

WHAT THAT GATE CAN AND CANNOT SEE, because an earlier version of this docstring claimed more than it
delivers. It is a median over substrates, so it is blind to a minority: the fraction that has to be
cross-assigned before it fires is (0.5 - q) / (1 - q), where q is the share of per-substrate medians
already below 0.5. Measured, that is 0.25 on the 291 and 0.31 on the 879 -- so roughly a quarter to
a third of either delivery could be mis-assigned with this statistic unmoved. It rules out a GLOBAL
shift or shuffle, which is what a positional join can plausibly get wrong, and nothing smaller.

A minority mis-assignment is caught instead by an instrument that resolves per substrate, in
revision/tests/test_the_metatox_column_over_the_whole_population.py: each substrate's similarity to
its own predictions against several strangers' sets, with the pass rate of the half this paper
already reports as the bar. It catches a tenth of a delivery cross-assigned, where this gate needs a
third.

The gate also mis-calibrates in the other direction on re-tautomerised substrates: it compares
against the substrate as the corpus stores it, while the service saw the submitted tautomer. For the
257 of 879 that differ, the measured median is 0.47 against the corpus form and 0.68 against what
was actually submitted, so a delivery made up of those alone would have been refused while being
perfectly correct.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
MAP = ROOT / "results" / "metatox_input" / "substrate_map.csv"


def _code_version() -> dict:
    """The producer's own digest, not just its name.

    This used to record the script name and the commit, which lets the provenance sweep recover the
    source from the history and verify by INFERENCE -- an assumption about how the work was done.
    stamp() records the digest of the source that actually ran, so the sweep can say whether the
    code has moved instead of guessing.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from _provenance import stamp
    return stamp(__file__)


def _spectrum(props) -> tuple[float, float] | None:
    """(Pa, Pi) for the Metabolite class, which is the score the delivery is named after.

    PASS writes the pair with a comma as the decimal separator: "0,370  0,167  Metabolite". Pa is
    the method's own confidence and therefore its ranking signal; ignoring it and taking the file's
    record order would score the method in an ordering it never produced, which is the one thing a
    frozen-prediction comparison must not do.
    """
    raw = props.get("PASS_ACTIVITY_SPECTRUM")
    if raw is None:
        return None
    for line in str(raw).splitlines():
        parts = line.replace(",", ".").split()
        if len(parts) >= 3 and parts[2].lower().startswith("metabolite"):
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


def parse_sdf(path: Path) -> dict:
    """{substrate key -> [(smiles, Pa, Pi)]}, ranked by the method's own confidence.

    The key is whatever precedes the underscore in the record's ID, kept as written. The first
    delivery numbered its records 1_1, 1_2, ... and carried no substrates, so its key is an index
    into the submission order and the join is positional and gated. The second numbered them
    SUB0001_1, using the ids from the submission set, so its key IS the join and no gate is needed
    to establish it -- though the same gate is run anyway, because an id that has been shifted by
    one is exactly as wrong as a position that has, and costs nothing to rule out.
    """
    out: dict[int, list[tuple]] = {}
    supplier = Chem.SDMolSupplier(str(path), sanitize=True)
    n_scored = n_total = n_declared = n_disagree = 0
    for mol in supplier:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        raw = props.get("ID")
        if raw is None or "_" not in str(raw):
            continue
        idx = str(raw).split("_")[0]
        # A bare number is the first delivery's index; anything else is an id from the submission
        # set. Normalising both to str here keeps one ranking path for the two, which is the point:
        # a second implementation of "order by Pa, de-duplicate by first appearance" is how two
        # readings of one column drift apart.
        if not idx:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if not smiles:
            continue
        n_total += 1
        pa_pi = _spectrum(props)
        if pa_pi is not None:
            n_scored += 1
        # PASS writes the spectrum only where it clears its own threshold, and says so in a second
        # tag. Cross-checking the two is the gate: if they disagree, the field is being read wrong.
        declared = "1 of 1" in str(props.get("PASS_RESULT_COUNT", ""))
        n_declared += declared
        if declared != (pa_pi is not None):
            n_disagree += 1
        pa, pi = pa_pi if pa_pi else (float("nan"), float("nan"))
        out.setdefault(idx, []).append((smiles, pa, pi))
    print(f"  metabolites carrying a Metabolite spectrum: {n_scored} of {n_total}", flush=True)
    print(f"  records PASS declares above its own threshold: {n_declared}, "
          f"disagreeing with the spectrum on {n_disagree}", flush=True)
    if n_disagree:
        raise SystemExit("the two tags disagree about which predictions PASS scored; the spectrum "
                         "field is not being read as PASS wrote it")
    # scored first, by the method's own confidence; the rest keep the order the file gives them,
    # since PASS expresses no preference among predictions it declines to score
    for idx in out:
        out[idx].sort(key=lambda t: (0, -t[1]) if t[1] == t[1] else (1, 0.0))
    return out


def median_similarity(pairs, gen) -> float:
    """Median over substrates of the median Tanimoto to that substrate's own predictions."""
    import statistics
    per = []
    for sub_smiles, preds in pairs:
        m = Chem.MolFromSmiles(sub_smiles)
        if m is None or not preds:
            continue
        fp = gen.GetFingerprint(m)
        sims = []
        for p in preds[:40]:
            q = Chem.MolFromSmiles(p[0] if isinstance(p, tuple) else p)
            if q is not None:
                sims.append(DataStructs.TanimotoSimilarity(fp, gen.GetFingerprint(q)))
        if sims:
            per.append(statistics.median(sims))
    return round(statistics.median(per), 4) if per else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--map", default=str(MAP),
                    help="the submission map this delivery answers; the first batch's is the "
                         "default and the second's is revision/metatox_submission_1170")
    ap.add_argument("--join", choices=("positional", "id"), default="positional",
                    help="positional: the delivery numbers its records 1..N in submission order, "
                         "which is an assumption and is gated. id: the delivery keys on the ids in "
                         "the map, which is a fact about the file and is gated anyway")
    ap.add_argument("--variant", default="SMIRKS rules, all 291 parents returned",
                    help="free text for a reader")
    ap.add_argument("--variant-key", default="smirks",
                    help="a controlled value for a machine. Two runs merged into one arm must "
                         "carry the SAME key: the free text differs legitimately between them "
                         "(one says 291 parents, the other 879) and cannot be compared, and a "
                         "check that merely looks for the word SMIRKS passes 'no SMIRKS' too")
    ap.add_argument("--out", default=str(ROOT / "results" / "metatox_smirks_preds.json"))
    args = ap.parse_args()

    by_index = parse_sdf(Path(args.sdf))
    rows = list(csv.DictReader(open(args.map)))
    print(f"predictions for {len(by_index)} substrate keys; "
          f"{sum(len(v) for v in by_index.values())} metabolites", flush=True)
    print(f"submission map: {len(rows)} substrates, {rows[0]['id']} .. {rows[-1]['id']}", flush=True)

    if args.join == "positional":
        if sorted(by_index, key=lambda k: int(k)) != [str(i) for i in range(1, len(rows) + 1)]:
            raise SystemExit("the substrate indices are not 1..N over the submission map; the "
                             "positional join is not available")
        pick = [by_index[str(i + 1)] for i in range(len(rows))]
        rot = [by_index[str(((i + 137) % len(rows)) + 1)] for i in range(len(rows))]
    else:
        # The ids the delivery uses must be exactly the ids the map defines. A delivery that answers
        # a substrate the map does not contain is not answering this submission, and one that is
        # missing ids has not covered it; either way the join is not this map's to make.
        unknown = sorted(set(by_index) - {r["id"] for r in rows})
        absent = sorted({r["id"] for r in rows} - set(by_index))
        if unknown or absent:
            raise SystemExit(
                f"the delivery's ids do not match the submission map: {len(unknown)} it contains "
                f"that the map does not (e.g. {unknown[:3]}), {len(absent)} the map defines that it "
                f"does not answer (e.g. {absent[:3]})")
        pick = [by_index[r["id"]] for r in rows]
        rot = [by_index[rows[(i + 137) % len(rows)]["id"]] for i in range(len(rows))]

    ordered = [(r["substrate_smiles"], pick[i]) for i, r in enumerate(rows)]
    rotated = [(r["substrate_smiles"], rot[i]) for i, r in enumerate(rows)]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    true_sim = median_similarity(ordered, gen)
    rot_sim = median_similarity(rotated, gen)
    print(f"\ngate: median self-similarity {true_sim} against {rot_sim} under a rotated assignment")
    if true_sim < 0.5 or true_sim < rot_sim + 0.25:
        raise SystemExit("the positional join does not separate from a rotated one; the substrate "
                         "order is not what this file assumes")

    # Keep the method's own order -- Pa descending -- and dedup by FIRST appearance. Sorting the
    # SMILES alphabetically, as an earlier revision did, scores a method in an ordering it never
    # produced, which makes every recall@k for it a statement about the alphabet.
    preds, scored = {}, {}
    for sub, items in ordered:
        seq, seen = [], set()
        for smiles, pa, pi in items:
            if smiles in seen:
                continue
            seen.add(smiles)
            seq.append(smiles)
        preds[sub] = seq
        scored[sub] = [[smiles, round(pa, 4), round(pi, 4)] for smiles, pa, pi in items
                       if smiles in seen and (seen.discard(smiles) or True)]
    above = {sub: [t[0] for t in v if t[1] > t[2]] for sub, v in scored.items()}
    sizes = [len(v) for v in preds.values()]
    n_above = [len(v) for v in above.values()]
    rep = {"config": {**_code_version(), "sdf": Path(args.sdf).name,
                      "variant": args.variant,
                      "variant_key": args.variant_key,
                      "join": f"{args.join} against {Path(args.map).relative_to(ROOT)}",
                      "gate": {"median_self_similarity": true_sim,
                               "median_under_rotated_assignment": rot_sim}},
           "n_substrates": len(preds),
           "n_predictions": int(sum(sizes)),
           "mean_output": round(sum(sizes) / len(sizes), 2),
           "median_output": sorted(sizes)[len(sizes) // 2],
           "ranking": "the method's own Pa for the Metabolite class, descending",
           "mean_output_above_threshold": round(sum(n_above) / len(n_above), 2),
           "predictions": preds,
           "predictions_with_scores": scored,
           "predictions_above_own_threshold": above}
    print(f"\nsubstrates {rep['n_substrates']}, predictions {rep['n_predictions']}, "
          f"mean output {rep['mean_output']}, median {rep['median_output']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
