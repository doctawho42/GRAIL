"""File-to-file metabolite prediction for way2drug: read a SMILES file, write a TSV of ranked
metabolites with scores, using the deployed model, the release ranking, and the released bank."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
from pathlib import Path

HEADER = ["parent_id", "rank", "metabolite_smiles", "score", "status"]


def read_input(path: str):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            ident, smiles = line.split("\t", 1)
        else:
            ident, smiles = str(n), line
        rows.append((ident.strip(), smiles.strip()))
    return rows


def predict_rows(model, rows, top_k, timeout_seconds):
    out = []
    # SIGALRM is Unix-only; way2drug runs on Linux, but guard so this never crashes elsewhere
    # (e.g. Windows) -- it just runs without a timeout there.
    use_alarm = bool(timeout_seconds) and timeout_seconds > 0 and hasattr(signal, "SIGALRM")

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"model.rank() exceeded {timeout_seconds}s")

    for ident, smiles in rows:
        if use_alarm:
            previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(timeout_seconds)
        try:
            ranked = model.rank(smiles, top_k=top_k)
        except TimeoutError:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "timeout"})
            continue
        except Exception:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "no_parse"})
            continue
        finally:
            if use_alarm:
                signal.alarm(0)                                    # cancel a pending alarm
                signal.signal(signal.SIGALRM, previous_handler)    # restore any prior handler
        if not ranked:
            out.append({"parent_id": ident, "rank": 0, "metabolite_smiles": "", "score": "", "status": "no_metabolites"})
            continue
        for rank, (met, score) in enumerate(ranked, 1):
            out.append({"parent_id": ident, "rank": rank, "metabolite_smiles": met,
                        "score": round(float(score), 6), "status": "ok"})
    return out


def write_tsv(path, out_rows):
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=str(Path(path).parent), suffix=".tmp")
    try:
        tmp.write("\t".join(HEADER) + "\n")
        for r in out_rows:
            tmp.write("\t".join(str(r[c]) for c in HEADER) + "\n")
        tmp.close()
        os.replace(tmp.name, path)   # atomic
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="grail-metabolites",
                                 description="Predict metabolites for a file of SMILES.")
    ap.add_argument("input", help="input file: one SMILES per line, or id<TAB>SMILES")
    ap.add_argument("output", help="output TSV path")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--timeout-seconds", type=int, default=120)
    ap.add_argument("--format", choices=["tsv"], default="tsv")
    args = ap.parse_args(argv)

    from grail_metabolism.deploy.model_adapter import load_released_model  # heavy import, deferred
    model = load_released_model(timeout_seconds=args.timeout_seconds)
    rows = read_input(args.input)
    out = predict_rows(model, rows, top_k=args.top_k, timeout_seconds=args.timeout_seconds)
    write_tsv(args.output, out)
    n_ok = len({r["parent_id"] for r in out if r["status"] == "ok"})
    print(f"wrote {args.output}: {len(rows)} substrates, {n_ok} with metabolites", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
