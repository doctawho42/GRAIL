# GRAIL metabolite prediction — deploy CLI

Predict probable metabolites with scores for a file of substrate SMILES.

## Usage
    grail-metabolites INPUT OUTPUT [--top-k 15] [--timeout-seconds 120]

- INPUT: one substrate per line, `SMILES` or `id<TAB>SMILES`. Blank lines ignored; a line with no id
  is given its 1-based line number.
- OUTPUT: TSV with header `parent_id  rank  metabolite_smiles  score  status`. `score` is the deployed
  combined confidence (filter * generator). `status` is one of `ok`, `no_parse`, `timeout`,
  `no_metabolites`.

## Example
    printf 'p1\tCCO\np2\tc1ccccc1O\n' > in.smi
    grail-metabolites in.smi out.tsv --top-k 15

## Environment
Install `requirements-deploy.txt` into a Python 3.10 environment, or use `environment.yml`. The model
files (released bank + released checkpoints) ship with the package.

## Performance
Per-substrate latency (measured, top-k 15): CCO ~1.7s; phenol (c1ccccc1O) ~11.6s; acetaminophen ~14.4s; larger substrates up to ~120s, bounded by --timeout-seconds (default 120).

## Bank
See NOTICE.md: the released bank excludes the BioTransformer templates at zero coverage cost.
