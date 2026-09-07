# `scripts/`

There are 363 Python files here. That number is a fact about how the work was done rather than a
design, and this file exists so a reader does not have to open them one at a time to find out which
ones matter. Every script carries a module docstring whose first line says what it answers; the
groups below are what those docstrings sort into.

## The three groups

**Gates (12).** Checks that hold the manuscripts to their artifacts, run by
`grail_metabolism/tests/test_paper_gates.py` and therefore by `make test`. Each one refuses rather
than warns, and each was written because something got past its absence. The important ones:

| script | what it refuses |
|---|---|
| `check_paper2_numbers.py` | a number in the manuscript that no artifact produced, or a numeral typed by hand |
| `audit_artifact_provenance.py` | an artifact whose producer has changed since it was written |
| `check_quantifiers.py` | a superlative that is not the extremum of its own series, or a separation verdict the interval does not support |
| `check_paper2_build.py` | an unresolved cross-reference, an overfull column, a Type 3 font, or a stale build |
| `check_licence_files.py` | a licence file that describes a different distribution from the one in the tree |
| `check_prereg.py` | a claimed effect with no registered hypothesis behind it |
| `sync_tracked_artifacts.py` | ignore rules that no longer match what the repository deposits |
| `check_register_tally.py` | a count of the register's confirmations that the register's own verdict column does not carry |

**Producers (102).** Each writes exactly one artifact under `results/`, stamps it with the digest of
the script that wrote it, and is named in `audit_artifact_provenance.py`. Most live in
`typed_edit/`. A number reaches a manuscript only through one of these, so the chain from a printed
figure back to the code that computed it is mechanical: the macro names its key, the key names its
artifact, the artifact names its producer and that producer's source digest.

The comparison itself is assembled by `typed_edit/deployment_table.py` (every arm at every budget),
`typed_edit/matched_length.py` (the same with list length held), `typed_edit/criterion_sweep.py`
(the same under each matching criterion) and `typed_edit/error_by_chemistry.py` (the same split by
transformation class). Comparator columns come from `typed_edit/sygma_by_dialect.py`,
`typed_edit/biotransformer_arm.py`, `typed_edit/metapredictor_beam_sweep.py` and
`typed_edit/gloryx_via_service.py`.

**Entry points named in the documentation (8).** `mine_rules.py` builds the template bank,
`measure_coverage.py` the reach ceiling, `run_benchmark.py` the field-standard comparison,
`run_multiseed.py` the mean and spread over seeds, `fix_splits.py` repairs the splits,
`verify_rebuilt_corpus.py` checks a corpus a reader rebuilt against the one evaluated here, and
`zenodo_deposit.py` packages what is too large to commit.

## The rest

The remaining files are one-off research probes: a question was asked once, a script answered it,
and the answer either reached a manuscript through a producer or did not survive. They are kept
because several are the only record of a measurement that was made and reported as negative, and
deleting them would leave those statements unsupported. They are not part of any tested surface and
nothing depends on them.

`legacy/` holds three utilities from before this repository had a package, kept for the same reason
and documented as unsupported in `legacy/README.md`.

## Conventions

A script that writes an artifact imports `_provenance.stamp` and records the digest of its own
source in what it writes; `record_inputs` does the same for the files it read. That is what lets a
checker say an artifact is stale rather than guessing from timestamps.

A gate is a script that exits non-zero. Each is expected to fail when the manuscript is perturbed in
the way it exists to catch, and several carry a `--self-test` for exactly that.
