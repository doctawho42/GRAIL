# ICLR 2027 submission: state, gates, and what is left

What the venue receives is two things: `paper/grail_iclr.pdf` and the supplementary archive that
`scripts/build_anonymous_archive.py` writes. Everything below is about those two.

## Run these before submitting

Each exits non-zero when its claim fails. None of them needs the dataset.

```bash
python scripts/check_page_limit.py               # nine pages, strict
python scripts/verify_paper_numbers.py           # every figure against its artifact
python scripts/check_coverage.py --appendix      # every numeral read by some check
python scripts/verify_citations.py --check       # every cited key has a resolution record
python scripts/build_anonymous_archive.py        # scans the PDF and the archive, then writes it
python scripts/polish_audit.py                   # register, and the manuscript's own prose only
python scripts/check_bibliography.py             # each entry resolvable
```

Current state: main text ends on page 9 of 9; 2,973 of 2,973 number checks agree; 0 of 1,214
numerals unread; 50 of 50 cited keys resolved with no disagreement; identity scan clean.

**The page limit is at the limit.** Page 9 carries 49 lines of prose against a median of 51 across
pages 2–8, so any addition pushes the text over. Re-run the page check after every edit, and then
re-run `verify_paper_numbers.py`: the number checks bind to the sentence around each figure, so an
edit made for length can decouple a published number from the artifact that verifies it. That
happened seven times in one night here, each time from replacing ", and" with a semicolon or
reordering a clause.

## What the venue requires

- **9 pages of main text, strict**, for the initial submission (10 at rebuttal and camera-ready).
  References are unlimited. The AI use statement is required and exempt; the ethics and
  reproducibility statements are recommended and exempt.
- **Double-blind.** The style prints "Anonymous authors" and ignores `\author` unless
  `\iclrfinalcopy` is set. This means an `\author` line naming people is *invisible* in a
  submission build: the PDF scans clean and de-anonymises the moment that switch is set. The
  archive gate checks the switch through the running head it changes — a submission reads "Under
  review as a conference paper", a camera-ready reads "Published as".
- **AI use statement**: every task on the venue's list must be placed in used / not used / not
  applicable. A sentence saying *where* a task applies is not a verdict on *whether* AI performed
  it, and the statement was rewritten because four tasks had a location instead of a verdict and
  two were not mentioned at all.

## Open, and needing the authors

1. **The public repository is the live exposure, and it is not something a gate can fix.** The
   manuscript names GRAIL in the main text and calls it ours throughout the appendices, and a
   repository of that name is public under an account handle that resolves to a person in one
   search. `docs/STATUS.md` recorded this as a known constraint. Every tracked file that carried
   the handle has been cleaned except `paper2/`, which is a signed manuscript for a non-anonymous
   venue and correctly cites the repository. The decision is whether the repository stays public
   under that name during review.

2. **Row 10b of `paper/SELF_CLAIMS.md` is recorded FAIL** and row 10a PARTIAL. Both predate this
   round and are documented there with what they found.

3. **The archive carries what the reproducibility statement promises** — checkpoints, the frozen
   predictions of every method compared, the audits, the harness, the split construction and the
   analysis code, in 13.2 MB. If anything is added to the manuscript's promise, add it to `ALLOW`
   in the builder and re-run: a pattern matching nothing is a hard error, so an entry that stops
   existing fails loudly rather than shipping an archive that quietly lacks it.

4. **`results/test_references.json` is deliberately not released.** It is rebuilt locally with
   `python scripts/dump_test_references.py` and is gitignored. It is required by anything that runs
   a predictor against the references; populations and the coverage ceiling read the released
   descriptors instead and work without it.

## Things that looked like slack and were not

Recorded because each cost time and would cost it again.

- `\looseness=-1` on all seventeen long paragraphs that lacked it changed the typeset length by
  **zero lines**. These paragraphs are already set as tightly as the measure allows, so TeX cannot
  find a shorter solution within tolerance.
- The style prints a line number in the margin of every line, and `pdftotext` returns those as
  text. A page carrying nothing but exempt statements still yields a token that reads as prose,
  which is how the first count of the overflow came out one line too high.
- The captions are not padding. They tell a reader how to read the figure, and cutting them costs
  comprehension for about four lines.
- The nine pages were reached by *moving*, not deleting: the two proofs (which Appendix D already
  carried word for word) and the docking table's three forced conventions. The conventions could
  not simply go — a paper arguing that evaluation choices must be declared cannot drop its own
  declaration to save a page.
