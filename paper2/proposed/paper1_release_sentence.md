# Proposed addition to the first paper, blocked on a page-budget decision

The recommendation in §Conclusion asks a leaderboard to publish "either a score per cell of that
grid or the predictions themselves". The attempt to run the four 2025–26 metabolite systems is
the concrete case for the last clause, and it is evidence rather than assertion because it comes
from a failed attempt.

**The body has zero slack.** Page 9 is full at 74 lines and the references begin on page 10.
Three lengths were tried — ten lines, six, and three — and a placement in the reproducibility
statement; every one pushes the references to page 11, which is a tenth body page. The sentence
therefore needs a compensating cut, and choosing what to cut is an editorial decision about
text the author wrote.

## The sentence, ready to paste

Into the reproducibility statement, after "Its weights and those of MetaPredictor are likewise
third-party and are cited rather than shipped.":

> The four metabolite systems published in 2025 and 2026 are all cited as available, and none of
> the four turned out to publish a path from a released artifact to a prediction on someone
> else's molecules: one releases training code and no weights, one releases weights and
> withholds the module that turns a predicted site into a structure, one is a web form, and one
> is a preprint whose code we could not find. That is why the fifth item of
> \S\ref{sec:conclusion} asks for the predictions themselves.

## The compensating cut this proposes

In the same statement: *"Applied to the eleven released retrosynthesis files, it recovers the
three test sets in seconds."* It is the least load-bearing sentence there — the checker's
existence and its inputs are already stated — and its length is close to what the addition
needs. Any equivalent cut elsewhere does as well; the point is that one is required.

Facts behind the sentence are in `results/comparator_acquisition.json`, each with the URL and
the date it was read, and `scripts/check_comparators.py` re-derives the local half.
