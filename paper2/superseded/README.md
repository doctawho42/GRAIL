# Superseded

`grail_service.tex` is the first manuscript for this work. It is kept because its numbers were
correct against the artifacts as they stood, and because the register in
`paper2/three_instances.md` refers to defects found in it.

It is not the manuscript. Two things retired it.

Its comparison reports one comparator. `results/four_method_291.json`, the artifact that defines
the population it is measured on, carries three, and two of them lead both GRAIL arms at the tight
budgets that manuscript claims. Correcting it required rewriting the abstract, the results and the
discussion around a different result, and that rewrite is `paper2/manuscript_draft.md`.

Its numbers predate the parent-drop convention. A prediction equal to the substrate consumes a
slot without being a prediction, and `four_method_291.json` drops it before the budget. Applying
that convention to every arm alike moved four of GRAIL's own figures.

The macro machinery it introduced survives and is the better part of it: every figure reached the
page through `paper2/numbers.tex`, generated from `results/`, and a checker refused any numeric
literal that was not on a short allow-list. The markdown manuscript keeps the discipline through
`scripts/check_draft_numbers.py`, which is weaker, and that is a real loss recorded here rather
than glossed.

---

`grail_nar.tex` is the same content laid out for the NAR Web Server Issue, which is no longer the
target. Two things retired it, neither of them the science.

NAR is fully open access and the charge is mandatory: 3,625 USD on the journal's own author
guidelines, with no subscription route. The only free paths are an institutional Read and Publish
agreement or the publisher's low- and middle-income-country discount.

And the Web Server Issue's criteria are eligibility gates rather than preferences. The software
must be functional on the date of the *proposal*, which would mean deploying before writing; the
manuscript must fit four to five printed pages, against the nine this one took; and a same-named
successor to a published tool is treated as an update, carrying a two-year interval and a
significant-changes bar. Cutting nine pages to five would have spent the evaluation protocol, the
preregistration and the three hypotheses that failed, which is the half of the paper that is
hardest to get elsewhere.

The target is now JCIM, where the same content is an Article with no page limit, where the
incumbent this work compares against was itself published, and where the subscription route
carries no charge.
