# Superseded

`grail_service.tex` is the first manuscript for this work. It is kept because its numbers were
correct against the artifacts as they stood, and because the register in
`paper2/three_instances.md` refers to defects found in it.

It is not the manuscript. Two things retired it.

Its comparison reports one comparator. `results/four_method_291.json`, the artifact that defines
the population it is measured on, carries three, and two of them lead both GRAIL arms at the tight
budgets that manuscript claims. Correcting it required rewriting the abstract, the results and the
discussion around a different result, and that rewrite is `paper2/webserver_draft.md`.

Its numbers predate the parent-drop convention. A prediction equal to the substrate consumes a
slot without being a prediction, and `four_method_291.json` drops it before the budget. Applying
that convention to every arm alike moved four of GRAIL's own figures.

The macro machinery it introduced survives and is the better part of it: every figure reached the
page through `paper2/numbers.tex`, generated from `results/`, and a checker refused any numeric
literal that was not on a short allow-list. The markdown manuscript keeps the discipline through
`scripts/check_draft_numbers.py`, which is weaker, and that is a real loss recorded here rather
than glossed.
