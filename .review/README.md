# Review packet

Two documents are under review and nothing else. Review what a journal referee would receive; do
not review the repository, the code, or the commit history.

- `.review/manuscript.txt`  the manuscript, extracted from paper2/grail_jcim.pdf
- `.review/si.txt`          the Supporting Information, from paper2/si.pdf
- `paper2/grail_jcim.pdf`   the manuscript as it renders
- `paper2/si.pdf`           the SI as it renders

Target: Journal of Chemical Information and Modeling (ACS), submitted as an Article.

Bracketed bold text such as [ZENODO DOI, ...] marks a value the authors have not yet supplied.
Treat these as the submission's open items, not as typographical errors.

Sources are readable if you need to check where a number comes from: `paper2/body.tex`,
`paper2/si.tex`, `paper2/numbers.tex` (every figure in the text reaches the page through a macro
generated from a pinned artifact under `results/`).
