"""A table's title is one sentence and the rest is a footnote, which is what ACS asks for.

JCIM's guidelines say each table must have a brief, one-phrase-or-sentence title, understandable
without reference to the text, and that details belong in footnotes rather than in the title. The
tables here carried captions of 84 to 150 words and four to six sentences, which explained the
table instead of naming it.

Rewriting seven generators by hand would put the same split in seven places, so the split lives
here: the caption's first sentence stays a caption, everything after it becomes a table note, and
the whole float is wrapped in a threeparttable so the notes sit under the table body where a
reader looks for them. The label moves up beside the caption, because a label that follows
\\end{threeparttable} binds to the wrong counter and prints ?? in every reference to it.

    from _acs_table import acs_table
    return acs_table(latex)
"""
from __future__ import annotations

import re

# A full stop that ends a sentence rather than an abbreviation or a decimal. The caption text is
# ordinary prose, so the cases that matter are "exh." and "int." in the arm names, "pct." and the
# numbers, all of which are followed by a lower-case letter or a digit rather than a capital.
_SENTENCE_END = re.compile(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z$\\])")


def _balanced(s: str, open_at: int) -> int:
    """Index just past the brace group that opens at `open_at`."""
    depth, i = 0, open_at
    while i < len(s):
        if s[i] == "{" and s[i - 1] != "\\":
            depth += 1
        elif s[i] == "}" and s[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced caption braces")


def acs_table(latex: str, extra_notes: tuple = ()) -> str:
    """Split one generated float's caption into an ACS title and its footnotes.

    `extra_notes` are appended as further \\item entries, for a generator that wants a note it
    never wrote into the caption. A float whose caption is already one sentence and that has no
    extra notes is returned unchanged, so this is safe to apply to every table.
    """
    m = re.search(r"\\caption\{", latex)
    if not m:
        return latex
    end = _balanced(latex, m.end() - 1)
    caption = latex[m.end():end - 1]

    flat = re.sub(r"\s+", " ", caption).strip()
    parts = _SENTENCE_END.split(flat, maxsplit=1)
    title, rest = parts[0].rstrip(". ") + ".", (parts[1].strip() if len(parts) > 1 else "")
    notes = ([rest] if rest else []) + [n for n in extra_notes if n]
    if not notes:
        return latex

    lm = re.search(r"\\label\{[^}]*\}\n?", latex)
    label = lm.group(0).strip() if lm else ""
    body = latex[:m.start()] + latex[end:]
    if lm:
        body = body.replace(lm.group(0), "", 1)

    # The float's opening line, then the wrapper, then the caption and label, then the tabular the
    # generator built, then the notes.
    om = re.search(r"\\begin\{table\*?\}(\[[^\]]*\])?\n", body)
    if not om:
        return latex
    head, tail = body[:om.end()], body[om.end():]
    cm = re.search(r"\\end\{table\*?\}", tail)
    if not cm:
        return latex
    inner, closing = tail[:cm.start()], tail[cm.start():]

    lead = ""
    im = re.match(r"((?:\\centering[^\n]*\n)|(?:\\centering\n))", inner)
    if im:
        lead, inner = im.group(1), inner[im.end():]

    items = "\n".join(f"\\item {n}" for n in notes)
    return (f"{head}{lead}\\begin{{threeparttable}}\n"
            f"\\caption{{{title}}}\n{label}\n"
            f"{inner.strip()}\n"
            f"\\begin{{tablenotes}}[flushleft]\\footnotesize\n{items}\n\\end{{tablenotes}}\n"
            f"\\end{{threeparttable}}\n{closing}")
