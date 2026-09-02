#!/usr/bin/env python3
"""Every key the manuscript cites resolves to a real work whose title and author it states.

SELF_CLAIMS row 9 claims this and was recorded PASS. The record behind it covers twenty keys,
dated when the paper surveyed one domain; the bibliography roughly tripled when the survey grew to
twenty-four leaderboards in four domains, and nothing re-ran. `check_bibliography.py` exits zero
throughout, because it asks whether an entry carries the fields a reader needs, not whether the
work those fields name exists. Two checks, neither of which is the claim.

This one resolves each cited key against a registry and compares what the entry says to what the
registry returns:

  * an entry with a DOI is fetched from Crossref by that DOI. Title, first author and year are
    compared, and a disagreement is reported rather than repaired.
  * an entry with an arXiv identifier is fetched from the arXiv API the same way.
  * an entry with neither is searched for by title. A search result is only ever recorded as
    `matched-by-title`, never as verified, and only when the normalised titles agree closely --
    because a search that returns something is not evidence that the something is right. The
    existing record's rule was "no DOI guessed" and this keeps it.

What it does NOT establish is whether the work supports what the manuscript attributes to it. That
is a reading, not a lookup, and the four entries carrying it were read by hand. Every record says
which of the two it got, so the stronger claim is never inferred from the weaker check.

    python scripts/verify_citations.py                 # resolve and write the record
    python scripts/verify_citations.py --check         # non-zero if a cited key has no record
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "citations_verified.json"
# A contact address is required by Crossref's polite pool. The submission is double-blind, so the
# anonymous one is used; it is a routing hint to their API, not a claim about who ran this.
UA = "grail-citation-check/1.0 (mailto:anonymous@example.com)"
# The venue's own template ships a bibliography of its example citations. They are not this
# paper's references and are excluded by file rather than by key, so a real key cannot hide there.
TEMPLATE_BIB = "iclr2027_conference.bib"


# LaTeX writes an accent as a command around the letter it modifies. Stripping the braces and the
# backslash alone leaves the accent character behind as a word boundary, so J{\"a}rvelin became
# "j arvelin" and did not match the registry's "Järvelin" -- a checker fault reported as a citation
# error, which is the way a row like this stops being read.
ACCENT = re.compile(r"\\[`'^\"~=.]\s*\{?([A-Za-z])\}?|\{\\[`'^\"~=.]\s*([A-Za-z])\}")


def _norm(s: str) -> str:
    """Comparable form of a title: no accents, no case, no punctuation, no LaTeX bracing."""
    s = ACCENT.sub(lambda m: m.group(1) or m.group(2), s or "")
    s = re.sub(r"[{}\\$]", "", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _words(s: str) -> set:
    return set(_norm(s).split())


def cited_keys() -> set:
    keys = set()
    for f in [ROOT / "paper/grail_iclr.tex"] + sorted(ROOT.glob("paper/app/*.tex")):
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", f.read_text()):
            keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    return keys


def _split_fields(body: str) -> dict:
    """The fields of one entry, split at brace depth zero.

    Not by line: this bibliography writes `volume = {15}, number = {9}, year = {2024}` on a single
    line, and a line-anchored pattern reads only the last of them. That is how a first pass saw
    zero DOI fields in a file carrying sixteen, and resolved by title search what it could have
    resolved by identifier.
    """
    fields, depth, buf, parts = {}, 0, "", []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    parts.append(buf)
    for part in parts:
        m = re.match(r"\s*(\w+)\s*=\s*(.*)", part, re.S)
        if not m:
            continue
        value = m.group(2).strip().strip(",").strip()
        if value[:1] in "{\"":
            value = value[1:-1] if value[-1:] in "}\"" else value[1:]
        fields[m.group(1).lower()] = " ".join(value.split())
    return fields


def bib_entries() -> dict:
    """Every entry of this paper's own bibliography, as {key: {field: value}}."""
    out = {}
    for path in sorted(ROOT.glob("paper/*.bib")):
        if path.name == TEMPLATE_BIB:
            continue
        text = path.read_text()
        # Brace matching, not a "\n}" anchor: an entry whose last field closes on the same line
        # as the entry brace is otherwise skipped entirely.
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
            i, depth = m.end() - 1, 1
            while depth and i + 1 < len(text):
                i += 1
                depth += (text[i] == "{") - (text[i] == "}")
            fields = _split_fields(text[m.end():i])
            fields["_type"] = m.group(1).lower()
            out[m.group(2).strip()] = fields
    return out


def _get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def from_crossref_doi(doi: str) -> dict | None:
    try:
        d = json.loads(_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    except Exception:
        return None
    authors = d.get("author") or []
    year = (d.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (d.get("title") or [""])[0],
            "first_author": (authors[0].get("family") if authors else "") or "",
            "year": year,
            "venue": (d.get("container-title") or [""])[0],
            "doi": d.get("DOI", "")}


def from_arxiv(eprint: str) -> dict | None:
    eprint = eprint.replace("arXiv:", "").strip()
    try:
        xml = _get(f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(eprint)}").decode()
    except Exception:
        return None
    t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    a = re.search(r"<entry>.*?<author>\s*<name>(.*?)</name>", xml, re.S)
    y = re.search(r"<entry>.*?<published>(\d{4})", xml, re.S)
    if not t:
        return None
    return {"title": " ".join(t.group(1).split()),
            "first_author": (a.group(1).split()[-1] if a else ""),
            "year": int(y.group(1)) if y else None,
            "venue": "arXiv", "doi": ""}


# Crossref indexes the peer reviews of an article as their own works, titled Review for "<title>".
# They score just under the acceptance threshold against the article's title, so a slightly looser
# threshold would have resolved a citation to a review of the work instead of the work. The type is
# what separates them, and it is checked rather than the threshold being tuned.
NOT_A_WORK = {"peer-review", "component", "grant", "journal-issue", "book-chapter"}


def by_title(title: str) -> dict | None:
    try:
        q = urllib.parse.urlencode({"query.bibliographic": _norm(title)[:200], "rows": 8})
        items = json.loads(_get(f"https://api.crossref.org/works?{q}"))["message"]["items"]
    except Exception:
        return None
    want = _words(title)
    for d in items:
        if d.get("type") in NOT_A_WORK:
            continue
        got = _words((d.get("title") or [""])[0])
        if not got or not want:
            continue
        # A search result is accepted only when the titles agree almost completely. Anything
        # looser resolves a key to a plausible neighbour, which is the defect being checked for.
        overlap = len(want & got) / max(len(want | got), 1)
        if overlap >= 0.85:
            authors = d.get("author") or []
            return {"title": (d.get("title") or [""])[0],
                    "first_author": (authors[0].get("family") if authors else "") or "",
                    "year": (d.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "venue": (d.get("container-title") or [""])[0],
                    "doi": d.get("DOI", ""), "_overlap": round(overlap, 3),
                    "_type": d.get("type", "")}
    return None


def _fold(name: str) -> str:
    """A surname reduced to the form its spellings share.

    A German umlaut is rendered either as the bare vowel or as the vowel plus e, and both are
    correct: the bibliography writes Buttensch{\\"o}n and Crossref returns Buttenschoen. Comparing
    the two as strings reports a citation error where there is a transliteration, and a row that
    reports errors it has not found is one nobody finishes reading.
    """
    n = _norm(name)
    for a, b in (("oe", "o"), ("ae", "a"), ("ue", "u"), ("ss", "s")):
        n = n.replace(a, b)
    return n.replace(" ", "")


def _same_name(entry_name: str, registry_name: str) -> bool:
    a, b = _fold(entry_name), _fold(registry_name)
    return bool(a) and bool(b) and (a in b or b in a)


def from_dblp(title: str) -> dict | None:
    """Machine-learning proceedings, which Crossref does not index.

    Five cited works are NeurIPS, ICLR, MLSys and ACL Anthology papers with no DOI. Crossref knows
    nothing of them, so a Crossref-only check reports them unresolved and a reader cannot tell that
    from a citation that does not exist. DBLP indexes exactly these venues; it rate-limits, so a
    failure is retried rather than recorded as an absence.
    """
    # DBLP throttles hard under a loop. The delays are long because a throttled request that is
    # recorded as "not found" is indistinguishable from a citation that does not exist, and the
    # first run of this resolver reported five such absences that were all rate limiting.
    time.sleep(3.0)
    for attempt in range(4):
        try:
            q = urllib.parse.urlencode({"q": _norm(title)[:120], "format": "json", "h": 5})
            hits = json.loads(_get(f"https://dblp.org/search/publ/api?{q}"))["result"]["hits"]
            break
        except Exception:
            time.sleep(6.0 * (attempt + 1))
    else:
        return None
    want = _words(title)
    for h in hits.get("hit", []):
        i = h.get("info", {})
        got = _words(i.get("title", ""))
        if not got:
            continue
        overlap = len(want & got) / max(len(want | got), 1)
        if overlap >= 0.85:
            au = i.get("authors", {}).get("author", [])
            au = au if isinstance(au, list) else [au]
            first = (au[0].get("text", "") if au and isinstance(au[0], dict) else "")
            return {"title": i.get("title", ""), "first_author": first.split()[-1] if first else "",
                    "year": int(i["year"]) if i.get("year", "").isdigit() else None,
                    "venue": i.get("venue", ""), "doi": i.get("doi", ""),
                    "_overlap": round(overlap, 3)}
    return None


def compare(entry: dict, found: dict) -> tuple:
    """What agrees between the bibliography and the registry, what does not, and what merely differs."""
    bad = []
    if _words(entry.get("title", "")) and _words(found["title"]):
        w, g = _words(entry["title"]), _words(found["title"])
        if len(w & g) / max(len(w | g), 1) < 0.7:
            bad.append(f"title: entry says {entry.get('title','')[:50]!r}, "
                       f"registry says {found['title'][:50]!r}")
    surname = re.split(r"\s+and\s+", entry.get("author", ""))[0].split(",")[0].split()[-1:] or [""]
    if found["first_author"] and surname[0] and not _same_name(surname[0], found["first_author"]):
        bad.append(f"first author: entry says {surname[0]!r}, registry says {found['first_author']!r}")
    ey = re.search(r"\d{4}", entry.get("year", "") or "")
    note = None
    if ey and found["year"] and abs(int(ey.group(0)) - int(found["year"])) > 1:
        msg = f"year: entry says {ey.group(0)}, registry says {found['year']}"
        # A year gap on an entry resolved by identifier is a disagreement. On one resolved by
        # title it usually is not: Sen's "Collective Choice and Social Welfare" is cited from
        # 1970 and Crossref returns the 1979 edition of the same book. Recorded, not called wrong.
        if found.get("_overlap") is None:
            bad.append(msg)
        else:
            note = msg + " (resolved by title; likely a different edition or printing)"
    return (not bad), bad, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="do not fetch; exit non-zero if a cited key has no record")
    args = ap.parse_args()

    keys, entries = cited_keys(), bib_entries()

    if args.check:
        if not OUT.exists():
            print(f"no {OUT.relative_to(ROOT)}; run this script without --check", file=sys.stderr)
            return 1
        rec = json.loads(OUT.read_text())
        have = {c["key"] for c in rec.get("citations", [])}
        missing = sorted(keys - have)
        stale = [c for c in rec.get("citations", []) if c.get("status") == "unresolved"]
        print(f"{len(keys)} cited keys, {len(have)} with a record, {len(missing)} without")
        for k in missing:
            print(f"   no record: {k}")
        for c in stale:
            print(f"   unresolved: {c['key']}  {c.get('note','')[:80]}")
        return 1 if (missing or stale) else 0

    # The twenty records already here were read by hand: each says whether the work supports what
    # the manuscript attributes to it, which no lookup establishes. They are carried forward rather
    # than replaced, and a first pass of this script overwrote them, which is why that is said here.
    prior = {}
    if OUT.exists():
        old_payload = json.loads(OUT.read_text())
        prior = {c["key"]: c for c in old_payload.get("citations", [])}
        prior_source = old_payload.get("source") or old_payload.get("config", {}).get("source", "")
    else:
        prior_source = ""

    out, t0 = [], time.time()
    for key in sorted(keys):
        e = entries.get(key)
        if e is None:
            out.append({"key": key, "status": "unresolved", "how": "none",
                        "note": "cited but absent from this paper's bibliography"})
            continue
        found, how = None, ""
        if e.get("doi"):
            found, how = from_crossref_doi(e["doi"]), "crossref-doi"
        if found is None:
            # The identifier is as often in a url as in an eprint field, and ten entries resolved
            # to nothing on a title search that carry one.
            blob = " ".join(str(e.get(k, "")) for k in ("eprint", "url", "journal", "note", "howpublished"))
            am = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+v?\d*)|arXiv[:\s]\s*([\d.]+v?\d*)", blob, re.I)
            ident = e.get("eprint") or (am.group(1) or am.group(2) if am else "")
            if ident:
                found, how = from_arxiv(ident), "arxiv-id"
        if found is None and e.get("title"):
            found, how = by_title(e["title"]), "crossref-title-search"
        if found is None and e.get("title"):
            found, how = from_dblp(e["title"]), "dblp-title-search"
        time.sleep(0.12)  # the polite pool asks for a modest rate

        if found is None:
            out.append({"key": key, "status": "unresolved", "how": how or "none",
                        "title": e.get("title", "")[:160],
                        "note": "no registry record found; not guessed"})
            continue
        ok, bad, note = compare(e, found)
        status = ("verified" if (ok and how not in ("crossref-title-search", "dblp-title-search"))
                  else "matched-by-title" if ok else "disagrees")
        rec = {"key": key, "status": status, "how": how,
               "doi": found.get("doi", "") or e.get("doi", ""),
               "registry_title": found["title"][:180],
               "registry_first_author": found["first_author"],
               "registry_year": found["year"], "venue": found.get("venue", "")[:90]}
        if bad:
            rec["disagreements"] = bad
        if note:
            rec["note"] = note
        was = prior.get(key)
        if was:
            # What the hand pass concluded, kept verbatim beside what the lookup found, so a
            # disagreement between the two is visible rather than resolved silently.
            rec["hand_checked"] = {k: v for k, v in was.items() if k != "key"}
            if was.get("status") in ("verified", "corrected") and rec["status"] == "matched-by-title":
                rec["status"] = "verified-by-hand"
        out.append(rec)
        print(f"  {status:16s} {how:22s} {key}")

    counts = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    payload = {
        "config": {
            "script": "verify_citations.py",
            "note": ("Each cited key resolved against Crossref by DOI, against arXiv by identifier, "
                     "or searched for by title. A title search is recorded as matched-by-title and "
                     "never as verified: a search returning something is not evidence it is the "
                     "right something. Whether a work supports what the manuscript attributes to "
                     "it is a reading and is not established here; the entries carrying that were "
                     "read by hand and are listed in the previous record's source field."),
            "seconds": round(time.time() - t0, 1),
        },
        "counts": counts,
        "n_cited": len(keys),
        "prior_source": prior_source,
        "citations": out,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\n{len(out)} keys: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
