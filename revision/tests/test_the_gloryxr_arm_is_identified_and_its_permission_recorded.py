"""GLORYxR is an unpublished tool run on unpublished weights, so what identifies it has to be here.

Run: python -m pytest revision/tests/test_the_gloryxr_arm_is_identified_and_its_permission_recorded.py -q

GLORYxR carries two columns in this work, run locally in both of its site-of-metabolism modes. It is
the only arm whose SOFTWARE and whose WEIGHTS are both unpublished: the package is a pre-publication
version with no paper to cite, and the model dumps it needs are not distributed with it and were
supplied privately by its first author for this work. Its first author has since confirmed in
writing that the tool may be included provided the paper says it is a pre-publication version, links
its repository, and states that it succeeds GLORYx.

That combination is the worst case for provenance, and the record was thin in exactly the places it
matters. The prediction artifacts name the repository and the interpreter and pin seven Python
packages, and they do NOT record which commit of GLORYxR ran, nor any digest of the dumps. The
dumps exist only in a session scratchpad. So at the moment this test was written, the key input to
two published columns was identified by a filesystem path that will not survive the session, and the
tool by a URL with no revision.

The tests below hold what closes that:

  the tool is identified by repository AND revision AND version, read from the checkout rather than
  typed, so the columns are tied to a state of someone else's code;
  every model dump is recorded by digest and size, because they are unpublished and a digest is the
  only thing that will still identify them when the path is gone -- and because their author intends
  to deposit them, after which anyone can check the deposit is what ran;
  the two byte-identical Phase 1 dumps are recorded as identical, MEASURED here, with the author's
  confirmation that the duplication is intended, so a reader who notices it is not left to guess
  whether this work loaded the wrong file;
  the permission is recorded with its conditions, and each condition is checked against the
  manuscript rather than asserted met.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ART = ROOT / "results" / "gloryxr_identity.json"
SI = ROOT / "paper2" / "si.tex"


def _blob() -> dict:
    assert ART.exists(), (
        f"{ART.relative_to(ROOT)} does not exist. Two published columns come from an unpublished "
        f"tool on unpublished weights, and nothing in this repository identifies either.")
    return json.loads(ART.read_text())


def test_the_tool_is_identified_by_revision_and_not_only_by_url():
    """A URL names a project. A revision names what ran.

    The prediction artifacts recorded the repository and seven package versions and no commit of
    GLORYxR itself, so the columns could not be tied to a state of the code that produced them. For
    a pre-publication tool that is the whole of its identity: there is no release, no DOI and no
    paper.
    """
    d = _blob()
    t = d.get("tool") or {}
    for f in ("name", "repository", "commit", "version", "licence"):
        assert t.get(f), f"the tool record does not carry {f}"
    assert re.fullmatch(r"[0-9a-f]{40}", t["commit"]), f"not a git revision: {t['commit']!r}"
    assert t["repository"].startswith("https://"), t["repository"]
    assert "github.com" in t["repository"], (
        "the recorded repository is not the one the permission asks the paper to link")
    # The prediction artifacts name the same source, or two records describe two different tools.
    for name in ("gloryxr_local_preds_default.json", "gloryxr_local_preds_strict.json"):
        p = ROOT / "results" / name
        if not p.exists():
            continue
        src = (json.loads(p.read_text()).get("obtained_from") or {}).get("source", "")
        assert src.rstrip("/").rstrip(".git") == t["repository"].rstrip("/").rstrip(".git"), (
            f"{name} names {src!r} and this record names {t['repository']!r}")


def test_every_model_dump_is_recorded_by_digest():
    """The dumps are unpublished and live on no permanent path, so the digest is all there will be.

    Their author intends to deposit them. Recording the digests now is what will let anyone check
    that a future deposit is the same weights these columns were computed from; recording only a
    directory name would leave that uncheckable the moment the directory is gone.
    """
    d = _blob()
    dumps = d.get("model_dumps")
    assert isinstance(dumps, list) and dumps, "no model dump is recorded"
    assert len(dumps) >= 2, f"only {len(dumps)} dump(s) recorded; the provider loads one per class"
    for m in dumps:
        for f in ("name", "sha256", "bytes"):
            assert m.get(f) is not None, f"a dump is recorded without {f}: {m}"
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), f"not a sha256: {m['sha256']!r}"
        assert m["bytes"] > 0
    assert d.get("model_provider"), "the provider that loaded them is not named"


def test_the_identical_pair_is_measured_and_its_intent_recorded():
    """Two dumps are the same file. Both halves of that have to be here.

    The measurement, because a reader who hashes the deposit will find it and must not be left
    wondering whether this work duplicated a file by mistake. The author's confirmation, because
    only its author can say the duplication is intended -- and it is, both classes being Phase 1
    reactions that the tool scores the same.

    Checked from the digests rather than from the claim: if a future set of dumps has no identical
    pair, the flag has to go with it.
    """
    d = _blob()
    dumps = d["model_dumps"]
    by_digest: dict[str, list[str]] = {}
    for m in dumps:
        by_digest.setdefault(m["sha256"], []).append(m["name"])
    measured = {h: ns for h, ns in by_digest.items() if len(ns) > 1}

    rec = d.get("identical_dumps")
    assert isinstance(rec, list), "the record does not report whether any dumps are identical"
    assert len(rec) == len(measured), (
        f"{len(measured)} group(s) of identical dumps are measurable from the digests recorded "
        f"here and {len(rec)} are reported")
    for group in rec:
        assert group.get("sha256") in measured, (
            f"a reported identical group is not identical by the recorded digests: {group}")
        assert sorted(group.get("names") or []) == sorted(measured[group["sha256"]]), (
            f"the names in a reported group are not the ones sharing that digest: {group}")
        assert group.get("confirmed_intended_by_the_author"), (
            "a duplicated model file is reported without the one statement that makes it a design "
            "rather than a mistake; only its author can supply that")
        assert group.get("what_the_author_said"), "the confirmation is recorded with no content"
    if measured:
        assert d.get("bytes_duplicated"), "the cost of the duplication is not recorded"


def test_the_permission_is_recorded_with_its_conditions_and_each_is_checked():
    """Permission to use unpublished work is a fact about the record, not a matter of trust.

    Its author granted inclusion on three conditions: that the paper says this is a pre-publication
    version, links the repository, and states that GLORYxR succeeds GLORYx. Each is checked here
    against paper2/si.tex, because a condition recorded as met and not met is worse than no record:
    it would make the permission look honoured while the manuscript failed it.
    """
    d = _blob()
    p = d.get("permission") or {}
    assert p.get("granted_by"), "the permission does not say who granted it"
    assert p.get("scope"), "the permission does not say what was granted"
    conds = p.get("conditions")
    assert isinstance(conds, list) and len(conds) >= 3, f"conditions recorded: {conds}"

    si = SI.read_text()
    for c in conds:
        assert c.get("condition"), f"a condition has no text: {c}"
        assert "met" in c, f"a condition does not record whether it is met: {c}"
        ev = c.get("evidence_in_the_manuscript")
        if c["met"]:
            assert ev, f"a condition is recorded as met with no evidence: {c['condition']}"
            assert ev in si, (
                f"the evidence recorded for {c['condition']!r} is not in paper2/si.tex: {ev!r}")
        else:
            assert c.get("why_not"), (
                f"a condition is recorded as unmet and does not say why: {c['condition']}")
    assert all(c["met"] for c in conds), (
        "a condition of the permission is unmet while the arm carries two columns in the paper: "
        + "; ".join(c["condition"] for c in conds if not c["met"]))
