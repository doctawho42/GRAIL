"""The paper's own gates have to run, or they rot without saying so.

`audit_claim_words.py` had been exiting non-zero for some time before this file existed: the
survey moved from the appendices into the body, the sentences it keys on moved with it, and
nothing ran the script, so a generated table stayed in the manuscript while the check that
produces it was failing. A gate nobody runs is not a gate.

These are slow enough to be worth naming and fast enough to keep in the default suite. They
need the manuscript and the committed artifacts, not the dataset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

GATES = [
    ("audit_claim_words.py", []),      # every `certified' traces to a declared family
    ("check_prereg.py", ["--self-test"]),
    # the registry itself has to stay valid: a hypothesis that loses its failure condition or
    # its family size stops being registered, and nothing else would say so
    ("check_prereg.py", ["--prereg", "paper2/preregistration.md"]),
    ("polish_audit.py", []),           # no injected instruction, reader address or diary prose
    # every number the registration leans on still traces to the code that wrote it
    # Both of these WRITE a tracked artifact when run with no arguments, so the suite was
    # regenerating results/artifact_provenance.json and results/withheld_template_carriers.json
    # every time it ran -- a test producing the artifacts other tests verify against. It cost an
    # hour of two people cross-checking a census neither had deliberately regenerated. Both
    # verdicts are computed before the write and neither reads its file back, so the output goes
    # to a scratch name under results/, which .gitignore already covers.
    # Consequence worth stating rather than discovering: `make test` no longer refreshes those two
    # artifacts. That is the point -- regenerating them is a producer's job and a decision, not a
    # side effect of running the tests.
    ("audit_artifact_provenance.py",
     ["--out", str(ROOT / "results" / ".gate_scratch_provenance.json")]),
    # a swept row whose numbers move with the clock names which ones, and the SI says so
    ("check_wallclock_fields.py", []),
    # the word around a number makes a claim too: superlatives and separation verdicts
    ("check_quantifiers.py", []),
    # Both documents compile, resolve every reference and overrun no column. This gate was
    # written, was correct, and was not on this list, so a manuscript shipped a cross-reference
    # that printed ?? through a green suite: three referees found it and the checker that would
    # have caught it had never been run. A gate nothing invokes is not a gate.
    ("check_paper2_build.py", []),
    # LICENSE, NOTICE.md and the package configuration describe the bank that ships, not
    # the one the paper measures. Both files went on describing the wrong one for a week.
    ("check_licence_files.py", []),
    # An input digest has to be the digest a READER computes. git normalises line endings on
    # commit and reports nothing, so eight tracked data files carried bytes here that no clone
    # would ever have, and three artifacts verified in this tree and reported a moved input in a
    # fresh one. The guarantee the manuscript quotes has to hold on the reader's side.
    ("check_working_tree_matches_index.py", []),
    # The manuscript's voice packs a claim, its qualification and its evidence into one sentence,
    # which is why it read heavily; splitting them cost nothing and does not stay done, because
    # every new paragraph is written in the same voice and no single edit ever looks wrong.
    ("check_prose_density.py", []),
    # no document may claim the withheld templates are not redistributed while a tracked file
    # carries them
    ("check_no_withheld_templates.py",
     ["--out", str(ROOT / "results" / ".gate_scratch_withheld.json")]),
    # the ignore rules name what the repository deposits, instead of being overridden by a
    # flag 372 times, which is how a released artifact goes missing from a release
    ("sync_tracked_artifacts.py", ["--check"]),
    # The prose counts the register's verdicts. One verdict cell was weakened from a confirmation
    # to a measurement and the two sentences that count confirmations were not, so the manuscript
    # said seven where its own table said six, a page apart, and no gate could see it because a
    # spelled-out word is invisible to the numbers gate.
    ("check_register_tally.py", []),
    ("check_register_tally.py", ["--self-test"]),
    # How often the prose reaches for two constructions, against what eight comparable papers do.
    # A reviewer called the manuscript machine-written; the reply that each hedge is individually
    # defensible answers a different question, because the complaint is about a rate. Measured,
    # "rather than" runs at 5.5 per 1000 words against a corpus range of 0.00-0.57.
    ("check_prose_rate.py", []),
]


# What a gate reads that a fresh clone may not have. A missing input is a skip, not a failure.
NEEDS = {
    "check_paper2_build.py": ("paper2/si.log", "paper2/grail_jcim.log"),
}

# A gate may be red on purpose, and that has to be said here rather than left to whoever reads
# the suite. Two of these were written to fail: the density gate gained the Supporting
# Information, which does not meet the tail ceilings the manuscript meets, and the rate gate
# exists precisely because the rate is out. A red gate nobody has declared is indistinguishable
# from a broken one, and a suite that is permanently red stops being read at all -- which is the
# same defect as a gate nobody runs, one step further along.
#
# The list is not a licence. If a declared gate PASSES, this test fails and demands its removal,
# so a debt cannot be paid and left on the books, and cannot be added to silence a real
# regression: the moment the prose reaches the ceiling, the entry has to go.
DECLARED_DEBT = {
    "check_prose_density.py":
        "paper2/si.tex: 12 per cent of sentences over 45 words against a ceiling of 10, longest "
        "96 against 75, and four sections over the section mean ceiling of 32. The file was "
        "absent from LIMITS while the reviewer named it explicitly; its mean (27.4) is inside "
        "the range eight comparable papers occupy and is not the defect. Closes in W4.",
    "check_prose_rate.py":
        "TWO criteria are red and both are accepted, with the second accepted by the author after "
        "being shown these numbers.\n"
        "LENGTH. paper2/body.tex is 10,048 words against 11,131 at the start of the pass: 1,083 "
        "removed, 9.7 per cent, against a 25 per cent target of 8,349. The author accepted roughly "
        "-14 per cent and recorded the rest here; the pass reached -9.7, and the 475-word gap "
        "between the acceptance and the outcome is part of the debt rather than hidden in it. That "
        "gap was then tested for closure by three instruments and none closes it without removing "
        "content. A near-duplicate search over all 403 sentences, Jaccard on content words at or "
        "above 0.40, returns TWO pairs worth about 60 words together. Of 28 classic filler "
        "constructions -- 'in order to', 'due to the fact that', 'it is important to note that' "
        "and the rest -- the file contains ZERO. And the longest sentences are dense rather than "
        "padded: the 54-word one is the list of released artifacts, which cannot shorten without "
        "dropping an artifact. The only blocks large enough to cover 475 words are a worked "
        "example carrying two concessions and a sensitivity analysis defending the matching "
        "criterion, and moving a concession into the Supporting Information to satisfy a word "
        "count is the burying a reviewer punishes, so transfer was rejected as the instrument. Why "
        "the remaining 1,699 words were not taken, each figure measured: every sentence in the file "
        "reaching for one of the eight constructions weighs about 1,600 words in total, so the "
        "construction work cannot supply the volume even if all of it were deleted; 440 words of "
        "the counted prose are ACS-mandated blocks -- Data and Software Availability 326, "
        "\\section*{Notes} 84, Author contributions 30 -- which duplicate the SI by design; and "
        "about 2,100 words are recorded concessions, where deleting one is a retraction rather "
        "than a cut.\n"
        "Those two figures are the only ones here this gate does not print, so the derivation is "
        "written out rather than the number trusted, and writing it out corrected the same figure "
        "THREE times. It was first 634, taken from a conversation. Re-deriving gave 636, and that "
        "was wrong twice over: a boundary regex of \\(?:sub)*section\\{ does not match "
        "\\section*{Notes}, so Notes was reported absent and its 84 words were silently absorbed "
        "into Author contributions; and suppinfo was credited 196 words although cpd.blocks drops "
        "environment bodies entirely, so NONE of its 209 ACS-mandated words are inside the 10,048 "
        "this target is measured against -- counting them as untouchable was double-counting an "
        "exclusion. The concession figure is given as 'about 2,100' deliberately: 2,102 is the sum "
        "over the 'text' field of every results/disclosure_inventory.json entry whose 'in' list "
        "contains paper2/body.tex, but that field is _plain-normalised and so is a THIRD word "
        "convention, not this gate's, and the three disagree by up to five per cent. To re-derive "
        "the 440, take each block between headings matched WITH \\*? and count it through "
        "cpd.blocks + cpd.sentences, the same splitter as the length figures above. A number a "
        "reader cannot re-derive is a number that stays wrong, and this one stayed wrong twice "
        "after being written down. "
        "Two adversarial passes then bounded what remained: sentence-level, 101 candidates of "
        "which 60 were adjudicated and 24 killed on claims stated nowhere else (40 per cent); "
        "paragraph-level, 29 proposals of which 28 were adjudicated and 20 killed (71 per cent), "
        "and all 8 nominal survivors carry reasons arguing the opposite of their boolean, so that "
        "route yielded nothing. What it did establish is that paragraph-sized blocks each hold "
        "something unique -- the only definition of the parent-drop convention, the only statement "
        "of the third correction, the only place the title's own term is defined. Relocation to "
        "the SI supplied the rest of what was safe: the runtime figure and its deployment detail "
        "moved, registry-neutral. The worked example and the site-consistency section were "
        "examined and declined -- the first is the paper's only concrete illustration, and the "
        "second's caveats are concessions whose place is beside the claim, not in a supplement.\n"
        "CONSTRUCTIONS. The criterion is no longer the per-phrase counts and the reason is a "
        "defect in them: they "
        "overlap by construction, so two residues can be neither summed nor compared, and a "
        "reword clears one probe while the construction stays on the page. Measured: a pass moved "
        "25 occurrences out of 'rather than' into 'and not', which occurs ZERO times in the "
        "49,637-word comparison corpus, while the share of sentences defining by exclusion moved "
        "barely at all. That share is now the criterion, over eight probes since 'instead of' was "
        "added on the author's decision: paper2/body.tex 28 of 403 sentences (6.95%) against a "
        "ceiling of 8 at digest 4b4e4cc9, paper2/si.tex 259 of 1097 (23.61%) against 24 at digest "
        "83ac498e, where the corpus runs at 18 of 1,841 sentences, 0.98%. Digest is sha256 of the "
        "file, first eight hex characters, because two agents reading the same file minutes apart "
        "reported 79 and 82 and only a file identity settles which state was read. The body figure "
        "fell 49 -> 28 in a pass that rewrote all 23 'and not' occurrences positively or deleted "
        "the negative half, so that construction now stands at ZERO in the manuscript as well as "
        "across the corpus, and epistemic self-grading fell 17 -> 9 with it. That pass removed 20 "
        "words, which is the point: it is not a volume instrument and was not used as one. At the "
        "earlier si.tex state, digest 36bad88c, the figure was 257 of 1080 and was "
        "reproduced by a second reader as 251, and the six-carrier difference is now explained: "
        "that reader's 'worth V-ing' was the probe's earlier form, missing 'naming', 'printing', "
        "'recording' and 'setting'. Restoring exactly that omission moves this file 257 -> 251, a "
        "delta of six, which is the figure this file's own docstring already recorded for that "
        "same omission. So the discrepancy was in the instrument and never in the text, and a "
        "synchronised re-measure would have reproduced it. Two earlier explanations were wrong "
        "and are kept here because what was ruled out is worth more than what was guessed: the "
        "file state (si.tex has not moved all session) and a truncated epistemic probe, which "
        "would cost 19 carriers rather than six. The ceiling is the "
        "most generous single paper (0.80 per 1000 "
        "words, PMC13292216) carried across each file's length, derived at every run from "
        "results/prose_rate_corpus.json and never typed here. Counts here are named with the file "
        "they describe and nothing else: body.tex moves under this pass, and two agents reading "
        "the same file minutes apart got 79 and 82 until the probe count and the file digest were "
        "stated alongside. Closes by deletion and positive assertion, not by rewording: W3 for "
        "the manuscript, W4 for the Supporting Information.",
}


@pytest.mark.parametrize("script,args", GATES,
                         ids=[f"{g[0]}{'-' + g[1][0].lstrip('-') if g[1] else ''}"
                              for g in GATES])
def test_paper_gate_exits_zero(script, args):
    path = ROOT / "scripts" / script
    if not path.exists():
        pytest.skip(f"{script} is not in this checkout")
    for a in args:
        if a.endswith(".md") and not (ROOT / a).exists():
            pytest.skip(f"{a} is not in this checkout")
    # Some gates read a build the repository does not carry: the manuscripts' LaTeX logs are not
    # committed, so a clone that has never compiled them has nothing for the build gate to check.
    # Skipping there is right -- the gate has no input, it has not passed -- and failing there
    # would make continuous integration red for a reason that says nothing about the code.
    for needed in NEEDS.get(script, ()):
        if not (ROOT / needed).exists():
            pytest.skip(f"{script} needs {needed}, which this checkout has not built")
    run = subprocess.run([sys.executable, str(path), *args], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)

    # A crash and a refusal both exit non-zero, so this has to come BEFORE the debt branch.
    # Without it, a gate that cannot even be parsed exits non-zero, matches its declared debt, and
    # is recorded as the failure it was expected to have: the suite would stop distinguishing "the
    # prose is unfixed" from "the gate is unparseable", and the debt mechanism -- written to keep
    # red gates honest -- would be the thing hiding a broken one. That is not hypothetical; a
    # docstring edit left check_prose_rate.py with two closing quote triples and a SyntaxError,
    # and this branch is what separates that from an unpaid debt.
    crashed = "Traceback (most recent call last)" in run.stderr or "SyntaxError" in run.stderr
    assert not crashed, (
        f"{script} did not run at all -- it crashed rather than reaching a verdict, so this tells "
        f"you nothing about the paper.\n--- stderr ---\n{run.stderr[-2000:]}")

    debt = DECLARED_DEBT.get(script)
    if debt is not None:
        assert run.returncode != 0, (
            f"{script} passes, so it is no longer a declared debt: remove its entry from "
            f"DECLARED_DEBT in this file. Leaving a paid debt on the books is how a gate stops "
            f"meaning anything.\nThe entry said: {debt}")
        pytest.xfail(f"declared debt -- {debt}")

    assert run.returncode == 0, (
        f"{script} exited {run.returncode}\n"
        f"--- stdout ---\n{run.stdout[-4000:]}\n--- stderr ---\n{run.stderr[-2000:]}")


STRATA = [
    # (artifact, in-arm file, complement file, the row flag that means `in the arm')
    ("h1_stratum.json", "sparse_at_rule_dense_at_type.txt",
     "sparse_at_rule_dense_at_type_complement.txt", "in_stratum"),
    ("h6_stratum.json", "trivial_automorphism.txt", "nontrivial_automorphism.txt", "trivial"),
]

ORBIT_ARM = ("h6_stratum.json", "orbit_ge3.txt", "orbit_ge3_complement.txt")


def test_h6_treatment_arm_is_the_orbit_three_substrates():
    """H6 registers its treatment prediction on this file, so it must be exactly that arm."""
    import json

    art = ROOT / "results" / ORBIT_ARM[0]
    txt, comp = ROOT / "strata" / ORBIT_ARM[1], ROOT / "strata" / ORBIT_ARM[2]
    if not art.exists() or not txt.exists():
        pytest.skip("the H6 arms have not been built in this checkout")
    d = json.loads(art.read_text())
    wanted = sorted(r["substrate"] for r in d["rows"] if r["largest_orbit"] >= 3)
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    assert listed == wanted, f"the arm holds {len(listed)}, orbit>=3 is {len(wanted)}"
    assert len(listed) == d["n_orbit_ge3"]
    assert len(listed) + len(other) == d["n_substrates"], "the arm and its complement do not partition"


@pytest.mark.parametrize("artifact,arm,complement,flag", STRATA,
                         ids=[s[1].replace(".txt", "") for s in STRATA])
def test_stratum_file_matches_its_artifact(artifact, arm, complement, flag):
    """A membership file and the run that produced it cannot drift apart.

    Each hypothesis is registered on a file of substrate SMILES. If one were edited, or
    regenerated under a changed definition, the hypothesis would quietly become a different
    one. Each file has to be exactly the in-arm substrates of its committed artifact, and the
    two files have to partition the split.
    """
    import json

    art = ROOT / "results" / artifact
    txt, comp = ROOT / "strata" / arm, ROOT / "strata" / complement
    if not art.exists() or not txt.exists() or not comp.exists():
        pytest.skip(f"{artifact} has not been built in this checkout")
    d = json.loads(art.read_text())
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    from_rows = sorted({r["substrate"] for r in d["rows"] if r[flag]})
    assert listed == from_rows, (
        f"{arm} holds {len(listed)} substrates, the artifact {len(from_rows)}")
    assert len(listed) + len(other) == d["n_substrates"], f"{arm} and {complement} do not partition"
    assert not set(listed) & set(other), f"{arm} and {complement} overlap"


def test_split_manifest_still_matches_the_data():
    """The freeze is a claim about data the repository does not hold; verify it holds anyway.

    If the external dataset is absent this skips, which is the honest outcome: nothing was
    checked. If it is present and any fingerprint moved, the preregistration is registered
    against a split that no longer exists.
    """
    manifest = ROOT / "paper2" / "split_manifest.json"
    data = ROOT / "grail_metabolism" / "data" / "test_triples_clean.txt"
    if not manifest.exists() or not data.exists():
        pytest.skip("the split manifest or the external dataset is not in this checkout")
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "typed_edit" / "freeze_split.py"),
                          "--verify"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, f"the split moved since the freeze\n{run.stdout[-3000:]}"


def test_h1_primary_stratum_is_the_intersection():
    """The primary membership file must be exactly the pairs every join key agrees on.

    H1 names this file as its primary definition in advance, so it cannot be the file that
    happened to come out best. The two files also have to partition the substrates that carry
    at least one typeable reference -- not the whole split, because 14 substrates carry none.
    """
    import json

    art = ROOT / "results" / "h1_join_sensitivity.json"
    txt = ROOT / "strata" / "sparse_at_rule_dense_at_type_intersection.txt"
    comp = ROOT / "strata" / "sparse_at_rule_dense_at_type_intersection_complement.txt"
    if not art.exists() or not txt.exists():
        pytest.skip("the join sensitivity has not been computed in this checkout")
    d = json.loads(art.read_text())
    keys = d["primary"]["keys"]
    wanted = sorted({r["substrate"] for r in d["rows"]
                     if all(r["in_stratum"][k] for k in keys)})
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    assert listed == wanted, f"the primary file holds {len(listed)}, the intersection {len(wanted)}"
    assert len(listed) == d["primary"]["substrates"]
    typeable = {r["substrate"] for r in d["rows"]}
    assert len(listed) + len(other) == len(typeable), "the two files do not partition"
    assert not set(listed) & set(other)
    # the ceiling the registration quotes has to be the one the arithmetic gives
    assert d["feasibility"]["max_gain_for_K"]["2.5"] == int(d["n_typeable_pairs"] // 2.5)


# --refresh-text rewrites the `text` snapshot in the concession registry for keys whose identity
# has not changed. That is a writer pointed at the one artifact this project has already destroyed
# once, so the refusals matter more than the feature: refreshing on a tree whose disclosures have
# MOVED would leave a green --check and a freshened snapshot, removing both signals a retraction
# leaves behind, and rewriting reviewed_and_accepted would rewrite what a human agreed to and the
# text they agreed it against. Asserting the flag works proves nothing about either; the test
# therefore perturbs the artifact and demands a refusal. It restores the bytes in a finally.
def test_refresh_text_refuses_what_it_must():
    import json
    art = ROOT / "results" / "disclosure_inventory.json"
    if not art.exists():
        pytest.skip("the concession registry is not in this checkout")
    orig = art.read_bytes()

    def run():
        return subprocess.run([sys.executable, "scripts/disclosure_inventory.py", "--refresh-text"],
                              cwd=ROOT, capture_output=True, text=True)
    try:
        # a disclosure the documents no longer hold: a retraction, and never a text refresh
        d = json.loads(orig)
        d["disclosures"]["a key no document holds any more, recorded so this cannot pass"] = {
            "text": "A sentence that was retracted and must not be refreshed away.",
            "in": ["paper2/si.tex"]}
        art.write_text(json.dumps(d, indent=1))
        r = run()
        assert r.returncode == 1, f"a retracted disclosure was not refused:\n{r.stdout}"
        assert "no longer agree" in r.stderr and "gone:" in r.stderr, r.stderr

        # a disclosure the record does not hold: the key sets disagree the other way
        d = json.loads(orig)
        d["disclosures"].pop(next(iter(d["disclosures"])))
        art.write_text(json.dumps(d, indent=1))
        r = run()
        assert r.returncode == 1, f"an unrecorded disclosure was not refused:\n{r.stdout}"
        assert "no longer agree" in r.stderr, r.stderr

        # the decisions and the text they were taken against are never rewritten
        d = json.loads(orig)
        if d.get("reviewed_and_accepted"):
            keep = "DELIBERATELY STALE record of what was agreed, which must survive a refresh."
            d["reviewed_and_accepted"][0]["text"] = keep
            art.write_text(json.dumps(d, indent=1))
            r = run()
            assert r.returncode == 0, r.stderr
            back = json.loads(art.read_text())
            assert back["reviewed_and_accepted"][0]["text"] == keep, \
                "--refresh-text rewrote what a human accepted"
            assert len(back["reviewed_and_accepted"]) == len(d["reviewed_and_accepted"])
    finally:
        art.write_bytes(orig)
        assert art.read_bytes() == orig, "the registry was not restored"
