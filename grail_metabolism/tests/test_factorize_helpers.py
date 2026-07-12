from scripts.factorize_recall import tautomer_hits


def test_tautomer_hits_counts_distinct_true_matches():
    # 'CCO' (ethanol) and 'OCC' are the same molecule -> one tautomer key.
    trues = ["CCO", "c1ccccc1"]          # ethanol, benzene
    preds = ["OCC", "CCO", "CCCC"]       # two spellings of ethanol + butane
    # only ethanol is recovered -> 1 of 2 trues
    assert tautomer_hits(preds, trues) == 1
    assert tautomer_hits([], trues) == 0


def test_tautomer_hits_merges_keto_enol():
    # The "tautomer-aware" advantage (ceiling 0.735 vs plain 0.718) rests on merging genuine
    # tautomers that a PLAIN InChIKey misses. Acetylacetone keto and its enol are one molecule up
    # to a proton shift. This is also a CANARY: _tautomer_inchikey wraps standardization in
    # try/except and silently falls back to plain _inchikey on any failure (e.g. a broken dep), so
    # every tautomer number would degrade to the plain value with no error — this test catches that.
    from grail_metabolism.metrics import _inchikey, _tautomer_inchikey
    keto, enol = "CC(=O)CC(C)=O", "CC(=O)C=C(O)C"
    assert _inchikey(keto) != _inchikey(enol)                     # plain InChIKey does NOT merge them
    assert _tautomer_inchikey(keto) == _tautomer_inchikey(enol)   # tautomer-InChIKey DOES
    assert tautomer_hits([enol], [keto]) == 1                     # predicting the enol recovers the keto true
