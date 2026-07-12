from scripts.factorize_recall import tautomer_hits


def test_tautomer_hits_counts_distinct_true_matches():
    # 'CCO' (ethanol) and 'OCC' are the same molecule -> one tautomer key.
    trues = ["CCO", "c1ccccc1"]          # ethanol, benzene
    preds = ["OCC", "CCO", "CCCC"]       # two spellings of ethanol + butane
    # only ethanol is recovered -> 1 of 2 trues
    assert tautomer_hits(preds, trues) == 1
    assert tautomer_hits([], trues) == 0
