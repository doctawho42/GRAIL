from grail_metabolism.model.reaction_types import canonical_type, build_type_vocab


def test_canonical_type_merges_periphery_variants():
    # two hydroxylations differing only in aromatic periphery share a radius-0 type
    a = "[cH:1][cH:2]>>[c:1][OH]"            # aromatic C-H -> C-OH (schematic)
    b = "[cH:1][c:2]([CH3])>>[c:1][OH]"
    ta, tb = canonical_type(a), canonical_type(b)
    assert ta is not None and ta == tb


def test_build_type_vocab_buckets_rare_rules():
    catalog = {
        "[CH2:1][OH:2]>>[CH:1]=[O:2]": {"count": 40},   # frequent -> its own type
        "[cH:1]>>[c:1][F]": {"count": 1},                # rare -> "other" (-1)
    }
    id2sig, rule2type = build_type_vocab(catalog, min_pairs=5)
    assert rule2type["[CH2:1][OH:2]>>[CH:1]=[O:2]"] >= 0
    assert rule2type["[cH:1]>>[c:1][F]"] == -1
