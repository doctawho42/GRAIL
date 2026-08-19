"""A small but real bank of metabolic SMIRKS plus drug-like substrates.

It exists for the self-test: it checks that the invariants hold. It does not stand in for
a run on the real bank, and its numbers do not predict that run.
"""

RULES = [
    # --- two notations for one and the same transformation ---
    ("aromatic hydroxylation (cH)",     "[cH:1]>>[c:1]O"),
    ("aromatic hydroxylation (c;H1)",   "[c;H1:1]>>[c:1]O"),
    # --- one transformation in two environments: same at r=0, split at r=1 ---
    ("methyl hydroxylation next to C",  "[CH3:1][C:2]>>[C;H2:1](O)[C:2]"),
    ("methyl hydroxylation next to N",  "[CH3:1][N:2]>>[C;H2:1](O)[N:2]"),
    # --- the rest ---
    ("benzylic hydroxylation",          "[c:1][CH2:2]>>[c:1][C;H1:2]O"),
    ("N-demethylation",                 "[N;X3:1][CH3:2]>>[N;X3:1].[C:2]O"),
    ("O-demethylation",                 "[c:1][O:2][CH3:3]>>[c:1][O:2]"),
    ("S-oxidation",                     "[S;X2:1]>>[S:1]=O"),
    ("N-oxidation",                     "[N;X3:1]>>[N+:1][O-]"),
    ("epoxidation",                     "[C:1]=[C:2]>>[C:1]1O[C:2]1"),
    ("ester hydrolysis",                "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[C:4]O"),
    ("amide hydrolysis",                "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]"),
    ("phenol sulfation",                "[c:1][OH:2]>>[c:1][O:2]S(=O)(=O)O"),
    ("phenol glucuronidation",          "[c:1][OH:2]>>[c:1][O:2]C1OC(C(=O)O)C(O)C(O)C1O"),
    ("N-acetylation",                   "[N;H2:1]>>[N:1]C(=O)C"),
    ("nitro reduction",                 "[N+;X3:1](=O)[O-]>>[N;H2:1]"),
]

SUBSTRATES = [
    ("diclofenac",   "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("propranolol",  "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("caffeine",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ibuprofen",    "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("paracetamol",  "CC(=O)Nc1ccc(O)cc1"),
    ("verapamil",    "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC"),
    ("nifedipine",   "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC"),
    ("testosterone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"),
]
