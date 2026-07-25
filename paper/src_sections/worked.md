**Worked example.** The quotient dependence is not hypothetical. Consider a predicted set
`{D-alanine, acetone (enol tautomer)}` scored against annotated references `{L-alanine, acetone
(keto tautomer)}`. Under strict `inchikey`, both predictions miss: the alanine pair differs at a
stereocenter, and the acetone pair differs only by a keto/enol tautomerization outside the subset
plain InChI normalizes, so it also misses under `inchi_no_stereo`. `inchikey_tautomer`
standardization additionally strips stereochemistry (canonical SMILES generated with
`isomericSmiles=False`), so tautomer-canonicalizing both sides collapses both gaps at once: the
acetone keto/enol pair becomes a hit *and* the D-/L-alanine stereocenter pair becomes a hit,
scoring full recall on this example. The same fixed predictions can therefore score as a
near-total miss or a full hit purely by changing the match quotient, with no change to the
underlying chemistry — the phenomenon §11 quantifies across methods and the shared external set.
