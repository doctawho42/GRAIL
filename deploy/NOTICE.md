# Deploy notice

This build uses `extended_smirks_released.txt` (6970 templates), which excludes the 611 templates
that appear verbatim in BioTransformer's published set. Removing them costs zero references
(`results/licence_removal_cost__clean_test.json`). The generator checkpoint shipped here is the
deployed generator subset to this bank; the filter is unchanged. No corpus and no full bank are
shipped or required.
