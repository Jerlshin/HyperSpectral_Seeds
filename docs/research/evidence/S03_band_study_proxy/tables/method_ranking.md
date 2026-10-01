**Selection methods, ranked on calib macro-F1 at small budgets. `effective` = beats a random subset of the same size by the pre-registered margin.**

| method | family | supervised | mean F1 (k≤40) | best F1 | at k | margin vs random | effective | mean Jaccard (k≤40) | select s |
|---|---|---|---|---|---|---|---|---|---|
| uniform | null_model | False | 0.2106 | 0.6787 | 192 | +0.1331 | yes | 1.000 | 0.00 |
| cluster_ward | redundancy | True | 0.2078 | 0.6663 | 256 | +0.0580 | yes | 0.766 | 0.50 |
| mrmr | redundancy | True | 0.2076 | 0.6663 | 256 | +0.1057 | yes | 0.637 | 0.01 |
| l1_path | embedded | True | 0.1924 | 0.6673 | 192 | +0.1027 | yes | 0.525 | 181.18 |
| random | null_model | False | 0.1827 | 0.6663 | 256 | +nan | NO | 0.032 | 0.00 |
| pca_loading | geometric | False | 0.1821 | 0.6663 | 256 | +0.0550 | yes | 0.796 | 0.01 |
| spa | geometric | False | 0.1689 | 0.6663 | 256 | +0.0350 | yes | 0.755 | 0.11 |
| tree_importance | embedded | True | 0.1333 | 0.6663 | 256 | +0.0207 | yes | 0.708 | 0.44 |
| pls_vip | chemometric | True | 0.1300 | 0.6663 | 256 | +0.0306 | yes | 0.930 | 0.07 |
| mi | univariate | True | 0.1112 | 0.6663 | 256 | +0.0527 | yes | 0.796 | 4.83 |
| fdr | univariate | True | 0.1014 | 0.6663 | 256 | +0.0173 | yes | 0.906 | 0.00 |
| variance | univariate | False | 0.0675 | 0.6663 | 256 | +0.0040 | NO | 0.994 | 0.00 |
