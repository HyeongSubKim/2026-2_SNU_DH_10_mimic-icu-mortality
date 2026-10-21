# Record2Vec, W = 24 h, embedder = UFNLP/gatortron-base  (test n = 1374, 28-day death 0.258)

| condition | AUROC | AUPRC | ΔAUROC vs A (95% CI) |
|---|---|---|---|
| A_tabpfn_structured | 0.837 | 0.673 | +0.000 [+0.000, +0.000] |
| B0_lr | 0.708 | 0.451 | -0.129 [-0.163, -0.094] |
| B0_tabpfn_pca32 | 0.671 | 0.405 | -0.166 [-0.201, -0.127] |
| C_B0_tabpfn_struct+pca32 | 0.838 | 0.664 | +0.000 [-0.006, +0.007] |
| F_tabpfn+B0_lr | 0.843 | 0.674 | +0.006 [+0.000, +0.011] |
| B_narr_lr | 0.704 | 0.461 | -0.133 [-0.160, -0.107] |
| B_narr_tabpfn_pca32 | 0.695 | 0.459 | -0.142 [-0.170, -0.115] |
| C_B_narr_tabpfn_struct+pca32 | 0.833 | 0.664 | -0.005 [-0.008, -0.001] |
| F_tabpfn+B_narr_lr | 0.829 | 0.658 | -0.008 [-0.013, -0.002] |
| B_struct_lr | 0.737 | 0.523 | -0.100 [-0.126, -0.073] |
| B_struct_tabpfn_pca32 | 0.703 | 0.457 | -0.134 [-0.163, -0.106] |
| C_B_struct_tabpfn_struct+pca32 | 0.835 | 0.665 | -0.003 [-0.006, +0.001] |
| F_tabpfn+B_struct_lr | 0.838 | 0.673 | +0.001 [-0.003, +0.005] |
| D_notes_lr | 0.807 | 0.624 | -0.030 [-0.053, -0.007] |
| D_notes_tabpfn_pca32 | 0.786 | 0.554 | -0.051 [-0.075, -0.027] |
| C_D_notes_tabpfn_struct+pca32 | 0.840 | 0.677 | +0.003 [-0.003, +0.009] |
| F_tabpfn+D_notes_lr | 0.847 | 0.696 | +0.010 [-0.002, +0.022] |
| D_termprompt_lr | 0.804 | 0.604 | -0.033 [-0.056, -0.009] |
| D_termprompt_tabpfn_pca32 | 0.771 | 0.548 | -0.067 [-0.092, -0.043] |
| C_D_termprompt_tabpfn_struct+pca32 | 0.839 | 0.681 | +0.002 [-0.004, +0.009] |
| F_tabpfn+D_termprompt_lr | 0.847 | 0.689 | +0.010 [-0.002, +0.022] |
| D_termfilt_lr | 0.769 | 0.552 | -0.068 [-0.092, -0.045] |
| D_termfilt_tabpfn_pca32 | 0.730 | 0.489 | -0.108 [-0.134, -0.082] |
| C_D_termfilt_tabpfn_struct+pca32 | 0.831 | 0.668 | -0.006 [-0.011, -0.000] |
| F_tabpfn+D_termfilt_lr | 0.834 | 0.679 | -0.003 [-0.013, +0.007] |
