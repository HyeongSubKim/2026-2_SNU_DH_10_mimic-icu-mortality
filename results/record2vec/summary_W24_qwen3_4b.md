# Record2Vec, W = 24 h, embedder = Qwen/Qwen3-Embedding-4B  (test n = 1374, 28-day death 0.258)

| condition | AUROC | AUPRC | ΔAUROC vs A (95% CI) |
|---|---|---|---|
| A_tabpfn_structured | 0.837 | 0.673 | +0.000 [+0.000, +0.000] |
| B0_lr | 0.760 | 0.541 | -0.078 [-0.102, -0.054] |
| B0_tabpfn_pca32 | 0.705 | 0.467 | -0.132 [-0.160, -0.105] |
| C_B0_tabpfn_struct+pca32 | 0.832 | 0.661 | -0.006 [-0.010, -0.002] |
| F_tabpfn+B0_lr | 0.835 | 0.668 | -0.002 [-0.009, +0.005] |
| B_narr_lr | 0.699 | 0.474 | -0.138 [-0.165, -0.111] |
| B_narr_tabpfn_pca32 | 0.683 | 0.454 | -0.154 [-0.182, -0.126] |
| C_B_narr_tabpfn_struct+pca32 | 0.831 | 0.663 | -0.006 [-0.010, -0.002] |
| F_tabpfn+B_narr_lr | 0.832 | 0.666 | -0.005 [-0.010, -0.001] |
| B_struct_lr | 0.736 | 0.508 | -0.101 [-0.127, -0.075] |
| B_struct_tabpfn_pca32 | 0.721 | 0.494 | -0.116 [-0.143, -0.088] |
| C_B_struct_tabpfn_struct+pca32 | 0.837 | 0.671 | -0.000 [-0.005, +0.004] |
| F_tabpfn+B_struct_lr | 0.836 | 0.669 | -0.001 [-0.007, +0.005] |
| D_notes_lr | 0.798 | 0.594 | -0.039 [-0.062, -0.017] |
| D_notes_tabpfn_pca32 | 0.775 | 0.551 | -0.062 [-0.086, -0.039] |
| C_D_notes_tabpfn_struct+pca32 | 0.841 | 0.680 | +0.004 [-0.004, +0.011] |
| F_tabpfn+D_notes_lr | 0.842 | 0.672 | +0.004 [-0.008, +0.016] |
| D_termprompt_lr | 0.816 | 0.625 | -0.021 [-0.044, +0.002] |
| D_termprompt_tabpfn_pca32 | 0.780 | 0.574 | -0.057 [-0.082, -0.034] |
| C_D_termprompt_tabpfn_struct+pca32 | 0.839 | 0.681 | +0.001 [-0.007, +0.009] |
| F_tabpfn+D_termprompt_lr | 0.849 | 0.703 | +0.012 [-0.001, +0.025] |
| D_termfilt_lr | 0.775 | 0.547 | -0.062 [-0.086, -0.039] |
| D_termfilt_tabpfn_pca32 | 0.740 | 0.483 | -0.097 [-0.122, -0.072] |
| C_D_termfilt_tabpfn_struct+pca32 | 0.830 | 0.664 | -0.007 [-0.013, -0.001] |
| F_tabpfn+D_termfilt_lr | 0.837 | 0.676 | -0.000 [-0.010, +0.009] |
