# Label efficiency and era transfer, W = 6 h, embedder = qwen3_4b (test AUROC)

| representation | n=16 | n=64 | n=256 | n=1024 | n=all |
|---|---|---|---|---|---|
| structured (TabPFN) | 0.631 ± 0.042 | 0.696 ± 0.031 | 0.761 ± 0.016 | 0.803 ± 0.009 | 0.829 |
| raw emb (LR) | 0.648 ± 0.038 | 0.693 ± 0.023 | 0.710 ± 0.011 | 0.743 ± 0.005 | 0.760 |
| narrative emb (LR) | 0.597 ± 0.053 | 0.649 ± 0.012 | 0.686 ± 0.020 | 0.718 ± 0.004 | 0.748 |
| structured emb (LR) | 0.548 ± 0.070 | 0.650 ± 0.040 | 0.691 ± 0.018 | 0.704 ± 0.012 | 0.724 |

| representation | carevue->metavision | metavision->metavision | metavision->carevue | carevue->carevue |
|---|---|---|---|---|
| structured (TabPFN) | 0.826 | 0.832 | 0.813 | 0.814 |
| raw emb (LR) | 0.764 | 0.764 | 0.737 | 0.759 |
| narrative emb (LR) | 0.745 | 0.726 | 0.738 | 0.739 |
| structured emb (LR) | 0.746 | 0.725 | 0.728 | 0.711 |
