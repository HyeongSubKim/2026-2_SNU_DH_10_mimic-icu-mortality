# Label efficiency and era transfer, W = 24 h, embedder = qwen3_4b (test AUROC)

| representation | n=16 | n=64 | n=256 | n=1024 | n=all |
|---|---|---|---|---|---|
| structured (TabPFN) | 0.609 ± 0.094 | 0.710 ± 0.025 | 0.754 ± 0.010 | 0.809 ± 0.012 | 0.834 |
| raw emb (LR) | 0.534 ± 0.065 | 0.602 ± 0.057 | 0.661 ± 0.014 | 0.708 ± 0.010 | 0.736 |
| narrative emb (LR) | 0.526 ± 0.075 | 0.589 ± 0.040 | 0.619 ± 0.014 | 0.655 ± 0.005 | 0.704 |

| representation | carevue->metavision | metavision->metavision | metavision->carevue | carevue->carevue |
|---|---|---|---|---|
| structured (TabPFN) | 0.833 | 0.837 | 0.825 | 0.828 |
| raw emb (LR) | 0.742 | 0.755 | 0.732 | 0.750 |
| narrative emb (LR) | 0.723 | 0.709 | 0.681 | 0.680 |
