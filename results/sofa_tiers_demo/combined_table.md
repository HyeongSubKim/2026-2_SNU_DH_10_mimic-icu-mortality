# 28-day mortality, first W hours — SOFA alone vs ML feature tiers

Cells: test AUROC [bootstrap 95 % CI] / test AUPRC. Same cohort, patient split and test set for every row.

| model / feature set | 6 h (test n=853, 28-d mortality 29.0 %) | 24 h (test n=1,374, 25.8 %) |
|---|---|---|
| **SOFA score alone** (no model) | 0.650 [0.609–0.689] / 0.443 | 0.657 [0.623–0.691] / 0.428 |
| XGBoost: 1 SOFA inputs | 0.794 [0.761–0.826] / 0.611 | 0.814 [0.788–0.836] / 0.623 |
| XGBoost: 2 SOFA inputs + lactate | 0.801 [0.769–0.831] / 0.619 | 0.823 [0.798–0.847] / 0.634 |
| XGBoost: 3 SOFA inputs + base excess | 0.801 [0.769–0.831] / 0.627 | 0.821 [0.796–0.844] / 0.637 |
| XGBoost: 4 SOFA inputs + lactate + base excess | 0.809 [0.779–0.839] / 0.630 | 0.828 [0.803–0.849] / 0.648 |
| XGBoost, + age/sex: 1 SOFA inputs | 0.814 [0.784–0.844] / 0.648 | 0.826 [0.802–0.848] / 0.639 |
| XGBoost, + age/sex: 2 SOFA inputs + lactate | 0.812 [0.782–0.843] / 0.640 | 0.831 [0.805–0.853] / 0.653 |
| XGBoost, + age/sex: 3 SOFA inputs + base excess | 0.823 [0.793–0.851] / 0.656 | 0.831 [0.807–0.854] / 0.650 |
| XGBoost, + age/sex: 4 SOFA inputs + lactate + base excess | 0.823 [0.793–0.853] / 0.662 | 0.837 [0.813–0.858] / 0.660 |
| TabPFN v2: 1 SOFA inputs | 0.808 [0.776–0.838] / 0.630 | 0.820 [0.792–0.843] / 0.632 |
| TabPFN v2: 2 SOFA inputs + lactate | 0.819 [0.787–0.849] / 0.656 | 0.822 [0.795–0.844] / 0.643 |
| TabPFN v2: 3 SOFA inputs + base excess | 0.816 [0.786–0.845] / 0.649 | 0.822 [0.795–0.845] / 0.645 |
| TabPFN v2: 4 SOFA inputs + lactate + base excess | 0.818 [0.787–0.848] / 0.657 | 0.823 [0.796–0.844] / 0.648 |
| TabPFN v2, + age/sex: 1 SOFA inputs | 0.824 [0.794–0.855] / 0.662 | 0.831 [0.805–0.853] / 0.649 |
| TabPFN v2, + age/sex: 2 SOFA inputs + lactate | 0.830 [0.800–0.858] / 0.674 | 0.833 [0.806–0.855] / 0.663 |
| TabPFN v2, + age/sex: 3 SOFA inputs + base excess | 0.833 [0.804–0.861] / 0.673 | 0.835 [0.809–0.858] / 0.661 |
| TabPFN v2, + age/sex: 4 SOFA inputs + lactate + base excess | 0.832 [0.803–0.861] / 0.681 | 0.837 [0.811–0.859] / 0.673 |
