# 28-day mortality from the first W hours: SOFA inputs ± lactate ± base excess (XGBoost)

## W = 6 h  (cohort 5,757; train/val/test 4,072/832/853; 28-day mortality in test 0.290)

| feature set | # features | val AUROC | test AUROC [95 % CI] | test AUPRC [95 % CI] |
|---|---|---|---|---|
| SOFA total score alone | 1 | – | 0.650 | 0.443 |
| age + SOFA total (logistic) | 2 | – | 0.708 | 0.511 |
| 1 SOFA inputs (XGBoost) | 56 | 0.742 | 0.794 [0.761, 0.826] | 0.611 [0.551, 0.676] |
| 2 + lactate (XGBoost) | 61 | 0.745 | 0.801 [0.769, 0.831] | 0.619 [0.553, 0.687] |
| 3 + base excess (XGBoost) | 61 | 0.755 | 0.801 [0.769, 0.831] | 0.627 [0.568, 0.689] |
| 4 + lactate + base excess (XGBoost) | 66 | 0.748 | 0.809 [0.779, 0.839] | 0.630 [0.567, 0.693] |

top features by gain (tier 4): vaso_norepinephrine 26, sofa_cardio 17, lactate_min 16, urine_total 12, sofa_total 12, vaso_drugs 11, lactate_last 10, gcs_last 10, base_excess_last 10, base_excess_mean 10

## W = 24 h  (cohort 9,229; train/val/test 6,505/1,350/1,374; 28-day mortality in test 0.258)

| feature set | # features | val AUROC | test AUROC [95 % CI] | test AUPRC [95 % CI] |
|---|---|---|---|---|
| SOFA total score alone | 1 | – | 0.657 | 0.428 |
| age + SOFA total (logistic) | 2 | – | 0.701 | 0.479 |
| 1 SOFA inputs (XGBoost) | 56 | 0.785 | 0.814 [0.788, 0.836] | 0.623 [0.571, 0.670] |
| 2 + lactate (XGBoost) | 61 | 0.793 | 0.823 [0.798, 0.847] | 0.634 [0.582, 0.683] |
| 3 + base excess (XGBoost) | 61 | 0.785 | 0.821 [0.796, 0.844] | 0.637 [0.586, 0.684] |
| 4 + lactate + base excess (XGBoost) | 66 | 0.791 | 0.828 [0.803, 0.849] | 0.648 [0.598, 0.694] |

top features by gain (tier 4): gcs_last 33, lactate_last 22, urine_total 18, sofa_total 16, lactate_min 15, sofa_coag 13, base_excess_last 13, urine_n 10, bilirubin_last 10, gcs_count 10

## Transformer baseline: TabPFN v2 (same cohorts, splits and feature sets; train = train + val, no tuning)

### W = 6 h (train+val 4,904, test 853)

| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |
|---|---|---|---|---|
| 1 SOFA inputs (TabPFN v2) | 56 | 0.808 [0.776, 0.838] | 0.630 [0.564, 0.693] | 20 s |
| 2 + lactate (TabPFN v2) | 61 | 0.819 [0.787, 0.849] | 0.656 [0.594, 0.720] | 18 s |
| 3 + base excess (TabPFN v2) | 61 | 0.816 [0.786, 0.845] | 0.649 [0.585, 0.712] | 16 s |
| 4 + lactate + base excess (TabPFN v2) | 66 | 0.818 [0.787, 0.848] | 0.657 [0.593, 0.718] | 17 s |

### W = 24 h (train+val 7,855, test 1,374)

| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |
|---|---|---|---|---|
| 1 SOFA inputs (TabPFN v2) | 56 | 0.820 [0.792, 0.843] | 0.632 [0.575, 0.679] | 17 s |
| 2 + lactate (TabPFN v2) | 61 | 0.822 [0.795, 0.844] | 0.643 [0.588, 0.694] | 22 s |
| 3 + base excess (TabPFN v2) | 61 | 0.822 [0.795, 0.845] | 0.645 [0.592, 0.695] | 24 s |
| 4 + lactate + base excess (TabPFN v2) | 66 | 0.823 [0.796, 0.844] | 0.648 [0.595, 0.697] | 24 s |

