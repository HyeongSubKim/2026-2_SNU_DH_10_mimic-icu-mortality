# 28-day mortality from the first W hours: SOFA inputs ± lactate ± base excess (XGBoost, + age/sex in every tier)

## W = 6 h  (cohort 5,757; train/val/test 4,072/832/853; 28-day mortality in test 0.290)

| feature set | # features | val AUROC | test AUROC [95 % CI] | test AUPRC [95 % CI] |
|---|---|---|---|---|
| SOFA total score alone | 1 | – | 0.650 | 0.443 |
| age + SOFA total (logistic) | 2 | – | 0.708 | 0.511 |
| 1 SOFA inputs (XGBoost) | 58 | 0.772 | 0.814 [0.784, 0.844] | 0.648 [0.590, 0.707] |
| 2 + lactate (XGBoost) | 63 | 0.779 | 0.812 [0.782, 0.843] | 0.640 [0.583, 0.699] |
| 3 + base excess (XGBoost) | 63 | 0.776 | 0.823 [0.793, 0.851] | 0.656 [0.594, 0.716] |
| 4 + lactate + base excess (XGBoost) | 68 | 0.784 | 0.823 [0.793, 0.853] | 0.662 [0.604, 0.718] |

top features by gain (tier 4): sofa_total 16, lactate_last 13, sofa_cardio 11, age 10, fio2_last 8, base_excess_mean 8, lactate_min 7, urine_total 7, gcs_last 7, urine_n 6

## W = 24 h  (cohort 9,229; train/val/test 6,505/1,350/1,374; 28-day mortality in test 0.258)

| feature set | # features | val AUROC | test AUROC [95 % CI] | test AUPRC [95 % CI] |
|---|---|---|---|---|
| SOFA total score alone | 1 | – | 0.657 | 0.428 |
| age + SOFA total (logistic) | 2 | – | 0.701 | 0.479 |
| 1 SOFA inputs (XGBoost) | 58 | 0.803 | 0.826 [0.802, 0.848] | 0.639 [0.588, 0.686] |
| 2 + lactate (XGBoost) | 63 | 0.812 | 0.831 [0.805, 0.853] | 0.653 [0.602, 0.702] |
| 3 + base excess (XGBoost) | 63 | 0.806 | 0.831 [0.807, 0.854] | 0.650 [0.599, 0.696] |
| 4 + lactate + base excess (XGBoost) | 68 | 0.810 | 0.837 [0.813, 0.858] | 0.660 [0.610, 0.706] |

top features by gain (tier 4): gcs_last 27, lactate_last 16, urine_ml_per_h 14, urine_total 13, sofa_total 12, lactate_min 12, age 11, base_excess_last 11, gcs_count 10, vaso_epinephrine 9

## Transformer baseline: TabPFN v2 (same cohorts, splits and feature sets; train = train + val, no tuning)

### W = 6 h (train+val 4,904, test 853)

| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |
|---|---|---|---|---|
| 1 SOFA inputs (TabPFN v2) | 58 | 0.824 [0.794, 0.855] | 0.662 [0.600, 0.722] | 22 s |
| 2 + lactate (TabPFN v2) | 63 | 0.830 [0.800, 0.858] | 0.674 [0.613, 0.734] | 15 s |
| 3 + base excess (TabPFN v2) | 63 | 0.833 [0.804, 0.861] | 0.673 [0.613, 0.732] | 18 s |
| 4 + lactate + base excess (TabPFN v2) | 68 | 0.832 [0.803, 0.861] | 0.681 [0.621, 0.739] | 19 s |

### W = 24 h (train+val 7,855, test 1,374)

| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |
|---|---|---|---|---|
| 1 SOFA inputs (TabPFN v2) | 58 | 0.831 [0.805, 0.853] | 0.649 [0.594, 0.699] | 19 s |
| 2 + lactate (TabPFN v2) | 63 | 0.833 [0.806, 0.855] | 0.663 [0.611, 0.709] | 21 s |
| 3 + base excess (TabPFN v2) | 63 | 0.835 [0.809, 0.858] | 0.661 [0.606, 0.710] | 21 s |
| 4 + lactate + base excess (TabPFN v2) | 68 | 0.837 [0.811, 0.859] | 0.673 [0.621, 0.717] | 24 s |

