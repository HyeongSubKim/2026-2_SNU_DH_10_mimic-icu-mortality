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
| 1 SOFA inputs (TabPFN v2) | 58 | 0.824 [0.794, 0.855] | 0.662 [0.600, 0.722] | 19 s |
| 2 + lactate (TabPFN v2) | 63 | 0.830 [0.800, 0.858] | 0.674 [0.613, 0.734] | 21 s |
| 3 + base excess (TabPFN v2) | 63 | 0.833 [0.804, 0.861] | 0.673 [0.613, 0.732] | 28 s |
| 4 + lactate + base excess (TabPFN v2) | 68 | 0.832 [0.803, 0.861] | 0.681 [0.621, 0.739] | 24 s |

### W = 24 h (train+val 7,855, test 1,374)

| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |
|---|---|---|---|---|
| 1 SOFA inputs (TabPFN v2) | 58 | 0.831 [0.805, 0.853] | 0.649 [0.594, 0.699] | 23 s |
| 2 + lactate (TabPFN v2) | 63 | 0.833 [0.806, 0.855] | 0.663 [0.611, 0.709] | 27 s |
| 3 + base excess (TabPFN v2) | 63 | 0.835 [0.809, 0.858] | 0.661 [0.606, 0.710] | 27 s |
| 4 + lactate + base excess (TabPFN v2) | 68 | 0.837 [0.811, 0.859] | 0.673 [0.621, 0.717] | 25 s |

## Conventional SOFA score references (same test sets; logistic regressions fitted on the train split)

| reference | 6 h AUROC [95 % CI] | 6 h AUPRC | 24 h AUROC [95 % CI] | 24 h AUPRC |
|---|---|---|---|---|
| SOFA total score alone | 0.650 [0.611, 0.689] | 0.443 | 0.657 [0.623, 0.693] | 0.428 |
| SOFA total + age (logistic) | 0.708 [0.670, 0.744] | 0.511 | 0.701 [0.670, 0.732] | 0.479 |
| SOFA total + age + sex (logistic) | 0.705 [0.665, 0.741] | 0.507 | 0.700 [0.669, 0.731] | 0.479 |
| 6 SOFA component scores + age + sex (logistic) | 0.711 [0.673, 0.748] | 0.507 | 0.707 [0.676, 0.739] | 0.487 |

SOFA total (worst values in the window): mean 8.1 at 6 h (died 9.5 vs alive 7.5), 9.3 at 24 h (died 10.8 vs alive 8.8).

## SOFA score alone (no model: the raw worst-in-window SOFA total is the risk score)

| | 6 h | 24 h |
|---|---|---|
| cohort / 28-day mortality | 5,757 / 27.9 % | 9,229 / 24.7 % |
| AUROC, whole cohort [95 % CI] | 0.655 [0.639, 0.670] | 0.650 [0.637, 0.663] |
| AUROC, test split only | 0.650 [0.609, 0.689] | 0.657 [0.623, 0.691] |
| AUPRC, whole cohort | 0.430 | 0.395 |
| component scores alone (AUROC) | cardio 0.634, renal 0.598, resp 0.574, CNS 0.569, coag 0.549, liver 0.539 | cardio 0.613, renal 0.613, resp 0.565, coag 0.560, liver 0.558, CNS 0.550 |
| 28-day mortality by SOFA 0–5 / 6–8 / 9–11 / >= 12 | 15.5 / 23.7 / 31.5 / 48.6 % | 14.2 / 18.6 / 22.4 / 39.8 % |
| SOFA >= 8: flagged, sens, spec, PPV | 54 %, 0.69, 0.52, 0.36 | 67 %, 0.79, 0.37, 0.29 |
| SOFA >= 12: flagged, sens, spec, PPV | 17 %, 0.29, 0.88, 0.49 | 27 %, 0.43, 0.79, 0.40 |
