# Curated note concepts (N01–N50) as features

## W = 6 h (test n = 853)

| feature set | # features | val AUROC | test AUROC | test AUPRC | ΔAUROC vs A [95 % CI] |
|---|---|---|---|---|---|
| A structured (tier 4 + age/sex) | 68 | 0.784 | 0.823 | 0.662 | +0.000 [+0.000, +0.000] |
| B concepts only (negation-aware) | 53 | 0.708 | 0.683 | 0.464 | -0.141 [-0.182, -0.096] |
| B0 concepts only (raw counts) | 53 | 0.699 | 0.672 | 0.459 | -0.151 [-0.193, -0.106] |
| C structured + concepts (negation-aware) | 121 | 0.801 | 0.819 | 0.656 | -0.005 [-0.022, +0.013] |
| C0 structured + concepts (raw counts) | 121 | 0.798 | 0.830 | 0.673 | +0.007 [-0.007, +0.022] |

Strongest single concepts (prevalence, 28-day mortality with / without, AUROC of the count alone):

| id | concept | prev | mort with | mort without | AUROC |
|---|---|---|---|---|---|
| N43 | worsening trajectory | 0.15 | 0.40 | 0.26 | 0.545 |
| N35 | fluid overload / edema | 0.43 | 0.31 | 0.26 | 0.542 |
| N37 | vasopressor / inotrope | 0.12 | 0.41 | 0.26 | 0.539 |
| N28 | unresponsive / poor neuro signs | 0.11 | 0.39 | 0.26 | 0.532 |
| N08 | cardiac arrest / CPR | 0.05 | 0.51 | 0.27 | 0.530 |
| N12 | hypotension / low output | 0.14 | 0.35 | 0.27 | 0.528 |
| N46 | altered mental status | 0.20 | 0.33 | 0.27 | 0.526 |
| N09 | hemorrhage | 0.20 | 0.33 | 0.27 | 0.526 |
| N04 | CHF | 0.17 | 0.34 | 0.27 | 0.525 |
| N02 | cirrhosis / ascites | 0.12 | 0.36 | 0.27 | 0.524 |
| N01 | metastatic cancer / chemo | 0.04 | 0.50 | 0.27 | 0.523 |
| N16 | respiratory failure / hypoxemia | 0.18 | 0.33 | 0.27 | 0.523 |

Leakage probe L01 (goals-of-care terms, not used): prevalence 0.045, mortality with 0.57 vs without 0.27, AUROC alone 0.533.

## W = 24 h (test n = 1,374)

| feature set | # features | val AUROC | test AUROC | test AUPRC | ΔAUROC vs A [95 % CI] |
|---|---|---|---|---|---|
| A structured (tier 4 + age/sex) | 68 | 0.810 | 0.837 | 0.660 | +0.000 [+0.000, +0.000] |
| B concepts only (negation-aware) | 53 | 0.734 | 0.751 | 0.505 | -0.086 [-0.115, -0.057] |
| B0 concepts only (raw counts) | 53 | 0.731 | 0.755 | 0.507 | -0.083 [-0.112, -0.054] |
| C structured + concepts (negation-aware) | 121 | 0.823 | 0.844 | 0.674 | +0.006 [-0.003, +0.016] |
| C0 structured + concepts (raw counts) | 121 | 0.824 | 0.847 | 0.686 | +0.010 [+0.001, +0.019] |

Strongest single concepts (prevalence, 28-day mortality with / without, AUROC of the count alone):

| id | concept | prev | mort with | mort without | AUROC |
|---|---|---|---|---|---|
| N43 | worsening trajectory | 0.27 | 0.33 | 0.22 | 0.562 |
| N37 | vasopressor / inotrope | 0.31 | 0.31 | 0.22 | 0.556 |
| N28 | unresponsive / poor neuro signs | 0.18 | 0.36 | 0.22 | 0.552 |
| N46 | altered mental status | 0.30 | 0.31 | 0.22 | 0.551 |
| N35 | fluid overload / edema | 0.59 | 0.27 | 0.22 | 0.544 |
| N02 | cirrhosis / ascites | 0.16 | 0.34 | 0.23 | 0.543 |
| N21 | atrial fibrillation | 0.14 | 0.36 | 0.23 | 0.543 |
| N12 | hypotension / low output | 0.26 | 0.30 | 0.23 | 0.540 |
| N27 | infection site | 0.48 | 0.28 | 0.22 | 0.540 |
| N09 | hemorrhage | 0.30 | 0.29 | 0.23 | 0.540 |
| N08 | cardiac arrest / CPR | 0.06 | 0.47 | 0.23 | 0.535 |
| N04 | CHF | 0.23 | 0.30 | 0.23 | 0.534 |

Leakage probe L01 (goals-of-care terms, not used): prevalence 0.101, mortality with 0.55 vs without 0.21, AUROC alone 0.581.
