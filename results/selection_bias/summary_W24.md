# Selection-bias check, W = 24 h (adult ICU stays alive at hour 24: n = 52,251; 1078 died within the window removed)

| population | exposure | exposed: deaths / N (%) | unexposed: deaths / N (%) | risk difference pp [95 % CI] | adj. OR demo† | adj. OR demo + severity‡ | IPW RD pp (demo) | IPW RD pp (demo + severity) |
|---|---|---|---|---|---|---|---|---|
| all adults | lactate recorded | 4,791 / 30,293 (15.8) | 1,897 / 21,958 (8.6) | +7.2 [+6.6, +7.7] | 2.14 [2.02, 2.27] | 1.53 [1.44, 1.63] | +7.6 [+6.8, +8.0] | +4.4 [+3.3, +5.0] |
| all adults | BE recorded | 5,045 / 32,679 (15.4) | 1,643 / 19,572 (8.4) | +7.0 [+6.5, +7.6] | 2.70 [2.54, 2.87] | 1.86 [1.73, 2.00] | +9.0 [+8.4, +9.6] | +3.9 [+2.8, +5.0] |
| all adults | SOFA computable | 2,537 / 10,590 (24.0) | 4,151 / 41,661 (10.0) | +14.0 [+13.1, +14.9] | 2.80 [2.64, 2.96] | 1.87 [1.74, 2.00] | +12.6 [+12.0, +13.2] | +7.9 [+5.9, +9.2] |
| SOFA-computable | lactate recorded | 2,281 / 9,229 (24.7) | 256 / 1,361 (18.8) | +5.9 [+3.7, +8.2] | 1.45 [1.24, 1.68] | 1.20 [1.03, 1.40] | +4.0 [+2.0, +7.0] | +2.6 [+0.2, +5.2] |
| SOFA-computable | BE recorded | skipped (exposed 10590, unexposed 0) | | | | | | |

† age, sex, first care unit, admission type, charting era.  ‡ plus vasopressor use and mechanical ventilation in the window.

## Baseline characteristics (all adults alive at hour W)

| characteristic | lactate yes | lactate no | BE yes | BE no | SOFA computable | not computable |
|---|---|---|---|---|---|---|
| N | 30,293 | 21,958 | 32,679 | 19,572 | 10,590 | 41,661 |
| 28-day mortality % | 15.8 | 8.6 | 15.4 | 8.4 | 24.0 | 10.0 |
| age, mean | 63.4 | 63.7 | 63.8 | 63.1 | 62.8 | 63.7 |
| male % | 57.0 | 55.6 | 57.7 | 54.2 | 58.2 | 56.0 |
| emergency admission % | 83.5 | 82.8 | 78.2 | 91.4 | 89.1 | 81.7 |
| MetaVision era % | 50.9 | 35.1 | 40.7 | 50.2 | 44.1 | 44.3 |
| MICU % | 42.0 | 35.9 | 34.5 | 47.6 | 46.4 | 37.6 |
| SICU % | 15.7 | 17.8 | 15.8 | 17.9 | 18.1 | 16.2 |
| CCU % | 9.4 | 21.4 | 11.8 | 18.8 | 13.0 | 14.8 |
| CSRU % | 18.5 | 16.6 | 25.5 | 4.7 | 11.7 | 19.2 |
| TSICU % | 14.4 | 8.3 | 12.4 | 11.0 | 10.8 | 12.1 |
| vasopressor in window % | 34.0 | 15.5 | 37.6 | 7.3 | 41.7 | 22.3 |
| ventilated in window % | 57.1 | 25.0 | 66.3 | 5.7 | 85.5 | 33.0 |
| GCS min, median | 9.0 | 14.0 | 8.0 | 15.0 | 7.0 | 14.0 |
| MAP min, median | 58.0 | 60.0 | 58.0 | 60.7 | 57.0 | 59.0 |
| creatinine max, median | 1.1 | 1.0 | 1.0 | 1.0 | 1.2 | 1.0 |
| SOFA computable % | 30.5 | 6.2 | 32.4 | 0.0 | 100.0 | 0.0 |
| SOFA total (if computable), median | 9.0 | 7.0 | 9.0 | nan | 9.0 | nan |

## Is prediction "easier" in the tested group? (test-split AUROC of a covariate-only model; SOFA score AUROC where computable)

| population | exposure | covariate model: exposed | unexposed | SOFA score: exposed (n) | unexposed (n) |
|---|---|---|---|---|---|
| all adults | lactate recorded | 0.760 | 0.755 | 0.650 (9,229) | 0.622 (1,361) |
| all adults | BE recorded | 0.770 | 0.719 | 0.649 (10,590) | – (0) |
| all adults | SOFA computable | 0.702 | 0.760 | 0.649 (10,590) | – (0) |
| SOFA-computable | lactate recorded | 0.727 | 0.557 | 0.650 (9,229) | 0.622 (1,361) |
