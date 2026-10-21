# Selection-bias check, W = 6 h (adult ICU stays alive at hour 6: n = 53,055; 274 died within the window removed)

| population | exposure | exposed: deaths / N (%) | unexposed: deaths / N (%) | risk difference pp [95 % CI] | adj. OR demo† | adj. OR demo + severity‡ | IPW RD pp (demo) | IPW RD pp (demo + severity) |
|---|---|---|---|---|---|---|---|---|
| all adults | lactate recorded | 4,884 / 27,988 (17.5) | 2,608 / 25,067 (10.4) | +7.0 [+6.5, +7.6] | 1.94 [1.84, 2.05] | 1.47 [1.39, 1.56] | +7.6 [+7.0, +8.1] | +4.3 [+3.6, +4.8] |
| all adults | BE recorded | 5,067 / 30,176 (16.8) | 2,425 / 22,879 (10.6) | +6.2 [+5.6, +6.8] | 2.36 [2.23, 2.49] | 1.70 [1.59, 1.81] | +8.8 [+8.1, +9.4] | +3.0 [+2.0, +3.6] |
| all adults | SOFA computable | 1,826 / 6,768 (27.0) | 5,666 / 46,287 (12.2) | +14.7 [+13.6, +15.8] | 2.50 [2.34, 2.66] | 1.58 [1.46, 1.70] | +13.8 [+12.9, +14.6] | +8.5 [+6.5, +11.8] |
| SOFA-computable | lactate recorded | 1,605 / 5,757 (27.9) | 221 / 1,011 (21.9) | +6.0 [+3.2, +8.8] | 1.36 [1.15, 1.61] | 1.18 [1.00, 1.40] | +4.8 [+1.6, +7.5] | +3.4 [-0.3, +6.5] |
| SOFA-computable | BE recorded | skipped (exposed 6768, unexposed 0) | | | | | | |

† age, sex, first care unit, admission type, charting era.  ‡ plus vasopressor use and mechanical ventilation in the window.

## Baseline characteristics (all adults alive at hour W)

| characteristic | lactate yes | lactate no | BE yes | BE no | SOFA computable | not computable |
|---|---|---|---|---|---|---|
| N | 27,988 | 25,067 | 30,176 | 22,879 | 6,768 | 46,287 |
| 28-day mortality % | 17.5 | 10.4 | 16.8 | 10.6 | 27.0 | 12.2 |
| age, mean | 63.4 | 63.9 | 63.9 | 63.3 | 62.6 | 63.8 |
| male % | 57.0 | 55.5 | 58.0 | 54.1 | 57.9 | 56.1 |
| emergency admission % | 83.8 | 82.8 | 77.2 | 91.5 | 90.9 | 82.3 |
| MetaVision era % | 52.5 | 35.0 | 40.3 | 49.5 | 43.6 | 44.3 |
| MICU % | 42.8 | 35.9 | 33.7 | 47.2 | 49.6 | 38.0 |
| SICU % | 15.8 | 17.7 | 15.5 | 18.1 | 19.2 | 16.3 |
| CCU % | 9.0 | 20.6 | 11.5 | 18.3 | 12.6 | 14.7 |
| CSRU % | 17.4 | 17.6 | 26.9 | 5.2 | 7.9 | 18.9 |
| TSICU % | 15.1 | 8.3 | 12.3 | 11.2 | 10.7 | 12.0 |
| vasopressor in window % | 26.9 | 12.4 | 30.1 | 6.8 | 33.3 | 18.1 |
| ventilated in window % | 51.7 | 24.5 | 62.8 | 7.2 | 85.6 | 32.0 |
| GCS min, median | 11.0 | 15.0 | 9.0 | 15.0 | 7.0 | 15.0 |
| MAP min, median | 64.0 | 67.0 | 64.0 | 67.3 | 63.0 | 66.0 |
| creatinine max, median | 1.1 | 1.0 | 1.0 | 1.0 | 1.2 | 1.0 |
| SOFA computable % | 20.6 | 4.0 | 22.4 | 0.0 | 100.0 | 0.0 |
| SOFA total (if computable), median | 8.0 | 6.0 | 8.0 | nan | 8.0 | nan |

## Is prediction "easier" in the tested group? (test-split AUROC of a covariate-only model; SOFA score AUROC where computable)

| population | exposure | covariate model: exposed | unexposed | SOFA score: exposed (n) | unexposed (n) |
|---|---|---|---|---|---|
| all adults | lactate recorded | 0.757 | 0.735 | 0.655 (5,757) | 0.589 (1,011) |
| all adults | BE recorded | 0.773 | 0.707 | 0.650 (6,768) | – (0) |
| all adults | SOFA computable | 0.726 | 0.746 | 0.650 (6,768) | – (0) |
| SOFA-computable | lactate recorded | 0.743 | 0.561 | 0.655 (5,757) | 0.589 (1,011) |
