# ICU mortality prediction on MIMIC-III (RF / XGBoost, 24 h ahead)

Hourly early-warning models that flag ICU patients who will die **within the next 24 hours**, trained on
MIMIC-III v1.4 (adult ICUs). The pipeline re-implements a NICU ventilation-prediction baseline
(hourly grid → event window labels → Optuna-tuned RandomForest) on public data, adds an XGBoost counterpart,
SHAP / ROC-PR analysis, an EDA report, and a stay-level "first W hours" experiment.

> MIMIC-III is a credentialed PhysioNet resource. No data, derived tables, model files or per-row predictions are
> in this repository — only code, aggregate metrics and figures. You need your own copy of
> `mimic-iii-clinical-database-1.4/`.

## Task

| | |
|---|---|
| Unit of prediction | one row per ICU stay per hour (`INTIME` → `OUTTIME`, or → death) |
| Event | in-ICU death: `ADMISSIONS.DEATHTIME` inside `[INTIME, OUTTIME]` |
| Positive rows | the **24 h** before death (`--pre-window 24`; the original protocol used 8 h) |
| Negative rows | every hour of stays that leave the ICU alive |
| Excluded | rows of event stays *before* the window (`is_pre_event=1`); stays that die within 24 h of ICU admission (window would start before admission); admissions that end in death outside this ICU stay (ambiguous) |
| Features (20) | HR, RR, SpO2, temperature, non-invasive & arterial SBP/DBP/MAP, FiO2, O2 flow, weight, height, base excess, daily total input / output / balance, age, sex — hourly means, forward-filled within the stay |
| Split | patient-level random 70 / 15 / 15 (seed 0; MIMIC dates are shifted per patient, so no calendar split) |
| Selection | Optuna on validation AUROC; test metrics reported once |

Cohort with the 24 h window: 50,276 stays (3,391 deaths), 5.09 M hourly rows, 82 k positive rows (1.8 % of training rows).

## Results (test set)

### Hourly model, death within 24 h  (`data/icu_death_w24h.csv`)

| model | val AUROC | test AUROC | test AUPRC | Base-Excess ablation† |
|---|---|---|---|---|
| XGBoost, native NaN (`--native-nan`) | 0.891 | **0.894** | **0.394** | 0.888 |
| XGBoost, train-median fill | 0.879 | 0.889 | 0.380 | 0.882 |
| RandomForest (notebook protocol, 100 trials) | 0.864 | 0.875 | 0.334 | 0.867 |

† `Base Excess` replaced by the train median (or NaN in native-NaN mode) at test time — the notebook's "key-feature" ablation.

Curves and SHAP: `results/shap_hourly_icu_death_w24h/` (ROC + PR, mean |SHAP| bars, beeswarms, `shap_mean_abs.csv`).
Top mean |SHAP| (native-NaN model): O2 flow 0.42, base excess 0.36, age 0.32, FiO2 0.34, I/O balance 0.24, SpO2 0.22, HR 0.22
— oxygen requirement, metabolic acidosis, age, fluid balance and hypoxaemia. A *missing* FiO2 (no oxygen charted) is a
strong negative signal. Compared with the 8 h model, the 24 h model leans more on slowly varying features (age, I/O balance)
and less on the acute vitals (SpO2, blood pressure).

### Reference: the same pipeline with an 8 h window (`--pre-window 8`, 0.75 % positive rows)

| model | test AUROC | test AUPRC |
|---|---|---|
| XGBoost, native NaN | 0.919 | 0.471 |
| XGBoost, median fill | 0.910 | 0.460 |
| RandomForest (200 trials) | 0.892 | 0.419 |

Shorter horizon → sharper physiological signal → higher AUROC; the 24 h model trades ~0.025 AUROC for a day of lead time.

### Stay-level "first W hours" experiment (`05_early_window.py`)

Observe only the first W hours after ICU admission and predict whether the stay ends in death
(stays that end inside the window are excluded, so the cohort shrinks and gets sicker as W grows).
Test AUROC / AUPRC on each window's own cohort:

| W (h) | 1 | 3 | 6 | 12 | 24 | 48 | 72 | 120 | 168 |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost AUROC | 0.782 | 0.815 | 0.832 | 0.861 | **0.880** | 0.867 | 0.836 | 0.809 | 0.812 |
| RandomForest AUROC | 0.765 | 0.808 | 0.824 | 0.847 | 0.861 | 0.842 | 0.824 | 0.772 | 0.761 |
| XGBoost AUPRC | 0.240 | 0.266 | 0.292 | 0.343 | 0.433 | 0.469 | 0.419 | 0.467 | 0.482 |
| test stays | 7,484 | 7,472 | 7,453 | 7,409 | 6,390 | 4,063 | 2,715 | 1,501 | 1,005 |
| event rate | .065 | .065 | .065 | .066 | .076 | .095 | .115 | .161 | .172 |

Discrimination saturates at **W = 12–24 h**; the decline afterwards is the cohort shift (long, sicker stays).
`results/early_window_death/summary.md` also reports AUROC by time-to-death after the window and on the common
cohort of stays longer than 168 h (`saturation.png`); curves and SHAP per window are in `results/shap_early_death/`.

### Endpoint comparison: in-ICU vs in-hospital vs 28-day mortality

A stay-level endpoint defined outside the ICU (28-day all-cause death from `PATIENTS.DOD`, which also covers deaths
after hospital discharge) yields far more positives and is what conventional severity scores are calibrated against.
Built with `01 --event none` (no stay is truncated or dropped by an in-ICU event) and run through the same
early-window models, identical features and patient split — XGBoost test AUROC / AUPRC:

| W (h) | in-ICU death | in-hospital death | 28-day death |
|---|---|---|---|
| 1 | 0.763 / 0.275 | 0.750 / 0.290 | 0.740 / 0.313 |
| 6 | 0.833 / 0.380 | 0.804 / 0.344 | 0.795 / 0.374 |
| 12 | 0.853 / 0.415 | 0.819 / 0.381 | 0.809 / 0.398 |
| **24** | **0.871 / 0.437** | **0.847 / 0.420** | **0.826 / 0.426** |
| 48 | 0.856 / 0.439 | 0.863 / 0.413 | 0.836 / 0.417 |

Positives (whole cohort): in-ICU 4,453 (8.3 %) → in-hospital 6,544 (12.2 %) → 28-day 7,776 (14.6 %); 3,454 of the
28-day deaths happen outside the ICU (1,828 on the ward, 1,626 after hospital discharge).

- **More positives, lower AUROC.** Ranking is consistent across every window: the further the endpoint is from the
  ICU stay, the lower the discrimination (−0.045 AUROC at W = 24 h from in-ICU to 28-day). The extra positives are
  late deaths (median 7.6 days after admission) whose course is shaped by treatment after the observation window —
  at W = 24 h the 28-day model scores 0.907 on deaths within the next 24 h but 0.806 on deaths later than 72 h.
- **AUPRC barely moves** (0.437 → 0.426 at W = 24 h) because the higher prevalence offsets the weaker ranking, so the
  28-day endpoint is not worse in absolute precision-recall terms.
- The 28-day model relies more on baseline severity (age, FiO2 range, RR) and less on the acute end-of-stay features
  that dominate the in-ICU model (minimum base excess, O2 flow).
- MIMIC-III records post-discharge deaths through a social-security death index, available for 45 % of stays; deaths
  that the index misses are counted as survivors, which biases the 28-day label towards the null.

Reference points from the literature for first-24 h severity scores on MIMIC-III (in-hospital mortality):
SAPS-II ≈ 0.78–0.86, OASIS ≈ 0.77, APS-III ≈ 0.78 — the 0.847 obtained here with 20 routinely charted variables is
in the same range, though cohort definitions differ and this is not a head-to-head comparison.

## Pipeline

```
export MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4      # csv.gz files from PhysioNet
python scripts/00_extract_icu_events.py                          # big tables -> data/*.parquet (CHARTEVENTS: ~20 min)
python scripts/01_build_dataset.py --event death --pre-window 24 # hourly grid + labels + split -> data/icu_death_w24h.csv
python scripts/04_train_xgb.py data/icu_death_w24h.csv icu_death_w24h_xgb_nan --native-nan --device cuda:0
python scripts/04_train_xgb.py data/icu_death_w24h.csv icu_death_w24h_xgb --device cuda:0
python scripts/02_train_rf.py  data/icu_death_w24h.csv icu_death_w24h_rf --n-trials 100 --n-jobs 16 --resume
python scripts/06_shap_curves.py --task hourly --prefix icu_death_w24h --device cuda:0
python scripts/03_eda.py --event death --dataset data/icu_death_w24h.csv   # results/eda_icu_death_w24h/EDA.md
python scripts/05_early_window.py --dataset data/icu_death_w24h.csv --windows 1 3 6 12 24 48 72 120 168
python scripts/06_shap_curves.py --task early --event death

# endpoint comparison (in-ICU vs in-hospital vs 28-day mortality) on one cohort
python scripts/01_build_dataset.py --event none                      # -> data/icu_allstays.csv, no event truncation
for O in icu_death hosp_death death_28d; do
  python scripts/05_early_window.py --dataset data/icu_allstays.csv --outcome $O --windows 1 3 6 12 24 48
done
```

| script | what it does |
|---|---|
| `00_extract_icu_events.py` | streams CHARTEVENTS (330 M rows) / LABEVENTS / INPUT / OUTPUT / PROCEDUREEVENTS and keeps the adult-ICU rows of the ITEMIDs we use (CareVue + MetaVision ids) |
| `01_build_dataset.py` | unit conversion (°F→°C, FiO2 fraction→%, inch→cm), clinical plausibility ranges, hourly means, daily I/O totals assigned to the next day, event time, window labels, patient split. `--event vent` reproduces the original ventilation task |
| `02_train_rf.py` | RandomForest, Optuna (max_depth 1–10, max_leaf_nodes 2–1000, n_estimators 100–500, `class_weight={0:1,1:50}`), val-AUROC selection, key-feature ablation. `--resume` keeps the study in sqlite; each trial fits in a forked child (the in-process multithreaded fit leaked ~2.5 GB RSS per trial) |
| `04_train_xgb.py` | same preparation and evaluation; GPU `hist`, early stopping on val, Optuna over depth / lr / min_child_weight / subsample / colsample / λ / scale_pos_weight; `--native-nan` skips the median fill |
| `06_shap_curves.py` | ROC / PR curves for all models of a task, exact TreeSHAP (XGBoost `pred_contribs`, RF via `shap.TreeExplainer` on a sample) |
| `03_eda.py` | raw MIMIC-III overview + cohort / missingness / label-time-structure report (`EDA.md` + figures) |
| `05_early_window.py` | stay-level early-warning experiment described above (XGB + RF, per-window models, saturation analysis). `--outcome {stay_event,icu_death,hosp_death,death_28d}` selects the endpoint |

### Fixes relative to the original notebook (all in `02_train_rf.py` / `01_build_dataset.py`)

1. Deterministic feature order (`sorted`) — the notebook used `list(set(...))`, which shuffles columns between Python
   processes and silently breaks a pickled model.
2. Forward-fill **per stay**, and **before** dropping the pre-window rows — the notebook filtered first, so the window
   rows of event patients lost their history and were median-filled; the model then learned "value == median ⇒ positive"
   (val AUROC inflated 0.75 → 0.94 in our leak diagnostic).
3. The Optuna best trial is the model that is evaluated (the notebook loaded a fixed trial file).
4. `average_precision_score` reported next to the notebook's `auc(recall, precision)`.

### Things to be aware of

- The hourly row that contains the event averages minutes *after* the event; for the ventilation task this leaks
  ventilator settings (FiO2 charted in 91 % of event-hour rows vs 6 % before). For death the charting rate *drops*
  toward the event, so this is not an issue for the mortality label, but keep it in mind for other events.
- Missingness is informative (charting frequency changes before death); the native-NaN XGBoost uses it, the
  median-fill variants mostly do not. Report both.
- CareVue (2001–08) and MetaVision (2008–12) stays differ in charting (`ICUSTAYS.DBSOURCE`); the EDA reports
  feature availability by era.
- `xgboost` is pinned to 3.0.5: 3.1+ PyPI wheels are built against CUDA 13 and silently fall back to CPU on
  CUDA-12 drivers.

## Environment

Python 3.12, `pip install -r requirements.txt`. GPU is optional (XGBoost / SHAP are much faster with one);
RandomForest tuning is CPU-bound (16 cores, ~5 min per trial on 3.5 M rows). Peak RAM for `01_build_dataset.py` ≈ 20 GB.
