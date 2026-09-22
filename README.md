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
| RandomForest (notebook protocol) | TBD | TBD | TBD | TBD |

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
(stays that end inside the window are excluded). XGBoost, test AUROC / AUPRC:

| W (h) | 1 | 3 | 6 | 12 | 24 | 48 | 72 | 120 | 168 |
|---|---|---|---|---|---|---|---|---|---|
| AUROC | 0.771 | 0.821 | 0.838 | 0.858 | **0.864** | 0.848 | 0.850 | 0.821 | 0.806 |
| AUPRC | 0.289 | 0.355 | 0.390 | 0.443 | 0.433 | 0.469 | 0.501 | 0.490 | 0.511 |

Discrimination saturates at **W = 12–24 h**; the decline afterwards is a cohort effect (only long, sicker stays
survive a long window: event rate 8 % → 18 %). See `results/early_window_death/summary.md` and `saturation.png`.

## Pipeline

```
export MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4      # csv.gz files from PhysioNet
python scripts/00_extract_icu_events.py                          # big tables -> data/*.parquet (CHARTEVENTS: ~20 min)
python scripts/01_build_dataset.py --event death --pre-window 24 # hourly grid + labels + split -> data/icu_death_w24h.csv
python scripts/04_train_xgb.py data/icu_death_w24h.csv icu_death_w24h_xgb_nan --native-nan --device cuda:0
python scripts/04_train_xgb.py data/icu_death_w24h.csv icu_death_w24h_xgb --device cuda:0
python scripts/02_train_rf.py  data/icu_death_w24h.csv icu_death_w24h_rf --n-trials 200 --n-jobs 16 --resume
python scripts/06_shap_curves.py --task hourly --prefix icu_death_w24h --device cuda:0
python scripts/03_eda.py --event death --dataset data/icu_death_w24h.csv   # results/eda_icu_death_w24h/EDA.md
python scripts/05_early_window.py --dataset data/icu_death_w24h.csv --windows 1 3 6 12 24 48 72 120 168
python scripts/06_shap_curves.py --task early --event death
```

| script | what it does |
|---|---|
| `00_extract_icu_events.py` | streams CHARTEVENTS (330 M rows) / LABEVENTS / INPUT / OUTPUT / PROCEDUREEVENTS and keeps the adult-ICU rows of the ITEMIDs we use (CareVue + MetaVision ids) |
| `01_build_dataset.py` | unit conversion (°F→°C, FiO2 fraction→%, inch→cm), clinical plausibility ranges, hourly means, daily I/O totals assigned to the next day, event time, window labels, patient split. `--event vent` reproduces the original ventilation task |
| `02_train_rf.py` | RandomForest, Optuna (max_depth 1–10, max_leaf_nodes 2–1000, n_estimators 100–500, `class_weight={0:1,1:50}`), val-AUROC selection, key-feature ablation. `--resume` keeps the study in sqlite; each trial fits in a forked child (the in-process multithreaded fit leaked ~2.5 GB RSS per trial) |
| `04_train_xgb.py` | same preparation and evaluation; GPU `hist`, early stopping on val, Optuna over depth / lr / min_child_weight / subsample / colsample / λ / scale_pos_weight; `--native-nan` skips the median fill |
| `06_shap_curves.py` | ROC / PR curves for all models of a task, exact TreeSHAP (XGBoost `pred_contribs`, RF via `shap.TreeExplainer` on a sample) |
| `03_eda.py` | raw MIMIC-III overview + cohort / missingness / label-time-structure report (`EDA.md` + figures) |
| `05_early_window.py` | stay-level early-warning experiment described above (XGB + RF, per-window models, saturation analysis) |

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
