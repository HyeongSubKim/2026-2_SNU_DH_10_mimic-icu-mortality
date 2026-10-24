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

### Cohort flow chart (SOFA / lactate / base-excess cohort)

`07_cohort_flow_counts.py` + `08_flowchart.py` produce CONSORT-style flow charts (`results/flowchart/`) for a cohort
defined by: adults (>= 18 y at ICU admission) -> SOFA computable in the first W h (all six organ systems measured) ->
lactate in the first W h -> base excess in the first W h -> not died within the first W h, with the final box split by
28-day all-cause mortality. W = 24 h: 61,532 ICU stays -> 53,329 adult -> 11,140 SOFA -> 9,731 lactate -> 9,731 base
excess -> 9,299 included (2,296 died within 28 days, 24.7 %). W = 6 h: 5,858 included (1,632 died, 27.9 %).

### SOFA cohort: SOFA inputs ± lactate ± base excess, 28-day mortality (`09_sofa_feature_tiers.py`, `10_tabpfn_tiers.py`)

On the flow-chart cohort (adults, SOFA computable, lactate, base excess, alive at W h), four nested feature sets from
the first W hours predict 28-day mortality. Test AUROC (bootstrap 95 % CI in `results/sofa_tiers/summary.md`):

| feature set | XGBoost 6 h | TabPFN v2 6 h | XGBoost 24 h | TabPFN v2 24 h |
|---|---|---|---|---|
| SOFA total score alone (reference) | 0.650 | – | 0.657 | – |
| 1 SOFA inputs (+ component scores) | 0.794 | 0.808 | 0.814 | 0.820 |
| 2 + lactate | 0.801 | 0.819 | 0.823 | 0.822 |
| 3 + base excess | 0.801 | 0.816 | 0.821 | 0.822 |
| 4 + lactate + base excess | 0.809 | 0.818 | **0.828** | 0.823 |

Cohorts: 5,757 stays (6 h; test 853, 29 % died) and 9,229 (24 h; test 1,374, 26 % died). Lactate / base excess add
~0.01 AUROC on top of the SOFA inputs (CIs overlap); the 24 h window is ~0.02 above 6 h; TabPFN v2 (pre-trained
tabular transformer, no tuning, ~20–50 s per fit) matches XGBoost at 24 h and is slightly better at 6 h — see
`results/sofa_tiers/transformer_feasibility.md` for the other transformer options and why they were not pursued.

With age and sex added to every tier (`--demographics`, `results/sofa_tiers_demo/`), test AUROC:

| model | SOFA total alone | SOFA + age + sex (logistic) | 1 SOFA inputs | 2 + lactate | 3 + base excess | 4 + lactate + BE |
|---|---|---|---|---|---|---|
| 6 h, XGBoost | 0.650 | 0.705 | 0.814 | 0.812 | 0.823 | 0.823 |
| 6 h, TabPFN v2 | 0.650 | 0.705 | 0.824 | 0.830 | 0.833 | 0.832 |
| 24 h, XGBoost | 0.657 | 0.700 | 0.826 | 0.831 | 0.831 | **0.837** |
| 24 h, TabPFN v2 | 0.657 | 0.700 | 0.831 | 0.833 | 0.835 | **0.837** |

Demographics add ~+0.01–0.02 AUROC and the ordering of the tiers is unchanged. The conventional SOFA score is the
reference: the same variables fed to a learner raise AUROC by ~0.13–0.17 over SOFA + age + sex (logistic).

### UTDE (ICML 2023) on the same cohort, and TabPFN + UTDE late fusion

UTDE = `MultimodalMIMIC` (Zhang et al., ICML 2023) run unchanged except for an adapter (channel count, val-AUROC checkpoint
selection, saving test probabilities, local Clinical-Longformer path): irregular hourly series of 21 channels + the last
5 notes (Clinical-Longformer, 1024 tokens), 6 epochs, original hyper-parameters. Data built by `13_build_utde_dataset.py`
from the identical cohort / split / label; late fusion by `14_late_fusion.py` (logistic stacker fitted on the validation
split; paired bootstrap for the differences). Test set:

| model | 6 h AUROC (AUPRC) | ΔAUROC vs TabPFN | 24 h AUROC (AUPRC) | ΔAUROC vs TabPFN |
|---|---|---|---|---|
| TabPFN v2, tier 4 + age/sex (structured) | 0.832 (0.681) | – | 0.837 (0.673) | – |
| UTDE (time series + notes) | 0.798 (0.618) | −0.034 [−0.066, −0.003] | 0.844 (0.680) | +0.007 [−0.013, +0.026] |
| **late fusion TabPFN + UTDE** | **0.845 (0.704)** | +0.014 [−0.002, +0.029] | **0.860 (0.707)** | **+0.023 [+0.011, +0.035]** |

UTDE alone trails the tabular model at 6 h (six time steps, bedside notes for 44 % of stays) and matches it at 24 h; the
two are complementary — fusing them is the best result so far on both windows and is significant at 24 h.

### Curated note-concept dictionary (N01–N50) as features (`15_note_concepts.py`)

Fifty clinician-curated concept groups (co-morbidity, acute state, neurology, base-excess causes, interventions, trajectory
language) are counted in the notes available by INTIME + W (negation-aware regex; goals-of-care terms L01 held out as a
leakage probe). XGBoost, test AUROC (AUPRC), paired ΔAUROC vs the structured model:

| feature set | 6 h | Δ | 24 h | Δ |
|---|---|---|---|---|
| A structured (tier 4 + age/sex) | 0.823 (0.662) | – | 0.837 (0.660) | – |
| B concepts only | 0.683 (0.464) | −0.140 | 0.751 (0.505) | −0.086 [−0.115, −0.057] |
| C structured + concepts (negation-aware) | 0.819 (0.656) | −0.004 | 0.844 (0.674) | +0.006 [−0.003, +0.016] |
| C0 structured + concepts (raw counts) | 0.830 (0.673) | +0.007 | 0.847 (0.686) | +0.010 [+0.001, +0.019] |

Single concepts are weak (best AUROC alone 0.56: "worsening" language, vasopressor mentions, unresponsiveness, altered
mental status); the dictionary adds ≈ +0.01 at 24 h and nothing at 6 h, versus +0.023 for UTDE late fusion.
L01 terms (DNR / CMO / hospice …) are present in 10 % of 24 h stays with 55 % vs 21 % mortality — must stay out of the model.

**Term-guided attention ablation (UTDE, 24 h).** Giving Longformer global attention to every token that matches the
N01–N50 dictionary (median 8 term tokens per note, in addition to [CLS]; `--term_global_attention` in the UTDE adapter)
changes nothing: UTDE 0.843 vs 0.844 (Δ −0.000 [−0.010, +0.010]), fused with TabPFN 0.857 vs 0.860 (Δ −0.003
[−0.009, +0.004]); the two models' test probabilities correlate at 0.93. The encoder already attends to these terms
when they matter; the dictionary's value is as interpretable features (+0.01) and as a prompt schema, not as an attention prior.

### Record2Vec (ICLR 2026) on the same cohort (`16_`–`19_record2vec_*.py`)

Record2Vec recipe: first-W-hour irregular record → canonical text serialisation (21 channels, raw units, hourly values) →
frozen LLM summary (Qwen2.5-7B-Instruct, local vLLM, temperature 0; two prompts: free narrative / organ-system schema; a third
prompt adds the notes available in the window = condition D) → frozen text embedder (Qwen3-Embedding-0.6B / **4B**, GatorTron-base)
→ logistic regression (C chosen on the validation split) or TabPFN on 32 PCs. Same cohort, split and label as above;
paired bootstrap vs structured TabPFN (A). Test AUROC (AUPRC), Qwen3-Embedding-4B unless stated:

| condition | 6 h | Δ vs A | 24 h | Δ vs A |
|---|---|---|---|---|
| A structured TabPFN (tier 4 + age/sex) | 0.832 (0.681) | – | 0.837 (0.673) | – |
| B0 raw serialisation → embedding → LR (no summary) | 0.757 (0.551) | −0.075 [−0.100, −0.050] | 0.760 (0.541) | −0.078 [−0.102, −0.054] |
| B narrative summary → embedding → LR | 0.744 (0.542) | −0.088 [−0.121, −0.056] | 0.699 (0.474) | −0.138 [−0.165, −0.111] |
| B schema summary → embedding → LR | 0.728 (0.530) | −0.105 [−0.137, −0.072] | 0.736 (0.508) | −0.101 [−0.127, −0.075] |
| C hybrid: structured + 32 PCs of the narrative embedding → TabPFN | 0.826 (0.675) | −0.005 [−0.014, +0.004] | 0.831 (0.663) | −0.006 [−0.010, −0.002] |
| F late fusion TabPFN + B0 LR | 0.826 (0.668) | −0.005 [−0.012, +0.001] | 0.835 (0.668) | −0.002 [−0.009, +0.005] |
| D summary of record + notes → embedding → LR | 0.763 (0.543) | −0.069 [−0.103, −0.037] | 0.798 (0.594) | −0.039 [−0.062, −0.017] |
| F late fusion TabPFN + D LR | 0.828 (0.664) | −0.004 [−0.023, +0.014] | 0.842 (0.672) | +0.004 [−0.008, +0.016] |
| D① term-guided prompt (N01–N50 categories must be covered; raw notes) → LR | 0.787 (0.603) | −0.045 [−0.078, −0.015] | 0.816 (0.625) | −0.021 [−0.044, +0.002] |
| F late fusion TabPFN + D① LR | 0.835 (0.688) | +0.003 [−0.015, +0.019] | 0.849 (0.703) | +0.012 [−0.001, +0.025] |
| D② term-filtered notes (only sentences matching N01–N50, L01 dropped) → LR | 0.767 (0.589) | −0.065 [−0.096, −0.034] | 0.775 (0.547) | −0.062 [−0.086, −0.039] |
| F late fusion TabPFN + D② LR | 0.831 (0.691) | −0.000 [−0.014, +0.013] | 0.837 (0.676) | −0.000 [−0.010, +0.009] |

Other embedders give the same ordering: narrative-summary LR 0.731 / 0.692 (0.6B) and 0.732 / 0.704 (GatorTron) at 6 / 24 h;
condition D is where the clinical embedder helps most: GatorTron D 0.768 / 0.807 at 6 / 24 h, fused with TabPFN
0.834 / 0.847 (+0.002 / +0.010 [−0.002, +0.022]) — the best text-path numbers, still short of UTDE fusion (0.845 / 0.860).

Label efficiency and pseudo-site transfer (`19_record2vec_transfer.py`, 4B embedder, LR C = 10; test AUROC, mean ± sd over 5 draws):

| representation | n = 16 | 64 | 256 | 1024 | all | CareVue→MetaVision | in-domain | MetaVision→CareVue | in-domain |
|---|---|---|---|---|---|---|---|---|---|
| 6 h structured (TabPFN) | 0.631 ± 0.042 | 0.696 ± 0.031 | 0.761 ± 0.016 | 0.803 ± 0.009 | 0.829 | 0.826 | 0.832 | 0.813 | 0.814 |
| 6 h raw-text embedding (LR) | 0.648 ± 0.038 | 0.693 ± 0.023 | 0.710 ± 0.011 | 0.743 ± 0.005 | 0.760 | 0.764 | 0.764 | 0.737 | 0.759 |
| 6 h narrative-summary embedding (LR) | 0.597 ± 0.053 | 0.649 ± 0.012 | 0.686 ± 0.020 | 0.718 ± 0.004 | 0.748 | 0.745 | 0.726 | 0.738 | 0.739 |
| 24 h structured (TabPFN) | 0.609 ± 0.094 | 0.710 ± 0.025 | 0.754 ± 0.010 | 0.809 ± 0.012 | 0.834 | 0.833 | 0.837 | 0.825 | 0.828 |
| 24 h raw-text embedding (LR) | 0.534 ± 0.065 | 0.602 ± 0.057 | 0.661 ± 0.014 | 0.708 ± 0.010 | 0.736 | 0.742 | 0.755 | 0.732 | 0.750 |
| 24 h narrative-summary embedding (LR) | 0.526 ± 0.075 | 0.589 ± 0.040 | 0.619 ± 0.014 | 0.655 ± 0.005 | 0.704 | 0.723 | 0.709 | 0.681 | 0.680 |

Reading: (i) the LLM-text representation trails the clinician-curated structured features by 0.08–0.14 AUROC, and adds
nothing on top of them (hybrid and late fusion are 0 or slightly negative); (ii) the summarisation step *loses* information
relative to embedding the raw serialisation (best summary prompt: 6 h −0.013, 24 h −0.024); prompt style matters as much
as the summariser (narrative 0.744 / 0.699 vs schema 0.728 / 0.736 at 6 / 24 h — the schema keeps more numbers, which pays
off on the longer record, but its template dominates the embedding on the short one); (iii) the paper's
two selling points do not appear here: with 16 labels the structured TabPFN is already as good as or better than the
embedding probes, and the CareVue→MetaVision drop is ≤ 0.01 for both representations, so there is no shift for the text
path to be robust to. Adding the notes to the summariser's input (D) is the one thing that
moves the text path (+0.02 / +0.06–0.10 over record-only summaries at 6 / 24 h), i.e. the notes carry information the vitals
do not, but a summary→embedding compression of them buys +0.004…+0.010 (ns) in fusion where UTDE's learned note encoder buys
+0.023. **Term dictionary in the text path (D①/D②).** Listing the N01–N50 categories in the summariser's prompt (the summary must
cover every category present) is the best Record2Vec variant at 24 h (0.816 alone, fusion 0.849, +0.012 [−0.001, +0.025])
and neutral at 6 h; restricting the notes to the sentences that match the dictionary (median 3,262 → 465 characters) is
clearly harmful (−0.02 … −0.04 vs raw notes): context outside the dictionary carries signal, which is the same lesson as
the UTDE term-attention ablation — the dictionary works as a "do not omit" checklist, not as a focus restriction.
Summary quality is not the cause — 96 % (narrative) / 99.5 % (schema) of the numbers quoted in the
summaries occur in the source record (2,000-stay check). Caveats: 7B summariser and ≤ 4B embedder (the paper used Gemini-2.0
Flash and trained downstream heads on 10× more stays); single summarisation run (temperature 0); numeric precision is the
likely bottleneck for a text embedding of a 21-channel record.

### Selection-bias check: tested vs untested stays (`20_selection_bias.py`, reviewer question)

"Does restricting to stays with lactate / BE results select patients who are easier to predict?" — checked on all adult
ICU stays alive at hour W (landmark), 28-day mortality, with crude risk differences, adjusted ORs (age, sex, unit,
admission type, era; ‡ + vasopressor and ventilation in the window), stabilised IPW, and discrimination inside each group.
Full tables: `results/selection_bias/summary_W{6,24}.md`.

| W | exposure (population) | exposed mortality (N) | unexposed mortality (N) | RD pp [95 % CI] | adj. OR † / ‡ | IPW RD pp † / ‡ |
|---|---|---|---|---|---|---|
| 6 h | SOFA computable (all adults) | 27.0 % (6,768) | 12.2 % (46,287) | +14.7 [+13.6, +15.8] | 2.50 / 1.58 | +13.8 / +8.5 |
| 6 h | lactate recorded (all adults) | 17.5 % (27,988) | 10.4 % (25,067) | +7.0 [+6.5, +7.6] | 1.94 / 1.47 | +7.6 / +4.3 |
| 6 h | lactate recorded (within SOFA-computable) | 27.9 % (5,757) | 21.9 % (1,011) | +6.0 [+3.2, +8.8] | 1.36 / 1.18 | +4.8 / +3.4 |
| 24 h | SOFA computable (all adults) | 24.0 % (10,590) | 10.0 % (41,661) | +14.0 [+13.1, +14.9] | 2.80 / 1.87 | +12.6 / +7.9 |
| 24 h | lactate recorded (all adults) | 15.8 % (30,293) | 8.6 % (21,958) | +7.2 [+6.6, +7.7] | 2.14 / 1.53 | +7.6 / +4.4 |
| 24 h | lactate recorded (within SOFA-computable) | 24.7 % (9,229) | 18.8 % (1,361) | +5.9 [+3.7, +8.2] | 1.45 / 1.20 | +4.0 / +2.6 |

BE is present in every SOFA-computable stay (an arterial gas gives PaO2 and BE together), so it adds no selection beyond
lactate. The tested groups are sicker (ventilated 57 % vs 25 %, vasopressors 34 % vs 16 %, GCS min 9 vs 14 at 24 h) and
the SOFA-computable gate — not the lactate gate — is the dominant selection (79 % of adults drop there, mortality ×2.4);
adjustment halves the differences but does not remove them. Discrimination is *not* higher in the selected group: the
same covariate model scores 0.70 (SOFA-computable) vs 0.76 (not computable) at 24 h and the SOFA score itself 0.65 vs
0.62 in lactate-tested vs untested stays, i.e. the restriction selects a sicker, more homogeneous population in which
prediction is if anything harder. Consequence for the study: the models are valid for ICU stays with an arterial gas and
lactate in the first W h (the population on which all tiers / UTDE / Record2Vec were compared), not for all ICU admissions.

### SHAP / attribution of the best model: late fusion TabPFN + UTDE (`21_fusion_shap.py`, `22_fusion_shap_figs.py`, `../MultimodalMIMIC/utde_attr.py` (patch: `utde/MultimodalMIMIC_sofa28.patch`))

Three levels, all on the test split. Figures in `results/fusion_shap/` (`fusion_beeswarm_W*.png`, `tabpfn_shap_W*.png`, `utde_beeswarm_W*.png`, `utde_attr_W*.png`).

**1. Fusion stacker (exact SHAP — the stacker is linear in the two logits).**

| W | coef TabPFN / UTDE | mean abs SHAP TabPFN / UTDE | UTDE share | logit corr. | cases where the two disagree | fusion / TabPFN / UTDE AUROC on those |
|---|---|---|---|---|---|---|
| 6 h | 0.55 / 0.54 | 0.60 / 0.55 | 48% | 0.67 | 26% | 0.598 / 0.659 / 0.389 |
| 24 h | 0.41 / 0.43 | 0.53 / 0.74 | 58% | 0.79 | 21% | 0.600 / 0.565 / 0.524 |

The stacker weights the two logits almost equally; the UTDE logit has the larger spread, so it carries 58 % of the fused
logit at 24 h, while the structured model leads at 6 h (52 %). The gain does not come from arbitrating opposite votes
(few cases sit on the cancellation diagonal) but from reinforcing the shared direction; where the two disagree the
fusion is better than either at 24 h and in between at 6 h (UTDE is confidently wrong on part of those at 6 h).

**2. TabPFN (tier 4 + age/sex) — permutation SHAP** (refit reproduces 0.832 / 0.837; all test cases, 853 / 1,374,
10 background rows, 2F+1 evaluations per case). Mean |SHAP| in probability units, grouped by source variable:
24 h — GCS 0.147 ≫ age 0.048 ≈ urine 0.046 > lactate 0.041 > MAP 0.035 > FiO2 0.029 ≈ BE 0.028 (single features:
`gcs_last` 0.111 ≫ `age` 0.048 > `urine_n` ≈ `gcs_count` ≈ `lactate_min` ≈ `lactate_last` ≈ `base_excess_last` 0.02);
6 h — GCS 0.082 ≈ age 0.071 > FiO2 0.047 ≈ lactate 0.047 ≈ urine 0.046 > BE 0.042 > platelets 0.039 (single: `age` 0.071 ≈
`gcs_last` 0.069 > `fio2_last` 0.031). Directions are clinical (low GCS, low urine, high lactate, low BE, high FiO2, high
age → death). Two observations: in the first 6 h prior risk (age) and oxygen requirement weigh as much as GCS, and only
by 24 h do the physiological trajectories dominate; and measurement-count features (`urine_n`, `gcs_count`,
`platelets_count`) sit in the top 12 at both windows — the model also learns the charting / testing pattern (cf. the
selection-bias section), a limitation to state.

**3. UTDE — occlusion attribution** (deep model: a channel's values and masks are zeroed in both inputs; notes are
replaced by the placeholder note; exact harness, reproduces 0.798 / 0.844).

| W | full | without notes | without time series | top channels by mean abs Δlogit | top channels by AUROC drop |
|---|---|---|---|---|---|
| 6 h | 0.798 | 0.786 | 0.621 | GCS 0.355, Temp 0.279, Urine 0.278, RR 0.257, Creatinine 0.178 | GCS 0.030, RR 0.015, Urine 0.013, Bilirubin 0.011 |
| 24 h | 0.844 | 0.819 | 0.681 | GCS 0.635, Urine 0.469, Lactate 0.354, Temp 0.320, RR 0.257 | GCS 0.038, Temp 0.016, Urine 0.010, Bilirubin 0.007 |

Both components rank GCS, urine output and lactate first, which is why their logits correlate at 0.7–0.8 and the fusion
gain is bounded (+0.014 / +0.023). What each adds: TabPFN — age/sex and the record-pattern counts (absent from UTDE);
UTDE — the notes (−0.025 AUROC when removed at 24 h, −0.012 at 6 h) and the temperature / respiratory-rate / heart-rate
trajectories, which the structured aggregates summarise only as min/max/mean. Easy follow-ups suggested by this: add
age/sex to the UTDE input, and raise `num_of_notes` (63 % of 24 h stays have more than the 5 notes UTDE keeps).

## License

Code in this repository is released under the Apache License 2.0 (see `LICENSE`). MIMIC-III data are not included and remain subject to the PhysioNet Credentialed Health Data Use Agreement; the UTDE reference implementation (XZhang97666/MultimodalMIMIC) is used as an external dependency via the patch in `utde/` and keeps its own terms.
