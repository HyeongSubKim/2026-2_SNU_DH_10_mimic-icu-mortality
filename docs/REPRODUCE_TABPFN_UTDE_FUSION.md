# Reproducing TabPFN, UTDE and the TabPFN + UTDE late fusion

SOFA cohort of MIMIC-III v1.4 (adults, SOFA computable within the first W h, lactate and base excess present, alive at
hour W), 28-day all-cause mortality, W = 6 h and 24 h, patient-level 70/15/15 split (seed fixed in `09`).
Test-set results that these steps reproduce (AUROC (AUPRC)):

| model | 6 h | 24 h |
|---|---|---|
| TabPFN v2, tier 4 + age/sex | 0.832 (.681) | 0.837 (.673) |
| UTDE (time series + notes) | 0.798 (.618) | 0.844 (.680) |
| late fusion TabPFN + UTDE | 0.845 (.704) | 0.860 (.707) |

## 0. Data access and environments

* MIMIC-III v1.4 from PhysioNet (credentialed; the CSV.gz files, including NOTEEVENTS). Nothing derived from patient
  data is in this repository — every step below regenerates it locally. Set `export MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4`.
* Three Python environments (the reference UTDE code needs old torch/transformers):

| env | used by | install |
|---|---|---|
| `base` (py ≥ 3.10) | 00, 07, 09, 12, 13 | `pip install -r requirements.txt` |
| `tabpfn` (py 3.12) | 10, 14, 21, 22 | `pip install -r requirements-tabpfn.txt` |
| `utde` (py 3.8) | UTDE training / attribution | `pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117 && pip install -r requirements-utde.txt` (set `PYTHONNOUSERSITE=1` so `~/.local` packages do not leak in) |

* Pre-trained weights, downloaded once into `$HF_HOME` (default `~/.cache/huggingface`):
  `Prior-Labs/TabPFN-v2-clf` (open checkpoint, no token) and `yikuan8/Clinical-Longformer`
  (`huggingface-cli download yikuan8/Clinical-Longformer`; transformers 4.16 cannot read the new hub cache layout, so the
  UTDE run script passes the snapshot directory explicitly via `CLINICAL_LONGFORMER_PATH`).
* One GPU with ≥ 24 GB (UTDE training with batch 2 × 5 notes × 1024 tokens); TabPFN runs on a few GB.

## 1. Shared cohort and structured features (base env, ~1 h)

```bash
python scripts/00_extract_icu_events.py                 # CHARTEVENTS / LABEVENTS / INPUT / OUTPUT subsets -> data/*.parquet (CHARTEVENTS pass ~30 min)
python scripts/07_cohort_flow_counts.py                 # cohort funnel; caches data/labevents_sofa.parquet, data/chartevents_gcs.parquet
python scripts/09_sofa_feature_tiers.py --demographics  # cohort, split, SOFA inputs + component scores, lactate/BE, age/sex
#   -> results/sofa_tiers_demo/features_W{6,24}.parquet (one row per ICU stay: SUBJECT_ID, dataset, y28, 68 features)
#   also fits the XGBoost tiers (GPU, Optuna 50 trials per tier; add --n-trials 5 for a quick pass)
```

`features_W{W}.parquet` is the single source of the cohort and split for everything below (5,757 stays at 6 h, 9,229 at 24 h).

## 2. TabPFN (tabpfn env, ~10 min)

```bash
HF_HOME=... python scripts/10_tabpfn_tiers.py --demographics --windows 6 24
#   -> results/sofa_tiers_demo/tabpfn_metrics.json, tabpfn_preds_W{W}.parquet (val + test probabilities per tier)
```

TabPFN v2 (`n_estimators 8`, no tuning) is fitted in-context on train + val and scored on test; the tier
`4 + lactate + base excess` is the model used downstream.

## 3. UTDE (base env for the data, utde env for training; ~3 h per window on one A6000)

```bash
# 3a. notes and UTDE pickles (base env)
python scripts/12_extract_note_text.py                                  # NOTEEVENTS -> data/noteevents_text.parquet (de-id tokens stripped)
python scripts/13_build_utde_dataset.py --windows 6 24 --out ../MultimodalMIMIC/Data/sofa28
#   21 hourly channels (z-scored with train statistics) + notes available by INTIME+W (STORETIME; radiology by CHARTTIME;
#   no discharge summaries) -> Data/sofa28/W{W}/{train,val,test}p2x_data.pkl, channels.json

# 3b. reference implementation + our adapter
bash utde/setup_utde.sh ../MultimodalMIMIC        # clones XZhang97666/MultimodalMIMIC @ 8e513ca, applies utde/MultimodalMIMIC_sofa28.patch

# 3c. train (original hyper-parameters: TS_Text, bioLongformer, 5 last notes x 1024 tokens, 6 epochs, lr 4e-4 / 2e-5, mTAND, self-cross fusion)
cd ../MultimodalMIMIC
GPU=0 W=24 PY=/path/to/utde-env/python bash run_sofa28.sh      # and W=6
#   -> run/sofa28_W{W}/ihm/TS_Text/.../auc/42.pth.tar (best val-AUROC epoch) and test_logits_42.npz in the same run dir
```

What the adapter changes (7 files, 243-line patch): channel count as an argument (`--orig_d_ts 21`), checkpoint selection
by validation AUROC (`--select_metric auc`), dumping test probabilities, loading Clinical-Longformer from a local snapshot,
list-typed arrays in the pickles, the optional `--term_global_attention` ablation, and the run / attribution scripts.
Model, loss and training loop are untouched.

## 4. Late fusion (tabpfn env, seconds)

```bash
python scripts/14_late_fusion.py --window 24 --utde-run ../MultimodalMIMIC/run/sofa28_W24/ihm/TS_Text/<...>/2e-05_2_3_0.0004_6_8_128_1_2/ --seed 42
python scripts/14_late_fusion.py --window 6  --utde-run ../MultimodalMIMIC/run/sofa28_W6/ihm/TS_Text/<...>/2e-05_2_3_0.0004_6_8_128_1_2/  --seed 42
#   -> results/late_fusion/summary_W{W}.md, metrics_W{W}.json, test_preds_W{W}.parquet
```

Logistic stacker on `[logit p_TabPFN, logit p_UTDE]` fitted on the **validation** split only (UTDE validation logits are
read from the checkpoint), applied to test; paired-bootstrap ΔAUROC against each component.

## 5. Optional: attribution (what each model and the fusion rely on)

```bash
# TabPFN permutation SHAP + exact SHAP of the stacker (tabpfn env; ~5 h per window on a shared GPU, add --max-cases 300 for a draft)
HF_HOME=... python scripts/21_fusion_shap.py --window 24
# UTDE modality / channel occlusion (utde env; ~1 h per window)
cd ../MultimodalMIMIC && GPU=0 W=24 PY=... ATTR_OUT=../mimic-icu-mortality/results/fusion_shap bash run_attr.sh
# figures
python scripts/22_fusion_shap_figs.py --window 6 24      # -> results/fusion_shap/*.png
```

## File map

| step | script | reads | writes |
|---|---|---|---|
| events | `scripts/00_extract_icu_events.py` | `$MIMIC3_RAW/*.csv.gz` | `data/{icustays,chartevents,inputevents_cv,inputevents_mv,outputevents,procedureevents_intub}.parquet` |
| cohort funnel | `scripts/07_cohort_flow_counts.py` | raw + data/ | `data/labevents_sofa.parquet`, `data/chartevents_gcs.parquet`, `results/flowchart/flow_counts.json` |
| features / split / XGB | `scripts/09_sofa_feature_tiers.py --demographics` | raw ICUSTAYS/PATIENTS/ADMISSIONS + data/ | `results/sofa_tiers_demo/features_W{W}.parquet`, `preds_W{W}.parquet`, `metrics.json` |
| TabPFN | `scripts/10_tabpfn_tiers.py --demographics` | features | `results/sofa_tiers_demo/tabpfn_preds_W{W}.parquet`, `tabpfn_metrics.json` |
| notes | `scripts/12_extract_note_text.py` | NOTEEVENTS | `data/noteevents_text.parquet` |
| UTDE data | `scripts/13_build_utde_dataset.py` | features + data/ | `../MultimodalMIMIC/Data/sofa28/W{W}/*.pkl` |
| UTDE model | `utde/setup_utde.sh`, `run_sofa28.sh` (in the adapter) | pickles, Clinical-Longformer | `run/sofa28_W{W}/.../auc/42.pth.tar`, `test_logits_42.npz` |
| fusion | `scripts/14_late_fusion.py` | TabPFN preds + UTDE run dir | `results/late_fusion/*` |
| attribution | `scripts/21_fusion_shap.py`, `utde_attr.py` (adapter), `scripts/22_fusion_shap_figs.py` | the above | `results/fusion_shap/*` |

## Notes and pitfalls

* Keep `results/**/*.parquet`, `*.npz`, `*.csv` with per-stay content and `data/` out of any public copy (PhysioNet DUA); the
  `.gitignore` here already does.
* `xgboost` must be the CUDA-12 wheel (`3.0.5`); 3.1+ wheels target CUDA 13 and silently fall back to CPU.
* The UTDE env must not see user-site packages (`PYTHONNOUSERSITE=1`), and the pickles store plain lists (numpy-2 pickles are
  unreadable by numpy 1.21).
* The fusion stacker must be fitted on validation predictions only; `14` reads the UTDE validation logits from the
  checkpoint to guarantee that.
