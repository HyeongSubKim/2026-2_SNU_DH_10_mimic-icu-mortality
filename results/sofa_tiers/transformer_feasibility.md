# Transformer models for the SOFA / lactate / base-excess cohort — feasibility

Setting: 5.8 k (6 h) / 9.2 k (24 h) ICU stays, ~60 aggregated tabular features, 28-day mortality (26–29 % positive),
patient-level split. Train sets are 4.1 k / 6.5 k rows.

| option | what it is | fit for this problem | status here |
|---|---|---|---|
| **TabPFN v2** (Hollmann et al., Nature 2025) | tabular transformer pre-trained on synthetic data; in-context learning, no training or tuning; handles NaN | **Best fit**: designed for <= 10 k rows x 500 features, which is exactly this cohort; strongest published small-data tabular baseline; one forward pass per tier (seconds) | installed (`tabpfn` 9.2.0), open v2 checkpoint from Hugging Face verified on GPU; `10_tabpfn_tiers.py` evaluates the same tiers/splits. The newer v2.5/v3 weights need a Prior Labs licence token |
| FT-Transformer / SAINT / TabTransformer (Gorishniy 2021, Somepalli 2021) | transformer trained from scratch on tabular rows (feature tokenisation + attention) | Feasible but unlikely to beat GBDT: in the benchmarks they match GBDT only with >= 10–50 k rows and need HP search; at 4–6 k rows they typically trail XGBoost by 0.01–0.03 AUROC | not run; `rtdl` package, ~1 h of GPU for HP search if wanted |
| Sequence transformer on the hourly series (STraTS, Transformer-encoder over the first W h) | model the raw time series instead of window aggregates | 6 h window = 6 time steps: too short to benefit; 24 h window plausible, but MIMIC-III benchmarks (Harutyunyan 2019 and follow-ups) show RNN/transformer sequence models ≈ GBDT on aggregated features for in-hospital mortality, and the training set here is ~20x smaller than those | not recommended as a first step; reconsider only with the full 50 k-stay cohort |
| EHR foundation models (CLMBR / MOTOR / EHRSHOT) | pre-trained on millions of longitudinal records, fine-tuned per task | require OMOP-formatted data and the pre-trained weights; MIMIC-III -> OMOP conversion is a multi-day effort; gains reported mostly for long-horizon, low-data tasks | out of scope for this project |

Recommendation: report XGBoost as the primary model and TabPFN v2 as the transformer comparison (same tiers, same
splits, zero tuning — a fair "can a transformer do better on this cohort?" test). If TabPFN matches or beats XGBoost,
an FT-Transformer run is the natural next step; if it does not, a trained-from-scratch transformer will not either.
