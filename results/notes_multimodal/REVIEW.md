# Adding MIMIC-III notes to the SOFA-cohort model — design review

Target model: 28-day mortality from the first W h (W = 6, 24) on the SOFA / lactate / base-excess cohort
(5,757 / 9,229 stays), structured XGBoost / TabPFN at AUROC 0.823 / 0.837 (tier 4 + age/sex).

## 1. Is there text to add? (leak-safe timing: STORETIME <= INTIME + W; radiology by CHARTTIME, it has no STORETIME)

| availability | 6 h cohort | 24 h cohort |
|---|---|---|
| any note, pre-ICU (−48 h .. 0) | 63 % (radiology 63 %, bedside 1 %) | 62 % |
| any note, pre-ICU + window | **91 %** | **96 %** |
| bedside note in window (nursing / physician / respiratory) | 44 % | 72 % |
| physician (admission) note in window — MetaVision only | 12 % | 18 % |
| words per stay, median (p90) | 614 (2,014) | 1,311 (6,625) |
| any note: died vs alive | 0.89 / 0.92 | 0.93 / 0.97 |
| in-window note: CareVue vs MetaVision | 0.82 / 0.74 | 1.00 / 0.87 |

Compared with the whole adult-ICU population (bedside note at 6 h: 4 %, at 12 h: 68 %) this cohort is far better
documented — it is the ABG / full-lab subgroup, i.e. the sickest, most intensively charted patients. Text is a real
modality here, even at 6 h: 43 % have a bedside note and 84 % a radiology report by hour 6.

Pre-ICU text (ED radiology reports: portable CXR, head CT; ED / ward notes) is available at t = 0 for 63 % and is the
same for both windows; it is the cleanest "extra information" because it does not depend on charting speed inside
the ICU.

## 2. Proposed design

Per stay, one text bundle = all eligible notes with availability time <= INTIME + W, ordered by time, each prefixed
with `[category, hours from admission]`, de-identification tokens stripped, discharge summaries / ECG / Echo excluded.

Text representations, cheapest first — run as a ladder and stop when the gain stops:
1. **TF-IDF (1–2-grams, 50 k vocabulary) → logistic regression**, trained on the train split → one text risk score.
   Minutes to run; shows whether there is signal at all.
2. **Frozen clinical encoder** (GatorTron-base or Clinical-Longformer; 512-token chunks, mean-pooled, averaged per
   stay) → 768-d embedding → PCA to 32–64 dims. ~15 k stays × ~2–10 chunks: < 15 min on one GPU. Embeddings go
   straight into XGBoost / TabPFN next to the structured features (early fusion) and into a text-only classifier.
3. **LLM zero-shot risk** (e.g. gpt-oss-120b via vLLM when two GPUs are free; the 7B models were weak on the
   discharge-summary task) → one probability feature. Expensive, mainly interpretability value.
4. Fine-tuning an encoder end-to-end: only if 1–2 show clear signal; ~6 k training stays is on the low side.

Fusion: report **structured only / text only / late fusion / early fusion**.
- Late fusion = logistic stacker on the validation split over (structured probability, text score, has_text).
  Handles missing text explicitly and keeps the structured model untouched.
- Early fusion = text PCs appended to the tier-4 (+ age/sex) feature table; XGBoost handles NaN for stays without text.

Evaluation: same patient split as 09/10; AUROC / AUPRC with bootstrap CI **and a paired bootstrap of the fused − structured
difference** (the expected gain is ~0.01–0.03, below the width of an unpaired CI); subgroup results for stays with /
without a bedside note and by CareVue / MetaVision; operating points at the matched alarm burden (table B).

## 3. Leakage and confounding controls

- Availability time: STORETIME for bedside notes (CHARTTIME lags the actual save by up to 4–7 h for nursing /
  physician notes); radiology has no STORETIME, so use CHARTTIME and, as a sensitivity analysis, require the report
  to be charted >= 1 h before the window end (report finalisation can lag the exam time).
- Time-filter the HADM join per stay (same admission, later ICU stay notes must not enter).
- Note presence is informative (era, unit, acuity): add `has_note`, `n_notes`, `n_words` and the DBSOURCE flag to the
  structured baseline so the text model is credited only for content.
- Goals-of-care language ("DNR", "comfort measures", "family meeting") is legitimately available and strongly
  predictive of 28-day death but partly self-fulfilling; report a sensitivity analysis with those phrases masked.
- Never let discharge summaries in (written after the outcome).

## 4. What gain to expect

Published multimodal results on MIMIC-III early mortality with notes + vitals (Khadanga 2019; Deznabi 2021;
ClinicRealm 2026) add about 0.01–0.04 AUROC over strong structured baselines; the gain shrinks as the structured model
gets stronger and the window longer. Expectation here: 6 h window +0.02–0.04 (structured data are thin, text is already
rich: ED imaging, admission notes), 24 h window +0.01–0.02 (structured model at 0.837 leaves less room). Radiology
text alone should carry most of the pre-ICU signal (pulmonary oedema, consolidation, intracranial haemorrhage).

## 5. Effort

Stage 1 (bundle builder + TF-IDF late fusion, both windows): ~half a day including the leak checks.
Stage 2 (GatorTron embeddings, early fusion into XGB / TabPFN, subgroup and paired-bootstrap reporting): one more day.
Reuse: `my_datasets/mimic-iii/build_mimic3_note_label.py` (de-id stripping, section parsing) and
`mimic_icu/data/noteevents_meta.parquet` (timing metadata already extracted).

## 6. Reference papers (baselines for this experiment)

Structured + notes (MIMIC-III, ICU mortality) — the direct multimodal baselines
- Khadanga S, Aggarwal K, Joty S, Srivastava J. Using clinical notes with time series data for ICU management. EMNLP-IJCNLP 2019, pp. 6432–6437. https://aclanthology.org/D19-1678/ (code: github.com/kaggarwal/ClinicalNotesICU)
- Deznabi I, Iyyer M, Fiterau M. Predicting in-hospital mortality by combining clinical notes with time-series data. Findings of ACL 2021. https://aclanthology.org/2021.findings-acl.352/ (code: github.com/Information-Fusion-Lab-Umass/ClinicalNotes_TimeSeries)
- Zhang D, Yin C, Zeng J, Yuan X, Zhang P. Combining structured and unstructured data for predictive models: a deep learning approach. BMC Med Inform Decis Mak 2020;20:280. https://doi.org/10.1186/s12911-020-01297-6
- Lyu W, Dong X, Wong R, Zheng S, Abell-Hart K, Wang F, Chen C. A multimodal transformer: fusing clinical notes with structured EHR data for interpretable in-hospital mortality prediction. AMIA Annu Symp Proc 2022:719–728. https://pmc.ncbi.nlm.nih.gov/articles/PMC10148371/
- Zhu Y, et al. ClinicRealm: re-evaluating large language models with conventional machine learning for non-generative clinical prediction tasks. npj Digit Med 2026;9:319. (structured EHR + notes, LLM vs ML; `src/multimodal/` fusion code)

Notes only
- van Aken B, et al. Clinical outcome prediction from admission notes using self-supervised knowledge integration. EACL 2021. https://arxiv.org/abs/2102.04110 (admission-note construction used for our MIMIC-III note task)

Structured-only benchmarks
- Harutyunyan H, Khachatrian H, Kale DC, Ver Steeg G, Galstyan A. Multitask learning and benchmarking with clinical time series data. Sci Data 2019;6:96. (MIMIC-III in-hospital mortality from the first 48 h; LSTM AUROC ≈ 0.86)

Clinical score baselines (SOFA, lactate, base excess)
- Vincent JL, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Med 1996;22:707–710.
- Raith EP, Udy AA, Bailey M, et al. Prognostic accuracy of the SOFA score, SIRS criteria, and qSOFA score for in-hospital mortality among adults with suspected infection admitted to the ICU. JAMA 2017;317(3):290–300. (SOFA crude AUROC 0.75; adjusted 0.815)
- Smith I, Kumar P, Molloy S, et al. Base excess and lactate as prognostic indicators for patients admitted to intensive care. Intensive Care Med 2001;27:74–83. (admission BE AUROC 0.73, lactate 0.78; BE < −4 & lactate > 1.5 → sens 80 %, spec 59 %)
- Follow-up: Base excess and lactate as prognostic indicators for patients admitted to intensive care — 15 years later. https://pmc.ncbi.nlm.nih.gov/articles/PMC4797108/

Tabular transformer
- Hollmann N, et al. Accurate predictions on small data with a tabular foundation model (TabPFN v2). Nature 2025.

### Newer architecture baselines (top-tier venues first)

| paper | venue | data / task | what it fuses | code |
|---|---|---|---|---|
| Zhang X, Li S, Chen Z, Yan X, Petzold L. Improving medical predictions by irregular multimodal EHR modeling (UTDE) | **ICML 2023** | MIMIC-III, 48 h in-hospital mortality + phenotyping (Harutyunyan cohort) | irregular time series (gated imputation/interpolation embeddings) + sequence of note embeddings as irregular series, interleaved cross-time attention | github.com/XZhang97666/MultimodalMIMIC |
| Xu M, et al. FlexCare: leveraging cross-task synergy for flexible multimodal healthcare prediction | **KDD 2024** | MIMIC-IV (+ MIMIC-CXR), several tasks, incomplete modalities | time series + notes (+ images), task-guided mixture of experts, asynchronous multi-task | github.com/mhxu1998/FlexCare |
| Zhu Y, et al. EMERGE: enhancing multimodal EHR predictive modeling with retrieval-augmented generation | CIKM 2024 | MIMIC-III / IV, in-hospital mortality, 30-day readmission | time series + notes + LLM/RAG patient summary, cross-attention fusion; compares against UTDE | github.com/yhzhu99/EMERGE |
| King R, Yang T, Mortazavi BJ. Multimodal pretraining of medical time series and notes | ML4H 2023 | MIMIC-III, mortality + phenotyping, few labels | contrastive + masked pretraining aligning measurements and notes; large gains when <= 1–10 % labelled | github.com/kingrc15/multimodal-clinical-pretraining |
| Bakumenko A, Hoelscher J, Smith H. Transparent early ICU mortality prediction with clinical transformer and per-case modality attribution | arXiv 2511.15847 (2025) | MIMIC-III, 48 h | late fusion: logistic stacker over BiLSTM (vitals) + fine-tuned ClinicalModernBERT (notes); AUROC 0.891 vs 0.876 best single modality | – |
| Ben Cheikh Larbi I, et al. Integrating text and time-series into (large) language models to predict medical outcomes | BioCreative IX 2025 (arXiv 2509.13696) | MIMIC-III, 48 h, admission notes + 13 features | instruction-tuned Llama 3.1 with DSPy prompt optimisation vs BioClinicalBERT; "comparable to specialised multimodal systems" | – |

Reading: every recent paper still benchmarks on the 48 h / in-hospital-mortality Harutyunyan cohort and uses **UTDE (ICML 2023)** as the
reference multimodal architecture; NeurIPS / ICLR 2024–25 work has moved to MIMIC-IV foundation models and LLM + time-series, not to
a new notes-plus-vitals fusion baseline. For this project: adopt UTDE as the "published architecture" comparison (its code takes an
hourly series + a list of timed notes, which is what the bundle builder produces), and the Bakumenko-style late fusion as the simple
strong baseline — which is the design proposed in §2. Numbers from these papers are not transferable (48 h window, in-hospital
death, ~21 k unselected stays vs our 6 / 24 h, 28-day death, SOFA-selected cohort); only the architectures are.
