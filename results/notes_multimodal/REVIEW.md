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
