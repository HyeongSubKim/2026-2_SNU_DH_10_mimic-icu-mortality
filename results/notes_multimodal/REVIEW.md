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

### Field trends at top-tier venues (2023–2026), for context

1. **Structured-EHR foundation models (event-stream / "patient timeline" pretraining)** — CLMBR-T-base + EHRSHOT (NeurIPS 2023 D&B),
   MOTOR time-to-event FM (ICLR 2024), Context Clues long-context FMs, Mamba 16k tokens (ICLR 2025), ETHOS zero-shot trajectory
   (npj Digit Med 2024), CEHR-GPT, "Generative medical event models improve with scale" (2025). Pretrain on millions of patients'
   coded events, few-shot / zero-shot downstream. Mostly structured codes; notes enter late.
2. **Benchmarks and data standards** — YAIB (ICLR 2024; dataset/cohort/preprocessing matter more than model class), MEDS + MEDS-DEV
   (decentralised, reproducible), DiReCT (MIMIC-IV multimodal), EHRSHOT. The 48 h in-hospital-mortality Harutyunyan task is being
   replaced by MIMIC-IV / multi-centre, many-task suites.
3. **LLM + time series** — time series as a native LLM modality: OpenTSLM (ICML 2026), "Portable representations for clinical time
   series with LLMs" (ICLR 2026: frozen LLM summarises the series into text, frozen text encoder embeds it), MILM, TRACE, LLM4EHR;
   instruction-tuned LLMs on notes + features (Ben Cheikh Larbi 2025). Emphasis on zero-/few-shot, cross-cohort portability and
   reasoning traces rather than a new fusion layer.
4. **Multimodal FMs** — autoregressive EHR FMs conditioned on ECG / CXR / notes via latent compression + gated cross-attention
   (2026 preprint); DiReCT reports that adding modalities does not automatically help ICU mortality.
5. **Where notes + vitals fusion sits now** — UTDE (ICML 2023) remains the reference architecture; newer fusion papers appear at
   KDD / CIKM / ML4H / AMIA (FlexCare, EMERGE, King 2023) rather than NeurIPS / ICLR / ICML.

Implication for this project: a novel fusion architecture is not where the field's attention is; a careful clinical question
(SOFA cohort, 6 h vs 24 h, 28-day death, lactate / BE, leak-safe note timing, matched-alarm-burden operating points) with strong
simple models (XGBoost, TabPFN, late fusion) and one published architecture (UTDE) as comparison is the defensible design.

### 7. Applying the ICLR 2026 "Record2Vec" recipe (Ji, Sun, Amaral, Goldenberg, Krishnan) to this cohort

What the paper does: 48 h irregular ICU window → canonical text serialisation → frozen LLM (Gemini-2.0 Flash / MedGemma /
Llama-3.1) writes a concise summary → frozen text embedder (Qwen3-embedding) → fixed vector → downstream predictor
(PatchTSMixer / MLP). Cohorts MIMIC-IV 57 k, HiRID 32 k, PPICU 39 k stays; tasks: forecasting, LoS, in-hospital mortality,
vasopressor / antibiotic use, lab ordering. Baselines: grid imputation + neural nets, TSDE (SSL), TimesFM (TS foundation
model), GenHPF. **No GBDT / tabular baseline on clinician-curated features.** In-distribution it wins 13/15 columns
(MIMIC-IV mortality AUROC 0.888); its real claims are cross-site transfer (10/10 columns; HiRID→PPICU mortality 0.72 vs
0.64 TimesFM) and few-shot (16 target labels ≈ full in-distribution model). Ablations: raw serialisation embedded
directly (no summary) is often best in-distribution, summaries win under shift; prompt style changes variance more than mean.
Code: github.com/Jerryji007/Record2Vec-ICLR2026.

Evaluation plan if we adopt it (all on the SOFA cohort, W = 6 / 24 h, 28-day death, same patient split):
1. Conditions — A structured XGBoost / TabPFN (existing, 0.823 / 0.837); B Record2Vec-style: LLM summary of the first W h
   → text embedding → LR / XGB / TabPFN; B0 control: raw serialisation embedded without summary; C hybrid: embedding PCs +
   structured features; D unified text: vitals summary + clinical notes through the same embedder (the multimodal variant).
2. In-distribution — test AUROC / AUPRC with **paired** bootstrap of each condition minus A; calibration (Brier, slope) because
   embedding-based classifiers are often less calibrated; matched-alarm-burden operating point (table B).
3. Shift and few-shot — this is where the method is supposed to win, and MIMIC-III gives a natural pseudo-site split:
   **train on CareVue stays (2001–08), test on MetaVision (2008–12) and vice versa** (different charting system, item IDs,
   documentation habits); report absolute AUROC and the relative drop vs in-distribution; few-shot = source model + 16 / 64 /
   512 target labels. Expect the structured model's drop to be larger than Record2Vec's.
4. Robustness of the text path — ≥ 3 summarisation runs (temperature 0 still varies with prompt wording), report mean ± sd;
   compare 2 prompt styles (structured trend summary vs free narrative).
5. Leakage — the summariser sees only data from the first W h; no outcome, no later labs; keep the same STORETIME rules for
   notes in condition D.
Cost: ~15 k windows × ~400 input / ~150 output tokens; local vLLM (gpt-oss-120b when two GPUs are free, or Qwen2.5-7B /
MedGemma-4B on one GPU) — a few hours; embedding minutes.

Expected outcome, stated before running:
- In-distribution: Record2Vec ≈ or slightly below the tuned tabular models (−0.01 … −0.03 AUROC). Reasons: our features are
  already clinician-curated worst / last / count aggregates of the same variables; GBDT / TabPFN are hard to beat on ~5 k rows
  (YAIB, Purushotham 2018); a text summary discards numeric precision; the paper's own baselines were neural nets on grid
  imputation, which are known to trail GBDT on tabular ICU data, and its downstream models were trained on 10× more samples.
- Hybrid C: +0.00 … +0.01 over A — a cheap check worth running.
- CareVue→MetaVision transfer and few-shot: this is the plausible win; a smaller relative drop would be a publishable
  finding even if in-distribution AUROC is equal.
- The real payoff is condition D: one frozen text encoder handles the vitals summary and the clinical notes, so the
  multimodal experiment collapses into a single pipeline with ~25× fewer tokens than raw serialisation.

### 8. Stage 3 — an agent-based variant (ML → LLM representation → agent)

Definition used here: an **agent** = an LLM that, given a goal and a set of tools, **plans, calls tools, observes the results
and iterates** before committing to an answer (ReAct-style loop), with state/memory across steps. A single frozen LLM call
(Record2Vec's summariser, ClinicRealm's zero-shot prompt) is *not* an agent; pasting tool outputs into one prompt is
"tool-augmented prompting", not an agent either. Autonomy over *which information to gather and when* is the defining property.

Reference systems (2024–26): ColaCare (WWW 2025; DoctorAgents wrap domain ML models + MetaAgent synthesises; MIMIC-IV
mortality AUROC 88.8 / AUPRC 54.1, AUPRC +0.9–4.5 % over the best domain model), EHRAgent (EMNLP 2024; code-generating
agent over EHR tables), AgentMD (npj Digit Med 2025; 2,164 clinical calculators as tools; 87.7 % vs 40.9 % GPT-4 accuracy on
risk calculation), AgentClinic (npj Digit Med 2026; sequential test ordering under incomplete information — accuracy
collapses relative to static QA), MedAgentBench (FHIR environment; best agent 69.7 % task success).

What to add for a defensible Stage 3 on this cohort:
1. **Tool layer, time-gated.** Every tool exposes only data with availability time ≤ INTIME + W: (a) structured risk model
   (XGBoost / TabPFN probability + SHAP top factors), (b) SOFA / lactate / BE calculators, (c) note retriever (bundle from §2),
   (d) Record2Vec summary of the vitals, (e) guideline / literature RAG (optional), (f) a "request additional measurement"
   tool that returns a value only if it was actually measured within the window.
2. **Orchestration.** One agent (ReAct, LangGraph or DSPy) with a fixed budget of steps; optional MDT-style multi-agent
   (ColaCare) — but add that only after the single agent is evaluated, otherwise the ablation is uninterpretable.
3. **Output contract.** A probability in [0, 1] plus a structured rationale (which tools, which evidence), so AUROC / AUPRC /
   calibration are computed exactly as for the ML rows.
4. **Baselines that isolate the agency.** (i) LLM zero-shot on the serialised window (no tools); (ii) tool-augmented single
   prompt (all tool outputs pasted once); (iii) the agent loop. The difference (iii) − (ii) is the value of *deciding what to
   look at*; (ii) − structured XGB is the value of the LLM as integrator.
5. **A task where agency can matter: active acquisition.** Reframe the lactate / BE question as a decision: at hour W the
   agent may "order" up to k extra tests from the measured-but-hidden set; score prediction quality vs number of tests ordered
   (value-of-information curve) against random / always-all / never policies. This is the one setting where an agent can beat
   a fixed-input model, because the ML model cannot choose its inputs.
6. **Evaluation protocol.** Same cohort, split and test set; paired bootstrap vs the structured model; calibration; matched
   alarm-burden operating points; **run-to-run action reliability** (repeat each case ≥ 3 times; agents are known to change tool
   sequences between runs); cost (tokens, tool calls, latency) per case; CareVue→MetaVision transfer as in §7; trajectory audit
   on a random sample of 100 cases for hallucinated evidence and leakage (did any tool return post-window data).
7. **Guardrails.** Deterministic decoding, tool schemas with explicit time filters, no free-form SQL on raw tables (EHRAgent-style
   code execution is a leakage risk), log every tool call.

Expected outcome: for the fixed-input 28-day prediction the agent should match, not beat, XGBoost / TabPFN (ColaCare's gains
are ~1–4 % AUPRC with much larger cohorts); the credible wins are (a) the active-acquisition curve, (b) evidence-grounded
rationales, (c) smaller transfer drop. Cost: ~15 k cases × (4–8 tool calls, ~3–6 k tokens) — a day of local vLLM time per
configuration; development ≈ 1–2 weeks.

### 9. Frozen TabPFN as a tabular encoder inside a UTDE-style fusion — feasibility check (W = 24 h, tier 4 + age/sex)

`TabPFNClassifier.get_embeddings(X, data_source)` returns a 192-d representation per row and per ensemble member
(shape n_estimators × n_rows × 192). Checked on our split (context = train split, 4 estimators):
- TabPFN itself: test AUROC 0.834.
- Logistic probe fitted on **query-mode** embeddings of the validation rows → test: 0.829–0.837 per estimator; 5-fold CV probe
  inside the test embeddings 0.829. The frozen embedding keeps essentially all of TabPFN's signal.
- Pitfalls: (1) embeddings of rows that are *inside the context* (`data_source='train'`) live in a different space for some
  estimators (probe train→test 0.73–0.83; naive mean over estimators 0.33) — always embed rows in query mode against a fixed
  context, and embed training rows with a K-fold context that excludes them; (2) do not average embeddings across estimators
  (each member permutes features/classes) — keep them per estimator or concatenate; (3) embeddings are deterministic across calls.
Design that follows: frozen TabPFN embedding (static, per stay) as the tabular branch; UTDE's note encoder (BERT + time
attention) as the text branch; concatenation or cross-attention + MLP head trained on train rows. Ladder to report:
TabPFN alone → logistic on frozen embedding → embedding + note branch → full UTDE; and the reverse direction (UTDE embedding
appended to the TabPFN feature table) as the alternative fusion.

### 10. Stage 1 result: UTDE reproduction + TabPFN/UTDE late fusion (done)

Validation AUROC by epoch — W=24: 0.769, 0.786, 0.788, 0.803, 0.810 (best, epoch 5); W=6: 0.720 → 0.735 → 0.764 (epoch 4), 0.751 (epoch 5).
Test: 6 h — TabPFN 0.832, UTDE 0.798, fusion 0.845 (+0.014 vs TabPFN [−0.002, +0.029]); 24 h — TabPFN 0.837, UTDE 0.844,
fusion 0.860 (+0.023 [+0.011, +0.035]). Stacker weights ≈ equal (0.41 / 0.43 at 24 h), i.e. the two models carry
different information. Compute: ~4.5 h (24 h) / ~5.5 h (6 h) on one A6000 (shared). Artefacts: ../MultimodalMIMIC/run/sofa28_W{6,24},
results/late_fusion/. Next: Record2Vec (Stage 2) plugged into the same fusion table.

### 11. Stage 3 result: Record2Vec on the SOFA cohort (done; details in README)

Run as planned in §7 (local Qwen2.5-7B-Instruct summariser, Qwen3-Embedding-0.6B / 4B and GatorTron embedders, LR / TabPFN
heads, conditions A / B0 / B (two prompts) / C / F / D, few-shot 16–1024, CareVue↔MetaVision transfer). Outcome against the
pre-stated expectations:
- In-distribution: worse than expected — Record2Vec summary embedding 0.744 / 0.699 (6 / 24 h) vs structured TabPFN
  0.832 / 0.837; the prediction was −0.01…−0.03, the result is −0.09…−0.14. Raw-serialisation embedding (0.76) beats the
  summary, consistent with the paper's own ablation ("raw often best in-distribution").
- Hybrid C / late fusion F: 0 or slightly negative (predicted +0.00…+0.01) — the text path carries no information the
  structured features lack.
- Transfer and few-shot: no win. The structured model drops ≤ 0.01 across charting eras, so the shift the method is meant
  to absorb is absent in MIMIC-III; in the 16-label regime TabPFN already matches the probes.
- Condition D (record + notes through one summariser): 0.763 / 0.798 (4B) and 0.768 / 0.807 (GatorTron) at 6 / 24 h — the notes
  add 0.02–0.10 over record-only summaries, but fusion with TabPFN gives only +0.002 / +0.010 (ns) vs +0.014 / +0.023 for UTDE.
- Term dictionary in the text path: a prompt-level checklist (D①) lifts the 24 h text model to 0.816 / fusion 0.849 (+0.012, ns),
  the best Record2Vec variant; filtering the notes down to dictionary sentences (D②) costs 0.02–0.04 — same lesson as the UTDE
  attention ablation: the terms are a completeness checklist, not a focus restriction.
Decision: Record2Vec is a negative result on this cohort; keep TabPFN + UTDE late fusion (0.845 / 0.860) as the Stage 1–2
reference and treat the LLM-text path as a candidate for the agent stage (§8) only where it adds information the
structured features do not have (notes), not as a replacement representation for the vitals.

### 12. Stage 3 experiment plan — agent-based prediction on the SOFA cohort (proposal, informed by §10–11)

What Stages 1–2 established, and what it implies for the agent design:
- The clinician-curated structured features + TabPFN/XGB are the strongest single representation (0.832 / 0.837) and an LLM
  text representation of the vitals cannot replace them (−0.08 … −0.14) → the agent must **call the structured model as a
  tool**, not re-derive risk from text.
- Notes carry information the vitals lack (UTDE fusion +0.023 at 24 h; note-concept dictionary +0.010; Record2Vec-D
  +0.06–0.10 over vitals-only text), but compressing them (summary → embedding) loses most of it → the agent's note access
  should be **question-driven retrieval** over the raw notes (ask "is there a goals-of-care / arrest / shock statement?"),
  not a one-shot summary.
- Term attention / term prompts change nothing when the model already sees the whole note → the dictionary's role in the
  agent is as a **checklist of questions**, not as an attention prior.
- There is no CareVue→MetaVision shift to exploit (≤ 0.01 drop) → transfer is not a selling point on MIMIC-III; the agent's
  credible wins are (a) choosing what to look at (active acquisition), (b) integrating heterogeneous evidence per case,
  (c) auditable rationales.

Experiments (all: same cohort / split / 28-day label; W = 6 and 24 h; test n = 853 / 1,374; paired bootstrap vs TabPFN):

| # | experiment | question | design | primary metric | decision gate |
|---|---|---|---|---|---|
| E0 | Tool layer + harness | can every tool be time-gated and logged? | tools: `risk_model(TabPFN prob + SHAP top-5)`, `sofa()/lactate()/be()`, `vitals_trend(channel)`, `note_search(query, k)` (BM25 + concept regex over notes with STORETIME ≤ INTIME + W; L01 terms masked), `order_test(name)` (returns value only if measured in window), `calculator(apache/saps)` | leakage audit: 0 post-window values in 100 traced cases | must pass before E1 |
| E1 | Agency ablation | is the ReAct loop worth more than one prompt? | (i) zero-shot on serialised window (have: 0.65 on ClinicRealm notes), (ii) tool-augmented single prompt (all tool outputs pasted once), (iii) ReAct agent, ≤ 8 steps, deterministic decoding, 3 repeats per case | AUROC / AUPRC; Δ(iii − ii) = value of deciding what to look at; action-sequence agreement across repeats | if (iii) ≤ (ii) on both windows → drop the loop, keep (ii) as "LLM integrator" |
| E2 | Agent vs best ML | does the agent beat TabPFN + UTDE fusion (0.845 / 0.860)? | agent of E1(iii); plus stacker of agent probability with TabPFN and UTDE (fit on val, as `14_late_fusion.py`) | paired ΔAUROC vs TabPFN and vs fusion; calibration slope / Brier; matched-alarm sensitivity (table B) | report regardless; expected: agent ≈ TabPFN, agent-in-stack +0.00…+0.01 |
| E3 | Active acquisition (the bet) | can choosing tests beat a fixed-input model? | hide lactate / BE / bilirubin / platelets / GCS at hour W; agent may `order_test` ≤ k ∈ {0,1,2,3}; compare to TabPFN with (a) nothing hidden, (b) all hidden, (c) random k tests, (d) greedy SHAP-ranked k tests | AUROC vs number of tests ordered (value-of-information curve); tests ordered per case; share of orders that return "not measured" | agent curve above random and SHAP-greedy at k = 1–2 = positive result; else negative |
| E4 | Note interrogation | does question-driven note reading recover UTDE's note gain? | agent restricted to `note_search` + `risk_model`; checklist = N01–N50 as candidate queries; output: evidence table (concept, quote, note time) + probability | Δ vs TabPFN; vs UTDE fusion; precision of quoted evidence on 100 audited cases | ≥ +0.015 at 24 h with ≥ 90 % evidence precision → notes path worth keeping |
| E5 | Multi-agent (optional) | ColaCare-style DoctorAgents (vitals / labs / notes) + MetaAgent | only if E1(iii) > E1(ii) | same as E2; cost per case | skip if E1 negative |

Models and cost: Qwen2.5-7B-Instruct / Qwen3-32B (local vLLM) as the agent LLM, gpt-oss-120b when two GPUs are free;
≈ 15 k cases × 3 repeats × ~5 tool calls × ~4 k tokens ≈ 1 B tokens ≈ 1–2 days of vLLM per configuration at 7B, so
E1–E2 on the **test split only** (2,227 cases) for the ablation, full train/val only where a stacker is fitted.
Implementation: LangGraph (or a plain ReAct loop) over the existing scripts — `09`/`10` (risk model, SHAP), `12`/`13`
(note text, STORETIME gating), `15` (concept regex), `16` (serialisation); tool calls logged to JSONL for the audit.
Timeline: E0 2–3 days, E1–E2 1 week, E3 1 week, E4 3–4 days; total ≈ 3–4 weeks of wall-clock with the GPUs shared.

Pre-registered expectations: E1(iii) − E1(ii) ≈ 0 for the fixed-input task; E2 agent ≈ TabPFN (−0.02 … +0.01), stack
+0.00 … +0.01; E3 is the only place a clear win is plausible (k = 1–2 tests recovering most of the all-tests AUROC);
E4 ≈ +0.01 with usable rationales. A negative E1/E2 with a positive E3/E4 is still a publishable story: "agency pays when
the model can choose its inputs or must cite evidence, not when the inputs are fixed".
