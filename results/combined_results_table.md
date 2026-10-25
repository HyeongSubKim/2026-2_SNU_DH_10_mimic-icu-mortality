# Combined results — SOFA cohort, 28-day mortality, test set (6 h n = 853, 24 h n = 1,374)

AUROC (AUPRC); Δ = paired-bootstrap ΔAUROC vs TabPFN v2 (tier 4 + age/sex) [95 % CI] unless marked.

| Stage | Method | Modality | 6 h AUROC (AUPRC) | Δ vs TabPFN | 24 h AUROC (AUPRC) | Δ vs TabPFN |
|---|---|---|---|---|---|---|
| 0 | SOFA score alone (raw score) | Unimodal (structured) | 0.650 (.443) | −0.183 [−0.222, −0.144] | 0.657 (.428) | −0.180 [−0.213, −0.146] |
| 1 | XGBoost, tier 4 + age/sex | Unimodal (structured) | 0.823 (.662) | −0.009 [−0.022, +0.005] | 0.837 (.660) | +0.000 [−0.009, +0.010] |
| 1 | **TabPFN v2, tier 4 + age/sex (baseline)** | Unimodal (structured) | **0.832 (.681)** | – | **0.837 (.673)** | – |
| 1 | XGBoost + note-concept dictionary N01–N50 (raw counts) | Multimodal (structured + note terms) | 0.830 (.673) | +0.007 [−0.007, +0.022] † | 0.847 (.686) | +0.010 [+0.001, +0.019] † |
| 1 | UTDE (ICML 2023) | Multimodal (time series + notes) | 0.798 (.618) | −0.034 [−0.066, −0.003] | 0.844 (.680) | +0.007 [−0.013, +0.026] |
| 1 | UTDE + term-guided global attention (N01–N50) | Multimodal | – | – | 0.843 (.687) | +0.006 ‡ |
| 1 | **Late fusion TabPFN + UTDE** | Multimodal | **0.845 (.704)** | +0.014 [−0.002, +0.029] | **0.860 (.707)** | **+0.023 [+0.011, +0.035]** |
| 1 | Late fusion TabPFN + UTDE (term attention) | Multimodal | – | – | 0.857 (.706) | +0.020 ‡ |
| 1 | **TabPFN + vital-sign aggregates (113 feats)** — fair structured baseline | Unimodal (structured) | 0.842 (.705) | +0.010 [−0.003, +0.024] | 0.852 (.684) | +0.015 [+0.007, +0.023] |
| 1 | UTDE, TabPFN's variable set (15 ch) + notes | Multimodal | 0.823 (.658) | −0.009 [−0.031, +0.013] | 0.826 (.658) | −0.011 [−0.025, +0.002] |
| 1 | Late fusion TabPFN + UTDE (15 ch) | Multimodal | 0.839 (.688) | +0.008 [−0.008, +0.023] | 0.841 (.680) | +0.003 [−0.002, +0.009] |
| 1 | Late fusion TabPFN+vitals + UTDE (21 ch) | Multimodal | 0.844 (.710) | +0.012 [−0.009, +0.032] | 0.863 (.709) | +0.026 [+0.013, +0.039] |
| 1 | **TabPFN-in-UTDE, 3 streams (TabPFN tokens + TS + notes)** | Multimodal | 0.836 (.700) | +0.004 [−0.012, +0.020] | **0.868 (.731)** | **+0.031 [+0.015, +0.046]** |
| 2 | Record2Vec B0: raw serialisation → Qwen3-Emb-4B → LR (no summary) | Unimodal (text of vitals) | 0.757 (.551) | −0.075 [−0.100, −0.050] | 0.760 (.541) | −0.078 [−0.102, −0.054] |
| 2 | Record2Vec B: LLM summary (narrative) → Qwen3-Emb-4B → LR | Unimodal (text of vitals) | 0.744 (.542) | −0.088 [−0.121, −0.056] | 0.699 (.474) | −0.138 [−0.165, −0.111] |
| 2 | Record2Vec B: LLM summary (schema) → Qwen3-Emb-4B → LR | Unimodal (text of vitals) | 0.728 (.530) | −0.105 [−0.137, −0.072] | 0.736 (.508) | −0.101 [−0.127, −0.075] |
| 2 | Record2Vec D: LLM summary of record + notes → Qwen3-Emb-4B → LR | Multimodal (unified text) | 0.763 (.543) | −0.069 [−0.103, −0.037] | 0.798 (.594) | −0.039 [−0.062, −0.017] |
| 2 | Record2Vec D: LLM summary of record + notes → GatorTron → LR | Multimodal (unified text) | 0.768 (.575) | −0.064 [−0.096, −0.031] | 0.807 (.624) | −0.030 [−0.053, −0.007] |
| 2 | Hybrid TabPFN (structured + 32 PCs of D embedding, GatorTron) | Multimodal | 0.832 (.681) | +0.000 [−0.009, +0.009] | 0.840 (.677) | +0.003 [−0.003, +0.009] |
| 2 | Late fusion TabPFN + Record2Vec D (GatorTron) | Multimodal | 0.834 (.688) | +0.002 [−0.013, +0.017] | 0.847 (.696) | +0.010 [−0.002, +0.022] |
| 2 | Record2Vec D①: term-guided prompt (N01–N50 checklist, raw notes) → Qwen3-Emb-4B → LR | Multimodal (unified text) | 0.787 (.603) | −0.045 [−0.078, −0.015] | 0.816 (.625) | −0.021 [−0.044, +0.002] |
| 2 | Late fusion TabPFN + Record2Vec D① | Multimodal | 0.835 (.688) | +0.003 [−0.015, +0.019] | 0.849 (.703) | +0.012 [−0.001, +0.025] |
| 2 | Record2Vec D②: term-filtered notes (dictionary sentences only) → Qwen3-Emb-4B → LR | Multimodal (unified text) | 0.767 (.589) | −0.065 [−0.096, −0.034] | 0.775 (.547) | −0.062 [−0.086, −0.039] |
| 2 | Late fusion TabPFN + Record2Vec D② | Multimodal | 0.831 (.691) | −0.000 [−0.014, +0.013] | 0.837 (.676) | −0.000 [−0.010, +0.009] |

† Δ vs XGBoost structured (same features, `15_note_concepts.py`), not vs TabPFN.
‡ point difference only; the paired CI was computed against the un-modified UTDE / fusion run: −0.000 [−0.010, +0.010] and −0.003 [−0.009, +0.004].

Sources: `results/sofa_tiers_demo/`, `results/note_concepts/`, `results/late_fusion/`, `results/record2vec/`.
