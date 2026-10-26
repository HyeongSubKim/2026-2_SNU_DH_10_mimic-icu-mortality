"""Draft slide deck: Previous works (TabPFN / UTDE) -> Proposed methods (late fusion, TabPFN-in-UTDE) -> Results (quantitative, SHAP).
Usage: python scripts/27_make_slides.py  -> results/slides/MIMIC_TabPFN_UTDE_fusion_draft.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
FONT = 'Malgun Gothic'; NAVY = RGBColor(0x1F, 0x3A, 0x5F); GREY = RGBColor(0x59, 0x59, 0x59); ACC = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2E, 0x8B, 0x57); PURPLE = RGBColor(0x6A, 0x5A, 0xCD); LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def txt(slide, x, y, w, h, text, size=14, bold=False, color=GREY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        lvl = 0
        while line.startswith('  '):
            line = line[2:]; lvl += 1
        p.level = lvl
        r = p.add_run(); r.text = ('• ' if (isinstance(text, list) and not line.startswith('—') and lvl == 0 and len(lines) > 1) else ('– ' if lvl else '')) + line
        r.font.size = Pt(size - 2 * lvl); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
    return tb


def header(slide, section, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.9)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    txt(slide, 0.4, 0.08, 9.5, 0.4, section, 12, False, RGBColor(0xC8, 0xD3, 0xE6))
    txt(slide, 0.4, 0.36, 12.4, 0.5, title, 22, True, RGBColor(0xFF, 0xFF, 0xFF))


def box(slide, x, y, w, h, text, fill=LIGHT, color=NAVY, size=11, bold=False, line=NAVY):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(1); tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line_ in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = line_; r.font.size = Pt(size); r.font.bold = bold and i == 0; r.font.color.rgb = color; r.font.name = FONT
    return s


def arrow(slide, x1, y1, x2, y2, color=GREY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); c.line.color.rgb = color; c.line.width = Pt(1.5)
    c.line._get_or_add_ln().append(c.line._get_or_add_ln().makeelement('{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd', {'type': 'triangle'}))
    return c


def table(slide, x, y, w, rows, col_w=None, size=11, bold_rows=(), h=None):
    nr, nc = len(rows), len(rows[0]); shape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h or 0.36 * nr)); t = shape.table
    if col_w:
        for j, cw in enumerate(col_w): t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ''; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val); r.font.size = Pt(size); r.font.name = FONT
            r.font.bold = (i == 0) or (i in bold_rows); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else GREY
            c.fill.solid(); c.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.02)
    return t


def pic(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


def foot(slide, note):
    txt(slide, 0.4, 7.05, 12.5, 0.35, note, 9, False, RGBColor(0x8A, 0x8A, 0x8A))


# ---------------------------------------------------------------- 0. title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
txt(s, 0.8, 2.2, 11.5, 1.2, 'ICU 입실 초기 데이터로 예측하는 28일 사망', 36, True, RGBColor(0xFF, 0xFF, 0xFF))
txt(s, 0.8, 3.3, 11.5, 0.8, 'TabPFN · UTDE · Late fusion · TabPFN-in-UTDE — MIMIC-III SOFA 코호트', 20, False, RGBColor(0xC8, 0xD3, 0xE6))
txt(s, 0.8, 5.6, 11.5, 0.5, '2026-2 SNU DH 10조  |  초안 (single seed)', 14, False, RGBColor(0xC8, 0xD3, 0xE6))

# ---------------------------------------------------------------- 0b. setup
s = prs.slides.add_slide(BLANK); header(s, 'Setup', '과제 · 코호트 · 평가 프로토콜')
txt(s, 0.4, 1.1, 6.0, 5.5, ['과제: ICU 입실 후 첫 W시간(6 h / 24 h) 자료로 28일 사망 예측 (PATIENTS.DOD; 창 내 사망은 제외 = landmark)',
    '코호트 (MIMIC-III v1.4): 성인 → SOFA 6개 장기 계산 가능 → lactate·BE 존재 → W시점 생존',
    '  6 h: 5,757 stays (사망 29 %) / 24 h: 9,229 stays (26 %)',
    '환자 단위 70/15/15 분할, 모든 모델이 동일한 분할·레이블 사용',
    '평가: test AUROC (AUPRC), paired bootstrap ΔAUROC [95 % CI] vs TabPFN',
    '입력 변수(통일 조건): SOFA 입력(PaO2, FiO2, platelets, bilirubin, creatinine, MAP, GCS, 승압제, 소변량, 환기) + lactate + BE + age/sex',
    '노트: INTIME + W 이전에 저장된(STORETIME) 노트만 — 누수 방지'], 13)
pic(s, R / 'flowchart/flowchart_6h_24h.png', 6.6, 1.2, w=6.4)
foot(s, '선택편향 점검: SOFA 계산 가능 여부가 지배적 선택(사망률 24.0 % vs 10.0 %); 선택군에서 예측이 더 쉬워지지는 않음(공변량 모델 AUROC 0.70 vs 0.76).')

# ---------------------------------------------------------------- 1. previous works: TabPFN
s = prs.slides.add_slide(BLANK); header(s, '1. Previous works', 'TabPFN v2 — 표 형식 데이터의 foundation model')
txt(s, 0.4, 1.1, 6.3, 5.6, ['Prior-data fitted transformer: 합성 데이터로 사전학습, 새 표에는 in-context로 적합 (경사 학습 없음, 수십 초)',
    '본 연구: 공개 체크포인트(Prior-Labs/TabPFN-v2-clf), 앙상블 8, 튜닝 없음; context = train + val',
    '입력 68열: SOFA 입력 변수의 창 내 min/max/mean/last/count, P/F비, 승압제 약제별 지표, 소변량, 환기, lactate·BE 집계, SOFA 구성점수·총점, age, sex',
    '결과 (test AUROC): 6 h 0.832, 24 h 0.837 — 같은 feature의 XGBoost 0.823 / 0.837 (foundation model 이점 ≤ 0.01)',
    'SOFA 총점 단독 0.650 / 0.657 → 학습 모델이 +0.18',
    '한계: 정적 집계만 봄 — 궤적·노트 정보 없음'], 13)
table(s, 7.0, 1.3, 5.9, [['feature set', 'XGB 6 h', 'TabPFN 6 h', 'XGB 24 h', 'TabPFN 24 h'],
    ['SOFA score 단독', '0.650', '', '0.657', ''], ['1 SOFA 입력 (+age/sex)', '0.814', '0.824', '0.826', '0.831'],
    ['2 + lactate', '0.812', '0.830', '0.831', '0.833'], ['3 + BE', '0.823', '0.833', '0.831', '0.835'], ['4 + lactate + BE', '0.823', '0.832', '0.837', '0.837']],
    col_w=[2.1, 0.95, 0.95, 0.95, 0.95], size=11, bold_rows=(5,))
txt(s, 7.0, 3.7, 5.9, 0.4, 'test AUROC, tier 1–4 (+ age/sex)', 10, False, GREY)
foot(s, 'Hollmann et al., TabPFN v2 (Nature 2025).  결과: results/sofa_tiers_demo/')

# ---------------------------------------------------------------- 1. previous works: UTDE
s = prs.slides.add_slide(BLANK); header(s, '1. Previous works', 'UTDE (Zhang et al., ICML 2023) — 불규칙 시계열 + 임상 노트')
txt(s, 0.4, 1.1, 6.3, 5.6, ['시계열 stream: 불규칙 측정치를 mTAND 시간-attention으로 고정 길이(tt_max) 토큰으로 보간 + 시간별 격자(ffill) 혼합(gated mixup)',
    '노트 stream: Clinical-Longformer(1,024 토큰) 임베딩 + 노트 작성 시각 attention; 마지막 5개 노트',
    '결합: self/cross-attention 3층 → MLP → 사망 확률',
    '본 연구 적용: 원 코드 그대로(패치 7파일), 21채널(vitals 포함) 또는 15채널(통일 변수), 6 epoch, 검증 AUROC로 체크포인트 선택',
    '결과 (test AUROC): 21채널 0.798 / 0.844; 노트 제거 시 −0.012 / −0.025; 시계열만 학습 0.797 / 0.827',
    '한계: 소표본(6.5k) 딥러닝 — 집계 feature + TabPFN/GBDT를 단독으로는 넘지 못함'], 13)
# architecture sketch
box(s, 7.2, 1.4, 2.4, 0.7, '불규칙 시계열\n(21 or 15 ch)', size=10); box(s, 7.2, 3.6, 2.4, 0.7, '임상 노트 ≤ 5\n(+ 작성 시각)', size=10)
box(s, 10.0, 1.4, 2.4, 0.7, 'mTAND + grid\n→ tt_max 토큰', size=10); box(s, 10.0, 3.6, 2.4, 0.7, 'Clinical-Longformer\n+ 시간 attention', size=10)
box(s, 8.6, 5.2, 2.6, 0.8, 'self / cross attention\n→ MLP → p(사망)', fill=RGBColor(0xE8, 0xF0, 0xFF), size=10, bold=True)
arrow(s, 9.6, 1.75, 10.0, 1.75); arrow(s, 9.6, 3.95, 10.0, 3.95); arrow(s, 11.2, 2.1, 9.9, 5.2); arrow(s, 11.2, 4.3, 9.9, 5.2)
foot(s, 'Zhang, X. et al. Improving medical predictions by irregular multimodal electronic health records modeling. ICML 2023.')

# ---------------------------------------------------------------- 2. proposed: late fusion
s = prs.slides.add_slide(BLANK); header(s, '2. Proposed methods', 'Late fusion — TabPFN과 UTDE의 확률을 검증셋에서 학습한 스태커로 결합')
box(s, 0.6, 1.5, 3.0, 1.0, 'TabPFN\n(구조화 68열)', fill=RGBColor(0xE6, 0xF4, 0xEA), line=GREEN, size=12, bold=True)
box(s, 0.6, 3.6, 3.0, 1.0, 'UTDE\n(시계열 + 노트)', fill=RGBColor(0xEE, 0xEA, 0xFA), line=PURPLE, size=12, bold=True)
box(s, 4.6, 2.55, 3.4, 1.0, 'logit(p_T), logit(p_U)\n→ 로지스틱 스태커 (val에서만 적합)', size=11)
box(s, 9.0, 2.55, 2.6, 1.0, 'p_fusion', fill=RGBColor(0xE8, 0xF0, 0xFF), size=12, bold=True)
arrow(s, 3.6, 2.0, 4.6, 2.9); arrow(s, 3.6, 4.1, 4.6, 3.2); arrow(s, 8.0, 3.05, 9.0, 3.05)
txt(s, 0.6, 5.0, 12.2, 2.0, ['두 모델을 독립적으로 학습한 뒤 최종 로짓 2개만 결합 — 가장 단순한 멀티모달 결합, 테스트 레이블 미사용',
    'TabPFN 검증 확률은 train만 context로 둔 적합에서, UTDE 검증 로짓은 체크포인트에서 읽음',
    '결과(21채널 UTDE): 6 h 0.845 (+0.014 [−0.002, +0.029]), 24 h 0.860 (+0.023 [+0.010, +0.035])',
    '— 그러나 변수를 통일하면(UTDE-SOFA) +0.008 / +0.003 (ns): 이득의 대부분은 UTDE에만 있던 활력징후 변수에서 온 것'], 13)
foot(s, 'scripts/14_late_fusion.py, 24_fusion_variants.py')

# ---------------------------------------------------------------- 2. proposed: TabPFN-in-UTDE
s = prs.slides.add_slide(BLANK); header(s, '2. Proposed methods', 'TabPFN-in-UTDE — frozen TabPFN 임베딩을 UTDE 구조 안의 토큰으로')
txt(s, 0.4, 1.05, 12.5, 0.9, ['TabPFN v2의 내부 표현(앙상블 멤버 8 × 192-d, query 모드; train 행은 5-fold context로 자기 자신 제외)을 stay당 토큰 8개로 꺼냄 — 선형 프로브만으로 TabPFN 성능 보존(0.834 vs 0.837)',
    '토큰을 Linear(192→128)로 투영해 UTDE의 cross-attention에 넣고, 노트 stream·결합·학습 일정은 UTDE 그대로 (end-to-end)'], 12)
# three variants
for k, (title_, ts_, tab_, col) in enumerate([('원래 UTDE', '시계열 토큰 (tt_max)', None, PURPLE), ('replace', None, 'TabPFN 토큰 8', GREEN), ('3-stream', '시계열 토큰 (tt_max)', 'TabPFN 토큰 8', ACC)]):
    x0 = 0.5 + k * 4.25
    box(s, x0, 2.1, 3.8, 0.45, title_, fill=RGBColor(0xFF, 0xFF, 0xFF), color=col, size=13, bold=True, line=col)
    yy = 2.75
    if ts_:
        box(s, x0, yy, 3.8, 0.55, ts_, fill=RGBColor(0xEE, 0xEA, 0xFA), line=PURPLE, size=11); yy += 0.65
    if tab_:
        box(s, x0, yy, 3.8, 0.55, tab_ + ' (frozen TabPFN)', fill=RGBColor(0xE6, 0xF4, 0xEA), line=GREEN, size=11); yy += 0.65
    box(s, x0, yy, 3.8, 0.55, '노트 토큰 5 (Longformer)', size=11); yy += 0.65
    box(s, x0, yy + 0.1, 3.8, 0.55, 'cross-attention → MLP', fill=RGBColor(0xE8, 0xF0, 0xFF), size=11)
txt(s, 0.5, 5.6, 12.3, 1.4, ['replace: 시계열 stream을 TabPFN 토큰으로 교체 → "TabPFN + 노트" 하나의 네트워크',
    '3-stream: 시계열 토큰 뒤에 TabPFN 토큰을 이어 붙임(길이 tt_max + 8; 위치 임베딩 테이블 확장) → 세 정보원을 한 번에 학습',
    '가설: 단일 모델이 늦은 융합과 같거나 나은 성능을 내면서 노트를 TabPFN 표현에 조건부로 읽을 수 있는가'], 12)
foot(s, 'UTDE 패치: --tab_emb, --tab_emb_mode {replace, concat}; scripts/23_tabpfn_embeddings.py')

# ---------------------------------------------------------------- 2. proposed: fair protocol
s = prs.slides.add_slide(BLANK); header(s, '2. Proposed methods', '공정 비교 프로토콜 — 변수를 맞춘 뒤 구조를 비교')
txt(s, 0.4, 1.1, 6.2, 5.6, ['문제: 원 UTDE 입력(21채널)에는 HR·RR·SpO2·체온·혈압·O2 flow가 있었고 TabPFN(SOFA 입력)에는 없었음 → 모델 비교에 변수 차이가 섞임',
    '통제 1 — UTDE-SOFA: UTDE 채널을 TabPFN의 변수로 제한(15채널: SOFA 입력 + lactate + BE + 환기 + age/sex 상수 채널)',
    '통제 2 — TabPFN + vitals: 반대로 TabPFN에 활력징후 집계 45열을 추가(113열)',
    '모든 비교는 같은 테스트셋에서 paired bootstrap(2,000회)',
    '해석 가능성: 세 모델 모두에 어트리뷰션 — TabPFN permutation SHAP, UTDE occlusion(모달리티·채널·TabPFN 토큰), 스태커 정확 SHAP + feature-level 분해'], 13)
table(s, 6.9, 1.3, 6.0, [['', 'TabPFN', 'UTDE 21ch', 'UTDE-SOFA'], ['SOFA 입력 + lactate/BE', '집계 68열', '시계열', '시계열'],
    ['age / sex', '○', '×', '상수 채널'], ['환기', 'vent_n', '×', '0/1 채널'], ['HR·RR·SpO2·Temp·혈압·O2flow', '×', '○ (9채널)', '×'], ['노트', '×', '○', '○']],
    col_w=[2.4, 1.2, 1.2, 1.2], size=11)
foot(s, 'scripts/13_build_utde_dataset.py --channel-set sofa, 25_tabpfn_vitals.py')

# ---------------------------------------------------------------- 3. results: unified table
s = prs.slides.add_slide(BLANK); header(s, '3. Results — Quantitative', '통일 변수(SOFA + lactate/BE + age/sex) 조건의 성능')
table(s, 0.4, 1.2, 12.5, [['model', '6 h AUROC (AUPRC)', 'Δ vs TabPFN [95 % CI]', '24 h AUROC (AUPRC)', 'Δ vs TabPFN [95 % CI]'],
    ['TabPFN (68 feature)', '0.832 (.681)', '–', '0.837 (.673)', '–'],
    ['UTDE-SOFA (15채널 + 노트)', '0.823 (.658)', '−0.009 [−0.031, +0.013]', '0.826 (.658)', '−0.011 [−0.025, +0.002]'],
    ['Late fusion TabPFN + UTDE-SOFA', '0.839 (.688)', '+0.008 [−0.008, +0.023]', '0.841 (.680)', '+0.003 [−0.002, +0.009]'],
    ['TabPFN-in-UTDE, replace', '0.827 (.677)', '−0.005 [−0.012, +0.002]', '0.836 (.671)', '−0.001 [−0.006, +0.004]'],
    ['TabPFN-in-UTDE, 3-stream', '0.835 (.687)', '+0.004 [−0.010, +0.016]', '0.856 (.709)', '+0.019 [+0.008, +0.030]']],
    col_w=[3.7, 2.0, 2.4, 2.0, 2.4], size=12, bold_rows=(5,))
txt(s, 0.4, 3.9, 12.5, 2.8, ['같은 변수에서는 UTDE 단독 < TabPFN, 늦은 융합의 이득은 유의하지 않음 (+0.008 / +0.003)',
    'replace(시계열을 TabPFN 토큰으로 교체)는 TabPFN과 구별되지 않음 (로짓 상관 0.98)',
    '3-stream만 24 h에서 유의하게 앞섬: +0.019 [+0.008, +0.030] vs TabPFN, +0.016 [+0.006, +0.026] vs 늦은 융합',
    '6 h에서는 어떤 구성도 TabPFN을 넘지 못함 — 노트·궤적의 가치는 하루치 기록이 쌓인 뒤 나타남',
    'test n = 853 / 1,374, single seed'], 13)
foot(s, 'results/late_fusion/variants_W{6,24}.md')

# ---------------------------------------------------------------- 3. results: full variables & fair baseline
s = prs.slides.add_slide(BLANK); header(s, '3. Results — Quantitative', '활력징후 변수를 포함한 비교 — 공정 기준선 TabPFN + vitals')
table(s, 0.4, 1.2, 12.5, [['model', '6 h', 'Δ vs TabPFN+vitals', '24 h', 'Δ vs TabPFN+vitals'],
    ['TabPFN (68)', '0.832 (.681)', '−0.010', '0.837 (.673)', '−0.015 [−0.023, −0.007]'],
    ['TabPFN + vitals 집계 (113) — 공정 기준선', '0.842 (.705)', '–', '0.852 (.684)', '–'],
    ['UTDE 21채널 + 노트', '0.798 (.618)', '−0.044', '0.844 (.680)', '−0.008 [−0.026, +0.010]'],
    ['Late fusion TabPFN + UTDE 21채널', '0.845 (.704)', '+0.004 [−0.011, +0.019]', '0.860 (.707)', '+0.008 [−0.004, +0.019]'],
    ['Late fusion TabPFN+vitals + UTDE 21채널', '0.844 (.710)', '+0.002 [−0.012, +0.016]', '0.863 (.709)', '+0.012 [+0.002, +0.021]'],
    ['3-stream (TabPFN 토큰 + 21채널 + 노트)', '0.836 (.700)', '−0.006 [−0.017, +0.004]', '0.868 (.731)', '+0.016 [+0.003, +0.028]']],
    col_w=[4.3, 1.7, 2.4, 1.7, 2.4], size=11, bold_rows=(2, 6))
txt(s, 0.4, 4.2, 12.5, 2.6, ['활력징후 집계만 TabPFN에 더해도 +0.010 / +0.015 — 구조화 모델의 공정 기준선은 0.842 / 0.852',
    '그 위의 멀티모달 이득: 늦은 융합 +0.002 (ns) / +0.012 (유의), 3-stream −0.006 / +0.016 (유의) — 24 h에서만, 크기 +0.01~0.02',
    '21채널 3-stream 0.868 vs 15채널 3-stream 0.856: 0.012가 활력징후 변수의 몫',
    '교훈: "멀티모달 vs 구조화" 비교는 변수를 맞춘 뒤에만 구조의 효과를 말할 수 있다'], 13)
foot(s, 'results/tabpfn_vitals/summary_W{6,24}.md')

# ---------------------------------------------------------------- 3. SHAP TabPFN
s = prs.slides.add_slide(BLANK); header(s, '3. Results — SHAP', 'TabPFN — permutation SHAP (전체 테스트, 확률 단위)')
pic(s, R / 'fusion_shap/tabpfn_shap_W24.png', 0.3, 1.05, w=8.2)
txt(s, 8.7, 1.1, 4.3, 5.8, ['24 h: gcs_last 0.111 ≫ age 0.048 > urine_n ≈ gcs_count ≈ lactate_min ≈ lactate_last ≈ base_excess_last (0.02)',
    '변수군: GCS 0.147 ≫ age 0.048 ≈ urine 0.046 > lactate 0.041 > MAP 0.035 > FiO2 ≈ BE 0.028',
    '6 h: age 0.071 ≈ gcs_last 0.069 > fio2_last 0.031 — 초기엔 사전 위험·산소요구가 신경학적 상태만큼 큼',
    '방향: 낮은 GCS·낮은 소변량·높은 lactate·낮은 BE·높은 FiO2·고령 → 사망 (임상적으로 일관)',
    '측정 횟수 feature(urine_n, gcs_count, platelets_count)가 상위 — 기록 패턴 학습 (limitation)'], 12)
foot(s, 'shap.PermutationExplainer, 배경 10행, 2F+1 평가/케이스; scripts/21_fusion_shap.py')

# ---------------------------------------------------------------- 3. SHAP UTDE-SOFA
s = prs.slides.add_slide(BLANK); header(s, '3. Results — SHAP', 'UTDE-SOFA — occlusion 어트리뷰션 (채널 제거 시 로짓 변화; 색 = 채널 평균)')
pic(s, R / 'unified_shap/utde_sofa_beeswarm_W24.png', 0.3, 1.05, h=5.9)
table(s, 8.7, 1.2, 4.3, [['24 h', '제거 시 AUROC'], ['전체', '0.826'], ['노트 제거', '0.803 (−0.023)'], ['시계열 제거', '0.643 (−0.183)'],
    ['GCS 채널', '−0.057'], ['Age 채널', '−0.020'], ['Lactate / PaO2', '−0.008'], ['Bilirubin', '−0.007']], col_w=[2.0, 2.3], size=11)
txt(s, 8.7, 4.3, 4.3, 2.6, ['6 h: Age (−0.023) ≈ GCS (−0.024) > bilirubin, BE; 노트 −0.018',
    'TabPFN과 같은 변수를 주면 UTDE도 같은 순서(GCS, age, lactate…)로 의존',
    '노트의 몫(−0.02)은 21채널 UTDE(−0.025)와 동일'], 12)
foot(s, '노트 제거 = placeholder 노트로 대체; 채널 제거 = 값·마스크 0. utde_attr.py')

# ---------------------------------------------------------------- 3. SHAP late fusion
s = prs.slides.add_slide(BLANK); header(s, '3. Results — SHAP', 'Late fusion — 스태커 정확 SHAP과 feature-level 분해 (24 h)')
pic(s, R / 'unified_shap/fusion_sofa_feature_beeswarm_W24.png', 0.3, 1.05, h=5.9)
pic(s, R / 'unified_shap/fusion_sofa_stacker_W24.png', 6.1, 1.1, w=6.9)
txt(s, 6.1, 3.6, 6.9, 3.4, ['스태커: TabPFN 0.55 / UTDE-SOFA 0.37 → 융합 로짓의 61 %가 TabPFN (6 h는 39 %); 로짓 상관 0.91',
    'feature-level(T = TabPFN SHAP × b₁, U = UTDE occlusion × b₂): gcs_last (T) 0.80 ≫ age (T) 0.32 > notes (U) 0.24 > GCS (U) 0.23 > Age (U) 0.18',
    'GCS와 age가 양쪽 경로로 중복 — UTDE 경로의 고유 기여는 노트뿐 → 융합 이득 +0.003',
    '6 h에서는 notes (U) 0.38 > Age (U) 0.29 > age (T) ≈ gcs_last (T) 0.21; 스태커가 UTDE 쪽(69 %)에 기댐'], 12)
foot(s, '노트 occlusion의 일괄 이동(placeholder 기준점, −1.48 logit)은 제거 후 표시. scripts/26_unified_shap.py')

# ---------------------------------------------------------------- 3. SHAP TabPFN-in-UTDE
s = prs.slides.add_slide(BLANK); header(s, '3. Results — SHAP', 'TabPFN-in-UTDE — replace는 노트를 버리고, 3-stream은 노트를 쓴다')
table(s, 0.4, 1.2, 7.4, [['제거 대상 (24 h)', 'replace: AUROC', '3-stream: AUROC'], ['없음 (전체)', '0.836', '0.856'],
    ['TabPFN 토큰', '0.562 (−0.274)', '0.765 (−0.091)'], ['노트', '0.834 (−0.002)', '0.840 (−0.016)'], ['시계열 stream', '(없음)', '0.852 (−0.005)'], ['토큰 + 노트', '0.494', '0.752']],
    col_w=[2.6, 2.4, 2.4], size=12)
txt(s, 0.4, 3.6, 7.4, 3.3, ['replace: TabPFN 토큰이 전부, 노트 제거 효과 ≈ 0 — 강한 모달리티가 노트 branch의 학습을 질식 (검증 AUROC epoch 2 이후 정체)',
    '3-stream: 노트 −0.016 (6 h −0.007) — 시계열 stream이 함께 학습되면 노트 branch가 살아남; 시계열 자체의 직접 기여는 −0.005',
    '채널별 기여는 모두 ≤ 0.006 (Vent, Age, GCS 순) — 시계열 정보는 TabPFN 토큰과 중복',
    '결론: 같은 정보에서 3-stream의 +0.019는 "TabPFN 표현을 조건으로 노트를 읽는 것"에서 나옴'], 12)
pic(s, R / 'unified_shap/tabemb3_sofa_beeswarm_W24.png', 8.0, 1.1, h=5.8)
foot(s, 'TabPFN 토큰 제거 = 0 벡터, 노트 제거 = placeholder. results/unified_shap/summary_W{6,24}.md')

# ---------------------------------------------------------------- summary
s = prs.slides.add_slide(BLANK); header(s, 'Summary', '결론 · 한계 · 다음 단계')
txt(s, 0.4, 1.1, 6.3, 5.8, ['— 결론',
    '구조화 feature + TabPFN이 가장 강한 단일 표현 (0.832 / 0.837); 변수를 맞추면 UTDE·늦은 융합은 이를 넘지 못함',
    '기존 융합 이득(+0.023)의 대부분은 UTDE만 보던 활력징후 변수 — TabPFN에 넣으면 0.852',
    'TabPFN을 UTDE 안에 넣을 때 시계열 stream을 남긴 3-stream만 유의한 이득(24 h +0.019; 공정 기준선 대비 +0.016)',
    '어트리뷰션: 모든 모델이 GCS·age·lactate·소변량에 의존; 노트는 24 h에서만, 3-stream에서만 실제로 쓰임',
    '— 한계',
    '단일 seed·단일 데이터셋(MIMIC-III); 측정 횟수 feature = 진료 패턴 학습; 적용 대상은 ABG·lactate가 시행된 ICU 환자'], 12)
txt(s, 7.0, 1.1, 5.9, 5.8, ['— 다음 단계',
    'seed 3회 반복으로 3-stream vs 늦은 융합(+0.016) 신뢰구간 확정',
    '3-stream에 TabPFN+vitals 토큰 사용 (공정 기준선과 같은 입력)',
    'UTDE 노트 수 5 → 10 (24 h stay의 63 %가 노트 5개 초과)',
    'Agent 단계: 구조화 위험 모델을 도구로, 노트는 질문 기반 검색으로, 능동 검사 획득 과제에서 평가',
    '— 산출물',
    'github.com/HyeongSubKim/mimic3-sofa28-mortality-multimodal (Apache-2.0): 재현 가이드, UTDE 패치, 결과표·그림'], 12)
out = R / 'slides' / 'MIMIC_TabPFN_UTDE_fusion_draft.pptx'; prs.save(out); print('saved', out, len(prs.slides), 'slides')
