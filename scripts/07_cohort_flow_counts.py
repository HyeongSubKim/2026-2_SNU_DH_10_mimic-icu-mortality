"""Cohort funnel for the flow chart: all ICU stays -> adults (>= 18 y) -> SOFA computable within W h -> lactate within
W h -> base excess within W h -> not died within the first W h, for W = 6 and 24.  Writes results/flowchart/flow_counts.json (drawn by 08_flowchart.py).

SOFA "computable" = all six organ systems have at least one measurement in the window: PaO2 & FiO2, platelets,
bilirubin, MAP or a vasopressor, GCS, creatinine or urine output. Labs are counted from 6 h before ICU admission.
Needs data/*.parquet from 00_extract_icu_events.py; the GCS and lab items it additionally needs are extracted on the
first run (CHARTEVENTS pass ~7 min) and cached in data/.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd

RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'flowchart'
OUT.mkdir(parents=True, exist_ok=True)
H = pd.Timedelta(1, 'h')

icu = pd.read_csv(RAW / 'ICUSTAYS.csv.gz', parse_dates=['INTIME', 'OUTTIME'])
pat = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB', 'DOD'], parse_dates=['DOB', 'DOD'])
adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'DEATHTIME'], parse_dates=['DEATHTIME'])
icu = icu.merge(pat, on='SUBJECT_ID').merge(adm, on='HADM_ID')
icu['age'] = (icu.INTIME.dt.year - icu.DOB.dt.year) - ((icu.INTIME.dt.month * 100 + icu.INTIME.dt.day) < (icu.DOB.dt.month * 100 + icu.DOB.dt.day))
F = {'all': {'stays': len(icu), 'patients': icu.SUBJECT_ID.nunique()}}

minor = icu.age < 18
bad_time = icu.OUTTIME.isna() | (icu.OUTTIME <= icu.INTIME)
F['excl_age'] = int(minor.sum()); F['excl_age_nicu'] = int((minor & (icu.FIRST_CAREUNIT == 'NICU')).sum())
F['excl_time'] = int((~minor & bad_time).sum())
st = icu[~minor & ~bad_time].copy()
F['adult'] = {'stays': len(st), 'patients': st.SUBJECT_ID.nunique()}
intime = st.set_index('ICUSTAY_ID').INTIME
ids_adult = set(st.ICUSTAY_ID)

# ---- measurement sources; the two extra extractions are cached in data/
LAB_ITEMS = {51265: 'platelets', 50885: 'bilirubin', 50912: 'creatinine', 50821: 'PaO2', 50816: 'FiO2_lab',
             50813: 'lactate', 50802: 'base_excess'}
GCS_ITEMS = {723, 454, 184, 223900, 223901, 220739}          # CareVue / MetaVision verbal, motor, eye
if not (DATA / 'labevents_sofa.parquet').exists():
    t0, parts = time.time(), []
    for ch in pd.read_csv(RAW / 'LABEVENTS.csv.gz', usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'], chunksize=5_000_000):
        parts.append(ch[ch.ITEMID.isin(LAB_ITEMS) & ch.VALUENUM.notna() & ch.HADM_ID.notna()])
    pd.concat(parts, ignore_index=True).to_parquet(DATA / 'labevents_sofa.parquet', index=False)
    print(f'labevents_sofa.parquet written ({time.time()-t0:.0f}s)', flush=True)
if not (DATA / 'chartevents_gcs.parquet').exists():
    t0, parts = time.time(), []
    for i, ch in enumerate(pd.read_csv(RAW / 'CHARTEVENTS.csv.gz', usecols=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'ERROR'],
                                       chunksize=5_000_000, low_memory=False)):
        parts.append(ch[ch.ITEMID.isin(GCS_ITEMS) & (ch.ERROR.fillna(0) != 1) & ch.VALUENUM.notna()].drop(columns='ERROR'))
        if i % 10 == 0:
            print(f'  CHARTEVENTS chunk {i}, {time.time()-t0:.0f}s', flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(DATA / 'chartevents_gcs.parquet', index=False)
    print(f'chartevents_gcs.parquet written ({time.time()-t0:.0f}s)', flush=True)

lab = pd.read_parquet(DATA / 'labevents_sofa.parquet').merge(st[['HADM_ID', 'ICUSTAY_ID']], on='HADM_ID')
lab['CHARTTIME'] = pd.to_datetime(lab.CHARTTIME)
bg = lab
ce = pd.read_parquet(DATA / 'chartevents.parquet', columns=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']); ce = ce[ce.VALUENUM.notna()]
gcs = pd.read_parquet(DATA / 'chartevents_gcs.parquet')
icv = pd.read_parquet(DATA / 'inputevents_cv.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'CHARTTIME': 't'})
imv = pd.read_parquet(DATA / 'inputevents_mv.parquet', columns=['ICUSTAY_ID', 'STARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'STARTTIME': 't'})
VASO = {30047, 30120, 30044, 30119, 30309, 30127, 30128, 30042, 30043, 30307, 30125,          # CareVue
        221906, 221289, 221662, 221653, 221749, 222315}                                      # MetaVision
inp = pd.concat([icv, imv]); inp = inp[inp.ITEMID.isin(VASO) & (inp.AMOUNT > 0)]
out = pd.read_parquet(DATA / 'outputevents.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'VALUE']); out = out[out.VALUE.notna()]


def stays_in(df, tcol, W, lab_window=False):
    df = df[df.ICUSTAY_ID.isin(ids_adult)]
    t = pd.to_datetime(df[tcol]); it = df.ICUSTAY_ID.map(intime)
    lo = it - 6 * H if lab_window else it
    return set(df.ICUSTAY_ID[(t >= lo) & (t <= it + W * H)])


for W in [6, 24]:
    organ = {'respiration': stays_in(lab[lab.ITEMID == 50821], 'CHARTTIME', W, True) &
                            (stays_in(ce[ce.ITEMID.isin({190, 3420, 3422, 223835})], 'CHARTTIME', W) | stays_in(lab[lab.ITEMID == 50816], 'CHARTTIME', W, True)),
             'coagulation': stays_in(lab[lab.ITEMID == 51265], 'CHARTTIME', W, True),
             'liver': stays_in(lab[lab.ITEMID == 50885], 'CHARTTIME', W, True),
             'cardiovascular': stays_in(ce[ce.ITEMID.isin({456, 52, 6702, 443, 220052, 220181, 225312})], 'CHARTTIME', W) | stays_in(inp, 't', W),
             'CNS': stays_in(gcs, 'CHARTTIME', W),
             'renal': stays_in(lab[lab.ITEMID == 50912], 'CHARTTIME', W, True) | stays_in(out, 'CHARTTIME', W)}
    sofa = ids_adult & set.intersection(*organ.values())
    no_sofa = ids_adult - sofa
    missing = {k: len(no_sofa - v) for k, v in organ.items()}          # stays lacking that organ among the excluded
    lact = stays_in(bg[bg.ITEMID == 50813], 'CHARTTIME', W, True)
    be = stays_in(bg[bg.ITEMID == 50802], 'CHARTTIME', W, True)
    s3 = sofa & lact; s4 = s3 & be
    # outcome already observed inside the window: in-hospital death (exact DEATHTIME) within the first W h
    died_in_window = set(st.ICUSTAY_ID[(st.DEATHTIME >= st.INTIME) & (st.DEATHTIME <= st.INTIME + W * H)])
    s5 = s4 - died_in_window
    died = set(st.ICUSTAY_ID[st.HOSPITAL_EXPIRE_FLAG == 1])
    # 28-day all-cause mortality from ICU admission (PATIENTS.DOD; an out-of-hospital DOD is a date, so compare on days)
    d0 = st.INTIME.dt.floor('D')
    died28 = set(st.ICUSTAY_ID[(st.DOD >= d0) & (st.DOD <= d0 + pd.Timedelta(28, 'D'))])
    pts = lambda s: int(st.SUBJECT_ID[st.ICUSTAY_ID.isin(s)].nunique())
    F[f'W{W}'] = {'sofa': {'stays': len(sofa), 'patients': pts(sofa)}, 'excl_sofa': len(no_sofa), 'excl_sofa_missing_organ': missing,
                  'lactate': {'stays': len(s3), 'patients': pts(s3)}, 'excl_lactate': len(sofa - lact),
                  'be': {'stays': len(s4), 'patients': pts(s4)}, 'excl_be': len(s3 - be),
                  'excl_died_in_window': len(s4 & died_in_window),
                  'final': {'stays': len(s5), 'patients': pts(s5)},
                  'final_died': len(s5 & died), 'final_alive': len(s5 - died),
                  'final_died_28d': len(s5 & died28), 'final_alive_28d': len(s5 - died28)}
json.dump(F, open(OUT / 'flow_counts.json', 'w'), indent=1)
print(json.dumps(F, indent=1))
