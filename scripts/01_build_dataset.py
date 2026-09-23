"""Build the hourly ICU dataset with the same labeling rules as the original NICU pipeline
(1.preprocessing -> 2-1.add_event -> 2-2.FiO2 -> 3.make_dataset_mode), on MIMIC-III adult ICU.

Event   : --event vent  (default) first mechanical-ventilation / intubation time in the ICU stay
                        (MIMIC-code ventilation_classification items + PROCEDUREEVENTS_MV 224385)
          --event death  ADMISSIONS.DEATHTIME inside [INTIME, OUTTIME] (in-ICU death); stays of admissions that
                        end in death outside this ICU stay are excluded (ambiguous), survivors are negatives
          --event none   no in-stay event: the full stay is kept and every row is a negative. Use this for
                        stay-level outcomes defined outside the ICU stay (28-day / in-hospital mortality),
                        which 05_early_window.py attaches with --outcome
Labels  : identical to 2-1.add_event NICU section, with W = --pre-window hours (default 24)
            event stay & first_event_time-W < ICU intime           -> stay dropped
            rec_time <  first_event_time-W                         -> is_event=1, is_pre_event=1 (excluded in training)
            first_event_time-W <= rec_time <= first_event_time     -> is_event=1, is_pre_event=0 (positive)
            rec_time >  first_event_time                           -> dropped
            no event                                               -> is_event=0, is_pre_event=1 (negative)
Split   : MIMIC dates are shifted per patient, so the original calendar split is replaced by a
          patient-level random split 70/15/15 (seed 0).
Output  : data/icu_<event>_w<W>h.csv (or data/icu_allstays.csv for --event none)  (columns mirror the original NICU csv where an adult analog exists)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SEED = 0
RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))   # folder with the PhysioNet csv.gz files
ap = argparse.ArgumentParser()
ap.add_argument('--event', choices=['vent', 'death', 'none'], default='death')
ap.add_argument('--pre-window', type=int, default=24, help='label window in hours before the event (positive rows)')
args = ap.parse_args()
EVENT, PRE_WINDOW_H = args.event, args.pre_window
t0 = time.time()


def log(msg):
    print(f'[{time.time()-t0:6.0f}s] {msg}', flush=True)


# ---------------------------------------------------------------- concept map
# same feature names as the original NICU csv where possible
CONCEPT = {}
for name, ids in {
    'HR 회/min': [211, 220045],
    'SBP mmHg': [442, 455, 220179],            # non-invasive
    'ASBP mmHg': [51, 6701, 220050],           # arterial
    'DBP mmHg': [8440, 8441, 220180],
    'ADBP mmHg': [8368, 8555, 220051],
    'Mean BP mmHg': [456, 443, 220181],
    'Mean ABP mmHg': [52, 6702, 220052, 225312],
    'RR 회/min': [615, 618, 220210, 224690],
    'TempF': [223761, 678], 'TempC': [223762, 676],
    'SpO2 %': [646, 220277],
    'FIO2_frac': [190], 'FIO2_pct': [3420, 3422, 223835],
    'weight kg': [762, 763, 3580, 224639, 226512],
    'height_in': [920, 1394, 4187, 3486, 226707], 'height_cm': [226730],
    'Flow rate L/min': [470, 471, 223834],
}.items():
    for i in ids:
        CONCEPT[i] = name

VENT_NUMERIC = {445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
                218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000, 226873, 224738,
                224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505, 506, 686,
                220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701, 223849}

PLAUSIBLE = {  # (low, high) inclusive; outside -> NaN   (replaces the original mean±5σ rule with clinical ranges)
    'HR 회/min': (0, 300), 'SBP mmHg': (0, 400), 'ASBP mmHg': (0, 400), 'DBP mmHg': (0, 300), 'ADBP mmHg': (0, 300),
    'Mean BP mmHg': (0, 300), 'Mean ABP mmHg': (0, 300), 'RR 회/min': (0, 100), 'BT ℃': (25, 45), 'SpO2 %': (0, 100),
    'FIO2_': (21, 100), 'weight kg': (20, 400), 'height cm': (100, 250), 'Flow rate L/min': (0, 100),
    'Base Excess': (-50, 50),
}

# ---------------------------------------------------------------- load
st = pd.read_parquet(DATA / 'icustays.parquet')
for c in ['INTIME', 'OUTTIME', 'DOB', 'ADMITTIME', 'DISCHTIME']:
    st[c] = pd.to_datetime(st[c])
st = st[st.OUTTIME.notna() & (st.OUTTIME > st.INTIME)]
log(f'stays {len(st)}')

ce = pd.read_parquet(DATA / 'chartevents.parquet')
ce['CHARTTIME'] = pd.to_datetime(ce['CHARTTIME'])
log(f'chartevents {len(ce)}')

# ---------------------------------------------------------------- event time
if EVENT == 'vent':   # first mech vent / intubation
    is_vent = ce.ITEMID.isin(VENT_NUMERIC) & ce.VALUENUM.notna()
    is_vent |= (ce.ITEMID == 720) & ce.VALUE.notna() & (ce.VALUE != 'Other/Remarks')
    is_vent |= (ce.ITEMID == 223848) & ce.VALUE.notna() & (ce.VALUE != 'Other')
    is_vent |= (ce.ITEMID == 467) & (ce.VALUE == 'Ventilator')
    is_vent |= (ce.ITEMID == 226732) & ce.VALUE.isin(['Endotracheal tube', 'Tracheostomy tube'])
    first_vent = ce.loc[is_vent].groupby('ICUSTAY_ID').CHARTTIME.min()
    proc = pd.read_parquet(DATA / 'procedureevents_intub.parquet')
    proc['STARTTIME'] = pd.to_datetime(proc['STARTTIME'])
    first_intub = proc.groupby('ICUSTAY_ID').STARTTIME.min()
    first_event = pd.concat([first_vent, first_intub], axis=1).min(axis=1).rename('first_event_time')
    st = st.merge(first_event, left_on='ICUSTAY_ID', right_index=True, how='left')
elif EVENT == 'death':   # in-ICU death
    death = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'DEATHTIME'], parse_dates=['DEATHTIME'])
    st = st.merge(death, on='HADM_ID', how='left').rename(columns={'DEATHTIME': 'first_event_time'})
    outside = st.first_event_time.notna() & ((st.first_event_time < st.INTIME) | (st.first_event_time > st.OUTTIME))
    st = st[~outside]
    log(f'stays of admissions that end in death outside this ICU stay (excluded): {outside.sum()}')
else:                    # no in-stay event; the outcome is attached later, at stay level
    st['first_event_time'] = pd.NaT
# original: only events inside the admission window are events
st.loc[(st.first_event_time < st.INTIME) | (st.first_event_time > st.OUTTIME), 'first_event_time'] = pd.NaT
log(f'stays with event: {st.first_event_time.notna().sum()} / {len(st)}')

# ---------------------------------------------------------------- hourly vitals
ce = ce[ce.ITEMID.isin(CONCEPT.keys()) & ce.VALUENUM.notna()].copy()
ce['concept'] = ce.ITEMID.map(CONCEPT)
ce['rec_time'] = ce.CHARTTIME.dt.floor('h')
# unit conversion
m = ce.concept == 'TempF'
ce.loc[m, 'VALUENUM'] = (ce.loc[m, 'VALUENUM'] - 32) * 5 / 9
ce.loc[m, 'concept'] = 'TempC'
m = ce.concept == 'FIO2_frac'
ce.loc[m, 'VALUENUM'] = ce.loc[m, 'VALUENUM'] * 100
ce.loc[m, 'concept'] = 'FIO2_pct'
m = (ce.concept == 'FIO2_pct') & (ce.VALUENUM <= 1.0)   # 223835 is sometimes charted as a fraction
ce.loc[m, 'VALUENUM'] = ce.loc[m, 'VALUENUM'] * 100
m = ce.concept == 'height_in'
ce.loc[m, 'VALUENUM'] = ce.loc[m, 'VALUENUM'] * 2.54
ce.loc[m, 'concept'] = 'height_cm'
ce['concept'] = ce.concept.replace({'TempC': 'BT ℃', 'FIO2_pct': 'FIO2_', 'height_cm': 'height cm'})
for c, (lo, hi) in PLAUSIBLE.items():
    m = (ce.concept == c) & ((ce.VALUENUM < lo) | (ce.VALUENUM > hi))
    ce.loc[m, 'VALUENUM'] = np.nan
ce = ce.dropna(subset=['VALUENUM'])
hourly = ce.groupby(['ICUSTAY_ID', 'rec_time', 'concept'], observed=True).VALUENUM.mean().unstack('concept')
hourly = hourly.reset_index()
del ce
log(f'hourly vitals {hourly.shape}')

# ---------------------------------------------------------------- base excess (LABEVENTS, by HADM)
lab = pd.read_parquet(DATA / 'labevents_be.parquet').dropna(subset=['VALUENUM'])
lab['CHARTTIME'] = pd.to_datetime(lab.CHARTTIME)
lab = lab.merge(st[['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']], on='HADM_ID')
lab = lab[(lab.CHARTTIME >= lab.INTIME) & (lab.CHARTTIME <= lab.OUTTIME)]
lab['rec_time'] = lab.CHARTTIME.dt.floor('h')
be = lab.groupby(['ICUSTAY_ID', 'rec_time']).VALUENUM.mean().rename('Base Excess').reset_index()
lo, hi = PLAUSIBLE['Base Excess']
be.loc[(be['Base Excess'] < lo) | (be['Base Excess'] > hi), 'Base Excess'] = np.nan
log(f'base excess rows {len(be)}')

# ---------------------------------------------------------------- daily I/O totals, shifted +1 day like the original
icv = pd.read_parquet(DATA / 'inputevents_cv.parquet')
icv = icv[icv.AMOUNTUOM.str.lower().isin(['ml', 'cc'])]
icv['date'] = pd.to_datetime(icv.CHARTTIME).dt.floor('D')
imv = pd.read_parquet(DATA / 'inputevents_mv.parquet')
imv = imv[imv.AMOUNTUOM.str.lower().isin(['ml'])]
imv['date'] = pd.to_datetime(imv.ENDTIME).dt.floor('D')
inp = pd.concat([icv[['ICUSTAY_ID', 'date', 'AMOUNT']], imv[['ICUSTAY_ID', 'date', 'AMOUNT']]])
total_in = inp.groupby(['ICUSTAY_ID', 'date']).AMOUNT.sum().rename('total_input')
out = pd.read_parquet(DATA / 'outputevents.parquet')
out = out[out.VALUEUOM.str.lower().isin(['ml'])]
out['date'] = pd.to_datetime(out.CHARTTIME).dt.floor('D')
total_out = out.groupby(['ICUSTAY_ID', 'date']).VALUE.sum().rename('total_output')
io = pd.concat([total_in, total_out], axis=1).reset_index()
io['io_balance'] = io.total_input.fillna(0) - io.total_output.fillna(0)
io['rec_time'] = io.date + pd.Timedelta(1, 'D')          # original: rec_time += 1 day
io = io.drop(columns='date')
del icv, imv, inp, out
log(f'io rows {len(io)}')

# ---------------------------------------------------------------- hourly grid per stay
st['t0'] = st.INTIME.dt.floor('h')
st['t1'] = st.OUTTIME.dt.floor('h')
# event stays: rows after the event are dropped anyway -> cut the grid at the event to save memory
st.loc[st.first_event_time.notna(), 't1'] = st.loc[st.first_event_time.notna(), 'first_event_time'].dt.floor('h')
n_hours = ((st.t1 - st.t0) / pd.Timedelta(1, 'h')).astype(int) + 1
grid = pd.DataFrame({
    'ICUSTAY_ID': np.repeat(st.ICUSTAY_ID.to_numpy(), n_hours),
    'rec_time': np.concatenate([pd.date_range(a, periods=n, freq='h').to_numpy() for a, n in zip(st.t0, n_hours)]),
})
log(f'grid rows {len(grid)}')

df = grid.merge(hourly, on=['ICUSTAY_ID', 'rec_time'], how='left')
df = df.merge(be, on=['ICUSTAY_ID', 'rec_time'], how='left')
df = df.merge(io, on=['ICUSTAY_ID', 'rec_time'], how='left')
del grid, hourly, be, io

# ---------------------------------------------------------------- static + labels (2-1.add_event rules)
static = st[['ICUSTAY_ID', 'SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'DOB', 'GENDER', 'first_event_time']].rename(
    columns={'INTIME': 'adm', 'OUTTIME': 'dis_date', 'DOB': 'birthdate'})
df = df.merge(static, on='ICUSTAY_ID', how='left')
df['sex'] = (df.GENDER == 'M').astype(float)
# year/day components instead of a timedelta: MIMIC shifts DOB of >89y patients by 300 years -> ns overflow
df['age'] = (df.rec_time.dt.year - df.birthdate.dt.year) + (df.rec_time.dt.dayofyear - df.birthdate.dt.dayofyear) / 365.25
df.loc[df.age > 120, 'age'] = 91.4          # MIMIC convention for the shifted >89y group
df = df.drop(columns=['GENDER'])

df['window_start'] = df.first_event_time - pd.Timedelta(PRE_WINDOW_H, 'h')
df['is_pre_event'] = 1
df['is_event'] = 0
ev = df.first_event_time.notna()
# original: event stays whose 8h window starts before admission are removed entirely
bad_stays = df.loc[ev & (df.window_start < df.adm), 'ICUSTAY_ID'].unique()
df = df[~df.ICUSTAY_ID.isin(bad_stays)]
ev = df.first_event_time.notna()
log(f'event stays dropped ({PRE_WINDOW_H}h window before intime): {len(bad_stays)}; remaining event stays {df.loc[ev, "ICUSTAY_ID"].nunique()}')
pre = ev & (df.rec_time < df.window_start)
win = ev & (df.rec_time >= df.window_start) & (df.rec_time <= df.first_event_time)
post = ev & (df.rec_time > df.first_event_time)
df = df[~post]
df.loc[pre[~post], ['is_event', 'is_pre_event']] = [1, 1]
df.loc[win[~post], ['is_event', 'is_pre_event']] = [1, 0]

# ---------------------------------------------------------------- split (patient level)
rng = np.random.RandomState(SEED)
subjects = df.SUBJECT_ID.unique()
rng.shuffle(subjects)
n = len(subjects)
split = {s: 'train' for s in subjects[:int(0.7 * n)]}
split.update({s: 'val' for s in subjects[int(0.7 * n):int(0.85 * n)]})
split.update({s: 'test' for s in subjects[int(0.85 * n):]})
df['dataset'] = df.SUBJECT_ID.map(split)

df = df.rename(columns={'ICUSTAY_ID': 'pid', 'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'})
df['rec_time_datetime'] = df.rec_time
df = df.sort_values(['pid', 'rec_time']).reset_index(drop=True)
out_path = DATA / ('icu_allstays.csv' if EVENT == 'none' else f'icu_{EVENT}_w{PRE_WINDOW_H}h.csv')
df.to_csv(out_path, index=False)

log(f'saved {out_path}  shape={df.shape}')
print(df.dataset.value_counts().to_dict())
print(pd.crosstab(df.is_event, df.is_pre_event))
print('stays', df.pid.nunique(), 'event stays', df.loc[df.is_event == 1, 'pid'].nunique())
print('missing rate per feature:')
print(df.drop(columns=['pid', 'subject_id', 'hadm_id', 'rec_time', 'rec_time_datetime', 'adm', 'dis_date', 'birthdate',
                       'first_event_time', 'window_start', 'dataset', 'is_event', 'is_pre_event']).isna().mean().round(3).to_string())
