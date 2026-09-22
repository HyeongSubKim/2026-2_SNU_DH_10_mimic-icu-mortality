"""EDA of MIMIC-III (raw) and of the ICU ventilation dataset built by 01_build_dataset.py.

Usage : python 03_eda.py [--event vent|death]
Output: results/eda[_<event>]/EDA.md + *.png
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))   # folder with the PhysioNet csv.gz files
ap = argparse.ArgumentParser()
ap.add_argument('--event', choices=['vent', 'death'], default='death')
ap.add_argument('--dataset', default=None, help='csv from 01_build_dataset.py (default data/icu_<event>_w24h.csv)')
ap.add_argument('--build-log', default=None, help='stdout of 01_build_dataset.py (cohort counts are read from it)')
args = ap.parse_args()
EVENT = args.event
DATASET = Path(args.dataset) if args.dataset else ROOT / 'data' / f'icu_{EVENT}_w24h.csv'
OUT = ROOT / 'results' / f'eda_{DATASET.stem}'
OUT.mkdir(parents=True, exist_ok=True)
BUILD_LOG = Path(args.build_log or ROOT / 'results' / f'01_build_{DATASET.stem}.log').read_text()
md = []


def h(title):
    md.append(f'\n## {title}\n')


def table(df, floatfmt='.3f'):
    md.append(df.to_markdown(floatfmt=floatfmt) + '\n')


def fig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=110)
    plt.close()
    md.append(f'![{name}]({name})\n')


# ============================================================ 1. raw MIMIC-III
h('1. MIMIC-III v1.4 raw overview')
pat = pd.read_csv(RAW / 'PATIENTS.csv.gz', parse_dates=['DOB', 'DOD'])
adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', parse_dates=['ADMITTIME', 'DISCHTIME'])
icu = pd.read_csv(RAW / 'ICUSTAYS.csv.gz', parse_dates=['INTIME', 'OUTTIME'])
sizes = {p.name: p.stat().st_size / 1e6 for p in RAW.glob('*.csv.gz')}
md.append(f'- tables: {len(sizes)} csv.gz, total {sum(sizes.values())/1e3:.1f} GB compressed; '
          f'largest CHARTEVENTS {sizes["CHARTEVENTS.csv.gz"]/1e3:.1f} GB, NOTEEVENTS {sizes["NOTEEVENTS.csv.gz"]/1e3:.1f} GB\n')
md.append(f'- PATIENTS {len(pat):,} / ADMISSIONS {len(adm):,} / ICUSTAYS {len(icu):,} '
          f'(subjects with ICU stay {icu.SUBJECT_ID.nunique():,})\n')
md.append(f'- dates are shifted per patient into 2100–2200; DOB of >89y patients shifted −300y (age → 300+)\n')

adm = adm.merge(pat[['SUBJECT_ID', 'GENDER', 'DOB', 'EXPIRE_FLAG']], on='SUBJECT_ID')
adm['age'] = (adm.ADMITTIME.dt.year - adm.DOB.dt.year) - ((adm.ADMITTIME.dt.dayofyear < adm.DOB.dt.dayofyear).astype(int))
adm['age_group'] = pd.cut(adm.age, [-1, 0, 17, 39, 59, 79, 89, 1000],
                          labels=['neonate', '1-17', '18-39', '40-59', '60-79', '80-89', '>89 (shifted)'])
t = adm.groupby('age_group', observed=True).agg(admissions=('HADM_ID', 'size'),
                                                  hosp_mortality=('HOSPITAL_EXPIRE_FLAG', 'mean'))
t['admissions_pct'] = t.admissions / t.admissions.sum() * 100
md.append('### admissions by age group (age at ADMITTIME)\n'); table(t)
t = adm.groupby('ADMISSION_TYPE').agg(admissions=('HADM_ID', 'size'), hosp_mortality=('HOSPITAL_EXPIRE_FLAG', 'mean'))
md.append('### admissions by type\n'); table(t)
md.append(f'### sex (PATIENTS): {pat.GENDER.value_counts().to_dict()}\n')
eth = adm.ETHNICITY.str.split(' - ').str[0].str.split('/').str[0].value_counts().head(6)
md.append('### ethnicity (top 6, collapsed)\n'); table(eth.to_frame('admissions'), floatfmt='.0f')
md.append(f'### in-hospital mortality: {adm.HOSPITAL_EXPIRE_FLAG.mean():.3f} '
          f'(neonates excluded: {adm.loc[adm.age>0, "HOSPITAL_EXPIRE_FLAG"].mean():.3f})\n')

icu = icu.merge(adm[['HADM_ID', 'age', 'HOSPITAL_EXPIRE_FLAG']], on='HADM_ID')
t = icu.groupby('FIRST_CAREUNIT').agg(stays=('ICUSTAY_ID', 'size'), subjects=('SUBJECT_ID', 'nunique'),
                                      los_median_d=('LOS', 'median'), los_mean_d=('LOS', 'mean'),
                                      age_median=('age', 'median'), hosp_mortality=('HOSPITAL_EXPIRE_FLAG', 'mean'))
t.loc['ALL'] = [len(icu), icu.SUBJECT_ID.nunique(), icu.LOS.median(), icu.LOS.mean(), icu.age.median(), icu.HOSPITAL_EXPIRE_FLAG.mean()]
md.append('### ICU stays by first care unit\n'); table(t, floatfmt='.2f')
t = pd.crosstab(icu.FIRST_CAREUNIT, icu.DBSOURCE)
md.append('### charting system (DBSOURCE) by unit — CareVue (2001–08) vs MetaVision (2008–12) use different ITEMIDs\n')
table(t, floatfmt='.0f')
t = icu.groupby('SUBJECT_ID').ICUSTAY_ID.size().value_counts().sort_index()
md.append(f'### ICU stays per subject: 1 stay {t[1]:,} subjects; 2: {t[2]:,}; 3+: {t[t.index>=3].sum():,}\n')

fig_, ax = plt.subplots(1, 3, figsize=(15, 4))
adm.loc[adm.age.between(1, 89), 'age'].hist(bins=45, ax=ax[0]); ax[0].set_title('age at admission (1–89y)')
ax[0].set_xlabel('years')
for u, g in icu.groupby('FIRST_CAREUNIT'):
    ax[1].hist(np.log10(g.LOS.clip(1e-2)), bins=40, histtype='step', label=u, density=True)
ax[1].set_title('ICU LOS by unit'); ax[1].set_xlabel('log10(days)'); ax[1].legend(fontsize=8)
icu.INTIME.dt.year.value_counts().sort_index().plot(ax=ax[2]); ax[2].set_title('ICU admissions per (shifted) year')
fig('raw_overview.png')

# ============================================================ 2. built dataset — cohort
h(f'2. Built dataset ({DATASET.name}, event = {EVENT}) — cohort')
df = pd.read_csv(DATASET)
for c in ['rec_time', 'adm', 'dis_date', 'first_event_time', 'window_start']:
    df[c] = pd.to_datetime(df[c], format='ISO8601')   # midnight stamps are written as date-only
df = df.drop(columns=['rec_time_datetime', 'birthdate'])
FEATS = sorted(set(df.columns) - {'pid', 'rec_time', 'adm', 'dis_date', 'first_event_time', 'window_start',
                                  'dataset', 'is_event', 'is_pre_event', 'subject_id', 'hadm_id'})
st = pd.read_parquet(ROOT / 'data' / 'icustays.parquet')
stay = df.groupby('pid').agg(subject_id=('subject_id', 'first'), dataset=('dataset', 'first'), adm=('adm', 'first'),
                             dis=('dis_date', 'first'), ev_time=('first_event_time', 'first'), rows=('rec_time', 'size'),
                             age=('age', 'first'), sex=('sex', 'first')).reset_index()
stay = stay.merge(st[['ICUSTAY_ID', 'FIRST_CAREUNIT', 'DBSOURCE', 'ADMISSION_TYPE', 'LOS']].rename(columns={'ICUSTAY_ID': 'pid'}), on='pid')
stay['event'] = stay.ev_time.notna().astype(int)
stay['hours_to_event'] = (stay.ev_time - stay.adm) / pd.Timedelta(1, 'h')

n_adult = len(st)
n_ev, n_base = map(int, re.search(r'stays with event: (\d+) / (\d+)', BUILD_LOG).groups())
n_drop = int(re.search(r'event stays dropped \([^)]*window before intime\): (\d+)', BUILD_LOG).group(1))
excl = re.search(r'death outside this ICU stay \(excluded\): (\d+)', BUILD_LOG)
md.append('### cohort flow\n')
md.append(f'| step | stays |\n|---|---|\n| ICUSTAYS total | {len(icu):,} |\n| adult units (NICU excluded) | {n_adult:,} |\n'
          + (f'| admissions ending in death outside this ICU stay (excluded) | {int(excl.group(1)):,} |\n' if excl else '')
          + f'| with event ({ {"vent": "first vent/intubation", "death": "in-ICU death"}[EVENT] } inside the stay) | {n_ev:,} / {n_base:,} |\n'
          f'| event stays dropped: label window starts before ICU intime | {n_drop:,} |\n'
          f'| **final** | **{len(stay):,}** (event {stay.event.sum():,} = {stay.event.mean()*100:.1f} %, '
          f'subjects {stay.subject_id.nunique():,}) |\n')
md.append(f'- rows {len(df):,}; label crosstab (is_event × is_pre_event):\n')
table(pd.crosstab(df.is_event, df.is_pre_event), floatfmt='.0f')
md.append('- training uses is_event=0 rows (negative) and is_event=1 & is_pre_event=0 rows (positive, the 8h window); '
          'is_event=1 & is_pre_event=1 rows (event stays before the window) are excluded.\n')

for col in ['FIRST_CAREUNIT', 'DBSOURCE', 'ADMISSION_TYPE', 'dataset']:
    t = stay.groupby(col).agg(stays=('pid', 'size'), event_rate=('event', 'mean'), hours_to_event_median=('hours_to_event', 'median'),
                              rows_per_stay_median=('rows', 'median'), age_median=('age', 'median'))
    md.append(f'### by {col}\n'); table(t, floatfmt='.2f')
stay['age_group'] = pd.cut(stay.age, [17, 39, 59, 79, 89, 100], labels=['18-39', '40-59', '60-79', '80-89', '>89'])
t = stay.groupby('age_group', observed=True).agg(stays=('pid', 'size'), event_rate=('event', 'mean'))
t2 = stay.groupby('sex').agg(stays=('pid', 'size'), event_rate=('event', 'mean')); t2.index = ['F', 'M']
md.append('### by age group / sex\n'); table(pd.concat([t, t2]), floatfmt='.3f')

fig_, ax = plt.subplots(1, 3, figsize=(15, 4))
stay.loc[stay.event == 1, 'hours_to_event'].clip(upper=240).hist(bins=60, ax=ax[0])
ax[0].set_title('hours from ICU intime to first vent (event stays, clipped 240h)'); ax[0].set_xlabel('hours')
for e, g in stay.groupby('event'):
    ax[1].hist(np.log10(g.rows), bins=40, histtype='step', density=True, label=f'event={e}')
ax[1].set_title('rows (hours) per stay in the dataset'); ax[1].set_xlabel('log10(hours)'); ax[1].legend()
t = stay.groupby('FIRST_CAREUNIT').event.mean()
t.plot.bar(ax=ax[2]); ax[2].set_title('event rate by unit'); ax[2].set_ylabel('fraction of stays')
fig('cohort.png')

# ============================================================ 3. features
h('3. Features')
neg = df.is_event == 0
pos = (df.is_event == 1) & (df.is_pre_event == 0)
pre = (df.is_event == 1) & (df.is_pre_event == 1)
lab = np.where(pos, 'pos(label window)', np.where(pre, 'pre-window(excluded)', 'neg'))

miss = pd.DataFrame({
    'raw_missing_all': df[FEATS].isna().mean(),
    'raw_missing_neg': df.loc[neg, FEATS].isna().mean(),
    'raw_missing_pos': df.loc[pos, FEATS].isna().mean(),
})
ff = df.sort_values(['pid', 'rec_time']).groupby('pid')[FEATS].ffill()
miss['after_ffill_all'] = ff.isna().mean()
miss['after_ffill_neg'] = ff[neg.values].isna().mean()
miss['after_ffill_pos'] = ff[pos.values].isna().mean()
miss['stays_ever_measured'] = df.groupby('pid')[FEATS].apply(lambda g: g.notna().any()).mean()
md.append('### missingness per feature (row level, hourly grid) — raw vs after per-stay forward fill\n'); table(miss)
md.append('- HR/RR/SpO2 are charted ~hourly; BP is hourly only when arterial (A*BP) or NIBP is used; '
          'FiO2 / O2 flow are charted mostly when oxygen is delivered; weight/height/base excess/I-O are sparse events '
          '(I/O is a daily total assigned to the next day 00:00, so at most 1 row/day is non-missing).\n')

# feature availability by charting system
t = pd.DataFrame({src: df.loc[df.pid.isin(stay.loc[stay.DBSOURCE == src, 'pid']), FEATS].notna().mean()
                  for src in ['carevue', 'metavision']})
md.append('### fraction of rows with a value, CareVue vs MetaVision (ITEMID mapping sanity check)\n'); table(t)

# distributions by label
desc = []
for f in FEATS:
    a, b = ff.loc[neg.values, f].dropna(), ff.loc[pos.values, f].dropna()
    desc.append({'feature': f, 'neg_median': a.median(), 'pos_median': b.median(), 'neg_IQR': f'{a.quantile(.25):.1f}–{a.quantile(.75):.1f}',
                 'pos_IQR': f'{b.quantile(.25):.1f}–{b.quantile(.75):.1f}',
                 'std_mean_diff': (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)})
desc = pd.DataFrame(desc).set_index('feature').sort_values('std_mean_diff', key=abs, ascending=False)
md.append('### value distribution, positive (label window) vs negative rows, after per-stay ffill\n'); table(desc, floatfmt='.2f')

plot_feats = [f for f in desc.index if f not in ('sex', 'total_input', 'total_output', 'io_balance')]
fig_, axes = plt.subplots(4, 4, figsize=(16, 13)); axes = axes.ravel()
for ax, f in zip(axes, plot_feats):
    a, b = ff.loc[neg.values, f].dropna(), ff.loc[pos.values, f].dropna()
    lo, hi = np.nanpercentile(pd.concat([a, b]), [0.5, 99.5])
    ax.hist(a.clip(lo, hi), bins=50, density=True, alpha=.5, label='neg')
    ax.hist(b.clip(lo, hi), bins=50, density=True, alpha=.5, label='pos')
    ax.set_title(f, fontsize=10); ax.legend(fontsize=7)
for ax in axes[len(plot_feats):]:
    ax.axis('off')
fig('feature_dist_by_label.png')

# correlation
c = ff[[f for f in FEATS if f != 'sex']].corr(method='spearman')
fig_, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns, fontsize=8)
plt.colorbar(im); ax.set_title('Spearman correlation (after ffill, pairwise complete)')
fig('feature_corr.png')
hi = c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
md.append('### strongest correlations (|rho| > 0.5)\n'); table(hi[hi.abs() > .5].sort_values(key=abs, ascending=False).to_frame('spearman'))

# ============================================================ 4. time structure
h('4. Time structure of the labels')
df['hrs'] = (df.rec_time - df.adm) / pd.Timedelta(1, 'h')
df['hrs_bin'] = pd.cut(df.hrs, [-1, 6, 12, 24, 48, 72, 168, 1e6], labels=['0-6', '6-12', '12-24', '24-48', '48-72', '72-168', '>168'])
t = df[neg | pos].groupby('hrs_bin', observed=True).agg(rows=('pid', 'size'), pos_rate=('is_event', 'mean'))
md.append('### positive-row rate by hours since ICU intime (training rows only)\n'); table(t, floatfmt='.4f')
if EVENT == 'vent':
    md.append('- positives concentrate in the first day: the notebook-style label has a built-in time-since-admission signal '
              '(time since admission alone is predictive).\n')

# per-window position: how far before the event is each positive row
df['h_before_event'] = ((df.first_event_time - df.rec_time) / pd.Timedelta(1, 'h'))
t = df.loc[pos].groupby(df.loc[pos, 'h_before_event'].round().clip(0, None))[FEATS[:0] + ['HR 회/min', 'RR 회/min', 'SpO2 %', 'FIO2_', 'Flow rate L/min']].agg(lambda s: s.notna().mean())
t.index.name = 'hours before event'
md.append('### raw charting rate inside the label window, by hours before the event (charting intensifies toward the event)\n'); table(t)

fig_, ax = plt.subplots(1, 2, figsize=(12, 4))
t = df[neg | pos].groupby(df[neg | pos].hrs.clip(upper=120).round()).is_event.mean()
ax[0].plot(t.index, t.values); ax[0].set_title('positive-row rate vs hours since intime'); ax[0].set_xlabel('hours'); ax[0].set_ylabel('P(pos)')
g = df.loc[pos].groupby(df.loc[pos, 'h_before_event'].round().clip(0, None))
for f in ['HR 회/min', 'RR 회/min', 'SpO2 %', 'FIO2_']:
    ax[1].plot(g[f].median().index, (g[f].median() / g[f].median().iloc[-1]).values, marker='.', label=f)
ax[1].invert_xaxis(); ax[1].set_title('median raw value inside window, relative to window start'); ax[1].set_xlabel('hours before event'); ax[1].legend()
fig('time_structure.png')

# ============================================================ 5. split
h('5. Split balance (patient-level 70/15/15, seed 0)')
t = df[neg | pos].groupby('dataset').agg(rows=('pid', 'size'), stays=('pid', 'nunique'), subjects=('subject_id', 'nunique'),
                                         pos_rows=('is_event', 'sum'), pos_rate=('is_event', 'mean'))
t['event_stays'] = df.loc[pos].groupby('dataset').pid.nunique()
md.append(''); table(t.loc[['train', 'val', 'test']], floatfmt='.4f')

(OUT / 'EDA.md').write_text('# MIMIC-III ICU ventilation dataset — EDA\n' + ''.join(md))
print('saved', OUT / 'EDA.md')
