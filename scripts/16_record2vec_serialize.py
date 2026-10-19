"""Record2Vec step 1: canonical text serialisation of the first W hours of each stay (raw units), for the SOFA cohort.

One record per stay: for every channel observed in the window, the channel name, unit and the list of
"h<hour>: value" pairs (hourly means; urine summed per hour; vasopressor = hours with an infusion). Also age / sex.
This is both the input to the LLM summariser (17) and the "no-summary" control that is embedded directly (18).
Output: results/record2vec/serialized_W{W}.parquet  (index ICUSTAY_ID; columns dataset, y28, n_tokens_est, text)
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'record2vec'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('b13', ROOT / 'scripts' / '13_build_utde_dataset.py')
b13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b13)

UNITS = {'HR': 'bpm', 'RR': '/min', 'SpO2': '%', 'Temp': '°C', 'SBP': 'mmHg', 'DBP': 'mmHg', 'MeanBP': 'mmHg', 'ASBP': 'mmHg (arterial)',
         'ADBP': 'mmHg (arterial)', 'MeanABP': 'mmHg (arterial)', 'FiO2': '%', 'O2flow': 'L/min', 'PaO2': 'mmHg', 'Platelets': 'K/uL',
         'Bilirubin': 'mg/dL', 'Creatinine': 'mg/dL', 'Lactate': 'mmol/L', 'BaseExcess': 'mmol/L', 'GCS': '/15', 'Urine': 'mL/h', 'Vasopressor': ''}
NAMES = {'HR': 'heart rate', 'RR': 'respiratory rate', 'SpO2': 'SpO2', 'Temp': 'temperature', 'SBP': 'systolic BP (non-invasive)',
         'DBP': 'diastolic BP (non-invasive)', 'MeanBP': 'mean BP (non-invasive)', 'ASBP': 'systolic BP (arterial line)',
         'ADBP': 'diastolic BP (arterial line)', 'MeanABP': 'mean arterial pressure (arterial line)', 'FiO2': 'FiO2', 'O2flow': 'O2 flow',
         'PaO2': 'PaO2', 'Platelets': 'platelets', 'Bilirubin': 'total bilirubin', 'Creatinine': 'creatinine', 'Lactate': 'lactate',
         'BaseExcess': 'base excess', 'GCS': 'GCS total', 'Urine': 'urine output', 'Vasopressor': 'vasopressor infusion'}
DEC = {'Temp': 1, 'Bilirubin': 1, 'Creatinine': 1, 'Lactate': 1, 'BaseExcess': 1, 'FiO2': 0, 'Platelets': 0, 'Urine': 0}


def serialize(g, W, age, sex):
    lines = [f'ICU patient, age {age:.0f}, sex {"male" if sex == 1 else "female"}. Measurements during the first {W} hours after ICU admission '
             f'(h0 = admission hour; labs drawn shortly before admission are listed at h0; missing hours = not measured).']
    for ch in b13.CHANNELS:
        s = g[g.channel == ch].sort_values('hour')
        if s.empty:
            continue
        if ch == 'Vasopressor':
            lines.append(f'- vasopressor infusion running at hours: {", ".join(f"h{int(h)}" for h in s.hour)}')
            continue
        d = DEC.get(ch, 0 if ch in ('HR', 'RR', 'SpO2', 'SBP', 'DBP', 'MeanBP', 'ASBP', 'ADBP', 'MeanABP', 'PaO2', 'GCS', 'O2flow') else 1)
        vals = ', '.join(f'h{int(h)}: {v:.{d}f}' for h, v in zip(s.hour, s.value))
        lines.append(f'- {NAMES[ch]} ({UNITS[ch]}): {vals}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--windows', type=int, nargs='+', default=[6, 24]); args = ap.parse_args()
    icu = pd.read_parquet(ROOT / 'data' / 'icustays.parquet', columns=['ICUSTAY_ID', 'HADM_ID', 'INTIME']).set_index('ICUSTAY_ID'); icu['INTIME'] = pd.to_datetime(icu.INTIME)
    for W in args.windows:
        f = pd.read_parquet(ROOT / 'results' / 'sofa_tiers_demo' / f'features_W{W}.parquet', columns=['SUBJECT_ID', 'dataset', 'y28', 'age', 'sex'])
        st = f.join(icu)
        long = b13.hourly_long(st, W)
        grp = {k: g for k, g in long.groupby('ICUSTAY_ID')}
        rows = []
        for sid, r in st.iterrows():
            g = grp.get(sid)
            if g is None:
                continue
            txt = serialize(g, W, r.age, r.sex)
            rows.append({'ICUSTAY_ID': sid, 'dataset': r.dataset, 'y28': int(r.y28), 'text': txt, 'n_tokens_est': len(txt) // 4})
        out = pd.DataFrame(rows).set_index('ICUSTAY_ID')
        out.to_parquet(OUT / f'serialized_W{W}.parquet')
        print(f'W={W}: {len(out):,} stays, est. tokens median {out.n_tokens_est.median():.0f}, p95 {out.n_tokens_est.quantile(.95):.0f}', flush=True)
        print(out.text.iloc[0][:900], '\n...', flush=True)


if __name__ == '__main__':
    main()
