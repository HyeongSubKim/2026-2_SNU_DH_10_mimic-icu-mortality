"""Operating points (flagged rate / sensitivity / specificity / PPV / NPV on the test split) for the SOFA score alone and
the XGBoost / TabPFN feature tiers of 09 / 10, under two threshold rules:
  A. Youden's J maximised on the validation split (for SOFA: over integer cut-offs)
  B. threshold matched to the alarm burden of SOFA >= 10 on the test split (same flagged fraction for every model)
Reads results/sofa_tiers*/preds_W{W}.parquet and tabpfn_preds_W{W}.parquet; writes operating_points.md in each dir.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TIERS = ['1 SOFA inputs', '2 + lactate', '3 + base excess', '4 + lactate + base excess']
SHORT = {'1 SOFA inputs': '1 SOFA 입력', '2 + lactate': '2 +lactate', '3 + base excess': '3 +BE', '4 + lactate + base excess': '4 +둘 다'}


def op(y, p, thr):
    pred = p >= thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    return {'flag': pred.mean(), 'sens': tp / max(tp + fn, 1), 'spec': tn / max(tn + fp, 1), 'ppv': tp / max(tp + fp, 1), 'npv': tn / max(tn + fn, 1)}


def youden_thr(y, p):
    cand = np.unique(p)
    j = [(op(y, p, t)['sens'] + op(y, p, t)['spec'] - 1, t) for t in cand]
    return max(j)[1]


def fmt(o):
    return f"{o['flag']*100:.0f}% / {o['sens']:.2f} / {o['spec']:.2f} / {o['ppv']:.2f}"


def main():
    for version in ['sofa_tiers', 'sofa_tiers_demo']:
        d = ROOT / 'results' / version
        if not (d / 'preds_W6.parquet').exists():
            print('skip', version); continue
        md = [f'# Operating points — {version}\n\nCell = flagged rate / sensitivity / specificity / PPV on the test split.\n']
        tables = {'A': {}, 'B': {}}; info = {}
        for W in [6, 24]:
            xg = pd.read_parquet(d / f'preds_W{W}.parquet'); tp = pd.read_parquet(d / f'tabpfn_preds_W{W}.parquet')
            val = (xg.dataset == 'val').to_numpy(); te = (xg.dataset == 'test').to_numpy(); y = xg.y28.to_numpy()
            s = xg.sofa_total.fillna(0).to_numpy()
            thr_sofa = youden_thr(y[val], s[val])
            flag_b = (s[te] >= 10).mean()                       # alarm burden of SOFA >= 10 on test
            info[W] = {'sofa_youden_cut': int(thr_sofa), 'matched_flag_rate': float(flag_b)}
            scorers = {('SOFA', 'SOFA'): s}
            for t in TIERS:
                scorers[('XGB', t)] = xg[t].to_numpy(); scorers[('TabPFN', t)] = tp[t].to_numpy()
            for (model, t), p in scorers.items():
                thr_a = thr_sofa if model == 'SOFA' else youden_thr(y[val], p[val])
                oa = op(y[te], p[te], thr_a)
                thr_b = 10 if model == 'SOFA' else np.quantile(p[te], 1 - flag_b)
                ob = op(y[te], p[te], thr_b)
                tables['A'].setdefault(t, {})[(model, W)] = (oa, thr_a); tables['B'].setdefault(t, {})[(model, W)] = (ob, thr_b)
        for rule, title in [('A', 'A. threshold = Youden-J optimum on the validation split'),
                            ('B', 'B. threshold matched to the flagged fraction of SOFA >= 10 on the test split')]:
            md.append(f'\n## {title}\n\n')
            if rule == 'A':
                md.append(f"SOFA cut-off chosen on validation: >= {info[6]['sofa_youden_cut']} (6 h), >= {info[24]['sofa_youden_cut']} (24 h).\n\n")
            else:
                md.append(f"Flagged fraction: {info[6]['matched_flag_rate']*100:.1f} % (6 h), {info[24]['matched_flag_rate']*100:.1f} % (24 h).\n\n")
            md.append('| tier | XGB 6h | TabPFN 6h | XGB 24h | TabPFN 24h |\n|---|---|---|---|---|\n')
            r = tables[rule]['SOFA']
            md.append(f"| **SOFA score 단독** | {fmt(r[('SOFA', 6)][0])} | {fmt(r[('SOFA', 6)][0])} | {fmt(r[('SOFA', 24)][0])} | {fmt(r[('SOFA', 24)][0])} |\n")
            for t in TIERS:
                r = tables[rule][t]
                md.append(f"| {SHORT[t]} | {fmt(r[('XGB', 6)][0])} | {fmt(r[('TabPFN', 6)][0])} | {fmt(r[('XGB', 24)][0])} | {fmt(r[('TabPFN', 24)][0])} |\n")
        (d / 'operating_points.md').write_text(''.join(md))
        print(''.join(md))


if __name__ == '__main__':
    main()
