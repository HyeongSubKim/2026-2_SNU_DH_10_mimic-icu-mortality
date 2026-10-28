"""Unimodal tier table with 95 % bootstrap CIs (XGBoost / TabPFN x tiers x 6 h / 24 h, with and without age/sex, SOFA score alone).
Reads results/sofa_tiers{,_demo}/{metrics.json,tabpfn_metrics.json} (CIs stored at training time) and bootstraps the SOFA
score from results/sofa_tiers_demo/preds_W{W}.parquet. Writes results/sofa_tiers_demo/unimodal_table_ci.{md,json}."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
TIERS = ['1 SOFA inputs', '2 + lactate', '3 + base excess', '4 + lactate + base excess']
LABEL = {'1 SOFA inputs': '1 SOFA 입력', '2 + lactate': '2 + lactate', '3 + base excess': '3 + BE', '4 + lactate + base excess': '4 + lactate + BE'}


def boot(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); a, b = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i])); b.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


fmt = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
out = {}; rows = []
for W in (6, 24):
    p = pd.read_parquet(R / 'sofa_tiers_demo' / f'preds_W{W}.parquet'); te = p.dataset == 'test'
    y = p.loc[te, 'y28'].values.astype(int); s = p.loc[te, 'sofa_total'].values.astype(float)
    ac, pc = boot(y, s); out[f'sofa_alone_W{W}'] = {'auroc': float(roc_auc_score(y, s)), 'auroc_ci95': ac, 'auprc': float(average_precision_score(y, s)), 'auprc_ci95': pc}
for demo, name in ((False, 'sofa_tiers'), (True, 'sofa_tiers_demo')):
    xg = json.load(open(R / name / 'metrics.json')); tp = json.load(open(R / name / 'tabpfn_metrics.json'))
    for t in TIERS:
        row = {'tier': LABEL[t], 'demo': demo}
        for W in ('6', '24'):
            x = xg[W]['tiers'][t]; tb = tp[W]['tiers'][t]
            row[f'xgb_{W}'] = {k: x[k] for k in ('test_auroc', 'test_auroc_ci95', 'test_auprc', 'test_auprc_ci95')}
            row[f'tabpfn_{W}'] = {k: tb[k] for k in ('test_auroc', 'test_auroc_ci95', 'test_auprc', 'test_auprc_ci95')}
        rows.append(row)
out['rows'] = rows; json.dump(out, open(R / 'sofa_tiers_demo' / 'unimodal_table_ci.json', 'w'), indent=1)

md = ['# Unimodal results with 95 % bootstrap CIs (test; 6 h n = 853, 24 h n = 1,374)', '',
      '## AUROC [95 % CI]', '', '| tier | XGB 6h | TabPFN 6h | XGB 24h | TabPFN 24h |', '|---|---|---|---|---|',
      f"| SOFA score 단독 | {fmt(out['sofa_alone_W6']['auroc'], out['sofa_alone_W6']['auroc_ci95'])} | (same) | {fmt(out['sofa_alone_W24']['auroc'], out['sofa_alone_W24']['auroc_ci95'])} | (same) |"]
for demo in (False, True):
    if demo:
        md.append('| **+age/sex** | | | | |')
    for r in [r for r in rows if r['demo'] == demo]:
        b = '**' if r['tier'].startswith('4') and demo else ''
        md.append(f"| {b}{r['tier']}{b} | " + ' | '.join(fmt(r[k]['test_auroc'], r[k]['test_auroc_ci95']) for k in ('xgb_6', 'tabpfn_6', 'xgb_24', 'tabpfn_24')) + ' |')
md += ['', '## AUPRC [95 % CI]', '', '| tier | XGB 6h | TabPFN 6h | XGB 24h | TabPFN 24h |', '|---|---|---|---|---|',
       f"| SOFA score 단독 | {fmt(out['sofa_alone_W6']['auprc'], out['sofa_alone_W6']['auprc_ci95'])} | (same) | {fmt(out['sofa_alone_W24']['auprc'], out['sofa_alone_W24']['auprc_ci95'])} | (same) |"]
for demo in (False, True):
    if demo:
        md.append('| **+age/sex** | | | | |')
    for r in [r for r in rows if r['demo'] == demo]:
        b = '**' if r['tier'].startswith('4') and demo else ''
        md.append(f"| {b}{r['tier']}{b} | " + ' | '.join(fmt(r[k]['test_auprc'], r[k]['test_auprc_ci95']) for k in ('xgb_6', 'tabpfn_6', 'xgb_24', 'tabpfn_24')) + ' |')
md += ['', 'CI: 1,000 bootstrap resamples of the test split (percentile). XGBoost: Optuna-tuned, GPU, native NaN; TabPFN v2: open checkpoint, 8 estimators, context = train + val.']
(R / 'sofa_tiers_demo' / 'unimodal_table_ci.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))
