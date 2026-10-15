"""Transformer baseline for the SOFA feature-tier experiment: TabPFN v2 (pre-trained tabular transformer, in-context
learning, no hyper-parameter search) on the exact feature tables / splits written by 09_sofa_feature_tiers.py.

Run with an environment that has `tabpfn` + torch (the ehr-llm-benchmark venv here). Uses the openly hosted
TabPFN-v2 classifier checkpoint (Prior-Labs/TabPFN-v2-clf on Hugging Face); TabPFN v2 handles up to ~10k training
rows and 500 features natively, which fits both cohorts.

Usage: HF_HOME=... python scripts/10_tabpfn_tiers.py [--windows 6 24] [--device cuda]
Output: results/sofa_tiers/tabpfn_metrics.json, appended section in summary.md
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'sofa_tiers'
META = ['SUBJECT_ID', 'dataset', 'y28', 'age', 'sex']
TIERS = {
    '1 SOFA inputs': lambda cols: [c for c in cols if not c.startswith(('lactate_', 'base_excess_'))],
    '2 + lactate': lambda cols: [c for c in cols if not c.startswith('base_excess_')],
    '3 + base excess': lambda cols: [c for c in cols if not c.startswith('lactate_')],
    '4 + lactate + base excess': lambda cols: list(cols),
}


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); p = np.asarray(p); a, b = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i])); b.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-estimators', type=int, default=8, help='TabPFN ensemble members (default of the library)')
    args = ap.parse_args()
    hf = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    ckpts = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))
    model_path = ckpts[0] if ckpts else None          # None -> library default (needs a Prior Labs token for v2.5+)
    print('checkpoint:', model_path, flush=True)

    results, md = {}, ['\n## Transformer baseline: TabPFN v2 (same cohorts, splits and feature sets; train = train + val, no tuning)\n\n']
    for W in args.windows:
        df = pd.read_parquet(OUT / f'features_W{W}.parquet')
        feats = [c for c in df.columns if c not in META]
        tr = df.dataset.isin(['train', 'val']).to_numpy(); te = (df.dataset == 'test').to_numpy()
        y = df.y28.to_numpy(); yt = y[te]
        results[W] = {'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'tiers': {}}
        md.append(f'### W = {W} h (train+val {int(tr.sum()):,}, test {int(te.sum()):,})\n\n| feature set | # features | test AUROC [95 % CI] | test AUPRC [95 % CI] | fit+predict time |\n|---|---|---|---|---|\n')
        for tier, pick in TIERS.items():
            cols = pick(feats)
            X = df[cols].to_numpy(dtype='float32')
            t0 = time.time()
            clf = TabPFNClassifier(device=args.device, model_path=model_path, n_estimators=args.n_estimators, random_state=0)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            dt = time.time() - t0
            auc_ci, ap_ci = boot_ci(yt, p)
            r = {'n_features': len(cols), 'test_auroc': float(roc_auc_score(yt, p)), 'test_auroc_ci95': auc_ci,
                 'test_auprc': float(average_precision_score(yt, p)), 'test_auprc_ci95': ap_ci, 'seconds': dt}
            results[W]['tiers'][tier] = r
            print(f"W={W} {tier:28s} feats {len(cols):3d} AUROC {r['test_auroc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] "
                  f"AUPRC {r['test_auprc']:.3f}  {dt:.0f}s", flush=True)
            md.append(f"| {tier} (TabPFN v2) | {len(cols)} | {r['test_auroc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] | "
                      f"{r['test_auprc']:.3f} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}] | {dt:.0f} s |\n")
        md.append('\n')
    json.dump(results, open(OUT / 'tabpfn_metrics.json', 'w'), indent=1)
    with open(OUT / 'summary.md', 'a') as f:
        f.write(''.join(md))
    print('saved', OUT / 'tabpfn_metrics.json')


if __name__ == '__main__':
    main()
