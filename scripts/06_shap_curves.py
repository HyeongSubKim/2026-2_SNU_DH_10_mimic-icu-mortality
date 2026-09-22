"""ROC / PR curves and SHAP values for the trained models of one task.

--task hourly --prefix <p>       : the hourly models saved as results/<p>_rf, <p>_xgb, <p>_xgb_nan (whichever exist)
--task early  --event death|vent : the early-window stay-level models (all W found; XGB + RF)

SHAP: XGBoost via the booster's own TreeSHAP (pred_contribs, exact, GPU); RF via shap.TreeExplainer on a 3,000-row
sample (exact TreeSHAP on 500 trees x 1,000 leaves is slow). mean|SHAP| is computed on <= 200k test rows
(all positives kept), beeswarm on a 20k random subset of those.
Output: results/shap_<task>_<prefix|event>/  roc_pr[_W].png, shap_bar_*.png, shap_beeswarm_*.png, shap_mean_abs.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.RandomState(0)


def sample_rows(y, n_max):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) + len(neg) <= n_max:
        return np.arange(len(y))
    neg = RNG.choice(neg, max(n_max - len(pos), 0), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def xgb_shap(booster, X, feats, device):
    booster.set_param({'device': device})
    it = booster.attr('best_iteration')
    kw = {'iteration_range': (0, int(it) + 1)} if it is not None else {}
    contrib = booster.predict(xgb.DMatrix(X, feature_names=feats), pred_contribs=True, **kw)
    return contrib[:, :-1]      # last column = bias


def rf_shap(model, X):
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    return sv[..., 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else sv[1]


def shap_plots(sv, X, feats, tag, out_dir, rows):
    m = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    rows.append(m.rename(tag))
    plt.figure(figsize=(7, 6))
    m.head(20)[::-1].plot.barh()
    plt.xlabel('mean |SHAP| (log-odds)'); plt.title(f'{tag}  (n={len(X):,})')
    plt.tight_layout(); plt.savefig(out_dir / f'shap_bar_{tag}.png', dpi=110); plt.close()
    k = RNG.choice(len(X), min(20000, len(X)), replace=False)
    shap.summary_plot(sv[k], X.iloc[k] if hasattr(X, 'iloc') else X[k], feature_names=feats, max_display=20, show=False)
    plt.title(tag); plt.tight_layout(); plt.savefig(out_dir / f'shap_beeswarm_{tag}.png', dpi=110); plt.close()


def curves(models, out_path, title):
    """models: list of (label, y, p)"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for label, y, p in models:
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f'{label}  AUROC={roc_auc_score(y, p):.3f}')
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f'{label}  AUPRC={average_precision_score(y, p):.3f}')
    ax[0].plot([0, 1], [0, 1], 'k--', lw=.8); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].set_title(f'ROC — {title}')
    y0 = models[0][1]; ax[1].axhline(y0.mean(), color='k', ls='--', lw=.8, label=f'prevalence={y0.mean():.3f}')
    ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].set_title(f'PR — {title}')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['hourly', 'early'], required=True)
    ap.add_argument('--event', default='death')
    ap.add_argument('--prefix', default='icu_death_w24h', help='hourly task: results/<prefix>_{rf,xgb,xgb_nan}')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-shap', type=int, default=200_000)
    ap.add_argument('--n-shap-rf', type=int, default=3000)
    args = ap.parse_args()
    out_dir = ROOT / 'results' / (f'shap_hourly_{args.prefix}' if args.task == 'hourly' else f'shap_early_{args.event}')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.task == 'hourly':
        runs = [('XGB_median_fill', f'{args.prefix}_xgb', 'xgb_pred'), ('XGB_native_NaN', f'{args.prefix}_xgb_nan', 'xgb_pred'),
                ('RF', f'{args.prefix}_rf', 'rf_pred')]
        cur = []
        for tag, name, pcol in runs:
            res = ROOT / 'results' / name
            if not (res / 'metrics.json').exists():
                print(f'{tag}: not finished, skipped'); continue
            met = json.load(open(res / 'metrics.json'))
            feats = met['features']
            csv = next(res.glob('baseline_*_pred.csv'))
            te = pd.read_csv(csv, usecols=feats + ['is_event', pcol])
            y, p = te.is_event.to_numpy(), te[pcol].to_numpy()
            cur.append((tag, y, p))
            print(f'{tag}: test rows {len(te):,} AUROC {roc_auc_score(y, p):.4f}', flush=True)
            if tag.startswith('XGB'):
                idx = sample_rows(y, args.n_shap)
                X = te.iloc[idx][feats]
                b = xgb.Booster(); b.load_model(ROOT / 'models' / name / f'{met["best_trial"]}.json')
                b.set_attr(best_iteration=str(met['best_iteration']))
                sv = xgb_shap(b, X, feats, args.device)
            else:
                idx = sample_rows(y, args.n_shap_rf)
                X = te.iloc[idx][feats]
                m = pickle.load(open(ROOT / 'models' / name / f'{met["best_trial"]}.pickle', 'rb'))
                sv = rf_shap(m, X.to_numpy(np.float32))
            shap_plots(sv, X, feats, tag, out_dir, rows)
        curves(cur, out_dir / 'roc_pr.png', f'{args.prefix}, hourly model, test')

    else:
        res = ROOT / 'results' / f'early_window_{args.event}'
        mdir = ROOT / 'models' / f'early_window_{args.event}'
        for W in sorted(int(f.stem[1:-5]) for f in res.glob('W*_test.parquet')):
            te = pd.read_parquet(res / f'W{W}_test.parquet')
            feats = [c for c in te.columns if c not in ('xgb_pred', 'rf_pred', 'y', 'hours_to_end')]
            y = te.y.to_numpy()
            curves([(f'XGB W={W}h', y, te.xgb_pred.to_numpy()), (f'RF W={W}h', y, te.rf_pred.to_numpy())],
                   out_dir / f'roc_pr_W{W}.png', f'{args.event}, first {W}h -> stay outcome, test')
            b = xgb.Booster(); b.load_model(mdir / f'W{W}_xgb.json')
            shap_plots(xgb_shap(b, te[feats], feats, args.device), te[feats], feats, f'XGB_W{W}h', out_dir, rows)
            rf = pickle.load(open(mdir / f'W{W}_rf.pickle', 'rb'))
            idx = sample_rows(y, args.n_shap_rf)
            Xr = te.iloc[idx][feats].fillna(rf['median'])
            shap_plots(rf_shap(rf['model'], Xr.to_numpy(np.float32)), Xr, feats, f'RF_W{W}h', out_dir, rows)
            print(f'W={W}: done', flush=True)

    pd.concat(rows, axis=1).to_csv(out_dir / 'shap_mean_abs.csv')
    print('saved', out_dir)


if __name__ == '__main__':
    main()
