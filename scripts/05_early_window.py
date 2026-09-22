"""Early-warning experiment: observe only the first W hours after ICU admission (W = 6, 12), predict the stay outcome.

Per stay, the hourly rows with hours_since_intime < W are aggregated (last / mean / min / max / n_obs per feature,
+ age, sex); the label is the stay-level event (in-ICU death for --event death). Stays that end (event or discharge)
within the first W hours are excluded because their outcome is already known inside the observation window.
Models: XGBoost (GPU, native NaN) and RandomForest (train-median fill), optuna on val AUROC, same patient split.

Usage: python 05_early_window.py [--event death] [--dataset data/icu_death_w24h.csv] [--windows 1 3 6 12 24] [--n-trials 50] [--device cuda:0]
Output: results/early_window_<event>/metrics.json, summary.md
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
rf_mod = import_module('02_train_rf')

ROOT = Path(__file__).resolve().parents[1]
STATIC = ['age', 'sex']


def build(df, W):
    stay = df.groupby('pid').agg(subject_id=('subject_id', 'first'), dataset=('dataset', 'first'), adm=('adm', 'first'),
                                 dis=('dis_date', 'first'), ev=('first_event_time', 'first'),
                                 age=('age', 'first'), sex=('sex', 'first'))
    stay['y'] = stay.ev.notna().astype(int)
    end = stay.ev.fillna(stay.dis)
    stay['hours_to_end'] = (end - stay.adm) / pd.Timedelta(1, 'h')
    keep = stay.hours_to_end > W
    n_excl = {'event_within_W': int((~keep & (stay.y == 1)).sum()), 'discharged_within_W': int((~keep & (stay.y == 0)).sum())}
    stay = stay[keep]

    feats = [c for c in sorted(set(df.columns) - set(rf_mod.INFO_CAND_COLS)) if c not in STATIC and c != 'hrs']
    obs = df[(df.hrs < W) & df.pid.isin(stay.index)]
    agg = obs.groupby('pid')[feats].agg(['last', 'mean', 'min', 'max', 'count'])
    agg.columns = [f'{a}__{b}' for a, b in agg.columns]
    X = stay[STATIC].join(agg)   # stays with no charted row in the window get all-NaN aggregates
    return stay, X, n_excl


def strat_auc(y, p, hours_after_window):
    """AUROC of deaths that occur within h hours after the window ends vs all survivors."""
    out = {}
    for name, (lo, hi) in {'death_le_24h': (0, 24), 'death_24_72h': (24, 72), 'death_gt_72h': (72, np.inf)}.items():
        m = (y == 0) | ((y == 1) & (hours_after_window > lo) & (hours_after_window <= hi))
        out[name] = {'n_pos': int(((y == 1) & m).sum()), 'auroc': float(roc_auc_score(y[m], p[m])) if ((y == 1) & m).sum() > 10 else None}
    return out


def fit_xgb(Xtr, ytr, Xva, yva, n_trials, device, seed):
    dtr, dva = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva)
    best = {'auc': -1}

    def obj(trial):
        params = {'max_depth': trial.suggest_int('max_depth', 2, 8), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                  'min_child_weight': trial.suggest_float('min_child_weight', 1, 100, log=True),
                  'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True),
                  'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 30, log=True)}
        b = xgb.train({**params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': device, 'seed': seed},
                      dtr, num_boost_round=2000, evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=False)
        auc = roc_auc_score(yva, b.predict(dva, iteration_range=(0, b.best_iteration + 1)))
        if auc > best['auc']:
            best.update(auc=auc, booster=b, it=(0, b.best_iteration + 1))
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    b, it = best['booster'], best['it']
    b.set_attr(best_iteration=str(it[1] - 1))
    return (lambda X: b.predict(xgb.DMatrix(X), iteration_range=it)), st.best_trial.params, st.best_value, \
        dict(sorted(b.get_score(importance_type='gain').items(), key=lambda kv: -kv[1])[:15]), b


def fit_rf(Xtr, ytr, Xva, yva, n_trials, seed, n_jobs):
    med = Xtr.median()
    Xtr_, Xva_ = Xtr.fillna(med).to_numpy(np.float32), Xva.fillna(med).to_numpy(np.float32)
    best = {'auc': -1}

    def obj(trial):
        params = {'max_depth': trial.suggest_int('max_depth', 2, 12), 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
                  'n_estimators': trial.suggest_int('n_estimators', 100, 500)}
        m = RandomForestClassifier(n_jobs=n_jobs, random_state=seed, class_weight={0: 1, 1: 10}, **params).fit(Xtr_, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva_)[:, 1])
        if auc > best['auc']:
            best.update(auc=auc, model=m)
        return auc

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    m = best['model']
    imp = dict(sorted(zip(Xtr.columns, m.feature_importances_.tolist()), key=lambda kv: -kv[1])[:15])
    return (lambda X: m.predict_proba(X.fillna(med).to_numpy(np.float32))[:, 1]), st.best_trial.params, st.best_value, imp, {'model': m, 'median': med}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', default='death')
    ap.add_argument('--dataset', default=None, help='csv from 01_build_dataset.py (default data/icu_<event>_w24h.csv)')
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 12])
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=25)
    args = ap.parse_args()
    out_dir = ROOT / 'results' / f'early_window_{args.event}'
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / 'models' / f'early_window_{args.event}'
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset or ROOT / 'data' / f'icu_{args.event}_w24h.csv')
    for c in ['rec_time', 'adm', 'dis_date', 'first_event_time']:
        df[c] = pd.to_datetime(df[c], format='ISO8601')
    df['hrs'] = (df.rec_time - df.adm) / pd.Timedelta(1, 'h')

    results = json.load(open(out_dir / 'metrics.json')) if (out_dir / 'metrics.json').exists() else {}
    results = {int(k): v for k, v in results.items()}
    md = []
    for W in args.windows:
        stay, X, n_excl = build(df, W)
        parts = {k: stay.dataset == k for k in ['train', 'val', 'test']}
        te = stay[parts['test']]
        h_after = te.hours_to_end.to_numpy() - W
        res = {'window_h': W, 'excluded': n_excl,
               'n_stays': {k: int(v.sum()) for k, v in parts.items()},
               'event_rate': {k: float(stay.y[v].mean()) for k, v in parts.items()},
               'n_features': X.shape[1], 'models': {}}
        print(f'W={W}h: stays {res["n_stays"]} event_rate {res["event_rate"]} excluded {n_excl}', flush=True)
        te_out = X[parts['test']].copy()
        for name, fn in [('xgb', lambda: fit_xgb(X[parts['train']], stay.y[parts['train']].to_numpy(), X[parts['val']], stay.y[parts['val']].to_numpy(),
                                                args.n_trials, args.device, args.seed)),
                         ('rf', lambda: fit_rf(X[parts['train']], stay.y[parts['train']].to_numpy(), X[parts['val']], stay.y[parts['val']].to_numpy(),
                                              args.n_trials, args.seed, args.n_jobs))]:
            predict, params, val_auc, imp, obj = fn()
            p = predict(X[parts['test']])
            y = te.y.to_numpy()
            if name == 'xgb':
                obj.save_model(model_dir / f'W{W}_xgb.json')
            else:
                pickle.dump(obj, open(model_dir / f'W{W}_rf.pickle', 'wb'))
            te_out[f'{name}_pred'] = p
            res['models'][name] = {'val_auroc': float(val_auc), 'test_auroc': float(roc_auc_score(y, p)),
                                   'test_auprc': float(average_precision_score(y, p)), 'by_time_to_death': strat_auc(y, p, h_after),
                                   'best_params': params, 'top_features': imp}
            print(f'  {name}: val {val_auc:.4f} test {res["models"][name]["test_auroc"]:.4f} AP {res["models"][name]["test_auprc"]:.4f}', flush=True)
        te_out['y'] = te.y.to_numpy()
        te_out['hours_to_end'] = te.hours_to_end.to_numpy()
        te_out.to_parquet(out_dir / f'W{W}_test.parquet')
        results[W] = res

        md.append(f'\n## W = {W} h\n')
        md.append(f'- stays train/val/test = {res["n_stays"]["train"]:,}/{res["n_stays"]["val"]:,}/{res["n_stays"]["test"]:,}; '
                  f'event rate {res["event_rate"]["test"]:.3f} (test); excluded: {n_excl["event_within_W"]} events and '
                  f'{n_excl["discharged_within_W"]} discharges inside the window; {res["n_features"]} features\n')
        md.append('| model | val AUROC | test AUROC | test AUPRC | death ≤24h after window | 24–72h | >72h |\n|---|---|---|---|---|---|---|\n')
        for name, r in res['models'].items():
            s = r['by_time_to_death']
            md.append(f'| {name} | {r["val_auroc"]:.3f} | {r["test_auroc"]:.3f} | {r["test_auprc"]:.3f} | '
                      + ' | '.join(f'{s[k]["auroc"]:.3f} (n={s[k]["n_pos"]})' if s[k]['auroc'] else f'— (n={s[k]["n_pos"]})'
                                   for k in ['death_le_24h', 'death_24_72h', 'death_gt_72h']) + ' |\n')
        md.append('\ntop features (xgb gain): ' + ', '.join(f'{k} {v:.0f}' for k, v in list(res['models']['xgb']['top_features'].items())[:10]) + '\n')
        md.append('top features (rf importance): ' + ', '.join(f'{k} {v:.3f}' for k, v in list(res['models']['rf']['top_features'].items())[:10]) + '\n')

    json.dump(dict(sorted(results.items())), open(out_dir / 'metrics.json', 'w'), indent=1, ensure_ascii=False)
    head = f'# Early-window prediction of {args.event} (observe first W hours, predict the stay outcome)\n'
    old = (out_dir / 'summary.md').read_text() if (out_dir / 'summary.md').exists() else head
    (out_dir / 'summary.md').write_text(old + ''.join(md) + saturation(results, out_dir))
    print('saved', out_dir)


def saturation(results, out_dir):
    """AUROC vs W on (a) each window's own test cohort and (b) the common cohort of stays that outlast the longest
    window, so that the curve is not confounded by the cohort shrinking with W."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Ws = sorted(results)
    te = {W: pd.read_parquet(out_dir / f'W{W}_test.parquet', columns=['y', 'hours_to_end', 'xgb_pred', 'rf_pred']) for W in Ws}
    common_idx = te[Ws[-1]].index
    rows = []
    for W in Ws:
        d = te[W]; c = d.loc[d.index.intersection(common_idx)]
        rows.append({'W': W, 'n_test': len(d), 'event_rate': d.y.mean(),
                     'xgb_auroc': roc_auc_score(d.y, d.xgb_pred), 'rf_auroc': roc_auc_score(d.y, d.rf_pred),
                     'xgb_auprc': average_precision_score(d.y, d.xgb_pred),
                     'n_common': len(c), 'xgb_auroc_common': roc_auc_score(c.y, c.xgb_pred), 'rf_auroc_common': roc_auc_score(c.y, c.rf_pred),
                     'xgb_auprc_common': average_precision_score(c.y, c.xgb_pred)})
    t = pd.DataFrame(rows).set_index('W')
    t.to_csv(out_dir / 'saturation.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, ls in [('xgb', '-'), ('rf', '--')]:
        ax[0].plot(t.index, t[f'{m}_auroc'], ls, marker='o', label=f'{m} (own cohort)')
        ax[0].plot(t.index, t[f'{m}_auroc_common'], ls, marker='s', alpha=.6, label=f'{m} (common cohort, n={t.n_common.iloc[-1]:,})')
    ax[0].set_xscale('log'); ax[0].set_xticks(t.index); ax[0].set_xticklabels(t.index); ax[0].set_xlabel('observation window W (h)')
    ax[0].set_ylabel('test AUROC'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title('AUROC vs W')
    ax[1].plot(t.index, t.xgb_auprc, marker='o', label='xgb AUPRC (own)'); ax[1].plot(t.index, t.xgb_auprc_common, marker='s', alpha=.6, label='xgb AUPRC (common)')
    ax[1].plot(t.index, t.event_rate, 'k:', marker='.', label='event rate (own cohort)')
    ax[1].set_xscale('log'); ax[1].set_xticks(t.index); ax[1].set_xticklabels(t.index); ax[1].set_xlabel('W (h)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title('AUPRC vs W')
    plt.tight_layout(); plt.savefig(out_dir / 'saturation.png', dpi=110); plt.close()
    md = '\n## Saturation (AUROC vs W)\n' + t.to_markdown(floatfmt='.3f') + '\n![saturation](saturation.png)\n'
    return md


if __name__ == '__main__':
    main()
