"""Fixed re-implementation of train_nicu_randomforest.ipynb.

Fixes vs. the notebook
  1. feature order is deterministic (sorted) -> pickled model is reusable across sessions
  2. ffill is done per patient (groupby pid) and never crosses train/val/test
  3. no hidden cell-to-cell state; everything is a function of the input csv
  4. the optuna best trial is the model that gets evaluated (notebook used 4.pickle)
  5. ffill happens BEFORE the pre-window rows are dropped (FIX 6 in prepare)
  6. AUPRC is reported both as auc(recall, precision) (notebook) and average_precision

Usage: python 02_train_rf.py data/icu_death_w24h.csv icu_death_w24h_rf [--n-trials 200] [--n-jobs 16] [--resume]
"""
import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

# same column-role lists as the notebook
INFO_CAND_COLS = ['pid', 'rec_time', 'adm', 'first_event_time', 'GUBUN', 'birthdate', 'dis_date', 'DSCH_DTE',
                  'window_start', 'first_event_time_8h', 'dataset', 'is_event', 'NEWS', 'NEWS_normalized', 'rec_time_datetime',
                  'is_pre_event', 'diff', 'subject_id', 'hadm_id']
NOT_KEY_FEAT = ['NO VENT (ppm)', 'Base Excess']  # notebook _FEAT_E


def get_score(pred, label):
    """Identical to the notebook's get_score (auroc, auprc, auprc2) + average precision."""
    auroc = roc_auc_score(label, pred)
    precision, recall, _ = precision_recall_curve(label, pred)
    recall_ = np.concatenate((recall, np.array([0])), axis=0)
    precision_ = np.concatenate((precision, np.array([precision[-1]])), axis=0)
    return auroc, auc(recall, precision), auc(recall_, precision_), average_precision_score(label, pred)


def prepare(tot_df, fill_median=True):
    var_cols = sorted(set(tot_df.columns) - set(INFO_CAND_COLS))  # FIX 1: deterministic order

    # FIX 2: forward-fill within patient only (rows are already time-ordered within pid)
    # FIX 6: ffill BEFORE dropping the pre-window rows of event patients. The notebook filtered first, so the
    #        8h-window rows lost their own history and were mostly median-filled -> "value==median" leaked the label.
    tot_df = tot_df.sort_values(['pid', 'rec_time']).copy()
    tot_df[var_cols] = tot_df.groupby('pid')[var_cols].ffill()
    tot_df = tot_df.query("~(is_event==1 and is_pre_event==1)").copy()

    train_median = tot_df.loc[tot_df.dataset == 'train', var_cols].median()
    if fill_median:
        tot_df[var_cols] = tot_df[var_cols].fillna(train_median)

    split = {}
    for name in ['train', 'val', 'test']:
        part = tot_df[tot_df.dataset == name]
        split[name] = (part[var_cols].to_numpy(dtype=np.float32), part['is_event'].to_numpy(), part)
    return split, var_cols, train_median


def _fit_one(params, X_train, y_train, X_val, y_val, path, n_jobs, q):
    model = RandomForestClassifier(n_jobs=n_jobs, random_state=25, class_weight={0: 1, 1: 50}, **params)
    model.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, -1])
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    q.put(val_auc)


def objective(trial, X_train, y_train, X_val, y_val, save_dir, n_jobs):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    }
    # each trial fits in a forked child: the multi-threaded fit leaves the parent's RSS growing by GBs per
    # trial otherwise (malloc arena fragmentation); the child exits and returns everything to the OS
    ctx = mp.get_context('fork')
    q = ctx.SimpleQueue()
    proc = ctx.Process(target=_fit_one, args=(params, X_train, y_train, X_val, y_val,
                                              save_dir / f'{trial.number}.pickle', n_jobs, q))
    proc.start()
    val_auc = q.get()
    proc.join()
    return val_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_csv')
    ap.add_argument('save_name')
    ap.add_argument('--n-trials', type=int, default=200)
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=25)
    ap.add_argument('--resume', action='store_true',
                    help='sqlite study in models/<save_name>/optuna.db; completed trials whose pickle exists but is '
                         'not in the study (an interrupted run without storage) are re-scored on val and added')
    args = ap.parse_args()

    save_dir = ROOT / 'models' / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / 'results' / args.save_name
    res_dir.mkdir(parents=True, exist_ok=True)

    tot_df = pd.read_csv(args.dataset_csv)
    split, var_cols, train_median = prepare(tot_df)
    X_train, y_train, _ = split['train']
    X_val, y_val, _ = split['val']
    X_test, y_test, test_df = split['test']
    print(f'features ({len(var_cols)}): {var_cols}')
    for k, (X, y, _) in split.items():
        print(f'{k}: rows={len(y)} pos={int(y.sum())} ({y.mean():.4f})')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f'sqlite:///{save_dir / "optuna.db"}' if args.resume else None
    study = optuna.create_study(direction='maximize', study_name='RandomForest Classifier', storage=storage,
                                load_if_exists=args.resume, sampler=optuna.samplers.TPESampler(seed=args.seed))
    if args.resume:
        done = len(study.trials)
        dists = {'max_depth': optuna.distributions.IntDistribution(1, 10),
                 'max_leaf_nodes': optuna.distributions.IntDistribution(2, 1000),
                 'n_estimators': optuna.distributions.IntDistribution(100, 500)}
        for p in sorted(save_dir.glob('*.pickle'), key=lambda p: int(p.stem)):
            if int(p.stem) < done:
                continue
            m = pickle.load(open(p, 'rb'))
            params = {k: m.get_params()[k] for k in dists}
            val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, -1])
            study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=val_auc))
            print(f'resumed trial {p.stem} from pickle: val AUROC={val_auc:.4f} {params}', flush=True)
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)  # a killed run leaves a RUNNING trial
    print(f'study has {n_done} completed trials; running {args.n_trials - n_done} more', flush=True)
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, save_dir, args.n_jobs),
                   n_trials=max(0, args.n_trials - n_done), show_progress_bar=True)
    best = study.best_trial
    print(f'best trial {best.number}: val AUROC={best.value:.4f} params={best.params}')

    # FIX 4: evaluate the best trial's model
    model = pickle.load(open(save_dir / f'{best.number}.pickle', 'rb'))
    # delete the non-best pickles to save space
    for p in save_dir.glob('*.pickle'):
        if p.stem != str(best.number):
            p.unlink()

    y_val_pred = model.predict_proba(X_val)[:, -1]
    y_test_pred = model.predict_proba(X_test)[:, -1]
    val_score = get_score(y_val_pred, y_val)
    test_score = get_score(y_test_pred, y_test)

    # key-feature ablation exactly as the notebook: replace NOT_KEY_FEAT with train median
    X_test_key = X_test.copy()
    for feat in NOT_KEY_FEAT:
        if feat in var_cols:
            X_test_key[:, var_cols.index(feat)] = train_median[feat]
    y_test_key_pred = model.predict_proba(X_test_key)[:, -1]
    test_key_score = get_score(y_test_key_pred, y_test)

    names = ['auroc', 'auprc', 'auprc2', 'average_precision']
    out = {
        'best_trial': best.number, 'best_params': best.params, 'val_auroc_optuna': best.value,
        'val': dict(zip(names, map(float, val_score))),
        'test': dict(zip(names, map(float, test_score))),
        'test_key_feat_ablation': dict(zip(names, map(float, test_key_score))),
        'features': var_cols, 'n_trials': args.n_trials,
        'importances': dict(zip(var_cols, map(float, model.feature_importances_))),
    }
    print(json.dumps({k: out[k] for k in ['val', 'test', 'test_key_feat_ablation', 'best_params']}, indent=1))
    json.dump(out, open(res_dir / 'metrics.json', 'w'), indent=1, ensure_ascii=False)
    json.dump({c: f'feat_{i}' for i, c in enumerate(var_cols)}, open(save_dir / 'col_dict.json', 'w'), indent=1, ensure_ascii=False)

    test_df = test_df.copy()
    test_df['rf_pred'] = y_test_pred
    test_df['rf_key_pred'] = y_test_key_pred
    test_df.to_csv(res_dir / f'baseline_randomforest_{args.save_name}_pred.csv', index=False)
    print('saved', res_dir)


if __name__ == '__main__':
    main()
