"""XGBoost counterpart of 02_train_rf.py: same data preparation (prepare / get_score imported from 02), same
val-AUROC optuna selection, same key-feature ablation; GPU training with early stopping on val.

--native-nan : skip the train-median fill and let XGBoost route missing values itself (missingness is informative
               here: charting frequency changes before the event). The ablation then sets Base Excess to NaN.

Usage: python 04_train_xgb.py data/icu_death_w24h.csv icu_death_w24h_xgb_nan [--n-trials 100] [--device cuda:0] [--native-nan]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
rf = import_module('02_train_rf')

ROOT = Path(__file__).resolve().parents[1]


def objective(trial, dtrain, dval, y_val, device):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 100, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100, log=True),
    }
    booster = xgb.train({**params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
                         'device': device, 'seed': 25}, dtrain, num_boost_round=2000,
                        evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    trial.set_user_attr('best_iteration', booster.best_iteration)
    val_auc = roc_auc_score(y_val, booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)))
    booster.save_model(trial.study.user_attrs['save_dir'] + f'/{trial.number}.json')
    return val_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_csv')
    ap.add_argument('save_name')
    ap.add_argument('--n-trials', type=int, default=100)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=25)
    ap.add_argument('--native-nan', action='store_true')
    args = ap.parse_args()

    save_dir = ROOT / 'models' / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / 'results' / args.save_name
    res_dir.mkdir(parents=True, exist_ok=True)

    tot_df = pd.read_csv(args.dataset_csv)
    split, var_cols, train_median = rf.prepare(tot_df, fill_median=not args.native_nan)
    X_train, y_train, _ = split['train']
    X_val, y_val, _ = split['val']
    X_test, y_test, test_df = split['test']
    print(f'features ({len(var_cols)}): {var_cols}  native_nan={args.native_nan}')
    for k, (X, y, _) in split.items():
        print(f'{k}: rows={len(y)} pos={int(y.sum())} ({y.mean():.4f})')
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=var_cols)
    dval = xgb.DMatrix(X_val, y_val, feature_names=var_cols)
    dtest = xgb.DMatrix(X_test, feature_names=var_cols)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', study_name='XGBoost', storage=f'sqlite:///{save_dir / "optuna.db"}',
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.set_user_attr('save_dir', str(save_dir))
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(lambda t: objective(t, dtrain, dval, y_val, args.device),
                   n_trials=max(0, args.n_trials - n_done), show_progress_bar=True)
    best = study.best_trial
    print(f'best trial {best.number}: val AUROC={best.value:.4f} params={best.params} '
          f'best_iteration={best.user_attrs["best_iteration"]}')
    booster = xgb.Booster(); booster.load_model(save_dir / f'{best.number}.json')
    for p in save_dir.glob('*.json'):
        if p.stem != str(best.number) and p.stem != 'col_dict':
            p.unlink()
    it = (0, best.user_attrs['best_iteration'] + 1)

    y_val_pred = booster.predict(dval, iteration_range=it)
    y_test_pred = booster.predict(dtest, iteration_range=it)
    X_test_key = X_test.copy()
    for feat in rf.NOT_KEY_FEAT:
        if feat in var_cols:
            X_test_key[:, var_cols.index(feat)] = np.nan if args.native_nan else train_median[feat]
    y_test_key_pred = booster.predict(xgb.DMatrix(X_test_key, feature_names=var_cols), iteration_range=it)

    names = ['auroc', 'auprc', 'auprc2', 'average_precision']
    gain = booster.get_score(importance_type='gain')
    out = {
        'best_trial': best.number, 'best_params': best.params, 'best_iteration': best.user_attrs['best_iteration'],
        'val_auroc_optuna': best.value, 'native_nan': args.native_nan,
        'val': dict(zip(names, map(float, rf.get_score(y_val_pred, y_val)))),
        'test': dict(zip(names, map(float, rf.get_score(y_test_pred, y_test)))),
        'test_key_feat_ablation': dict(zip(names, map(float, rf.get_score(y_test_key_pred, y_test)))),
        'features': var_cols, 'n_trials': args.n_trials,
        'importances_gain': {c: float(gain.get(c, 0.0)) for c in var_cols},
    }
    print(json.dumps({k: out[k] for k in ['val', 'test', 'test_key_feat_ablation', 'best_params']}, indent=1))
    json.dump(out, open(res_dir / 'metrics.json', 'w'), indent=1, ensure_ascii=False)
    test_df = test_df.copy()
    test_df['xgb_pred'] = y_test_pred
    test_df['xgb_key_pred'] = y_test_key_pred
    test_df.to_csv(res_dir / f'baseline_xgboost_{args.save_name}_pred.csv', index=False)
    print('saved', res_dir)


if __name__ == '__main__':
    main()
