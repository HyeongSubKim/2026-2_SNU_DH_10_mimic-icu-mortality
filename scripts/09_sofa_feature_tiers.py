"""28-day mortality from the first W hours of the ICU stay, on the flow-chart cohort (07_cohort_flow_counts.py), with
four nested feature sets:

  1. SOFA inputs        raw SOFA variables aggregated over the window (PaO2, FiO2, PaO2/FiO2, platelets, bilirubin, MAP,
                        vasopressors, GCS, creatinine, urine output, ventilation) + the six SOFA component scores and total
  2. + lactate          1 + lactate (min / max / mean / last / n)
  3. + base excess      1 + base excess (min / max / mean / last / n)
  4. + lactate + BE     everything

Cohort per window W: adults, SOFA computable in W h, lactate in W h, base excess in W h, alive at W h.
Outcome: death from any cause within 28 days of ICU admission (PATIENTS.DOD). Patient-level 70/15/15 split (seed 0).
Model: XGBoost (GPU, native NaN), Optuna on validation AUROC, early stopping; test AUROC / AUPRC with bootstrap 95 % CI.
Reference rows: SOFA total score alone, and age + SOFA total (logistic regression).
--demographics adds age and sex to every tier (results go to results/sofa_tiers_demo/).

Usage: MIMIC3_RAW=... python scripts/09_sofa_feature_tiers.py [--windows 6 24] [--n-trials 50] [--device cuda:0]
Output: results/sofa_tiers/{metrics.json, summary.md, features_W{W}.parquet, roc_W{W}.png}
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
H = pd.Timedelta(1, 'h')
SEED = 0

MAP_ITEMS = {456, 52, 6702, 443, 220052, 220181, 225312}
FIO2_ITEMS = {190, 3420, 3422, 223835}
VENT_ITEMS = {720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 224685,
              224684, 224686, 218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000,
              226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505,
              506, 686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701}
VASO = {30047: 'norepinephrine', 30120: 'norepinephrine', 221906: 'norepinephrine',
        30044: 'epinephrine', 30119: 'epinephrine', 30309: 'epinephrine', 221289: 'epinephrine',
        30043: 'dopamine', 30307: 'dopamine', 221662: 'dopamine',
        30042: 'dobutamine', 30306: 'dobutamine', 221653: 'dobutamine',
        30127: 'phenylephrine', 30128: 'phenylephrine', 221749: 'phenylephrine',
        30125: 'vasopressin', 222315: 'vasopressin'}
GCS_EYE, GCS_MOTOR, GCS_VERBAL = {184, 220739}, {454, 223901}, {723, 223900}
LAB = {50821: 'pao2', 50816: 'fio2_lab', 51265: 'platelets', 50885: 'bilirubin', 50912: 'creatinine',
       50813: 'lactate', 50802: 'base_excess'}


# ----------------------------------------------------------------------------------------------- cohort
def load_cohort_tables():
    icu = pd.read_csv(RAW / 'ICUSTAYS.csv.gz', parse_dates=['INTIME', 'OUTTIME'])
    pat = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB', 'DOD', 'GENDER'], parse_dates=['DOB', 'DOD'])
    adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'DEATHTIME'], parse_dates=['DEATHTIME'])
    st = icu.merge(pat, on='SUBJECT_ID').merge(adm, on='HADM_ID')
    st['age'] = (st.INTIME.dt.year - st.DOB.dt.year) - ((st.INTIME.dt.month * 100 + st.INTIME.dt.day) < (st.DOB.dt.month * 100 + st.DOB.dt.day))
    st['age'] = st.age.astype(float); st.loc[st.age > 120, 'age'] = 91.4
    st = st[(st.age >= 18) & st.OUTTIME.notna() & (st.OUTTIME > st.INTIME)].copy()
    d0 = st.INTIME.dt.floor('D')
    st['y28'] = ((st.DOD >= d0) & (st.DOD <= d0 + pd.Timedelta(28, 'D'))).astype(int)
    st['sex'] = (st.GENDER == 'M').astype(int)
    # patient-level split, fixed across windows and tiers
    rng = np.random.RandomState(SEED)
    subj = st.SUBJECT_ID.unique(); rng.shuffle(subj)
    n = len(subj)
    split = {s: 'train' for s in subj[:int(.7 * n)]}
    split.update({s: 'val' for s in subj[int(.7 * n):int(.85 * n)]})
    split.update({s: 'test' for s in subj[int(.85 * n):]})
    st['dataset'] = st.SUBJECT_ID.map(split)
    return st.set_index('ICUSTAY_ID')


def load_events(st):
    ids = set(st.index)
    lab = pd.read_parquet(DATA / 'labevents_sofa.parquet').merge(st.reset_index()[['HADM_ID', 'ICUSTAY_ID']], on='HADM_ID')
    lab['t'] = pd.to_datetime(lab.CHARTTIME); lab['name'] = lab.ITEMID.map(LAB)
    ce = pd.read_parquet(DATA / 'chartevents.parquet', columns=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUE'])
    ce = ce[ce.ICUSTAY_ID.isin(ids)]; ce['t'] = pd.to_datetime(ce.CHARTTIME)
    gcs = pd.read_parquet(DATA / 'chartevents_gcs.parquet'); gcs = gcs[gcs.ICUSTAY_ID.isin(ids)]; gcs['t'] = pd.to_datetime(gcs.CHARTTIME)
    icv = pd.read_parquet(DATA / 'inputevents_cv.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'CHARTTIME': 't'})
    imv = pd.read_parquet(DATA / 'inputevents_mv.parquet', columns=['ICUSTAY_ID', 'STARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'STARTTIME': 't'})
    vaso = pd.concat([icv, imv]); vaso = vaso[vaso.ITEMID.isin(VASO) & (vaso.AMOUNT > 0) & vaso.ICUSTAY_ID.isin(ids)].copy()
    vaso['t'] = pd.to_datetime(vaso.t); vaso['drug'] = vaso.ITEMID.map(VASO)
    out = pd.read_parquet(DATA / 'outputevents.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'VALUE'])
    out = out[out.VALUE.notna() & out.ICUSTAY_ID.isin(ids)].copy(); out['t'] = pd.to_datetime(out.CHARTTIME)
    for df in (lab, ce, gcs, vaso, out):          # chart/input/output parquets carry ICUSTAY_ID as float
        df['ICUSTAY_ID'] = df.ICUSTAY_ID.astype('int64')
    return lab, ce, gcs, vaso, out


def window(df, st, W, lab_window=False):
    it = df.ICUSTAY_ID.map(st.INTIME)
    lo = it - 6 * H if lab_window else it
    return df[(df.t >= lo) & (df.t <= it + W * H)]


def agg(df, col, name, stats=('min', 'max', 'mean', 'last', 'count')):
    g = df.sort_values('t').groupby('ICUSTAY_ID')[col]
    return pd.concat({f'{name}_{s}': getattr(g, s)() for s in stats}, axis=1)


# ----------------------------------------------------------------------------------------------- features + SOFA
def build_features(st, ev, W):
    lab, ce, gcs, vaso, out = ev
    labw = window(lab, st, W, True)
    labw = labw.assign(VALUENUM=np.where(labw.name.eq('fio2_lab') & (labw.VALUENUM <= 1), labw.VALUENUM * 100, labw.VALUENUM))
    cew = window(ce[ce.VALUENUM.notna()], st, W)
    fio2 = cew[cew.ITEMID.isin(FIO2_ITEMS)].copy()
    fio2['VALUENUM'] = np.where(fio2.VALUENUM <= 1, fio2.VALUENUM * 100, fio2.VALUENUM)
    fio2 = pd.concat([fio2[['ICUSTAY_ID', 't', 'VALUENUM']], labw.loc[labw.name == 'fio2_lab', ['ICUSTAY_ID', 't', 'VALUENUM']]])
    fio2 = fio2[(fio2.VALUENUM >= 21) & (fio2.VALUENUM <= 100)]
    pao2 = labw[labw.name == 'pao2'][['ICUSTAY_ID', 't', 'VALUENUM']]
    # PaO2/FiO2: each PaO2 paired with the most recent FiO2 charted up to 4 h before it (21 % if none)
    pf = pd.merge_asof(pao2.sort_values('t'), fio2.sort_values('t').rename(columns={'VALUENUM': 'fio2'}),
                       on='t', by='ICUSTAY_ID', direction='backward', tolerance=4 * H)
    pf['fio2'] = pf.fio2.fillna(21.0); pf['pf'] = pf.VALUENUM / (pf.fio2 / 100)
    mapw = cew[cew.ITEMID.isin(MAP_ITEMS) & cew.VALUENUM.between(20, 200)]
    vent = window(ce[ce.ITEMID.isin(VENT_ITEMS)], st, W).groupby('ICUSTAY_ID').size().rename('vent_n')
    # GCS total per charting time (sum of the three components when charted together)
    g = window(gcs, st, W).copy()
    g['comp'] = np.select([g.ITEMID.isin(GCS_EYE), g.ITEMID.isin(GCS_MOTOR), g.ITEMID.isin(GCS_VERBAL)], ['e', 'm', 'v'], default='x')
    g = g.pivot_table(index=['ICUSTAY_ID', 't'], columns='comp', values='VALUENUM', aggfunc='first').dropna().reset_index()
    g['VALUENUM'] = g.e + g.m + g.v
    vw = window(vaso, st, W); ow = window(out, st, W)

    X = pd.concat([
        agg(pao2, 'VALUENUM', 'pao2'), agg(fio2, 'VALUENUM', 'fio2'), agg(pf, 'pf', 'pf', ('min', 'mean', 'last')),
        agg(labw[labw.name == 'platelets'], 'VALUENUM', 'platelets'),
        agg(labw[labw.name == 'bilirubin'], 'VALUENUM', 'bilirubin'),
        agg(labw[labw.name == 'creatinine'], 'VALUENUM', 'creatinine'),
        agg(mapw, 'VALUENUM', 'map'), agg(g, 'VALUENUM', 'gcs', ('min', 'last', 'count')),
        vw.groupby('ICUSTAY_ID').agg(vaso_n=('AMOUNT', 'size'), vaso_drugs=('drug', 'nunique')),
        pd.get_dummies(vw.drop_duplicates(['ICUSTAY_ID', 'drug']).set_index('ICUSTAY_ID').drug, prefix='vaso').groupby(level=0).max().astype(int),
        ow.groupby('ICUSTAY_ID').agg(urine_total=('VALUE', 'sum'), urine_n=('VALUE', 'size')),
        vent,
        agg(labw[labw.name == 'lactate'], 'VALUENUM', 'lactate'),
        agg(labw[labw.name == 'base_excess'], 'VALUENUM', 'base_excess'),
    ], axis=1).reindex(st.index)
    for c in ['vaso_n', 'vaso_drugs', 'vent_n', 'urine_n'] + [c for c in X.columns if c.startswith('vaso_') and c not in ('vaso_n', 'vaso_drugs')]:
        X[c] = X[c].fillna(0)
    X['vent'] = (X.vent_n > 0).astype(int)
    X['urine_ml_per_h'] = X.urine_total / W

    # SOFA component scores from the worst value in the window (standard cut-offs; vasopressor dose not available,
    # so dopamine/dobutamine -> 2, norepinephrine/epinephrine -> 3; urine criterion only used for the 24 h window)
    def cut(v, bins, scores):
        return pd.cut(v, bins=bins, labels=scores, right=False).astype(float)
    s = pd.DataFrame(index=X.index)
    resp = cut(X.pf_min, [-np.inf, 100, 200, 300, 400, np.inf], [4, 3, 2, 1, 0])
    resp = np.where((resp >= 3) & (X.vent == 0), 2, resp)                     # 3-4 require ventilation
    s['sofa_resp'] = resp
    s['sofa_coag'] = cut(X.platelets_min, [-np.inf, 20, 50, 100, 150, np.inf], [4, 3, 2, 1, 0])
    s['sofa_liver'] = cut(X.bilirubin_max, [-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf], [0, 1, 2, 3, 4])
    cardio = np.where(X.map_min < 70, 1, 0).astype(float)
    cardio = np.where(X.get('vaso_dopamine', 0).astype(bool) | X.get('vaso_dobutamine', 0).astype(bool), np.maximum(cardio, 2), cardio)
    cardio = np.where(X.get('vaso_norepinephrine', 0).astype(bool) | X.get('vaso_epinephrine', 0).astype(bool), np.maximum(cardio, 3), cardio)
    s['sofa_cardio'] = cardio
    s['sofa_cns'] = cut(X.gcs_min, [-np.inf, 6, 10, 13, 15, np.inf], [4, 3, 2, 1, 0])
    renal = cut(X.creatinine_max, [-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf], [0, 1, 2, 3, 4])
    if W >= 24:
        renal = np.maximum(renal, np.where(X.urine_total < 200, 4, np.where(X.urine_total < 500, 3, 0)))
    s['sofa_renal'] = renal
    s['sofa_total'] = s.sum(axis=1, min_count=1)
    return pd.concat([X, s], axis=1)


def cohort_mask(X, st, W):
    organs = [X.pao2_count > 0, X.fio2_count > 0, X.platelets_count > 0, X.bilirubin_count > 0,
              (X.map_count > 0) | (X.vaso_n > 0), X.gcs_count > 0, (X.creatinine_count > 0) | (X.urine_n > 0)]
    sofa_ok = np.logical_and.reduce(organs)
    died_in_w = (st.DEATHTIME >= st.INTIME) & (st.DEATHTIME <= st.INTIME + W * H)
    return sofa_ok & (X.lactate_count > 0) & (X.base_excess_count > 0) & ~died_in_w.to_numpy()


TIERS = {
    '1 SOFA inputs': lambda cols: [c for c in cols if not c.startswith(('lactate_', 'base_excess_'))],
    '2 + lactate': lambda cols: [c for c in cols if not c.startswith('base_excess_')],
    '3 + base excess': lambda cols: [c for c in cols if not c.startswith('lactate_')],
    '4 + lactate + base excess': lambda cols: list(cols),
}


# ----------------------------------------------------------------------------------------------- modelling
def fit_xgb(Xtr, ytr, Xva, yva, n_trials, device, seed=25):
    dtr, dva = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva)
    best = {'auc': -1}

    def obj(trial):
        params = {'max_depth': trial.suggest_int('max_depth', 2, 8), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                  'min_child_weight': trial.suggest_float('min_child_weight', 1, 50, log=True),
                  'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True)}
        b = xgb.train({**params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': device, 'seed': seed},
                      dtr, num_boost_round=1000, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False)
        auc = roc_auc_score(yva, b.predict(dva, iteration_range=(0, b.best_iteration + 1)))
        if auc > best['auc']:
            best.update(auc=auc, booster=b, it=(0, b.best_iteration + 1), params=params)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed)).optimize(obj, n_trials=n_trials)
    b, it = best['booster'], best['it']
    return (lambda X: b.predict(xgb.DMatrix(X), iteration_range=it)), best


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); p = np.asarray(p); aucs, aps = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i])); aps.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))], [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--demographics', action='store_true', help='add age and sex to every feature set')
    args = ap.parse_args()
    OUT = ROOT / 'results' / ('sofa_tiers_demo' if args.demographics else 'sofa_tiers')
    OUT.mkdir(parents=True, exist_ok=True)

    st = load_cohort_tables()
    ev = load_events(st)
    results, md = {}, [f'# 28-day mortality from the first W hours: SOFA inputs ± lactate ± base excess (XGBoost{", + age/sex in every tier" if args.demographics else ""})\n']
    for W in args.windows:
        X = build_features(st, ev, W)
        keep = cohort_mask(X, st, W)
        Xc, sc = X[keep], st[keep]
        if args.demographics:
            Xc = Xc.join(sc[['age', 'sex']])
        Xc.join(sc[[c for c in ['SUBJECT_ID', 'dataset', 'y28', 'age', 'sex'] if c not in Xc.columns]]).to_parquet(OUT / f'features_W{W}.parquet')
        y = sc.y28.to_numpy(); part = {k: (sc.dataset == k).to_numpy() for k in ['train', 'val', 'test']}
        res = {'window_h': W, 'n': {k: int(v.sum()) for k, v in part.items()}, 'event_rate_test': float(y[part['test']].mean()),
               'n_features_total': int(Xc.shape[1]), 'tiers': {}}
        print(f"W={W}h cohort {len(Xc):,} (train/val/test {res['n']}), 28-day mortality test {res['event_rate_test']:.3f}", flush=True)
        curves = []
        # reference: SOFA total alone, and age + SOFA total logistic regression
        yt = y[part['test']]
        sofa_t = Xc.sofa_total.to_numpy()
        res['reference'] = {'sofa_total_alone': {'test_auroc': float(roc_auc_score(yt, sofa_t[part['test']])),
                                                 'test_auprc': float(average_precision_score(yt, sofa_t[part['test']]))}}
        Z = np.c_[sc.age.to_numpy(), sofa_t]; Z = np.nan_to_num(Z, nan=0.0)
        lr = LogisticRegression(max_iter=1000).fit(Z[part['train']], y[part['train']])
        p_lr = lr.predict_proba(Z[part['test']])[:, 1]
        res['reference']['age_plus_sofa_logistic'] = {'test_auroc': float(roc_auc_score(yt, p_lr)), 'test_auprc': float(average_precision_score(yt, p_lr))}
        curves.append(('SOFA total alone', yt, sofa_t[part['test']]))
        preds = sc[['SUBJECT_ID', 'dataset', 'y28']].copy(); preds['sofa_total'] = sofa_t
        for tier, pick in TIERS.items():
            cols = pick(Xc.columns)
            predict, best = fit_xgb(Xc.loc[part['train'], cols], y[part['train']], Xc.loc[part['val'], cols], y[part['val']], args.n_trials, args.device)
            p = predict(Xc.loc[part['test'], cols])
            preds[tier] = predict(Xc[cols])          # val + test predictions, for operating-point analysis
            auc_ci, ap_ci = boot_ci(yt, p)
            gain = best['booster'].get_score(importance_type='gain')
            res['tiers'][tier] = {'n_features': len(cols), 'val_auroc': float(best['auc']), 'test_auroc': float(roc_auc_score(yt, p)),
                                  'test_auroc_ci95': auc_ci, 'test_auprc': float(average_precision_score(yt, p)), 'test_auprc_ci95': ap_ci,
                                  'best_params': best['params'], 'best_iteration': best['it'][1] - 1,
                                  'top_features_gain': dict(sorted(gain.items(), key=lambda kv: -kv[1])[:12])}
            curves.append((tier, yt, p))
            print(f"  {tier:28s} feats {len(cols):3d}  val {best['auc']:.3f}  test AUROC {res['tiers'][tier]['test_auroc']:.3f} "
                  f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]  AUPRC {res['tiers'][tier]['test_auprc']:.3f}", flush=True)
        results[W] = res
        preds.to_parquet(OUT / f'preds_W{W}.parquet')
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for name, yy, pp in curves:
            fpr, tpr, _ = roc_curve(yy, pp); ax.plot(fpr, tpr, label=f'{name}  {roc_auc_score(yy, pp):.3f}', lw=1.3, ls='--' if 'alone' in name else '-')
        ax.plot([0, 1], [0, 1], 'k:', lw=.7); ax.set_xlabel('1 - specificity'); ax.set_ylabel('sensitivity')
        ax.set_title(f'28-day mortality, first {W} h (test, n={int(part["test"].sum()):,})'); ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(OUT / f'roc_W{W}.png', dpi=160); plt.close()

        md.append(f'\n## W = {W} h  (cohort {len(Xc):,}; train/val/test {res["n"]["train"]:,}/{res["n"]["val"]:,}/{res["n"]["test"]:,}; '
                  f'28-day mortality in test {res["event_rate_test"]:.3f})\n\n')
        md.append('| feature set | # features | val AUROC | test AUROC [95 % CI] | test AUPRC [95 % CI] |\n|---|---|---|---|---|\n')
        r = res['reference']
        md.append(f"| SOFA total score alone | 1 | – | {r['sofa_total_alone']['test_auroc']:.3f} | {r['sofa_total_alone']['test_auprc']:.3f} |\n")
        md.append(f"| age + SOFA total (logistic) | 2 | – | {r['age_plus_sofa_logistic']['test_auroc']:.3f} | {r['age_plus_sofa_logistic']['test_auprc']:.3f} |\n")
        for tier, t in res['tiers'].items():
            md.append(f"| {tier} (XGBoost) | {t['n_features']} | {t['val_auroc']:.3f} | {t['test_auroc']:.3f} [{t['test_auroc_ci95'][0]:.3f}, {t['test_auroc_ci95'][1]:.3f}] "
                      f"| {t['test_auprc']:.3f} [{t['test_auprc_ci95'][0]:.3f}, {t['test_auprc_ci95'][1]:.3f}] |\n")
        t4 = res['tiers']['4 + lactate + base excess']
        md.append('\ntop features by gain (tier 4): ' + ', '.join(f'{k} {v:.0f}' for k, v in list(t4['top_features_gain'].items())[:10]) + '\n')
    json.dump(results, open(OUT / 'metrics.json', 'w'), indent=1)
    (OUT / 'summary.md').write_text(''.join(md))
    print('saved', OUT)


if __name__ == '__main__':
    main()
