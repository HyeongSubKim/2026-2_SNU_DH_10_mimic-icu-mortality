"""Selection-bias check requested by the reviewer: do stays with a lactate / base-excess result in the first W h differ in
28-day mortality and baseline characteristics from stays without one?  All adult ICU stays alive at hour W (landmark), W = 6 / 24.

For each exposure (lactate recorded, BE recorded, SOFA computable = the gate that removes most stays):
  crude 28-day mortality by group, risk difference with 95 % Wald CI; baseline table; adjusted logistic regression
  (age, sex, first care unit, admission type, charting era [+ vasopressor, ventilation as severity proxies]); IPW
  (stabilised, trimmed) risk difference with bootstrap CI; and whether prediction is "easier" in the tested group =
  AUROC of a covariate model and of the SOFA score inside each group.
Outputs results/selection_bias/W{W}.json, summary_W{W}.md.  Reuses the feature builder of 09_sofa_feature_tiers.py.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'selection_bias'; OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('s09', ROOT / 'scripts' / '09_sofa_feature_tiers.py')
s09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s09)
H = pd.Timedelta(1, 'h')
UNITS = ['MICU', 'SICU', 'CCU', 'CSRU', 'TSICU']


def rd_ci(y, g):
    a, b = y[g == 1], y[g == 0]
    p1, p0 = a.mean(), b.mean(); se = np.sqrt(p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(b))
    return {'n1': int(len(a)), 'd1': int(a.sum()), 'p1': float(p1), 'n0': int(len(b)), 'd0': int(b.sum()), 'p0': float(p0),
            'rd_pp': float(100 * (p1 - p0)), 'rd_ci': [float(100 * (p1 - p0 - 1.96 * se)), float(100 * (p1 - p0 + 1.96 * se))]}


def design(df, extra):
    Z = pd.DataFrame({'age': df.age, 'sex': df.sex, 'emergency': (df.ADMISSION_TYPE == 'EMERGENCY').astype(int),
                      'elective': (df.ADMISSION_TYPE == 'ELECTIVE').astype(int), 'metavision': (df.DBSOURCE == 'metavision').astype(int)}, index=df.index)
    for u in UNITS[1:]:
        Z[f'unit_{u}'] = (df.FIRST_CAREUNIT == u).astype(int)
    if extra:
        Z['vaso'] = (df.vaso_n > 0).astype(int); Z['vent'] = df.vent
    return Z


def adj_or(df, g, extra):
    Z = design(df, extra); Z.insert(0, 'exposed', g)
    m = sm.Logit(df.y28.values, sm.add_constant(Z.values.astype(float))).fit(disp=0, maxiter=200)
    ci = m.conf_int()[1]
    return {'or': float(np.exp(m.params[1])), 'ci': [float(np.exp(ci[0])), float(np.exp(ci[1]))], 'p': float(m.pvalues[1])}


def ipw_rd(df, g, extra, n_boot=50):
    Z = design(df, extra).values.astype(float); y = df.y28.values; g = g.astype(int)
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    def one(idx):
        ps = LogisticRegression(max_iter=300, tol=1e-3).fit(Zs[idx], g[idx]).predict_proba(Zs[idx])[:, 1]
        ps = np.clip(ps, np.percentile(ps, 1), np.percentile(ps, 99))
        pg = g[idx].mean(); w = np.where(g[idx] == 1, pg / ps, (1 - pg) / (1 - ps))
        yy, gg = y[idx], g[idx]
        return 100 * (np.average(yy[gg == 1], weights=w[gg == 1]) - np.average(yy[gg == 0], weights=w[gg == 0]))
    rng = np.random.RandomState(0); allidx = np.arange(len(y))
    est = one(allidx); b = [one(rng.choice(allidx, len(y))) for _ in range(n_boot)]
    return {'rd_pp': float(est), 'ci': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def baseline(df, g):
    rows = {}
    for name, sub in (('exposed', df[g == 1]), ('unexposed', df[g == 0])):
        r = {'n': int(len(sub)), 'age_mean': float(sub.age.mean()), 'male_pct': float(100 * sub.sex.mean()),
             'emergency_pct': float(100 * (sub.ADMISSION_TYPE == 'EMERGENCY').mean()), 'metavision_pct': float(100 * (sub.DBSOURCE == 'metavision').mean()),
             'vaso_pct': float(100 * (sub.vaso_n > 0).mean()), 'vent_pct': float(100 * sub.vent.mean()),
             'gcs_min_median': float(sub.gcs_min.median()), 'map_min_median': float(sub.map_min.median()),
             'creatinine_max_median': float(sub.creatinine_max.median()), 'sofa_computable_pct': float(100 * sub.sofa_ok.mean()),
             'sofa_total_median_if_computable': float(sub.loc[sub.sofa_ok, 'sofa_total'].median()), 'y28_pct': float(100 * sub.y28.mean())}
        r.update({f'unit_{u}_pct': float(100 * (sub.FIRST_CAREUNIT == u).mean()) for u in UNITS})
        rows[name] = r
    return rows


def auroc_by_group(df, g):
    """covariate-only logistic model (fit on the train split of all stays) and the SOFA score, scored inside each group"""
    Z = design(df, True); Z['gcs_min'] = df.gcs_min; Z['map_min'] = df.map_min; Z['creat_max'] = df.creatinine_max
    Z = Z.fillna(Z[df.dataset == 'train'].median())
    m = LogisticRegression(max_iter=2000).fit(Z[df.dataset == 'train'], df.y28[df.dataset == 'train'])
    te = (df.dataset == 'test').values; p = m.predict_proba(Z)[:, 1]
    r = {}
    for name, mask in (('exposed', g == 1), ('unexposed', g == 0)):
        sel = te & mask; r[f'covariate_model_auroc_{name}'] = float(roc_auc_score(df.y28[sel], p[sel])) if df.y28[sel].nunique() == 2 else None
        sel = mask & df.sofa_ok.values
        r[f'sofa_score_auroc_{name}'] = float(roc_auc_score(df.y28[sel], df.sofa_total[sel])) if sel.sum() > 50 and df.y28[sel].nunique() == 2 else None
        r[f'sofa_score_n_{name}'] = int(sel.sum())
    return r


def main():
    st = s09.load_cohort_tables()
    st['ADMISSION_TYPE'] = pd.read_parquet(ROOT / 'data' / 'icustays.parquet').set_index('ICUSTAY_ID').ADMISSION_TYPE.reindex(st.index)
    print(f'adult stays {len(st):,}', flush=True)
    ev = None if all((ROOT / 'data' / f'selection_bias_features_W{W}.parquet').exists() for W in (6, 24)) else s09.load_events(st)
    for W in (6, 24):
        cache = ROOT / 'data' / f'selection_bias_features_W{W}.parquet'
        if cache.exists():
            X = pd.read_parquet(cache)
        else:
            X = s09.build_features(st, ev, W); X.to_parquet(cache)
        df = pd.concat([st, X], axis=1)
        organs = [X.pao2_count > 0, X.fio2_count > 0, X.platelets_count > 0, X.bilirubin_count > 0,
                  (X.map_count > 0) | (X.vaso_n > 0), X.gcs_count > 0, (X.creatinine_count > 0) | (X.urine_n > 0)]
        df['sofa_ok'] = np.logical_and.reduce(organs)
        died_in_w = (st.DEATHTIME >= st.INTIME) & (st.DEATHTIME <= st.INTIME + W * H)
        df = df[~died_in_w.values].copy()                       # landmark: alive at hour W
        df['lact'] = (df.lactate_count > 0).astype(int); df['be'] = (df.base_excess_count > 0).astype(int)
        res = {'window': W, 'n_adult_alive_at_W': int(len(df)), 'died_in_window_removed': int(died_in_w.sum()), 'analyses': {}}
        pops = {'all adults': df, 'SOFA-computable': df[df.sofa_ok]}
        for pop_name, pop in pops.items():
            for exp_name, g in (('lactate recorded', pop.lact.values), ('BE recorded', pop.be.values), ('SOFA computable', pop.sofa_ok.values.astype(int))):
                if pop_name == 'SOFA-computable' and exp_name == 'SOFA computable':
                    continue
                if min((g == 1).sum(), (g == 0).sum()) < 50:          # e.g. BE is present in (almost) every SOFA-computable stay
                    res['analyses'][f'{pop_name} | {exp_name}'] = {'skipped': f'exposed {int((g == 1).sum())}, unexposed {int((g == 0).sum())}'}; continue
                k = f'{pop_name} | {exp_name}'
                r = {'crude': rd_ci(pop.y28.values, g), 'baseline': baseline(pop, g),
                     'adj_or_demo': adj_or(pop, g, False), 'adj_or_demo_severity': adj_or(pop, g, True),
                     'ipw_rd_demo': ipw_rd(pop, g, False), 'ipw_rd_demo_severity': ipw_rd(pop, g, True), 'discrimination': auroc_by_group(pop, g)}
                res['analyses'][k] = r
                c = r['crude']; print(f"W={W} {k}: {c['p1']*100:.1f}% ({c['n1']:,}) vs {c['p0']*100:.1f}% ({c['n0']:,}), RD {c['rd_pp']:+.1f} pp "
                      f"{c['rd_ci']}, adjOR {r['adj_or_demo']['or']:.2f} / {r['adj_or_demo_severity']['or']:.2f}, IPW RD {r['ipw_rd_demo']['rd_pp']:+.1f} / {r['ipw_rd_demo_severity']['rd_pp']:+.1f}", flush=True)
        json.dump(res, open(OUT / f'W{W}.json', 'w'), indent=1)
        md = [f'# Selection-bias check, W = {W} h (adult ICU stays alive at hour {W}: n = {len(df):,}; {int(died_in_w.sum())} died within the window removed)', '',
              '| population | exposure | exposed: deaths / N (%) | unexposed: deaths / N (%) | risk difference pp [95 % CI] | adj. OR demo† | adj. OR demo + severity‡ | IPW RD pp (demo) | IPW RD pp (demo + severity) |',
              '|---|---|---|---|---|---|---|---|---|']
        for k, r in res['analyses'].items():
            if 'skipped' in r:
                md.append(f"| {k.replace(' | ', ' | ')} | skipped ({r['skipped']}) | | | | | | |"); continue
            pop_name, exp_name = k.split(' | '); c = r['crude']; o1 = r['adj_or_demo']; o2 = r['adj_or_demo_severity']; i1 = r['ipw_rd_demo']; i2 = r['ipw_rd_demo_severity']
            md.append(f"| {pop_name} | {exp_name} | {c['d1']:,} / {c['n1']:,} ({c['p1']*100:.1f}) | {c['d0']:,} / {c['n0']:,} ({c['p0']*100:.1f}) | "
                      f"{c['rd_pp']:+.1f} [{c['rd_ci'][0]:+.1f}, {c['rd_ci'][1]:+.1f}] | {o1['or']:.2f} [{o1['ci'][0]:.2f}, {o1['ci'][1]:.2f}] | "
                      f"{o2['or']:.2f} [{o2['ci'][0]:.2f}, {o2['ci'][1]:.2f}] | {i1['rd_pp']:+.1f} [{i1['ci'][0]:+.1f}, {i1['ci'][1]:+.1f}] | {i2['rd_pp']:+.1f} [{i2['ci'][0]:+.1f}, {i2['ci'][1]:+.1f}] |")
        md += ['', '† age, sex, first care unit, admission type, charting era.  ‡ plus vasopressor use and mechanical ventilation in the window.', '',
               '## Baseline characteristics (all adults alive at hour W)', '',
               '| characteristic | lactate yes | lactate no | BE yes | BE no | SOFA computable | not computable |', '|---|---|---|---|---|---|---|']
        keys = [('n', 'N'), ('y28_pct', '28-day mortality %'), ('age_mean', 'age, mean'), ('male_pct', 'male %'), ('emergency_pct', 'emergency admission %'),
                ('metavision_pct', 'MetaVision era %'), ('unit_MICU_pct', 'MICU %'), ('unit_SICU_pct', 'SICU %'), ('unit_CCU_pct', 'CCU %'), ('unit_CSRU_pct', 'CSRU %'),
                ('unit_TSICU_pct', 'TSICU %'), ('vaso_pct', 'vasopressor in window %'), ('vent_pct', 'ventilated in window %'), ('gcs_min_median', 'GCS min, median'),
                ('map_min_median', 'MAP min, median'), ('creatinine_max_median', 'creatinine max, median'), ('sofa_computable_pct', 'SOFA computable %'),
                ('sofa_total_median_if_computable', 'SOFA total (if computable), median')]
        B = [res['analyses'][f'all adults | {e}']['baseline'] for e in ('lactate recorded', 'BE recorded', 'SOFA computable')]
        for key, label in keys:
            vals = [b[grp][key] for b in B for grp in ('exposed', 'unexposed')]
            md.append(f'| {label} | ' + ' | '.join(f'{v:,.0f}' if key == 'n' else f'{v:.1f}' for v in vals) + ' |')
        md += ['', '## Is prediction "easier" in the tested group? (test-split AUROC of a covariate-only model; SOFA score AUROC where computable)', '',
               '| population | exposure | covariate model: exposed | unexposed | SOFA score: exposed (n) | unexposed (n) |', '|---|---|---|---|---|---|']
        for k, r in res['analyses'].items():
            if 'skipped' in r:
                continue
            d = r['discrimination']; f = lambda v: '–' if v is None else f'{v:.3f}'
            md.append(f"| {k.replace(' | ', ' | ')} | {f(d['covariate_model_auroc_exposed'])} | {f(d['covariate_model_auroc_unexposed'])} | "
                      f"{f(d['sofa_score_auroc_exposed'])} ({d['sofa_score_n_exposed']:,}) | {f(d['sofa_score_auroc_unexposed'])} ({d['sofa_score_n_unexposed']:,}) |")
        open(OUT / f'summary_W{W}.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md), flush=True)


if __name__ == '__main__':
    main()
