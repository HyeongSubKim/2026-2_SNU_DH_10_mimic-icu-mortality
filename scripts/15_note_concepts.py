"""Do the curated note concepts (N01–N50 term dictionary from the team) add predictive value?

For every stay of the SOFA cohort and window W, the notes available by INTIME + W (same timing rule as the UTDE builder:
STORETIME for bedside notes, CHARTTIME for radiology, from 48 h before admission) are scanned for the 50 concept groups.
Features per group: number of matching notes (`c_<id>`), plus has_note / n_notes / n_words. Negation is handled with a
NegEx-style window (a negation cue within 6 tokens before the term cancels the hit); raw counts are kept as a variant.
L01 (goals-of-care terms) is NOT a feature; it is reported separately as a leakage probe.

Models (same split as 09/10): XGBoost (optuna 50 trials) and TabPFN v2 on
   A  tier 4 + age/sex (structured, reference)
   B  concept features only
   C  A + concept features
and paired-bootstrap ΔAUROC(C − A). Also a univariate table: prevalence and 28-day mortality with / without each concept.
Usage: python scripts/15_note_concepts.py --windows 6 24 --device cuda:0
Output: results/note_concepts/
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'note_concepts'
OUT.mkdir(parents=True, exist_ok=True)
H = pd.Timedelta(1, 'h')
NOTE_CATS = ['Nursing/other', 'Nursing', 'Physician', 'Respiratory', 'General', 'Radiology']

CONCEPTS = {
 'N01': ('metastatic cancer / chemo', 'metastasis; metastatic; metastatic cancer; chemotherapy'),
 'N02': ('cirrhosis / ascites', 'cirrhosis; ascites; paracentesis'),
 'N03': ('chronic lung disease', 'pulmonary fibrosis; home oxygen; home o2'),
 'N04': ('CHF', 'chf; congestive heart failure'),
 'N05': ('frailty / malnutrition', 'dementia; bedridden; nursing home; frail; frailty; cachexia; malnutrition; weight loss'),
 'N06': ('immunocompromised / heme malignancy', 'immunocompromised; immunosuppressed; neutropenia; neutropenic; leukemia; lymphoma; myeloma; hematologic malignancy; bone marrow transplant; bmt; transplant; hiv; aids'),
 'N07': ('MDRO carrier', 'carrier; colonized; mrsa; vre; esbl; cre; crab; mdr; pseudomonas; mdro'),
 'N08': ('cardiac arrest / CPR', 'cardiac arrest; cpr; defibrillation; rosc'),
 'N09': ('hemorrhage', 'hemorrhage; bleeding; melena; hematemesis'),
 'N10': ('coagulopathy / DIC', 'coagulopathy; dic; disseminated intravascular coagulation; thrombocytopenia'),
 'N11': ('shock', 'shock; septic shock; cardiogenic shock; hemorrhagic shock; hypovolemic shock; distributive shock; obstructive shock; refractory shock; circulatory collapse; hemodynamic instability'),
 'N12': ('hypotension / low output', 'hypotension; persistent hypotension; refractory hypotension; despite fluids; fluid-unresponsive; low cardiac output; low ejection fraction; oliguria'),
 'N13': ('hypoperfusion', 'mottling; mottled; cold extremities; delayed capillary refill; hypoperfusion; poor perfusion; tissue hypoxia; occult hypoperfusion; malperfusion; dysoxia; anaerobic metabolism'),
 'N14': ('bowel ischemia', 'ischemic bowel; mesenteric ischemia; bowel infarction; ischemic colitis; gangrenous'),
 'N15': ('obstructive cause', 'massive pulmonary embolism; massive pe; tension pneumothorax'),
 'N16': ('respiratory failure / hypoxemia', 'respiratory failure; respiratory distress; severe hypoxemia; profound hypoxia; decreased oxygen delivery; oxygen delivery failure; severe anemia'),
 'N17': ('mechanical ventilation', 'intubat*; mechanical ventilation; vented; ett'),
 'N18': ('extubation', 'extubat*'),
 'N19': ('multi-organ failure', 'multiple organ failure; multiorgan failure; mof'),
 'N20': ('sepsis / source', 'sepsis; urosepsis; meningitis; pancreatitis; influenza; mrsa'),
 'N21': ('atrial fibrillation', 'atrial fibrillation; afib; a-fib; af'),
 'N22': ('volume overload', 'volume overload; fluid overload'),
 'N23': ('trauma / post-op', 'trauma; post-cardiac surgery; s/p cabg; post-op'),
 'N24': ('ARDS', 'ards; acute respiratory distress syndrome; bilateral infiltrates'),
 'N25': ('acute brain injury', 'intracranial hemorrhage; ich; subarachnoid hemorrhage; sah; subdural; stroke; cva; herniation; anoxic brain injury; hypoxic ischemic'),
 'N26': ('pneumonia / aspiration', 'pneumonia; aspiration; aspiration pneumonia; vap; hcap; hap'),
 'N27': ('infection site', 'disseminated; peritonitis; c.difficile; cdad; c.diff; bacteremia; bsi; perforation; pneumonia; pn; aspiration pneumonia; osteomyelitis; septic arthritis; pancreatitis; meningitis; encephalitis; surgical infection; wound infection; clabsi; cellulitis; cholangitis; urosepsis'),
 'N28': ('unresponsive / poor neuro signs', 'unresponsive; unresponsiveness; fixed pupils; pupils fixed; fixed and dilated; no gag'),
 'N29': ('alert (low risk)', 'alert; awake; perrl'),
 'N30': ('sedated', 'sedated; sedation'),
 'N31': ('ketoacidosis', 'ketoacidosis; dka; diabetic ketoacidosis; ketonuria; starvation ketosis'),
 'N32': ('renal acidosis / AKI', 'aki; acute kidney injury; renal failure; uremic acidosis; esrd; renal tubular acidosis; acetazolamide; topiramate'),
 'N33': ('bicarbonate loss', 'diarrhea; high-output ileostomy; enteric fistula; biliary drainage; ng suction'),
 'N34': ('toxic / metabolic', 'salicylate; methanol; ethylene glycol; metformin; rhabdomyolysis; status epilepticus'),
 'N35': ('fluid overload / edema', 'fluid overload; volume overload; anasarca; third space; third spacing; positive balance; pulmonary edema; edema; edematous'),
 'N36': ('fluid resuscitation', 'fluid resuscitation; fluid bolus; crystalloid; lactated ringer; lr; plasma-lyte; albumin'),
 'N37': ('vasopressor / inotrope', 'vasopressor; pressors; norepinephrine; levophed; epinephrine; vasopressin; phenylephrine; neo; dobutamine; milrinone'),
 'N38': ('bicarbonate given', 'sodium bicarbonate; bicarb drip; bicarb bolus; tham'),
 'N39': ('massive transfusion / emergency surgery', 'massive transfusion; mtp; prbc; ffp; damage control; laparotomy'),
 'N40': ('renal replacement', 'renal replacement therapy; crrt; cvvh; cvvhd; hemodialysis; hd'),
 'N41': ('chloride load', 'normal saline; 0.9% ns; large volume saline; hyperchloremic'),
 'N42': ('ECMO', 'ecmo'),
 'N43': ('worsening trajectory', 'worsening; deteriorating; decompensation; persistent acidosis; refractory; not improving; still acidotic'),
 'N44': ('improving trajectory', 'improving; resolved; corrected; normalized; weaned; clearing; downtrend; responsive to fluids'),
 'N45': ('post-op recovery (low risk)', 'pod; diet; ebl'),
 'N46': ('altered mental status', 'altered mental status; decreased consciousness; lethargic; somnolent; obtunded; unresponsive'),
 'N47': ('delirium', 'delirium; confused; confusion; disoriented; acute confusion'),
 'N48': ('immunocompromised state', 'immunocompromised; immunosuppressed; neutropenia; neutropenic; hiv; aids'),
 'N49': ('hematologic malignancy', 'leukemia; lymphoma; myeloma; hematologic malignancy'),
 'N50': ('transplant', 'bone marrow transplant; bmt; stem cell transplant; solid organ transplant; kidney transplant; liver transplant; heart transplant; lung transplant'),
}
LEAK = ('goals of care (leakage probe, excluded)', 'dnr; dni; comfort care; comfort measures; cmo; hospice; family meeting; poor prognosis; withdrawal of care; withdraw care; goals of care')
NEG = re.compile(r"\b(no|not|without|denies|denied|negative for|neg for|rule out|r/o|ruled out|unlikely|absence of|free of|never|none)\b", re.I)


def compile_terms(spec):
    pats = []
    for t in [s.strip().lower() for s in spec.split(';') if s.strip()]:
        t = re.escape(t).replace(r'\ ', r'\s+').replace(r'\-', r'[-\s]?')
        pats.append(t[:-2] + r'\w*' if t.endswith(r'\*') else t)
    return re.compile(r'(?<![\w/])(?:' + '|'.join(pats) + r')(?![\w/])', re.I)


PATTERNS = {k: compile_terms(v[1]) for k, v in CONCEPTS.items()}
LEAK_PAT = compile_terms(LEAK[1])


def count_hits(text, pat, negate=True):
    n = 0
    for m in pat.finditer(text):
        if negate:
            window = text[max(0, m.start() - 40):m.start()]
            if NEG.search(window):
                continue
        n += 1
    return n


def note_bundle(st, W):
    nt = pd.read_parquet(DATA / 'noteevents_text.parquet')
    nt = nt[nt.CATEGORY.isin(NOTE_CATS) & nt.HADM_ID.isin(set(st.HADM_ID))].merge(st.reset_index()[['HADM_ID', 'ICUSTAY_ID', 'INTIME']], on='HADM_ID')
    avail = pd.to_datetime(nt.STORETIME).fillna(pd.to_datetime(nt.CHARTTIME))
    nt['h'] = (avail - nt.INTIME) / H
    return nt[(nt.h > -48) & (nt.h <= W)]


def concept_features(nt, index):
    rows = {}
    for sid, g in nt.groupby('ICUSTAY_ID'):
        texts = g.TEXT.tolist()
        r = {'has_note': 1, 'n_notes': len(texts), 'n_words': int(sum(len(t.split()) for t in texts))}
        for k, pat in PATTERNS.items():
            r[f'c_{k}'] = sum(1 for t in texts if count_hits(t, pat) > 0)
            r[f'raw_{k}'] = sum(1 for t in texts if pat.search(t))
        r['leak_L01'] = sum(1 for t in texts if count_hits(t, LEAK_PAT) > 0)
        rows[sid] = r
    F = pd.DataFrame.from_dict(rows, orient='index').reindex(index)
    F['has_note'] = F.has_note.fillna(0); F = F.fillna(0)
    return F


def fit_xgb(Xtr, ytr, Xva, yva, n_trials, device, seed=25):
    dtr, dva = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva); best = {'auc': -1}

    def obj(trial):
        p = {'max_depth': trial.suggest_int('max_depth', 2, 8), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'min_child_weight': trial.suggest_float('min_child_weight', 1, 50, log=True), 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True)}
        b = xgb.train({**p, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': device, 'seed': seed},
                      dtr, num_boost_round=1000, evals=[(dva, 'val')], early_stopping_rounds=40, verbose_eval=False)
        a = roc_auc_score(yva, b.predict(dva, iteration_range=(0, b.best_iteration + 1)))
        if a > best['auc']:
            best.update(auc=a, b=b, it=(0, b.best_iteration + 1))
        return a
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed)).optimize(obj, n_trials=n_trials)
    return lambda X: best['b'].predict(xgb.DMatrix(X), iteration_range=best['it']), best


def paired(y, pa, pb, n=2000, seed=0):
    r = np.random.RandomState(seed); d = []
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tabpfn', action='store_true', help='also run TabPFN (needs the tabpfn env)')
    args = ap.parse_args()
    icu = pd.read_parquet(DATA / 'icustays.parquet', columns=['ICUSTAY_ID', 'HADM_ID', 'INTIME']).set_index('ICUSTAY_ID'); icu['INTIME'] = pd.to_datetime(icu.INTIME)
    META = ['SUBJECT_ID', 'dataset', 'y28']
    results, md = {}, ['# Curated note concepts (N01–N50) as features\n']
    for W in args.windows:
        df = pd.read_parquet(ROOT / 'results' / 'sofa_tiers_demo' / f'features_W{W}.parquet')
        st = df[META].join(icu)
        F = concept_features(note_bundle(st, W), df.index)
        F.to_parquet(OUT / f'concepts_W{W}.parquet')
        y = df.y28.to_numpy(); part = {k: (df.dataset == k).to_numpy() for k in ['train', 'val', 'test']}
        struct = [c for c in df.columns if c not in META]
        ccols = [f'c_{k}' for k in CONCEPTS] + ['has_note', 'n_notes', 'n_words']
        rcols = [f'raw_{k}' for k in CONCEPTS] + ['has_note', 'n_notes', 'n_words']
        Xall = df[struct].join(F)
        sets = {'A structured (tier 4 + age/sex)': struct, 'B concepts only (negation-aware)': ccols, 'B0 concepts only (raw counts)': rcols,
                'C structured + concepts (negation-aware)': struct + ccols, 'C0 structured + concepts (raw counts)': struct + rcols}
        res = {'n_test': int(part['test'].sum()), 'models': {}}
        preds = {}
        for name, cols in sets.items():
            pred, best = fit_xgb(Xall.loc[part['train'], cols], y[part['train']], Xall.loc[part['val'], cols], y[part['val']], args.n_trials, args.device)
            p = pred(Xall.loc[part['test'], cols]); preds[name] = p
            res['models'][name] = {'n_features': len(cols), 'val_auroc': float(best['auc']), 'test_auroc': float(roc_auc_score(y[part['test']], p)),
                                   'test_auprc': float(average_precision_score(y[part['test']], p))}
            print(f'W={W} {name:45s} feats {len(cols):3d} val {best["auc"]:.3f} test {res["models"][name]["test_auroc"]:.3f} / {res["models"][name]["test_auprc"]:.3f}', flush=True)
        yt = y[part['test']]
        for name in sets:
            res['models'][name]['d_vs_A'] = paired(yt, preds[name], preds['A structured (tier 4 + age/sex)'])
        # univariate concept table (whole cohort)
        uni = []
        for k, (label, _) in CONCEPTS.items():
            hit = F[f'c_{k}'].to_numpy() > 0
            uni.append({'id': k, 'concept': label, 'prevalence': hit.mean(), 'mort_with': y[hit].mean() if hit.any() else np.nan,
                        'mort_without': y[~hit].mean(), 'auroc_alone': roc_auc_score(y, F[f'c_{k}'].to_numpy()) if hit.any() else np.nan})
        hit = F.leak_L01.to_numpy() > 0
        uni.append({'id': 'L01', 'concept': LEAK[0], 'prevalence': hit.mean(), 'mort_with': y[hit].mean() if hit.any() else np.nan, 'mort_without': y[~hit].mean(),
                    'auroc_alone': roc_auc_score(y, F.leak_L01.to_numpy()) if hit.any() else np.nan})
        uni = pd.DataFrame(uni).set_index('id'); uni.to_csv(OUT / f'univariate_W{W}.csv')
        gain = best  # last model's booster = C0; recompute for C
        results[W] = res
        md.append(f'\n## W = {W} h (test n = {res["n_test"]:,})\n\n| feature set | # features | val AUROC | test AUROC | test AUPRC | ΔAUROC vs A [95 % CI] |\n|---|---|---|---|---|---|\n')
        for name, r in res['models'].items():
            md.append(f"| {name} | {r['n_features']} | {r['val_auroc']:.3f} | {r['test_auroc']:.3f} | {r['test_auprc']:.3f} | {r['d_vs_A'][0]:+.3f} [{r['d_vs_A'][1][0]:+.3f}, {r['d_vs_A'][1][1]:+.3f}] |\n")
        top = uni.drop(index='L01').sort_values('auroc_alone', ascending=False).head(12)
        md.append('\nStrongest single concepts (prevalence, 28-day mortality with / without, AUROC of the count alone):\n\n| id | concept | prev | mort with | mort without | AUROC |\n|---|---|---|---|---|---|\n')
        for k, r in top.iterrows():
            md.append(f"| {k} | {r.concept} | {r.prevalence:.2f} | {r.mort_with:.2f} | {r.mort_without:.2f} | {r.auroc_alone:.3f} |\n")
        L = uni.loc['L01']
        md.append(f"\nLeakage probe L01 (goals-of-care terms, not used): prevalence {L.prevalence:.3f}, mortality with {L.mort_with:.2f} vs without {L.mort_without:.2f}, AUROC alone {L.auroc_alone:.3f}.\n")
    json.dump(results, open(OUT / 'metrics.json', 'w'), indent=1); (OUT / 'summary.md').write_text(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
