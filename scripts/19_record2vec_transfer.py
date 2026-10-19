"""Record2Vec step 4: label efficiency and pseudo-site transfer of the representations (structured features vs text embeddings).

Few-shot: fit on n labelled train stays (stratified draws, 5 seeds), test on the full test split.
Transfer: fit on CareVue stays (train+val), test on MetaVision test stays, and the reverse; in-domain = same era train -> same era test.
Classifiers: TabPFN (structured features, median fill) and L2 logistic regression (--C, default 10 = the value selected on the validation split in 18) on the L2-normalised embeddings.
Usage: python scripts/19_record2vec_transfer.py --window 6 --embedder qwen3 --texts raw narrative structured
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'record2vec'
SHOTS = [16, 64, 256, 1024, None]


def fit_predict(kind, Xtr, ytr, Xte, device, ckpt, C=10.0):
    if kind == 'tabpfn':
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=device, model_path=ckpt, n_estimators=8, random_state=0, ignore_pretraining_limits=True).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--embedder', default='qwen3')
    ap.add_argument('--texts', nargs='+', default=['raw', 'narrative', 'structured'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--C', type=float, default=10.0)
    args = ap.parse_args(); W = args.window
    hf = Path(os.environ.get('HF_HOME', '/mnt/nas5/AIBL-Research/hskim/.cache/huggingface'))
    ckpt = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))[0]

    ser = pd.read_parquet(OUT / f'serialized_W{W}.parquet'); ids = ser.index.values
    y = ser.y28.values.astype(int); split = ser.dataset.values
    era = pd.read_parquet(ROOT / 'data' / 'icustays.parquet').set_index('ICUSTAY_ID').loc[ids, 'DBSOURCE'].values
    feat = pd.read_parquet(ROOT / 'results' / 'sofa_tiers_demo' / f'features_W{W}.parquet')
    Xs = feat.loc[ids, [c for c in feat.columns if c not in ('SUBJECT_ID', 'dataset', 'y28')]].values.astype(float)
    Xs = np.where(np.isnan(Xs), np.nanmedian(Xs[split == 'train'], axis=0), Xs)
    reps = {'structured (TabPFN)': ('tabpfn', Xs)}
    for t in args.texts:
        p = OUT / f'emb_W{W}_{t}_{args.embedder}.npy'
        if p.exists():
            reps[f'{t} emb (LR)'] = ('lr', np.load(p))
    tr, te = split == 'train', split == 'test'
    res = {'window': W, 'embedder': args.embedder, 'few_shot': {}, 'transfer': {}}

    for name, (kind, X) in reps.items():
        row = {}
        for n in SHOTS:
            aucs = []
            for seed in range(5 if n else 1):
                if n is None:
                    idx = np.where(tr)[0]
                else:
                    rng = np.random.RandomState(seed); pos, neg = np.where(tr & (y == 1))[0], np.where(tr & (y == 0))[0]
                    k = max(2, int(round(n * y[tr].mean()))); idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)]
                aucs.append(roc_auc_score(y[te], fit_predict(kind, X[idx], y[idx], X[te], args.device, ckpt, args.C)))
            row[str(n or 'all')] = (float(np.mean(aucs)), float(np.std(aucs)))
        res['few_shot'][name] = row
        tdict = {}
        for src, dst in (('carevue', 'metavision'), ('metavision', 'carevue')):
            fit_cross = (split != 'test') & (era == src); fit_in = (split != 'test') & (era == dst); test = te & (era == dst)
            tdict[f'{src}->{dst}'] = float(roc_auc_score(y[test], fit_predict(kind, X[fit_cross], y[fit_cross], X[test], args.device, ckpt, args.C)))
            tdict[f'{dst}->{dst}'] = float(roc_auc_score(y[test], fit_predict(kind, X[fit_in], y[fit_in], X[test], args.device, ckpt, args.C)))
        res['transfer'][name] = tdict
        print(name, row, tdict, flush=True)

    json.dump(res, open(OUT / f'transfer_W{W}_{args.embedder}.json', 'w'), indent=1)
    md = [f'# Label efficiency and era transfer, W = {W} h, embedder = {args.embedder} (test AUROC)', '',
          '| representation | ' + ' | '.join(f'n={s or "all"}' for s in SHOTS) + ' |', '|---|' + '---|' * len(SHOTS)]
    for name, row in res['few_shot'].items():
        md.append(f'| {name} | ' + ' | '.join(f'{m:.3f} ± {s:.3f}' if s else f'{m:.3f}' for m, s in row.values()) + ' |')
    keys = list(next(iter(res['transfer'].values())).keys())
    md += ['', '| representation | ' + ' | '.join(keys) + ' |', '|---|' + '---|' * len(keys)]
    for name, t in res['transfer'].items():
        md.append(f'| {name} | ' + ' | '.join(f'{t[k]:.3f}' for k in keys) + ' |')
    open(OUT / f'transfer_W{W}_{args.embedder}.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
