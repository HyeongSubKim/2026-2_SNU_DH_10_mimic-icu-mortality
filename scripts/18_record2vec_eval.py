"""Record2Vec step 3: embed the texts with a frozen encoder and evaluate classifiers against the structured baseline.

Conditions (same SOFA cohort, same train/val/test split as 09/10):
  A   structured tier-4 + age/sex features      TabPFN (ref, 10_tabpfn_tiers.py) and XGB (09)
  B0  embedding of the raw serialisation (no LLM summary)         -> LR / TabPFN on the embedding
  B   embedding of the LLM summary (narrative / structured prompt) -> LR / TabPFN
  C   hybrid: structured features + top-k PCs of the summary embedding -> TabPFN
  D   embedding of the summary written from record + notes        -> LR / TabPFN   (if summ_W{W}_notes.jsonl exists)
Embedder: Qwen/Qwen3-Embedding-0.6B (last-token pooling, L2-normalised) by default; --embedder gatortron uses UFNLP/gatortron-base mean pooling.
Outputs results/record2vec/emb_W{W}_{text}_{embedder}.npy, metrics_W{W}_{embedder}.json, summary_W{W}_{embedder}.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'record2vec'
TIER = '4 + lactate + base excess'
EMBEDDERS = {'qwen3': 'Qwen/Qwen3-Embedding-0.6B', 'qwen3_4b': 'Qwen/Qwen3-Embedding-4B', 'gatortron': 'UFNLP/gatortron-base'}


def paired_boot(y, p_a, p_b, n=2000, seed=0):
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def embed(texts, name, device, batch=16, max_len=2048):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBEDDERS[name], padding_side='left' if name.startswith('qwen3') else 'right')
    model = AutoModel.from_pretrained(EMBEDDERS[name], torch_dtype=torch.bfloat16 if name.startswith('qwen3') else torch.float32).to(device).eval()
    max_len = min(max_len, 512) if name == 'gatortron' else max_len            # GatorTron has 512 positions
    order = np.argsort([-len(t) for t in texts]); out = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
            h = model(**enc).last_hidden_state
            if name.startswith('qwen3'):
                e = h[:, -1]                                                   # left padding -> last token is EOS
            else:
                m = enc['attention_mask'].unsqueeze(-1).to(h.dtype); e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = e[j]
            if s % (batch * 50) == 0:
                print(f'    embedded {s}/{len(texts)}', flush=True)
    return np.stack(out)


def tabpfn_fit_predict(Xtr, ytr, Xev, device):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device=device, model_path=TABPFN_CKPT, n_estimators=8, random_state=0, ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1]


def lr_fit_predict(Xtr, ytr, Xva, yva, Xte):
    best = None
    for C in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
        m = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(Xtr, ytr)
        a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, C, m)
    return best[2].predict_proba(Xte)[:, 1], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--embedder', default='qwen3', choices=list(EMBEDDERS))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--pcs', type=int, default=32)
    ap.add_argument('--embed-only', action='store_true', help='only write the embedding caches (run in an env with transformers>=4.51)')
    args = ap.parse_args()
    W = args.window
    global TABPFN_CKPT
    import glob, os
    hf = Path(os.environ.get('HF_HOME', '/mnt/nas5/AIBL-Research/hskim/.cache/huggingface'))
    TABPFN_CKPT = (sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt'))) + [None])[0]

    feat = pd.read_parquet(ROOT / 'results' / 'sofa_tiers_demo' / f'features_W{W}.parquet')
    tp = pd.read_parquet(ROOT / 'results' / 'sofa_tiers_demo' / f'tabpfn_preds_W{W}.parquet')[TIER]   # val + test probabilities
    ser = pd.read_parquet(OUT / f'serialized_W{W}.parquet')
    ids = ser.index.values
    y = ser.y28.values.astype(int); split = ser.dataset.values
    tr, va, te = split == 'train', split == 'val', split == 'test'
    print(f'W={W}: n={len(ids)} train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}, pos rate test {y[te].mean():.3f}', flush=True)

    # structured feature matrix (tier 4 + age/sex) in the same row order
    tier_cols = [c for c in feat.columns if c not in ('SUBJECT_ID', 'dataset', 'y28')]
    Xs = feat.loc[ids, tier_cols].values.astype(float)
    Xs_f = np.where(np.isnan(Xs), np.nanmedian(Xs[tr], axis=0), Xs)

    texts = {'raw': ser.text.to_dict()}
    for style in ('narrative', 'structured', 'notes'):
        p = OUT / f'summ_W{W}_{style}.jsonl'
        if p.exists():
            d = {json.loads(l)['ICUSTAY_ID']: json.loads(l)['summary'] for l in open(p)}
            if all(i in d for i in ids):
                texts[style] = d
            else:
                print(f'  {style}: {sum(i in d for i in ids)}/{len(ids)} summaries, skipped', flush=True)

    res = {'window': W, 'embedder': args.embedder, 'n_test': int(te.sum()), 'conditions': {}}
    preds = {}
    # A: references
    tpt = tp.loc[ids].values
    preds['A_tabpfn_structured'] = tpt[te]; tp_va = tpt[va]
    for text_name, tdict in texts.items():
        cache = OUT / f'emb_W{W}_{text_name}_{args.embedder}.npy'
        if cache.exists():
            E = np.load(cache)
        else:
            print(f'  embedding {text_name} ({len(ids)} texts) with {args.embedder}', flush=True)
            E = embed([tdict[i] for i in ids], args.embedder, args.device); np.save(cache, E)
        if args.embed_only:
            continue
        cond = {'raw': 'B0', 'narrative': 'B_narr', 'structured': 'B_struct', 'notes': 'D_notes'}[text_name]
        p_lr, C = lr_fit_predict(E[tr], y[tr], E[va], y[va], E[te]); preds[f'{cond}_lr'] = p_lr
        pca = PCA(n_components=args.pcs, random_state=0).fit(E[tr]); Z = pca.transform(E)
        preds[f'{cond}_tabpfn_pca{args.pcs}'] = tabpfn_fit_predict(Z[tr], y[tr], Z[te], args.device)
        preds[f'C_{cond}_tabpfn_struct+pca{args.pcs}'] = tabpfn_fit_predict(np.hstack([Xs_f[tr], Z[tr]]), y[tr], np.hstack([Xs_f[te], Z[te]]), args.device)
        # late fusion of structured TabPFN and the embedding LR (stacker fitted on val, as in 14)
        p_lr_va = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(E[tr], y[tr]).predict_proba(E[va])[:, 1]
        st = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[logit(np.clip(tp_va, 1e-6, 1 - 1e-6)), logit(np.clip(p_lr_va, 1e-6, 1 - 1e-6))], y[va])
        preds[f'F_tabpfn+{cond}_lr'] = st.predict_proba(np.c_[logit(np.clip(preds['A_tabpfn_structured'], 1e-6, 1 - 1e-6)), logit(np.clip(p_lr, 1e-6, 1 - 1e-6))])[:, 1]
        res['conditions'][cond] = {'lr_C': C}

    if args.embed_only:
        return
    yt = y[te]; ref = preds['A_tabpfn_structured']
    rows = []
    for k, p in preds.items():
        d, ci = paired_boot(yt, p, ref) if k != 'A_tabpfn_structured' else (0.0, [0.0, 0.0])
        rows.append({'model': k, 'auroc': roc_auc_score(yt, p), 'auprc': average_precision_score(yt, p), 'd_auroc_vs_A': d, 'ci': ci})
        np.save(OUT / f'pred_W{W}_{k}_{args.embedder}.npy', p)
    res['results'] = rows
    json.dump(res, open(OUT / f'metrics_W{W}_{args.embedder}.json', 'w'), indent=1)
    md = [f'# Record2Vec, W = {W} h, embedder = {EMBEDDERS[args.embedder]}  (test n = {te.sum()}, 28-day death {yt.mean():.3f})', '',
          '| condition | AUROC | AUPRC | ΔAUROC vs A (95% CI) |', '|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['model']} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['d_auroc_vs_A']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
    open(OUT / f'summary_W{W}_{args.embedder}.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    main()
