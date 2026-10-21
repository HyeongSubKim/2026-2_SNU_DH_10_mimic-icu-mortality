"""Attribution for the trained UTDE model (TS_Text) on the SOFA-28d test split: modality-level Shapley values (2 players:
time series, notes -> exact from 4 forward passes) and per-channel occlusion (zero a channel's values and mask in both the
irregular and the imputed-grid inputs).  Notes are "removed" by replacing them with the placeholder note used for stays
without notes during training (in-distribution).  Same CLI arguments as run_sofa28.sh; W given by --tt_max.
Writes <out_dir>/utde_attr_W{W}.npz with per-case probabilities for every configuration.
"""
import os, sys, json, pickle, numpy as np, torch
from util import *
from model import *
from data import *
from checkpoint import *

PLACEHOLDER = 'No clinical note available in this window.'
CHANNELS = 'HR,RR,SpO2,Temp,SBP,DBP,MeanBP,ASBP,ADBP,MeanABP,FiO2,O2flow,PaO2,Platelets,Bilirubin,Creatinine,Lactate,BaseExcess,GCS,Urine,Vasopressor'.split(',')


def main():
    out_dir = os.environ['ATTR_OUT']
    args = parse_args(); set_seed(args.seed); make_save_dir(args)
    device = torch.device('cuda')
    BioBert, _, tokenizer = loadBert(args, device)
    _, _, loader = data_perpare(args, 'test', tokenizer)
    model = MULTCrossModel(args=args, device=device, orig_d_ts=args.orig_d_ts, orig_reg_d_ts=2 * args.orig_d_ts, orig_d_txt=768,
                           ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes, Biobert=BioBert).to(device)
    ck = torch.load(os.path.join(args.ck_file_path, 'auc', f'{args.seed}.pth.tar'), map_location=device)
    model.load_state_dict(ck['network']); model.eval()
    enc = tokenizer.encode_plus(PLACEHOLDER, add_special_tokens=True, return_attention_mask=True)
    ph_ids = torch.tensor(enc['input_ids']); ph_mask = torch.tensor(enc['attention_mask']); ph_mask[0] += 1   # Longformer global [CLS]

    batches = list(loader)                      # tokenise once; every occlusion pass re-uses the same tensors
    print('batches materialised', len(batches), flush=True)

    def run(drop_notes=False, drop_ts=False, drop_channel=None):
        probs, labels = [], []
        for batch in batches:
            ts, ts_mask, ts_tt, reg_ts, ids, am, nt, ntm, label = batch
            ts, ts_mask, reg_ts = ts.clone(), ts_mask.clone(), reg_ts.clone()
            D = args.orig_d_ts
            if drop_ts:
                ts.zero_(); ts_mask.zero_(); reg_ts.zero_()
            if drop_channel is not None:
                c = drop_channel; ts[:, :, c] = 0; ts_mask[:, :, c] = 0; reg_ts[:, :, c] = 0; reg_ts[:, :, D + c] = 0
            if drop_notes:
                B = ts.shape[0]; L = len(ph_ids)
                ids = torch.zeros(B, args.num_of_notes, L, dtype=torch.long); am = torch.zeros_like(ids)
                ids[:, 0] = ph_ids; am[:, 0] = ph_mask
                nt = torch.zeros(B, args.num_of_notes); nt[:, 0] = 1.0
                ntm = torch.zeros(B, args.num_of_notes, dtype=torch.long); ntm[:, 0] = 1
            with torch.no_grad():
                p = model(x_ts=ts.to(device), x_ts_mask=ts_mask.to(device), ts_tt_list=ts_tt.to(device), input_ids_sequences=ids.to(device),
                          attn_mask_sequences=am.to(device), note_time_list=nt.to(device), note_time_mask_list=ntm.to(device), reg_ts=reg_ts.to(device))
            probs.append(p.float().cpu().numpy()); labels.append(label.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    res = {}
    res['full'], y = run(); print('full done, auc', roc_auc_score(y, res['full']), flush=True)
    res['no_notes'], _ = run(drop_notes=True); print('no_notes auc', roc_auc_score(y, res['no_notes']), flush=True)
    res['no_ts'], _ = run(drop_ts=True); print('no_ts auc', roc_auc_score(y, res['no_ts']), 'nan', int(np.isnan(res['no_ts']).sum()), flush=True)
    res['none'], _ = run(drop_notes=True, drop_ts=True); print('none done', flush=True)
    for c, name in enumerate(CHANNELS):
        res[f'drop_{name}'], _ = run(drop_channel=c); print(f'drop {name} auc {roc_auc_score(y, res[f"drop_{name}"]):.4f}', flush=True)
    np.savez(os.path.join(out_dir, f'utde_attr_W{args.tt_max}.npz'), y=y, channels=np.array(CHANNELS), **res)
    print('saved', flush=True)


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    main()
