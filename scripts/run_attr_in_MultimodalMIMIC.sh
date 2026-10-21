#!/usr/bin/env bash
# UTDE attribution on the SOFA-28d test split. Usage: GPU=1 W=24 bash run_attr.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_HOME=/mnt/nas5/AIBL-Research/hskim/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export CLINICAL_LONGFORMER_PATH=$(ls -d $HF_HOME/hub/models--yikuan8--Clinical-Longformer/snapshots/*/ | head -1)
export ATTR_OUT=../mimic-icu-mortality/results/fusion_shap
W=${W:-24}
[ -f "$ATTR_OUT/utde_attr_W${W}.npz" ] && { echo "exists: utde_attr_W${W}.npz, skipping"; exit 0; }
/mnt/nas5/AIBL-Research/hskim/.envs/mulehr/bin/python -W ignore utde_attr.py --num_train_epochs 6 --modeltype 'TS_Text' \
    --kernel_size 1 --train_batch_size 2 --eval_batch_size 16 --seed 42 \
    --gradient_accumulation_steps 16 --num_update_bert_epochs 2 --bertcount 3 \
    --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
    --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3 \
    --output_dir "run/sofa28_W${W}/" --embed_dim 128 \
    --model_name "bioLongformer" --task 'ihm' --file_path "Data/sofa28/W${W}" \
    --num_labels 2 --num_heads 8 --embed_time 64 --tt_max $W --orig_d_ts 21 --select_metric auc \
    --irregular_learn_emb_text --irregular_learn_emb_ts --reg_ts --TS_mixup --mixup_level batch_seq_feature
