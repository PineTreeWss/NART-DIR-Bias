#!/bin/bash
#SBATCH --job-name          DT
#SBATCH --time              48:00:00
#SBATCH --cpus-per-task     16
#SBATCH --gres              gpu:4
#SBATCH --mem               250G
#SBATCH --output            DA-T-En-De.%j.out
#SBATCH --partition         a100_batch
data_dir=../../data-bin/wmt16-de-en-original-skd-preprocessed
checkpoint_dir=../../checkpoints/wmt16-de-en-bi-skd-P0-Upp8

fairseq-train ${data_dir}  \
    --user-dir ../DA-Transformer/fs_plugins \
    --task translation_lev_modified  --noise full_mask \
    --arch bi_glat_decomposed_link_base \
    --decoder-learned-pos --encoder-learned-pos \
    --share-all-embeddings --activation-fn gelu \
    --apply-bert-init \
    --links-feature feature:position:share --decode-strategy lookahead \
    --max-source-positions 128 --max-target-positions 1024 --src-upsample-scale 8.0 \
    \
    --criterion bi_nat_dag_loss \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.5:0.1@200k --glance-strategy number-random \
    \
    --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 \
    --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
    --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --ddp-backend c10d --torch-dag-loss --torch-dag-best-alignment --torch-dag-logsoftmax-gather  \
    \
    --no-epoch-checkpoints \
    --max-tokens 1639  --update-freq 10 --grouped-shuffling \
    --max-update 300000 --max-tokens-valid 1639 \
    --save-interval 1  --save-interval-updates 2215  \
    --keep-interval-updates 200 \
    --skip-invalid-size-inputs-valid-test \
    --seed 0 \
    --save-dir ${checkpoint_dir} 