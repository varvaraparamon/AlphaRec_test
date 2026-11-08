#!/usr/bin/env bash
# Чекпоинты и stats.txt пишутся в results/, полный лог — в results/msd.log.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

MSYS_NO_PATHCONV=1 docker run --rm --gpus all --shm-size=4g \
    -v "$(pwd -W 2>/dev/null || pwd)/data/General:/app/data/General" \
    -v "$(pwd -W 2>/dev/null || pwd)/results:/app/weights" \
    alpharec \
    python main.py --rs_type General --clear_checkpoints --saveID tau_0.15_bge_mlp_ \
        --dataset msd --model_name AlphaRec --n_layers 2 --patience 20 --cuda 0 --no_wandb \
        --train_norm --pred_norm --neg_sample 256 --lm_model bge_large --model_version mlp \
        --tau 0.15 --infonce 1 \
    2>&1 | grep --line-buffered -v "finish recommend one batch" | tee results/msd.log
