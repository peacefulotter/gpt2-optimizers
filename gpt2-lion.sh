#!/bin/bash

export WANDB_API_KEY="20748bf5af66adabd3460ff209b21b9e6c13f954"
python ./src/main.py \
    --model base \
    --opt lion \
    --n_embd 768 \
    --n_head 12 \
    --wandb \
    --wandb_project llm-baselines \
    --wandb_run_prefix h768_nh12_nlyr12_sl512_d005_lion \
    --n_layer 12 \
    --batch_size 50 \
    --sequence_length 512 \
    --acc_steps 4 \
    --dropout 0.05
