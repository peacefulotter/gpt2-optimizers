#!/bin/bash

export WANDB_API_KEY="MY_KEY"
python ./src/main.py \
    --model base \
    --opt signsgd \
    --n_embd 768 \
    --n_head 12 \
    --wandb \
    --wandb_project llm-baselines \
    --wandb_run_prefix h768_nh12_nlyr12_sl512_d005_signsgd \
    --lr 0.0001 \
    --n_layer 24 \
    --batch_size 50 \
    --sequence_length 512 \
    --acc_steps 4 \
    --dropout 0.05