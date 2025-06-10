#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
# export WANDB_MODE=offline

deepspeed src/train.py \
    --deepspeed ./scripts/zero2.json \
    --text_model nlpie/tiny-clinicalbert \
    --pass_num 2 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --dataloader_num_workers 8 \
    --output_dir output/Text3DSAM \
    --run_name Training'