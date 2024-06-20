#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

DATA_DIR=MOT17 
DATA_SPLIT=images/test
NUM_GPUS=2

EXP_NAME=motrv2_train_mot17_random_drop_finetune_sam_ofs_kf
args=$(cat configs/motrv2_train_mot17_random_drop_finetune_sam_ofs_kf.args)
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} \
    submit_dance.py ${args} --exp_name outputs/${EXP_NAME}--${DATA_DIR}-${DATA_SPLIT} \
    --data_dir ${DATA_DIR}/${DATA_SPLIT} \
    --local_world_size ${NUM_GPUS} \
    --resume /home/nqthuc/Documents/MOT/MOTRv2/exps/motrv2_train_mot17_random_drop_finetune/run13/checkpoint.pth