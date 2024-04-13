#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

DATA_DIR=DanceTrack 
DATA_SPLIT=test
NUM_GPUS=2

EXP_NAME=sam_feat_selector__motion_pred_v3
args=$(cat configs/motrv2_sam_feat_selector__motion_pred_v3.args)
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} \
    submit_dance.py ${args} --exp_name outputs/${EXP_NAME}--${DATA_DIR}-${DATA_SPLIT} \
    --resume $1 --data_dir ${DATA_DIR}/${DATA_SPLIT} \
    --local_world_size ${NUM_GPUS}


