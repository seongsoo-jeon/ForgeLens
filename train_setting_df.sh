#!/bin/bash

EXP_NAME="train_setting_df"

TRAIN_ROOT="./dataset_path/Training/deepfake"
VAL_ROOT="./dataset_path/Training/val"
EVAL_ROOT="./dataset_path/Training/eval"

COMMON_ARGS="--experiment_name ${EXP_NAME} \
    --train_data_root ${TRAIN_ROOT} \
    --val_data_root ${VAL_ROOT} \
    --train_classes . \
    --val_classes . \
    --num_workers 4 \
    --seed 3407 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2"

python train.py \
    ${COMMON_ARGS} \
    --training_stage 1 \
    --stage1_batch_size 32 \
    --stage1_epochs 50 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7

python train.py \
    ${COMMON_ARGS} \
    --training_stage 2 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.000002 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7

python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${EVAL_ROOT} \
    --eval_stage 1 \
    --num_workers 0 \
    --seed 3407

python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${EVAL_ROOT} \
    --eval_stage 2 \
    --num_workers 0 \
    --seed 3407