#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-baseline/50eps/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_rego
PY_ARGS=${@:1}

python -u train_deform_detr_rego.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --use_rego \
    --dim_feedforward 2048 \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 50 \
    --lr_drop 40 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --coco_path '/home/qinc/Dataset/ISAID/iSAID_patches' \
    --num_workers 12 \
    --batch_size 2 \
    --lr 5e-5 \
    --dataset_file 'isaid' \
    --pretrained_coco ./ckpt/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    ${PY_ARGS}
