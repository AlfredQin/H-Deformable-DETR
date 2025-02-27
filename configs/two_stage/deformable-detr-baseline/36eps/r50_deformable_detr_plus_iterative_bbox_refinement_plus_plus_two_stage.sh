#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-baseline/36eps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 36 \
    --lr_drop 30 \
    --coco_path '/home/qinc/Dataset/ISAID/iSAID_patches' \
    --num_workers 12 \
    --batch_size 2 \
    --lr 5e-5 \
    --dataset_file 'isaid' \
    --pretrained_coco ./ckpt/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    ${PY_ARGS}
