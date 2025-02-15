#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python pytorch/main.py \
    --name "mri_grappa" \
    --epochs 2000 \
    --batch-size 1 \
    --lr 1e-4 \
    --lr-decay 1 \
    --optim adam \
    --log-freq 50 \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_grappa \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type four_times_mask \
    --mri-train-type inverse \
    model SL \
    --class-type toeplitz_symmetric \
    --r 4 \
    --is-complex \
    --dim 2
    