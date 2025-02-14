#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python pytorch/main.py \
    --name "mri_grappa" \
    --epochs 6000 \
    --batch-size 128 \
    --lr 3e-3 \
    --optim adam \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_grappa \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type two_times_mask \
    --mri-train-type forward \
    --single-coil \
    model SL \
    --class-type toeplitz_symmetric \
    --r 2 \
    