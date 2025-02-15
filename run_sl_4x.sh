#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python pytorch/main.py \
    --name "mri_2x" \
    --epochs 4000 \
    --batch-size 1 \
    --lr 8e-5 \
    --lr-decay 0.99995 \
    --optim ams \
    --log-freq 200 \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_4x \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type four_times_mask \
    --operator-type circulant \
    --mri-train-type inverse \
    model SL \
    --class-type toeplitz_symmetric \
    --r 4 \
    --is-complex \
    --dim 2
    