#!/bin/bash

python pytorch/main.py \
    --name "mri_grappa" \
    --epochs 100 \
    --batch-size 1 \
    --lr 1e-3 \
    --optim adam \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_grappa \
    --case-name file_brain_AXT1_202_2020377 \
    --slice-idx 3 \
     model SL 
    