#!/bin/bash

# Synthetic (Lite-HRNet-18, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-18_speedplus-640x640_syn_to_syn.py

# Synthetic (Lite-HRNet-30, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-30_speedplus-640x640_syn_to_syn.py