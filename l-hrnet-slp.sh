#!/bin/bash

# Sunlamp (Lite-HRNet-18, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-18_speedplus-640x640_slp_to_slp.py

# Sunlamp (Lite-HRNet-30, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-30_speedplus-640x640_slp_to_slp.py