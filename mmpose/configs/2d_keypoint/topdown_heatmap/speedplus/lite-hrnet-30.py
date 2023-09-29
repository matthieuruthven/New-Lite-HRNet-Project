# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from mmpose.configs._base_.default_runtime import *

from mmengine.dataset import DefaultSampler
from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, MultiStepLR
from torch.optim import Adam

from mmpose.codecs import UDPHeatmap
from mmpose.datasets import (SpeedPlusDataset, GenerateTarget,
                             LoadImage, PackPoseInputs)
from mmpose.datasets.transforms.common_transforms import RandomBBoxTransform
from mmpose.evaluation import PCKAccuracy
from mmpose.models import (HeatmapHead, LiteHRNet, KeypointMSELoss,
                           PoseDataPreprocessor, TopdownPoseEstimator)

# runtime
train_cfg.update(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type=Adam,
    lr=2e-3,
))

# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=100,
        milestones=[45, 60],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=4)

# hooks
default_hooks.update(checkpoint=dict(save_best='speedplus/PCK', rule='greater'))

# codec settings
codec = dict(
    type=UDPHeatmap, input_size=(640, 640), heatmap_size=(160, 160), sigma=2)

# model settings
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type=LiteHRNet,
        in_channels=3,
        extra=dict(
            stem=dict(  
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320)))),
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_mpii_256x256-faae8bd8_20210622.pth')
    ),
    head=dict(
        type=HeatmapHead,
        in_channels=40,
        out_channels=11,
        deconv_out_channels=None,
        loss=dict(type=KeypointMSELoss, use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = SpeedPlusDataset
data_mode = 'topdown'
data_root = '../space_cluster/speedplus_cropped'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]
val_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='train.json',
        data_prefix=dict(img='image'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='validation.json',
        data_prefix=dict(img='image'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=PCKAccuracy,
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
