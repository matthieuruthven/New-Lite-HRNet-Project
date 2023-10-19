_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=100, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[45, 60],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=4)

# hooks
default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(640, 640), heatmap_size=(160, 160), sigma=2)

# TODO: Load weights from pre-trained model
# resume = True
# load_from = 'weights/litehrnet18_mpii_256x256-cabd7984_20210623.pth'

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
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
                    (40, 80, 160, 320),
                )),
            with_head=True,
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=40,
        out_channels=11,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/data/mruthven/lite-hrnet-dsnt/weights/litehrnet30_mpii_256x256-faae8bd8_20210622.pth'
    ))

# base dataset settings
dataset_type = 'SpeedPlusDataset'
data_mode = 'topdown'
data_root_train = '/data/mruthven/speedplus_640/synthetic'
data_root_val = '/data/mruthven/speedplus_640/synthetic'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='MyRandomFlip', prob=0.75, direction=['horizontal','vertical','diagonal']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        data_mode=data_mode,
        ann_file='train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        data_mode=data_mode,
        ann_file='validation.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='PCKAccuracy')
test_evaluator = val_evaluator