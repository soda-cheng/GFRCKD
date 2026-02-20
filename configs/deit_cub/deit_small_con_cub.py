_base_ = [
    '../_base_/default_runtime.py'
]
#pretrained = 'configs/pretrain/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth'
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='small',
        img_size=224,
        patch_size=16,
        drop_rate=0.04,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        #type='RecycleClsHead',
        num_classes=200,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1)
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        #dict(type='CutMix', alpha=1.0)
    ]),
)


# dataset settings
auto_scale_lr = dict(base_batch_size=128)
# dataset settings
dataset_type = 'CUB'
data_preprocessor = dict(
    num_classes=200,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=300),
    dict(type='RandomCrop', crop_size=224, padding=14),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=300),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='/data1/mazc/gjp/ljy/data/cub200/CUB_200_2011/',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='/data1/mazc/gjp/ljy/data/cub200/CUB_200_2011/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=1.0))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[90, 140], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
val_cfg = dict()
test_cfg = dict()
