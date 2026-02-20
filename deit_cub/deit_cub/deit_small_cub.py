_base_ = [
    '../_base_/default_runtime.py'
]
pretrained = 'configs/pretrain/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth'
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='small',
        img_size=224,
        patch_size=16,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')
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
auto_scale_lr = dict(base_batch_size=64)
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
    batch_size=64,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='...',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='...',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    optimizer=dict(
        type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0005, nesterov=True))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=15,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=15,
        end=120,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
val_cfg = dict()
test_cfg = dict()
