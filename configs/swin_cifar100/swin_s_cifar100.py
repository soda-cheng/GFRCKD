_base_ = [
    '../_base_/default_runtime.py'
]
#pretrained = 'configs/pretrain/deit-small-distilled_3rdparty_pt-4xb256_in1k_20211216-4de1d725.pth'
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=32,
        patch_size=4,
        init_cfg=None
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        #type='RecycleDeiTClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False
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
auto_scale_lr = dict(base_batch_size=256)
dataset_type = 'CIFAR100'
data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=8),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/data1/mazc/gjp/ljy/data/cifar100/',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/data1/mazc/gjp/ljy/data/cifar100/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=1.0))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 200], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()
