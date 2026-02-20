_base_ = [
    '../_base_/default_runtime.py'
]
# model setting
model = dict(
    type='ImageClassifier',
    pretrained='configs/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth',
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
        hidden_dim=192),
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=32,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
                ]),
    neck=None)

# schedule setting
# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=1.0))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()


# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)

# dataset settings
dataset_type = 'CIFAR100'
data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/data/data/',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=192,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/data/data/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
