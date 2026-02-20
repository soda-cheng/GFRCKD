_base_ = [
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='small',
        img_size=224,
        patch_size=16,
        drop_rate=0.06,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        #type='RecycleClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision')
    ),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2))
    )

# data settings
# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        #data_root='/data/goujp/data/imagenet/ILSVRC2012_img_train',
        data_root='/data1/mazc/gjp/ljy/data/imagenet/ILSVRC2012_img_train',
        #ann_file='/data/goujp/yicheng/data/imagenet/meta/train.txt',
        ann_file='/data1/mazc/gjp/yc/data/imagenet/meta/train.txt',
        data_prefix='',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        #data_root='/data/goujp/data/imagenet/val',
        data_root='/data1/mazc/gjp/yc/data/imagenet/ILSVRC2012_img_val',
        #ann_file='/data/goujp/yicheng/data/imagenet/meta/val.txt',
        ann_file='/data1/mazc/gjp/yc/data/imagenet/meta/val.txt',
        data_prefix='',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule setting
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        type='AdamW',
        # for batch in each gpu is 128, 8 gpu
        # lr = 5e-4 * 128 * 8 / 512 = 0.001
        #lr=5e-4 * 128 * 8 / 512,
        lr=0.0005,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=20)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)