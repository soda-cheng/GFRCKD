_base_ = [
    '../_base_/default_runtime.py'
]

# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})
pretrained='configs/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'
# model setting
model = dict(
    type='ImageClassifier',
    #train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
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
        patch_size=8,
        drop_rate=0.1,
        init_cfg=[dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear'),
        	dict(
                type='Pretrained',
                checkpoint=pretrained,
                _delete_=True,
                prefix='backbone')
                ]),
    neck=None)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

# schedule setting
# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=1.0))
# learning policy
param_scheduler = [
	#dict(type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1),
	dict(type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=50),
    	dict(type='CosineAnnealingLR', T_max=200, by_epoch=False,
        	begin=50, end=200)]

# train, val, test setting
# ipu cfg
# model partition config
ipu_model_cfg = dict(
    train_split_edges=[
        dict(layer_to_call='backbone.patch_embed', ipu_id=0),
        dict(layer_to_call='backbone.layers.3', ipu_id=1),
        dict(layer_to_call='backbone.layers.6', ipu_id=2),
        dict(layer_to_call='backbone.layers.9', ipu_id=3)
    ],
    train_ckpt_nodes=['backbone.layers.{}'.format(i) for i in range(12)])

# device config
options_cfg = dict(
    randomSeed=42,
    partialsType='half',
    train_cfg=dict(
        executionStrategy='SameAsIpu',
        Training=dict(gradientAccumulation=32),
        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
    ),
    eval_cfg=dict(deviceIterations=1, ),
)

# add model partition config and device config to runner
runner = dict(
    type='IterBasedRunner',
    ipu_model_cfg=ipu_model_cfg,
    options_cfg=options_cfg,
    max_iters=200)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=50))

fp16 = dict(loss_scale=256.0, velocity_accum_type='half', accum_type='half')
train_cfg = dict(by_epoch=False, max_iters=200)
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
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    #dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    #dict(type='ToHalf', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='PackClsInputs'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    #dict(type='ToHalf', keys=['img']),
    dict(type='Collect', keys=['img'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/data/data/',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
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
