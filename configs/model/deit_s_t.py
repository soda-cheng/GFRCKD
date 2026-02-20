_base_ = [
    '../deit/deit-tiny_pt-4xb256_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

# config settings
wsld = False
dkd = False
kd = True
nkd = False
vitkd = True

# method details
model = dict(
    _delete_ = True,
    type='MARKDdistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'configs/pretrain/deit-small-distilled_3rdparty_pt-4xb256_in1k_20211216-4de1d725.pth',
    teacher_cfg = 'configs/deit/deit-small_pt-4xb256_in1k.py',
    student_cfg = 'configs/deit/deit-tiny_pt-4xb256_in1k.py',
    distill_cfg = [dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_vitkd',
                                       use_this = vitkd,
                                       student_dims = 192,
                                       teacher_dims = 384,
                                       alpha_vitkd=0.00003,
                                       beta_vitkd=0.000003,
                                       lambda_vitkd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = kd,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),

                   ]
    )
