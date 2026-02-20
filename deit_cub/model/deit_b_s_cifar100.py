_base_ = [
    '../deit_cifar100/deit_s_cifar100.py'
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
markd = True

# method details
model = dict(
    _delete_ = True,
    type='MARKDdistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'configs/pretrain/deit-base-distilled_3rdparty_pt-16xb64_in1k_20211216-42891296.pth',
    teacher_cfg = 'configs/deit_cifar100/deit_b_cifar100.py',
    student_cfg = 'configs/deit_cifar100/deit_s_cifar100.py',
    distill_cfg = [dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_markd',
                                       use_this = markd,
                                       student_dims = 384,
                                       teacher_dims = 768,
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
                                      alpha=0.5,)
                                ]
                        )
                   ]
    )
