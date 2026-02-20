_base_ = [
    '../deit_cifar100/deit_t_cifar100.py'
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
vitkd = False

# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'configs/pretrain/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth',
    teacher_cfg = 'configs/deit_cifar100/deit_s_cifar100.py',
    student_cfg = 'configs/deit_cifar100/deit_t_cifar100.py',
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
                                      alpha=0.5,)
                                ]
                        )
                   ]
    )
