_base_ = [
    '../vit_cifar100/vits_pre.py'
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
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'configs/pretrain/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth',
    teacher_cfg = 'configs/vit_cifar100/vitb_pre.py',
    student_cfg = 'configs/vit_cifar100/vits_pre.py',
    distill_cfg = [dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_vitkd',
                                       use_this = vitkd,
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
