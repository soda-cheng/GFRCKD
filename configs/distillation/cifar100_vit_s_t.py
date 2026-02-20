_base_ = [
    '../vit_cifar100/vitt-pre.py'
]
##Under GFRCKD, if want the model as a student, please convert its header to RecycleClsHead
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

# config settings
wsld = False
dkd = False
kd = False
nkd = False
vitkd = True
gfrckd = False
msekd = False

# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    #type='GFRCKDdistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'configs/pretrain/.pth',
    teacher_cfg = 'configs/vit_cifar100/vits-pre.py',
    student_cfg = 'configs/vit_cifar100/vitt-pre.py',
    distill_cfg = [ dict(methods=[dict(type='GFRCKDLoss',
                                       name='loss_gfrckd',
                                       use_this = gfrckd,
                                       student_dims = 192,
                                       teacher_dims = 384,
                                       alpha_gfrckd=0.00008,
                                       lambda_gfrckd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='MSEKDLoss',
                                       name='loss_msekd',
                                       use_this=msekd,
                                       student_dims=192,
                                       teacher_dims=384,
                                       alpha_vitkd=0.00003,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='ViTKDLoss',
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
                    dict(methods=[dict(type='WSLDLoss',
                                       name='loss_wsld',
                                       use_this = wsld,
                                       temp=2.0,
                                       alpha=2.5,
                                       num_classes=1000,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DKDLoss',
                                       name='loss_dkd',
                                       use_this = dkd,
                                       temp=1.0,
                                       alpha=1.0,
                                       beta=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='NKDLoss',
                                       name='loss_nkd',
                                       use_this = nkd,
                                       temp=1.0,
                                       gamma=1.0,
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
                         )]
    )
