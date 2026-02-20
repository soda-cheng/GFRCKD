_base_ = [
    '../deit_cub/deit_small_cub.py'
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
vitkd = False
gfrckd = True
msekd = False
# method details
model = dict(
    _delete_ = True,
    #type='ClassificationDistiller',
    type='GFRCKDdistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained='configs/pretrain/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth',
    teacher_cfg = 'configs/deit_cub/deit_base_cub.py',
    student_cfg = 'configs/deit_cub/deit_small_cub.py',
    distill_cfg = [ dict(methods=[dict(type='GFRCKDLoss',
                                       name='loss_gfrckd',
                                       use_this = gfrckd,
                                       student_dims = 384,
                                       teacher_dims = 768,
                                       alpha_gfrckd=0.000003,
                                       lambda_gfrckd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='MSEKDLoss',
                                       name='loss_msekd',
                                       use_this=msekd,
                                       student_dims=384,
                                       teacher_dims=768,
                                       alpha_msekd=0.00003,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='ViTKDLoss',
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
