import os

fp_16=dict(
    enable=True,
    opt_level="O1",
    loss_scale="dynamic",
    max_loss_scale=128.
)

# model settings
model = dict(
    type='SOHOSingleStreamPre',
    backbone_pre="torchvision://resnet18",
    language_pre="open-mmlab://bert-base-uncased",
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        with_cp=True,
        style='pytorch'),
    language=dict(
        type='VLBertModel',
        num_hidden_layers=3,
        with_cp=True,
    ),
    neck=dict(
        type='SimpleVDforPreGate',
        in_channels=512,
        out_channels=768,
        num_tokens=2048,
        decay=0.4,
        mask_prob=0.015,
        norm_cfg=dict(type='SyncBN'),
    ),

    head=dict(
        type='MLM_MVM_ITM_head',
        hidden_size=768,
    ))

# dataset settings
dataset_type = 'VisualLanguagePretrainDataset'
data_root = os.path.join(os.getenv("IN_OUT_PATH", './'), 'data/vg_coco_pre/')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromZip'),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'language_tokens', 'mask_labels', 'next_label', 'language_attention','img_target' ]),
]
test_pipeline = [

]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            max_length=16,
            sentence_group=4,
            use_qa=False,
            ann_file='coco_cap_train_pre.json',
            img_prefix=data_root + 'train2014/',
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            max_length=16,
            use_qa=False,
            sentence_group=4,
            ann_file='coco_cap_val_pre.json',
            img_prefix=data_root + 'val2014/',
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            max_length=16,
            sentence_group=4,
            use_qa=False,
            ann_file='vg_cap_pre.json',
            img_prefix=data_root + 'images/',
            pipeline=train_pipeline),
    ],
)

paramwise_options = {
           '\Abackbone.': dict(opt='sgd',lr_mult=100),
            }
optimizer = dict(type='AdamwSGD', lr=0.0001, adamw_weight_decay=0.01, momentum=0.9, sgd_weight_decay=0.0001,paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=dict(max_norm=10.0, norm_type=2),update_interval=2)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[25, 35])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = os.path.join(os.getenv("IN_OUT_PATH", './'),
                        'work_dirs/Pretrain/soho_res18_pre')
load_from = None
resume_from = None
workflow = [('train', 1)]
