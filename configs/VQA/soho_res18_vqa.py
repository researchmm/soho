import os

"""
86.07 47.15 58.5 68.57/17
86.09 47.77 58.52 68.66/18
"""

fp_16=dict(
    enable=True,
    opt_level="O0",
    loss_scale="dynamic",
    max_loss_scale=128
)

# model settings
model = dict(
    type='SOHOSingleStreamVQA',
    backbone_pre="torchvision://resnet18",
    language_pre="open-mmlab://bert-base-uncased",
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        with_cp=False,
        style='pytorch'),
    language=dict(
        type='VLBertModel',
        num_hidden_layers=3,
        with_cp=True,
    ),
    neck=dict(
        type='SimpleVDforVQA',
        in_channels=512,
        out_channels=768,
        num_tokens=2048,
        norm_cfg=dict(type='SyncBN'),
        # activation='relu'
    ),

    head=dict(
        type='SOHO_DownStream_VQA_head',
        hidden_size=768,
    ))

# dataset settings
dataset_type = 'VisualLanguageDownstreamVQA'
data_root = os.path.join(os.getenv("IN_OUT_PATH", './'), 'data/coco/')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromZip'),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 800)], keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'language_tokens', 'question_ids', 'vqa_labels', 'language_attention', ]),
]
test_pipeline = [
dict(type='LoadImageFromZip'),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'language_tokens', 'question_ids', 'language_attention', ]),
]
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            max_length=16,
            sentence_group=1,
            ann_file='train_data_qa_caption_new_box.json',
            img_prefix=data_root + 'train2014/',
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            max_length=16,
            sentence_group=1,
            ann_file='val_data_qa_caption_new_box.json',
            img_prefix=data_root + 'val2014/',
            pipeline=train_pipeline),
    ],
    test=dict(
        type=dataset_type,
        data_root=data_root,
        max_length=16,
        sentence_group=1,
        ann_file='test_data_qa.json',
        img_prefix=data_root + 'test2015/',
        pipeline=test_pipeline,
        test_mode=True,
    )
)
# optimizer
paramwise_options = {
           '\Abackbone.': dict(opt='sgd',lr_mult=100)}
optimizer = dict(type='AdamwSGD', lr=0.0001, adamw_weight_decay=0.01, momentum=0.9, sgd_weight_decay=0.0001,paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=dict(max_norm=4.0, norm_type=2),update_interval=2)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[12, 16])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = os.path.join(os.getenv("IN_OUT_PATH", './'),
                        'work_dirs/VQA/soho_res18_vqa')
#load_from = os.path.join(os.getenv("INIT_PATH", './'),"work_dirs/init_weight/Res18_soho_pretraining.pth")
load_from = 'pretrained/epoch_40.pth'
resume_from = None
workflow = [('train', 1)]

custom_hooks = [
]
