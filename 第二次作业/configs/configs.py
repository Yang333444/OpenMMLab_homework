# 模型配置
# model settings
#复制于mmpretrain/configs/_base_/models/resnet18.py
model = dict(
    #模型的类型
    type='ImageClassifier',
    # 骨干网络，主要用来特征提取
    backbone=dict(
        #主干网络为Resnet
        type='ResNet',
        #深度网络层数为18
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    # neck用来衔接backbone与head的输入输出
    # 把主干网络提取的每张图的特征转换为1维向量，
    neck=dict(type='GlobalAveragePooling'),
    # head主要用来执行具体的任务
    # 头，将我们得到的一维向量映射到任务所需要的类别树上，
    head=dict(
        # 头的类型
        type='LinearClsHead',
        #分类类别
        num_classes=30,
        in_channels=512,
        #损失函数
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #计算得分最高的5类
        topk=(1, 5),
    init_cfg=dict(type="Pretrained", checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth")
    ))


# 数据集配置
# 复制于mmpretrain/configs/_base_/datasets/imagenet_bs32.py
# dataset settings
# python的配置文件在加载完成之后，中间变量的连接关系都没有了
# 数据集类型
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    #需要与模型的分类类别匹配
    num_classes=30,
    # RGB format normalization parameters（归一化参数）
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    # 加载图片
    dict(type='LoadImageFromFile'),
    # 随机翻转图片
    dict(type='RandomResizedCrop', scale=224),
    # 标签信息
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 打包成样本
    dict(type='PackInputs'),
]

# 每一步都是对样本的处理操作
test_pipeline = [
    # 加载图片
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    # 样本数
    batch_size=32,
    # 处理样本的进程数
    num_workers=5,
    # 数据集信息的配置
    dataset=dict(
        type=dataset_type,
        data_root='data/train',
        # 数据处理流程
        pipeline=train_pipeline),
    # 采样器的配置
    sampler=dict(type='DefaultSampler', shuffle=True),
)

#同上类似
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/valdate',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=1)

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# 规划配置
# 复制于mmpretrain/configs/_base_/schedules/imagenet_bs256.py

# optimizer（优化器）
optim_wrapper = dict(
    # SGD:随机梯度下降
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# learning policy（学习策略）（参数规划器）
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# train, val, test setting（训练、验证、测试的流程设置）
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()


#运行参数配置
# 复制于mmpretrain/configs/_base_/default_runtime.py
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`(为了复现的设置)
randomness = dict(seed=None, deterministic=False)

