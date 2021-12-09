_base_ = "./deeplabv3plus_r50-d8_wo_aux.py"
network = dict(
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(depth=101),
    train_flow=[("s", 10)],
)
