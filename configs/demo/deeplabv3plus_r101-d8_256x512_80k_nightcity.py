_base_ = "./demo1.py"
network = dict(
    pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101), train_flow=[("s",10)]
)
