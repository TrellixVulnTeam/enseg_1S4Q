_base_ = "./demo1.py"
network = dict(train_flow=[("s", 10)])
crop_size = (512, 256)
optimizer = dict(_delete_=True,type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()