# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="CustomizedTextLoggerHook", by_epoch=False),
        dict(type="VisualizationHook", by_epoch=False),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
runner = dict(
    type="DynamicIterBasedRunner", is_dynamic_ddp=True, pass_training_status=True
)
find_unused_parameters = True
log_level = "INFO"
load_from = None
resume_from = None 
workflow = [("train", 1)]
cudnn_benchmark = True





























'''###____pretty_text____###'''



'''
log_config = dict(
    interval=100,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        dict(type='VisualizationHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)
find_unused_parameters = False
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
'''
