# optimizer
# learning policy


lr_config = dict(policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000)

total_iters = 270000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=27000, metric="mIoU", pre_eval=True, save_best="mIoU")






























'''###____pretty_text____###'''



'''
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
total_iters = 270000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(
    interval=27000, metric='mIoU', pre_eval=True, save_best='mIoU')
'''
