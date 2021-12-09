import os.path as osp
import time

import mmcv
import torch
from enseg import __version__
from enseg.core import EvalHook, build_optimizers
from enseg.datasets import build_dataloader, build_dataset
from enseg.models.builder import build_network
from enseg.utils import (
    collect_env,
    get_available_gpu,
    get_root_logger,
    parse_args,
    set_random_seed,
)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, IterBasedRunner
from mmcv.utils import Config, build_from_cfg, get_git_hash


def main():
    # ---------------------------------#
    """ Parsing configuration """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = get_available_gpu([0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    logger.info(f"Config:\n{cfg.pretty_text}")
    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: " f"{args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    # ---------------------------------#
    """ Build network, dataloader, and runner """
    # build network
    network = build_network(cfg["network"])  # This function might affect random events
    set_random_seed(cfg.seed, deterministic=args.deterministic)
    network.init_weights()
    # build dataset and dataloader
    dataset = build_dataset(cfg.data.train)
    set_random_seed(cfg.seed, deterministic=args.deterministic)
    dataloader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        drop_last=True,
    )
    dataloaders = [dataloader]
    # make checkpoint config
    if cfg.checkpoint_config is not None:
        # save enseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            enseg_version=f"{__version__}+{get_git_hash()[:7]}",
            config=cfg.pretty_text,
            CLASSES=dataset.CLASSES,
            PALETTE=dataset.PALETTE,
        )
    network.CLASSES = dataset.CLASSES
    meta.update(cfg.checkpoint_config.meta)
    # TODO enable MMDistributedDataParallel
    network = MMDataParallel(network.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    # Build the optimizer separately for each sub model of the network(function: enseg.core.build_optimizers), not as a whole (function: mmcv.runner.build_optimizer)
    optimizer = build_optimizers(network, cfg.optimizer)  # Custom optimizer
    # build runner; Custom runner are not supported
    runner = IterBasedRunner(
        network,
        batch_processor=None,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta,
    )

    # ---------------------------------#
    """ Register hooks """
    # register training hooks
    # Optimizer operations are performed on the network, not the runner
    runner.register_training_hooks(
        cfg.lr_config,
        None,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    runner.timestamp = timestamp
    # build validate dataset
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    # register eval hooks
    eval_cfg = cfg.get("evaluation", {})
    eval_cfg["by_epoch"] = False
    eval_hook = EvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")
    # register custom hook
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # ---------------------------------#
    """ Resume/load and run """
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # ---------------------------------#
    """ Run """
    # set_random_seed(cfg.seed)
    runner.run(dataloaders, cfg.workflow, cfg.total_iters)


if __name__ == "__main__":
    main()
