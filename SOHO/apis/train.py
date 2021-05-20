import random
import re
import os

import numpy as np
import torch
import torch.distributed as dist
from commons.parallel import MMDataParallel, MMDistributedDataParallel
from commons.runner import DistSamplerSeedHook, EpochBasedRunner, obj_from_dict,load_checkpoint

from SOHO.datasets import build_dataloader
from SOHO.hooks import build_hook, DistOptimizerHook
from SOHO.utils import get_root_logger, print_log
from .. import optimizers
from apex import amp


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)
    if cfg.load_from:
        load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False, logger=logger)
        print_log("*****loading from {} to init the model****".format(cfg.load_from),logger)

    # start training
    if distributed:
        _dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)
    else:
        _non_dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        opt=obj_from_dict(optimizer_cfg, optimizers,
                      dict(params=model.parameters()))
        return opt
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'): # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                "{} not in optimizer_cfg".format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print_log('paramwise_options -- {}: {}={}'.format(
                                name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        opt=optimizer_cls(params, **optimizer_cfg)
        return opt


def _dist_train(model, dataset, cfg, logger=None, timestamp=None, meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True,drop_last=getattr(cfg.data, 'drop_last', False))
        for ds in dataset
    ]

    # put model on gpus
    optimizer = build_optimizer(model, cfg.optimizer)

    gpu_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    model = model.cuda()
    if cfg.fp_16.enable:
        model,optimizer = amp.initialize(model, optimizer,
                                          opt_level=cfg.fp_16.opt_level,
                                          loss_scale=cfg.fp_16.loss_scale,
                                          max_loss_scale=cfg.fp_16.max_loss_scale)
        print_log('**** Initializing mixed precision done. ****')
    model = MMDistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        )

    # build runner

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        resume_from=cfg.resume_from,
        fp_16=cfg.fp_16,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        common_params = dict(dist_mode=True)
        runner.register_hook(build_hook(hook, common_params))

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False)) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        resume_from=cfg.resume_from,
        load_from=cfg.load_from,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        common_params = dict(dist_mode=False)
        runner.register_hook(build_hook(hook, common_params))

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
