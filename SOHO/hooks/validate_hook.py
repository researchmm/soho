from commons.runner import Hook

import time
import torch
from torch.utils.data import Dataset
from SOHO.utils import nondist_forward_collect, dist_forward_collect
from .registry import HOOKS

@HOOKS.register_module
class ValidateHook(Hook):
    def __init__(self,
                 dataset,
                 dist_mode=True,
                 initial=False,
                 interval=1,
                 **eval_kwargs):
        from SOHO import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            drop_last=False,
            shuffle=False)
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if isinstance(self.interval,list):
            if runner.epoch in self.interval:
                self._run_validate(runner)
        else:
            if not self.every_n_epochs(runner, self.interval):
                return
            self._run_validate(runner)

    def _run_validate(self, runner):
        runner.model.eval()
        func = lambda **x: runner.model(mode='test', **x)
        results=None
        if self.dist_mode:
            results = dist_forward_collect(
                func, self.data_loader, runner.rank,
                len(self.dataset))  # dict{key: np.ndarray}
        else:
            results = nondist_forward_collect(func, self.data_loader,
                                              len(self.dataset))
        if runner.rank == 0:
            self._evaluate(runner, results)
        else:
            time.sleep(30)
        runner.model.train()

    def _evaluate(self, runner, results):
        self.dataset.evaluate(
            results,
            epoch=runner.epoch,
            logger=runner.logger,
            **self.eval_kwargs['eval_param'])
        # eval_res = self.dataset.evaluate(
        #     results,
        #     epoch=runner.epoch(),
        #     keyword=keyword,
        #     logger=runner.logger,
        #     **self.eval_kwargs['eval_param'])
        # for name, val in eval_res.items():
        #     runner.log_buffer.output[name] = val
        # runner.log_buffer.ready = True