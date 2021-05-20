from commons.runner import OptimizerHook
from apex import amp


class DistOptimizerHook(OptimizerHook):

    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        if runner.fp_16.enable:
            runner.outputs['loss'] /= self.update_interval
            if self.every_n_iters(runner, self.update_interval):
                with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if self.grad_clip is not None:
                    grad_norm=self.clip_grads(
                        amp.master_params(runner.optimizer)
                    )
                    if grad_norm is not None:
                        runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                                 runner.outputs['num_samples'])
                runner.optimizer.step()
                runner.optimizer.zero_grad()
            else:
                with amp.scale_loss(runner.outputs['loss'], runner.optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()


        else:
            runner.outputs['loss'] /= self.update_interval
            runner.outputs['loss'].backward()
            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    grad_norm=self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                                 runner.outputs['num_samples'])
                runner.optimizer.step()
                runner.optimizer.zero_grad()
