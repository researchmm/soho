import math
import torch
from torch.optim.optimizer import Optimizer

class AdamwLARS(Optimizer):
    """
    combine the SGD and Adamw into one optimizer to handle the CNN and transformer
    """
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-6,adamw_weight_decay=0.0,momentum=0.,dampening=0,lars_weight_decay=0.,eta=0.001,nesterov=False,correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if adamw_weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(adamw_weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lars_weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(lars_weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, adamw_weight_decay=adamw_weight_decay, momentum=momentum,
                        dampening=dampening, lars_weight_decay=lars_weight_decay, eta=eta,nesterov=nesterov,
                        correct_bias=correct_bias)
        super(AdamwLARS, self).__init__(params, defaults)

    def step(self,closure=None):
        """
        performs a single step optimization step, for backbone we use the SGD, for transformer we use Adamw
        :param closure:
        :return:
        """
        loss=None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            opt=group.get('opt',None)
            if opt in ['lars',]:
                lars_weight_decay = group['lars_weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                eta = group['eta']
                lars_exclude = group.get('lars_exclude',False)

                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad.data
                    if lars_exclude:
                        local_lr=1.
                    else:
                        weight_norm = torch.norm(p).item()
                        grad_norm = torch.norm(d_p).item()
                        #compute locall leanring rate for this layer
                        local_lr = eta*weight_norm/(grad_norm+weight_norm*lars_weight_decay)
                    actual_lr = local_lr* group['lr']
                    d_p = d_p.add(p.data, alpha=lars_weight_decay).mul(actual_lr)
                    if momentum !=0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer']=torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    p.data.add_(-d_p)
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    state = self.state[p]

                    # state initializetion
                    if len(state)==0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1

                    # decay the first and second moment runing average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    step_size = group['lr']
                    if group['correct_bias']:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state['step']
                        bias_correction2 = 1.0 - beta2 ** state['step']
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)

                    # just adding the square of the weights to the loss function is not
                    # the correct way of using L2 regularization weight decay with Adam
                    # since that will interact with m and v parameters in stranger ways
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group['adamw_weight_decay'] >0.0:
                        p.data.add_(-group['lr'] * group['adamw_weight_decay'], p.data)
        return loss


