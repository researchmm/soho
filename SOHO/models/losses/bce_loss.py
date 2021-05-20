import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class BCELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = F.binary_cross_entropy_with_logits

    def forward(self,
                input,
                target,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            input,
            target,
            reduction=reduction,
            **kwargs)
        return loss_cls