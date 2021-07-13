from ..builder import HEADS,build_loss
from ..utils import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..losses import CrossEntropyLoss

@HEADS.register_module
class MLM_MVM_ITM_head(nn.Module):
    def __init__(self,
                 loss_mlm=dict(type="CrossEntropyLoss", loss_weight=1.0, reduction=None),
                 loss_mvm=dict(type="CrossEntropyLoss", loss_weight=1.0, reduction=None),
                 loss_itm=dict(type="CrossEntropyLoss", loss_weight=1.0),
                 hidden_size=768):
        super(MLM_MVM_ITM_head, self).__init__()

        #self.criterion_mlm = build_loss(loss_mlm)
        #self.criterion_mvm = build_loss(loss_mvm)
        #self.criterion_itm = build_loss(loss_itm)

        self.criterion_mlm = nn.CrossEntropyLoss(reduction='none')
        self.criterion_mvm = nn.CrossEntropyLoss(reduction='none')
        self.criterion_itm = nn.CrossEntropyLoss(reduction='none')

        self.match_class = nn.Linear(hidden_size, 2,bias=False)

    def init_weights(self):
        if  isinstance(self.match_class,nn.Linear):
            kaiming_normal_(self.match_class.weight.data)


    def forward(self,fusion_feature):
        return self.match_class(fusion_feature[:,0,:])

    def loss(self, mask_l_pred, mask_l_label, mask_v_pred, mask_v_label, next_pred, next_label):
        losses = dict()

        loss_mlm = self.criterion_mlm(mask_l_pred, mask_l_label).sum()
        loss_mvm = self.criterion_mvm(mask_v_pred, mask_v_label).sum()
        eps = 1e-6

        mask_l = mask_l_label>=0
        mask_l_labels_input = mask_l_label[mask_l]
        mask_l_score_input = mask_l_pred[mask_l]
        if len(mask_l_labels_input)<1:
            pass
        else:
            losses['acc_l'] = accuracy(mask_l_score_input.detach(), mask_l_labels_input.detach())

        mask_v = mask_v_label>=0
        mask_v_labels_input = mask_v_label[mask_v]
        #print(mask_v_labels_input.shape, mask_v_pred.shape)
        mask_v_score_input = mask_v_pred[mask_v]
        if len(mask_v_labels_input)<1:
            losses['acc_v'] = torch.Tensor([0.]).to(device=loss_mlm.device)
        else:
            #print(mask_v_score_input.detach().argmax(dim=1), mask_v_labels_input.detach())
            losses['acc_v'] = accuracy(mask_v_score_input.detach(), mask_v_labels_input.detach())

        losses['loss_mlm'] = loss_mlm / (mask_l.float().sum() + eps)
        losses['loss_mvm'] = loss_mvm / (mask_v.float().sum() + eps)


        losses['select'] = torch.tensor([mask_l_labels_input.detach().size(0) / mask_l_label.detach().size(0)], device=mask_l_pred.device)
        losses['loss_match'] = self.criterion_itm(next_pred, next_label)
        losses['acc_match'] = accuracy(next_pred.detach(), next_label.detach())

        return losses