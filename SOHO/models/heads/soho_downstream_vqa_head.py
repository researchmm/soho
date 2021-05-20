from ..builder import HEADS,build_loss
from ..utils import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_


@HEADS.register_module
class SOHO_DownStream_VQA_head(nn.Module):
    def __init__(self,
                 loss1=dict(type="BCELoss", loss_weight=1.0),
                 hidden_size=768,
                 num_answes=3129):
        super(SOHO_DownStream_VQA_head, self).__init__()
        # 3129 for vqa and 1852 for GQA
        self.criterion1 = build_loss(loss1)

        self.vqa = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size * 4),
                            nn.ReLU(),
                            nn.Dropout(0.5, inplace=True),
                            nn.Linear(hidden_size * 4, num_answes)
        )

    def init_weights(self):
        for m in self.vqa:
            if isinstance(m,nn.Linear):
                kaiming_normal_(m.weight.data)


    def forward(self,fusion_feature):
        return self.vqa(fusion_feature[:,0,:])

    def loss(self,cls_pred,cls_label):
        losses = dict()
        loss = self.criterion1(cls_pred,cls_label)
        loss *= cls_label.size(1)
        losses['loss_cls']=loss
        losses['score'] = self.compute_score_with_logits(cls_pred, cls_label)
        return losses

    def compute_score_with_logits(self, score, labels):
        logits = torch.max(score.detach(), 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots = one_hots.type_as(labels)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = torch.sum(one_hots * labels, dim=1)
        return scores
