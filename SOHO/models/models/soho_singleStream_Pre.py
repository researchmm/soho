import torch
import torch.nn as nn
import torch.nn.functional as F

from commons.utils import print_log
from .base import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class SOHOSingleStreamPre(BaseModel):
    def __init__(self,
                 backbone,
                 neck=None,
                 language=None,
                 head=None,
                 backbone_pre=None,
                 language_pre=None):
        super(SOHOSingleStreamPre, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        if language is not None:
            self.language = builder.build_language(language)
        else:
            self.language=None

        if head is not None:
            self.head = builder.build_head(head)
        else:
            self.head = None

        l_emb_weight = self.language.bert.embeddings.word_embeddings.weight
        self.mask_l_fc = nn.Linear(l_emb_weight.size(1), l_emb_weight.size(0), bias=False)
        self.mask_l_fc.weight = l_emb_weight
        
        num_tokens = self.neck.vq.num_tokens
      

        self.init_weights(backbone_pre,language_pre)


    def init_weights(self, backbone_pre=None,language_pre=None):
        if backbone_pre is not None:
            print_log('load model from: {}'.format(backbone_pre), logger='root')

        self.backbone.init_weights(pretrained=backbone_pre)
        if self.neck:
            self.neck.init_weights()

        if self.language and language_pre is not None:
            print_log('load language model from: {}'.format(language_pre), logger='root')
        self.language.init_weights(pretrained=language_pre)

        if self.head:
            self.head.init_weights()

    def forward_backbone(self,img):
        x = self.backbone(img)
        return x

    def forward_train(self, img, img_meta, language_tokens, mask_labels, next_label, language_attention, **kwargs):
        bs = img.size(0)
        x = self.forward_backbone(img)

        num_sentence = language_tokens[0].size(0)
        language_tokens = torch.cat(language_tokens,dim=0)
        language_attention = torch.cat(language_attention,dim=0)

        mask_labels = torch.cat(mask_labels,dim=0)

        mask_labels = mask_labels.view(-1,language_tokens.size(1))
        next_label=torch.cat(next_label,dim=0)
        next_label = next_label.view(-1)


        assert language_tokens.size(0) == bs*num_sentence

        x = [item.unsqueeze(dim=1).expand((-1, num_sentence, -1, -1, -1)) for item in x]
        x = [item.contiguous().view(-1, item.size(2), item.size(3), item.size(4)) for item in x]

        visual_tokens, visual_attention,mask_v_labels  = self.neck(x, img_meta)

        fusion_feature=self.language(language_tokens,language_attention,
                                     visual_tokens=visual_tokens,visual_attention_mask=visual_attention)

        mask_l_token_pred = self.mask_l_fc(fusion_feature)# + self.l_bias
        mask_l_token_pred = mask_l_token_pred[:, :language_tokens.size(1), :].contiguous().view(-1, mask_l_token_pred.size(2))

        #with torch.no_grad():
        mask_v_token_pred = fusion_feature[:,language_tokens.size(1):,:].contiguous().view(-1,fusion_feature.size(2))
        mask_v_token_norm = F.normalize(mask_v_token_pred, dim=-1)
        emb = self.neck.vq.embed.data
        emb_norm = F.normalize(emb)
        mask_v_score = torch.matmul(mask_v_token_norm, emb_norm.t())

        # mask_v_token_pred = fusion_feature[:,language_tokens.size(1):,:].contiguous().view(-1,fusion_feature.size(2))
        # mask_v_score = self.mask_v_fc(mask_v_token_pred.detach())

        mask_v_token_pred = mask_v_score



        next_pred=self.head(fusion_feature)
        #tmp=self.neck.vq.curr_temp

        losses = self.head.loss(
            mask_l_token_pred, mask_labels.view(-1),
            mask_v_token_pred, mask_v_labels.view(-1),
            next_pred, next_label)

        #losses['neck_tmp']=torch.tensor([tmp],device=next_pred.device)
        # losses['total_num']=torch.tensor([self.neck.total_num*1.0],device=next_pred.device)


        return losses

    def forward_test(self, img, **kwargs):
        pass
