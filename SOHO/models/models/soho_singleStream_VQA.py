import torch
import torch.nn as nn

from SOHO.utils import print_log
from .base import BaseModel
from .. import builder
from ..registry import MODELS



@MODELS.register_module
class SOHOSingleStreamVQA(BaseModel):
    def __init__(self,
                 backbone,
                 neck=None,
                 language=None,
                 head=None,
                 backbone_pre=None,
                 language_pre=None):
        super(SOHOSingleStreamVQA, self).__init__()

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

    def forward_train(self, img, language_tokens,language_attention,vqa_labels,img_meta, **kwargs):
        bs = img.size(0)
        x = self.forward_backbone(img)
        num_sentence = language_tokens[0].size(0)
        language_tokens = torch.cat(language_tokens,dim=0)
        language_attention = torch.cat(language_attention,dim=0)
        vqa_labels = torch.cat(vqa_labels,dim=0)

        assert language_tokens.size(0) ==bs*num_sentence


        visual_tokens, visual_attention = self.neck(x,img_meta)

        fusion_feature=self.language(language_tokens,language_attention,
                                     visual_tokens=visual_tokens,visual_attention_mask=visual_attention)

        next_pred=self.head(fusion_feature)
        losses=self.head.loss(next_pred,vqa_labels)
        return losses

    def forward_test(self, img, language_tokens,language_attention,img_meta,vqa_labels=None,question_ids=None, **kwargs):
        bs = img.size(0)
        x = self.forward_backbone(img)
        num_sentence = language_tokens[0].size(0)
        language_tokens = torch.cat(language_tokens, dim=0)
        language_attention = torch.cat(language_attention, dim=0)

        assert language_tokens.size(0) == bs * num_sentence


        visual_tokens, visual_attention = self.neck(x,img_meta)

        fusion_feature = self.language(language_tokens, language_attention,
                                       visual_tokens=visual_tokens, visual_attention_mask=visual_attention)

        next_pred = self.head(fusion_feature)
        outs=[question_ids[0],next_pred]
        keys=["ids","pred"]
        out_tensors = [out.cpu() for out in outs]
        return dict(zip(keys,out_tensors))
