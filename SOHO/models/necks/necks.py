import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import math
import numpy as np
from ..registry import NECKS
from  ..utils import ConvModule,build_norm_layer
from commons.cnn import kaiming_init
from .utils import SOHO_Pre_VD


@NECKS.register_module
class SimpleVDforPre(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 decay=0.1,
                 max_decay=0.99,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 mask_prob=0.015,
                 begin_align=False,
                 pos_align=True
                 ):
        super(SimpleVDforPre, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            activation=activation,
            norm_cfg=norm_cfg,
            inplace=False
        )

        self.ln = nn.LayerNorm(out_channels)
        self.mask_emb = nn.Embedding(1, out_channels)
        self.vq = SOHO_Pre_VD(num_tokens, out_channels, decay=decay,max_decay=max_decay)
        self.num_tokens = num_tokens
        self.out_channels = out_channels
        self.mask_prob = mask_prob
        self.total_num=0
        self.pos_align = pos_align
        self.begin_align = begin_align

        if begin_align:
            self.begin_line = nn.Linear(out_channels,out_channels)

        if pos_align:
            self.pos_line = nn.Linear(out_channels,out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta)
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, img, img_meta):
        img = img[-1]
        x = F.max_pool2d(img, 2, stride=2)
        xq = self.conv(x)
        xq_img = xq

        batch_size, c, h, w = xq.size()
        b=batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        if self.begin_align:
            inputs_flatten=self.begin_line(inputs_flatten)

        quantized_pt, indices = self.vq(inputs_flatten)
        if self.pos_align:
            quantized_pt = self.pos_line(quantized_pt)

        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1))
        embedded_pt = embedded_pt.permute(0,2,1).view(b,-1,h,w)

        embedded_pt = embedded_pt+xq_img

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)

        indices = indices.view(batch_size, 1, h, w).float()
        indices = indices * visual_mask - 100 * (1 - visual_mask)

        tmp = np.random.randint(h * w)
        tmp_label = indices[:, :, tmp // w, tmp % w].view(batch_size, 1, 1, 1)
        masked_indices = (indices == tmp_label).float()
        masked_indices = masked_indices * visual_mask

        probability_matrix = torch.full(tmp_label.shape, self.mask_prob)
        masked_indices2 = torch.bernoulli(probability_matrix).to(device=img.device).float()
        masked_indices = masked_indices * masked_indices2

        # mask_emb = torch.zeros_like(embedded_pt).to(device=xq.device).float()
        mask_emb = self.mask_emb.weight.view(1, self.out_channels, 1, 1)
        embedded_pt = embedded_pt * (1 - masked_indices) + mask_emb * masked_indices
        embedded_pt += pos

        xq = self.ln(embedded_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        labels = indices * masked_indices - 100 * (1 - masked_indices)
        labels = labels.long().view(batch_size, -1)

        xq = xq.view(xq.size(0), xq.size(1), -1).contiguous()
        xq = xq.transpose(1, 2)

        visual_mask = visual_mask.view(batch_size, -1).long()



        return xq, visual_mask,labels

@NECKS.register_module
class SimpleVDforPreGate(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 decay=0.1,
                 max_decay=0.99,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 mask_prob=0.015,
                 begin_align=False,
                 pos_align=True
                 ):
        super(SimpleVDforPreGate, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            activation=activation,
            norm_cfg=norm_cfg,
            inplace=False
        )

        self.ln = nn.LayerNorm(out_channels)
        self.mask_emb = nn.Embedding(1, out_channels)
        self.vq = SOHO_Pre_VD(num_tokens, out_channels, decay=decay,max_decay=max_decay)
        self.num_tokens = num_tokens
        self.out_channels = out_channels
        self.mask_prob = mask_prob
        self.total_num=0
        self.pos_align = pos_align
        self.begin_align = begin_align
        self.gate_fc = ConvModule(
            out_channels*2,
            2,
            1,
            conv_cfg=conv_cfg,
            activation=activation,
            norm_cfg=norm_cfg,
            inplace=False
        )

        if begin_align:
            self.begin_line = nn.Linear(out_channels,out_channels)

        if pos_align:
            self.pos_line = nn.Linear(out_channels,out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta)
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, img, img_meta):
        img = img[-1]
        x = F.max_pool2d(img, 2, stride=2)
        xq = self.conv(x)
        xq_img = xq

        batch_size, c, h, w = xq.size()
        b=batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        if self.begin_align:
            inputs_flatten=self.begin_line(inputs_flatten)

        quantized_pt, indices = self.vq(inputs_flatten)
        if self.pos_align:
            quantized_pt = self.pos_line(quantized_pt)

        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1))
        embedded_pt = embedded_pt.permute(0,2,1).view(b,-1,h,w)

        tmp_feature = torch.cat([embedded_pt,xq_img],dim=1)
        # print("tmp_feature.size()={}".format(tmp_feature.size()))
        tmp_s = self.gate_fc(tmp_feature)
        tmp_score = F.softmax(tmp_s,dim=1)
        emb_score = tmp_score[:,0,:,:].unsqueeze(dim=1)
        img_score = tmp_score[:,1,:,:].unsqueeze(dim=1)


        embedded_pt = embedded_pt*emb_score+xq_img*img_score

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)

        indices = indices.view(batch_size, 1, h, w).float()
        indices = indices * visual_mask - 100 * (1 - visual_mask)

        tmp = np.random.randint(h * w)
        tmp_label = indices[:, :, tmp // w, tmp % w].view(batch_size, 1, 1, 1)
        masked_indices = (indices == tmp_label).float()
        masked_indices = masked_indices * visual_mask

        probability_matrix = torch.full(tmp_label.shape, self.mask_prob)
        masked_indices2 = torch.bernoulli(probability_matrix).to(device=img.device).float()
        masked_indices = masked_indices * masked_indices2

        # mask_emb = torch.zeros_like(embedded_pt).to(device=xq.device).float()
        mask_emb = self.mask_emb.weight.view(1, self.out_channels, 1, 1)
        embedded_pt = embedded_pt * (1 - masked_indices) + mask_emb * masked_indices
        embedded_pt += pos

        xq = self.ln(embedded_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        labels = indices * masked_indices - 100 * (1 - masked_indices)
        labels = labels.long().view(batch_size, -1)

        xq = xq.view(xq.size(0), xq.size(1), -1).contiguous()
        xq = xq.transpose(1, 2)

        visual_mask = visual_mask.view(batch_size, -1).long()



        return xq, visual_mask,labels


@NECKS.register_module
class SimpleVDforVQA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 ):
        super(SimpleVDforVQA, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=activation,
            inplace=False
        )

        self.ln = nn.LayerNorm(out_channels)
        self.out_channels = out_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta)
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, img, img_meta):
        img = img[-1]
        x = F.max_pool2d(img, 2, stride=2)

        xq = self.conv(x)
        xq_img = xq

        batch_size, c, h, w = xq.size()
        b = batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        quantized_pt = inputs_flatten
        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1)) #b,w*h,c
        embedded_pt = embedded_pt.permute(0, 2, 1).view(b, -1, h, w) # b,c,h,w

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)

        xq = embedded_pt + pos
        xq = xq.view(b,-1,h*w).permute(0,2,1) # b,h*w,c
        xq = self.ln(xq)
        visual_mask = visual_mask.view(batch_size, -1).long()


        return xq, visual_mask
