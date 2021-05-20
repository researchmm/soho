import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import math
from ..registry import NECKS
from  ..utils import ConvModule,build_norm_layer
from commons.cnn import kaiming_init
from .utils import SOHO_direct_VD
import torch.distributed as dist


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
            # norm_cfg=dict(type='BN'),
            norm_cfg=norm_cfg,
            activation=activation,
            inplace=False
        )

        self.ln = nn.LayerNorm(out_channels)
        self.vq = SOHO_direct_VD()
        self.num_tokens = num_tokens
        self.out_channels = out_channels
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h, w = img_meta[0]['pad_shape'][:2]
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

        batch_size, c, h, w = xq.size()
        b = batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        # indices = self.vq(xq)
        quantized_pt, indices = self.vq(inputs_flatten)

        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1)) #b,w*h,c
        embedded_pt = embedded_pt.permute(0, 2, 1).view(b, -1, h, w) # b,c,h,w

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)




        xq = embedded_pt+pos
        xq = xq.view(b,-1,h*w).permute(0,2,1) # b,h*w,c
        xq = self.ln(xq)
        visual_mask = visual_mask.view(batch_size, -1).long()


        return xq, visual_mask