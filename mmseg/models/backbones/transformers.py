import numbers
import collections
import logging
import functools
import torch
from torch import nn
from torch.nn import functional as F

from mmseg.ops import resize
import json
import numpy as np
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.shape[0],) + x.shape[2:]).byte()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None].cuda() / dim_t
        pos_y = y_embed[:, :, :, None].cuda() / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PosEncoder(nn.Module):
    def __init__(self, c=1, num_downsample=0, size=None):
        super(PosEncoder, self).__init__()

        self.size = size
        ds_conv = []

        for _ in range(num_downsample):
            ds_conv.append(nn.Conv2d(c,
                                     c,
                                     3,
                                     stride=2,
                                     padding=1,
                                     bias=False))
        ds_conv.append(nn.Conv2d(c, 1, 1))
        self.ds_conv = nn.Sequential(*ds_conv)

        if self.size:
            self.pos_dim = size[0] * size[1] // (4 ** num_downsample)
            self.pos_conv = nn.Conv1d(self.pos_dim, self.pos_dim, 1)

    def forward(self, token_coef):
        N, L, H, W = token_coef.shape
        token_coef = token_coef.reshape(N * L, 1, H, W)

        # interpolation to deal with input with varying sizes
        if self.size:
            token_coef = F.interpolate(token_coef, size=(self.size[0], self.size[1]))

        # downsampling
        token_coef = self.ds_conv(token_coef)
        token_coef = token_coef.view(N, L, -1).permute(0, 2, 1)

        # compress and compute the position encoding.
        if self.size:
            token_coef = self.pos_conv(token_coef)  # N, Cp, L
        return token_coef


class Tokenizer(nn.Module):
    '''
        l: number of tokenizers
        c: channels of feature map
        ct: channels of tokenizers
    '''

    def __init__(self, l, ct, c, head=3, groups=3):
        super(Tokenizer, self).__init__()

        # c -> l, get 2d attention score, It can be seen as a convolutional filter that divides the feature map X
        #   into various regions that corresponds to different semantic concepts.
        self.conv_token_coef = nn.Sequential(nn.Conv2d(c, l, kernel_size=1, padding=0, bias=False),
                                             nn.BatchNorm2d(l))

        # c -> c, get 2d feature (value), maybe not useful
        self.conv_value = nn.Sequential(nn.Conv2d(c, c, kernel_size=1, padding=0, bias=False, groups=groups),
                                        nn.BatchNorm2d(c))

        self.pos_encoding = PosEncoder(size=(16, 16), num_downsample=1)
        # self.pos_sine = PositionEmbeddingSine(num_pos_feats=2)

        self.conv_token = nn.Sequential(nn.Conv1d(c + self.pos_encoding.pos_dim, ct, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm1d(ct))
        self.head = head
        self.c = c
        self.ct = ct

    # feature: N, C, H, W, token: N, CT , L
    def forward(self, feature):
        # pos_sine = self.pos_sine(feature)
        # feature = torch.cat((feature, pos_sine), dim=1)
        token_coef = self.conv_token_coef(feature)  # c -> l, get 2d tokens

        N, L, H, W = token_coef.shape
        token_coef = token_coef.view(N, 1, L, H * W)
        token_coef = token_coef.permute(0, 1, 3, 2)  # N, 1, HW, L
        token_coef = token_coef / np.sqrt(self.c)  # get stable gradient
        token_coef = F.softmax(token_coef, dim=2)  # get attention score along HW

        value = self.conv_value(feature).view(N, self.head, self.c // self.head, H * W)  # N, h, C//h, HW

        # (N, h, C//h, HW) * (N, 1, HW, L) -> N, h, C//h, L
        tokens = torch.matmul(value, token_coef).view(N, self.c, -1)  # N, C, L

        pos_encoding = self.pos_encoding(token_coef.permute(0, 3, 1, 2).reshape(N, L, H, W))  # N, cp (HW), L
        tokens = torch.cat((tokens, pos_encoding), dim=1)  # N, C + Cp (HW), L
        tokens = self.conv_token(tokens)  # N, Ct, L
        return tokens


class Transformer(nn.Module):
    def __init__(self, ct, head=3, kqv_groups=3):
        super(Transformer, self).__init__()
        self.k_conv = nn.Sequential(nn.Conv1d(ct, ct // 2, kernel_size=1, padding=0, bias=False, groups=kqv_groups),
                                    nn.BatchNorm1d(ct // 2))
        self.q_conv = nn.Sequential(nn.Conv1d(ct, ct // 2, kernel_size=1, padding=0, bias=False, groups=kqv_groups),
                                    nn.BatchNorm1d(ct // 2))
        self.v_conv = nn.Sequential(nn.Conv1d(ct, ct, kernel_size=1, padding=0, bias=False, groups=kqv_groups),
                                    nn.BatchNorm1d(ct))
        self.ff_conv = nn.Sequential(nn.Conv1d(ct, ct, kernel_size=1, padding=0, bias=False),
                                     nn.BatchNorm1d(ct))
        self.head = head
        self.ct = ct

    def forward(self, tokens):
        N = tokens.shape[0]  # # N, ct, l
        k = self.k_conv(tokens).view(N, self.head, self.ct // 2 // self.head, -1)  # N, h, ct // 2 // h, l
        q = self.q_conv(tokens).view(N, self.head, self.ct // 2 // self.head, -1)  # N, h, ct // 2 // h, l
        v = self.v_conv(tokens).view(N, self.head, self.ct // self.head, -1)  # N, h, ct // h, l

        # (N, h, l, ct // 2 // h) * (N, h, ct // 2 // h, l) -> N, h, l, l
        kq = torch.matmul(k.permute(0, 1, 3, 2), q)
        kq = F.softmax(kq / np.sqrt(kq.shape[2]), dim=2)

        # (N, h, ct // h, l) * (N, h, l, l) -> N, h, ct // h, l
        kqv = torch.matmul(v, kq).view(N, self.ct, -1)  # N, ct, l
        tokens = tokens + kqv
        tokens = tokens + self.ff_conv(tokens)  # Maybe useless
        return tokens  # N, ct, l


class Projector(nn.Module):
    def __init__(self, CT, C, head=3, groups=3):
        super(Projector, self).__init__()
        self.head = head

        self.proj_value_conv = nn.Sequential(nn.Conv1d(CT, C, 1),
                                             nn.BatchNorm1d(C))
        self.proj_key_conv = nn.Sequential(nn.Conv1d(CT, C, 1),
                                           nn.BatchNorm1d(C))
        # self.proj_query_conv = nn.Conv2d(C, CT, 1, groups=groups)

    def forward(self, feature, token):
        N, _, L = token.shape  # N, ct, l.  feature: N, C, H, W.
        h = self.head

        proj_v = self.proj_value_conv(token).view(N, h, -1, L)  # N, h, c/h, l
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)  # N, h, c/h, l
        proj_q = feature
        # proj_q = self.proj_query_conv(feature)   # N, ct, H, W

        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)  # （N, h, C // h, HW) -> （N, h, HW, C // h）
        proj_coef = F.softmax(torch.matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)  # N, h, HW , L
        proj = torch.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))  # N, h, C//h, HW
        _, _, H, W = feature.shape
        proj = proj.view(N, -1, H, W)
        return feature + proj.view(N, -1, H, W)
